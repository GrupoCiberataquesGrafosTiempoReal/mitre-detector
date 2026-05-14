import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv

# ==============================================================================
# 1. ARQUITECTURA DEL MODELO
# ==============================================================================
class AdvancedEdgeExpert(torch.nn.Module):
    def __init__(self, node_in_channels, edge_in_channels, hidden_channels, out_classes, conv_type='SAGE', dropout_rate=0.3):
        super(AdvancedEdgeExpert, self).__init__()
        self.conv_type = conv_type
        
        if conv_type == 'SAGE':
            self.conv1 = SAGEConv(node_in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, hidden_channels)
            self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        elif conv_type == 'GAT':
            heads = 4
            self.conv1 = GATConv(node_in_channels, hidden_channels // heads, heads=heads)
            self.conv2 = GATConv(hidden_channels, hidden_channels // heads, heads=heads)
            self.conv3 = GATConv(hidden_channels, hidden_channels, heads=1)

        clf_input_dim = (hidden_channels * 2) + edge_in_channels
        self.edge_classifier = nn.Sequential(
            nn.Linear(clf_input_dim, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, out_classes) 
        )

    def forward(self, x, edge_index_msg, edge_index_pred, edge_attr_pred):
        x = F.relu(self.conv1(x, edge_index_msg))
        x = F.relu(self.conv2(x, edge_index_msg))
        x = self.conv3(x, edge_index_msg)
        
        src, dst = edge_index_pred[0], edge_index_pred[1]
        edge_features = torch.cat([x[src], x[dst], edge_attr_pred], dim=-1)
        return self.edge_classifier(edge_features)

# ==============================================================================
# 2. CLASE DETECTOR
# ==============================================================================
class DetectorMITRE:
    def __init__(self, ruta_modelos="../data/processed"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[*] Cargando SOC AI en: {self.device}")
        
        # 1. Cargar Artefactos
        self.scaler_edges = joblib.load(f"{ruta_modelos}/scaler_edges.pkl")
        self.scaler_nodes = joblib.load(f"{ruta_modelos}/scaler_nodes.pkl")
        self.encoder_tactics = joblib.load(f"{ruta_modelos}/encoder_tactics.pkl")
        self.num_clases = len(self.encoder_tactics.classes_)
        self.id_benigno = list(self.encoder_tactics.classes_).index('Benigno')
        
        # 2. Cargar Columnas Esperadas (Para construir el vector automáticamente)
        with open(f"{ruta_modelos}/columnas_modelo.json", "r") as f:
            self.columnas_esperadas = json.load(f)
            
        self.variables_base = ['duration', 'orig_bytes', 'resp_bytes', 'orig_pkts', 'resp_pkts', 'missed_bytes', 'time_since_last_conn']
        self.columnas_dummies = self.columnas_esperadas[len(self.variables_base):]
        
        # 3. Cargar Nodos Históricos
        self.node_df = pd.read_pickle(f"{ruta_modelos}/node_features_historicas.pkl")
        self.x_nodos = torch.tensor(self.scaler_nodes.transform(self.node_df.values), dtype=torch.float).to(self.device)
        self.ip_to_id = {ip: idx for idx, ip in enumerate(self.node_df.index)}
        
        n_node_feat = self.x_nodos.shape[1]
        n_edge_feat = self.scaler_edges.n_features_in_
        
        # 4. Cargar Modelos
        self.bin_web = self._load_mod(n_node_feat, n_edge_feat, 128, 2, 'SAGE', f"{ruta_modelos}/bin_web.pth")
        self.bin_infra = self._load_mod(n_node_feat, n_edge_feat, 64, 2, 'SAGE', f"{ruta_modelos}/bin_infra.pth")
        self.bin_auth = self._load_mod(n_node_feat, n_edge_feat, 128, 2, 'GAT', f"{ruta_modelos}/bin_auth.pth")
        self.bin_gen = self._load_mod(n_node_feat, n_edge_feat, 256, 2, 'GAT', f"{ruta_modelos}/bin_gen.pth")
        
        self.multi_web = [self._load_mod(n_node_feat, n_edge_feat, 128, self.num_clases, 'SAGE', f"{ruta_modelos}/multi_web_sage.pth")]
        self.multi_infra = [
            self._load_mod(n_node_feat, n_edge_feat, 64, self.num_clases, 'SAGE', f"{ruta_modelos}/multi_infra_sage.pth"),
            self._load_mod(n_node_feat, n_edge_feat, 64, self.num_clases, 'GAT', f"{ruta_modelos}/multi_infra_gat.pth")
        ]
        self.multi_auth = [
            self._load_mod(n_node_feat, n_edge_feat, 128, self.num_clases, 'SAGE', f"{ruta_modelos}/multi_auth_sage.pth"),
            self._load_mod(n_node_feat, n_edge_feat, 128, self.num_clases, 'GAT', f"{ruta_modelos}/multi_auth_gat.pth")
        ]
        self.multi_gen = [
            self._load_mod(n_node_feat, n_edge_feat, 256, self.num_clases, 'SAGE', f"{ruta_modelos}/multi_gen_sage.pth"),
            self._load_mod(n_node_feat, n_edge_feat, 256, self.num_clases, 'GAT', f"{ruta_modelos}/multi_gen_gat.pth")
        ]

    def _load_mod(self, in_n, in_e, hid, out, t, path):
        m = AdvancedEdgeExpert(in_n, in_e, hid, out, t).to(self.device)
        m.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        m.eval()
        return m

    def _construir_vector(self, registro):
        """Convierte los datos en bruto al vector matemático esperado"""
        vector = []
        for col in self.variables_base:
            vector.append(float(registro.get(col, 0.0) or 0.0))
            
        estado_actual = str(registro.get('conn_state', 'OTH'))
        for col_dummy in self.columnas_dummies:
            estado_dummy = col_dummy.split('state_')[1]
            vector.append(1.0 if estado_actual == estado_dummy else 0.0)
            
        return vector

    def predecir(self, registro):
        """
        Recibe un diccionario simple con datos del paquete y devuelve predicción.
        """
        # 1. Autoconstrucción del vector
        vector_num = self._construir_vector(registro)
        edge_attr = torch.tensor(self.scaler_edges.transform([vector_num]), dtype=torch.float).to(self.device)
        
        # 2. Mapeo de nodos
        id_s = self.ip_to_id.get(registro.get('src_ip_zeek', ''), 0)
        id_d = self.ip_to_id.get(registro.get('dest_ip_zeek', ''), 0)
        edge_idx = torch.tensor([[id_s], [id_d]], dtype=torch.long).to(self.device)
        
        srv = str(registro.get('service', '')).lower()
        
        # 3. Enrutamiento
        if srv in ['ssl', 'http']:
            m_bin, m_multi = self.bin_web, self.multi_web
        elif srv in ['dns', 'ntp', 'dhcp']:
            m_bin, m_multi = self.bin_infra, self.multi_infra
        elif srv in ['smb', 'gssapi', 'ntlm', 'dce_rpc']:
            m_bin, m_multi = self.bin_auth, self.multi_auth
        else:
            m_bin, m_multi = self.bin_gen, self.multi_gen

        with torch.no_grad():
            # --- FASE 1: PORTERO ---
            out_bin = m_bin(self.x_nodos, edge_idx, edge_idx, edge_attr)
            if torch.argmax(out_bin, dim=1).item() == 0:
                return {
                    "es_ataque": False, 
                    "tactic": "Benigno", 
                    "confianza": float(F.softmax(out_bin, dim=1)[0][0].item())
                }

            # --- FASE 2: MULTICLASE (Soft Voting) ---
            probs_acum = torch.zeros(self.num_clases).to(self.device)
            for m in m_multi:
                probs_acum += F.softmax(m(self.x_nodos, edge_idx, edge_idx, edge_attr), dim=1)[0]
            
            probs_final = probs_acum / len(m_multi)
            probs_final[self.id_benigno] = 0.0 # Forzamos clase maliciosa
            
            idx_pred = torch.argmax(probs_final).item()
            return {
                "es_ataque": True,
                "tactic": str(self.encoder_tactics.inverse_transform([idx_pred])[0]),
                "confianza": float(probs_final[idx_pred].item())
            }