import torch
import joblib
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv

class AdvancedEdgeExpert(torch.nn.Module):
    def __init__(self, node_in_channels, edge_in_channels, hidden_channels, out_classes, conv_type='SAGE', dropout_rate=0.3):
        super(AdvancedEdgeExpert, self).__init__()
        self.conv_type = conv_type
        
        # --- DIVERSIDAD ARQUITECTÓNICA ---
        if conv_type == 'SAGE':
            self.conv1 = SAGEConv(node_in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, hidden_channels)
            self.conv3 = SAGEConv(hidden_channels, hidden_channels) # NUEVA CAPA
        elif conv_type == 'GAT':
            # GAT usa cabezales de atención (heads). Multiplican la salida, 
            # así que dividimos los canales para mantener la dimensión final igual.
            heads = 4
            self.conv1 = GATConv(node_in_channels, hidden_channels // heads, heads=heads)
            self.conv2 = GATConv(hidden_channels, hidden_channels // heads, heads=heads) # AHORA MANTIENE HEADS
            self.conv3 = GATConv(hidden_channels, hidden_channels, heads=1) # NUEVA CAPA FINAL

        
        # --- PERCEPTRÓN DINÁMICO ---
        clf_input_dim = (hidden_channels * 2) + edge_in_channels
        self.edge_classifier = nn.Sequential(
            nn.Linear(clf_input_dim, hidden_channels),
            nn.BatchNorm1d(hidden_channels), # Batch Normalization ayuda mucho a la convergencia
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, out_classes) 
        )

    def forward(self, x, edge_index_msg, edge_index_pred, edge_attr_pred):
        x = self.conv1(x, edge_index_msg)
        x = F.relu(x)
        x = self.conv2(x, edge_index_msg)
        x = F.relu(x)
        x = self.conv3(x, edge_index_msg)
        
        src = edge_index_pred[0]
        dst = edge_index_pred[1]
        
        edge_features = torch.cat([x[src], x[dst], edge_attr_pred], dim=-1)
        
        return self.edge_classifier(edge_features)

class DetectorMITRE:
    def __init__(self, ruta_modelos="modelos_produccion"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Inicializando SOC IA en: {self.device}")
        
        # 1. Cargar Artefactos
        self.scaler_edges = joblib.load(f"{ruta_modelos}/scaler_edges.pkl")
        self.scaler_nodes = joblib.load(f"{ruta_modelos}/scaler_nodes.pkl")
        self.encoder_tactics = joblib.load(f"{ruta_modelos}/encoder_tactics.pkl")
        self.num_clases = len(self.encoder_tactics.classes_)
        self.id_benigno = self.encoder_tactics.transform(['Benigno'])[0]
        
        # 2. Cargar Nodos Históricos
        self.node_df = pd.read_pickle(f"{ruta_modelos}/node_features_historicas.pkl")
        self.x_nodos = torch.tensor(
            self.scaler_nodes.transform(self.node_df.values), dtype=torch.float
        ).to(self.device)
        self.ip_to_id = {ip: idx for idx, ip in enumerate(self.node_df.index)}
        
        num_node_feats = self.x_nodos.shape[1]
        num_edge_feats = self.scaler_edges.n_features_in_
        
        # ==================== CARGAR MODELOS FASE 1 (BINARIOS) ====================
        self.bin_web = self._load_model(num_node_feats, num_edge_feats, 128, 2, 'SAGE', 0.3, f"{ruta_modelos}/bin_web.pth")
        self.bin_infra = self._load_model(num_node_feats, num_edge_feats, 64, 2, 'SAGE', 0.2, f"{ruta_modelos}/bin_infra.pth")
        self.bin_auth = self._load_model(num_node_feats, num_edge_feats, 128, 2, 'GAT', 0.4, f"{ruta_modelos}/bin_auth.pth")
        self.bin_gen = self._load_model(num_node_feats, num_edge_feats, 256, 2, 'GAT', 0.5, f"{ruta_modelos}/bin_gen.pth")
        
        # ==================== CARGAR MODELOS FASE 2 (MULTICLASE) ====================
        self.multi_web = [
            self._load_model(num_node_feats, num_edge_feats, 128, self.num_clases, 'SAGE', 0.3, f"{ruta_modelos}/multi_web_sage.pth")
        ]
        self.multi_infra = [
            self._load_model(num_node_feats, num_edge_feats, 64, self.num_clases, 'SAGE', 0.2, f"{ruta_modelos}/multi_infra_sage.pth"),
            self._load_model(num_node_feats, num_edge_feats, 64, self.num_clases, 'GAT', 0.2, f"{ruta_modelos}/multi_infra_gat.pth")
        ]
        self.multi_auth = [
            self._load_model(num_node_feats, num_edge_feats, 128, self.num_clases, 'SAGE', 0.4, f"{ruta_modelos}/multi_auth_sage.pth"),
            self._load_model(num_node_feats, num_edge_feats, 128, self.num_clases, 'GAT', 0.4, f"{ruta_modelos}/multi_auth_gat.pth")
        ]
        self.multi_gen = [
            self._load_model(num_node_feats, num_edge_feats, 256, self.num_clases, 'SAGE', 0.5, f"{ruta_modelos}/multi_gen_sage.pth"),
            self._load_model(num_node_feats, num_edge_feats, 256, self.num_clases, 'GAT', 0.5, f"{ruta_modelos}/multi_gen_gat.pth")
        ]

    def _load_model(self, in_nodes, in_edges, hidden, out_cls, conv_type, dropout, path):
        model = AdvancedEdgeExpert(in_nodes, in_edges, hidden, out_cls, conv_type, dropout).to(self.device)
        model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        model.eval()
        return model

    def _obtener_ids_nodos(self, ip_src, ip_dst):
        id_src = self.ip_to_id.get(ip_src, 0)
        id_dst = self.ip_to_id.get(ip_dst, 0)
        return torch.tensor([[id_src], [id_dst]], dtype=torch.long).to(self.device)
        
    def predecir_conexion(self, registro_conexion):
        """
        Ejecuta la inferencia en cascada. 
        registro_conexion debe incluir un 'vector_numerico' compatible con scaler_edges.
        """
        # 1. Preparar Tensores
        features_crudas = np.array([registro_conexion['vector_numerico']])
        edge_attr = torch.tensor(self.scaler_edges.transform(features_crudas), dtype=torch.float).to(self.device)
        edge_index = self._obtener_ids_nodos(registro_conexion['src_ip'], registro_conexion['dst_ip'])
        
        servicio = str(registro_conexion.get('service', '')).lower()
        
        # 2. Enrutador de Modelos
        if servicio in ['ssl', 'http']:
            mod_bin, mods_multi = self.bin_web, self.multi_web
        elif servicio in ['dns', 'ntp', 'dhcp']:
            mod_bin, mods_multi = self.bin_infra, self.multi_infra
        elif servicio in ['smb', 'gssapi', 'ntlm', 'dce_rpc', 'gssapi,smb,ntlm', 'smb,dce_rpc,ntlm,gssapi', 'ntlm,dce_rpc,smb,gssapi']:
            mod_bin, mods_multi = self.bin_auth, self.multi_auth
        else:
            mod_bin, mods_multi = self.bin_gen, self.multi_gen
            
        with torch.no_grad():
            # --- FASE 1: PORTERO (BINARIO) ---
            out_bin = mod_bin(self.x_nodos, edge_index, edge_index, edge_attr)
            es_ataque = torch.argmax(out_bin, dim=1).item() == 1
            
            if not es_ataque:
                # Es benigno, cortamos la cascada aquí
                return {
                    "es_ataque": False,
                    "tactica_mitre": "Benigno",
                    "confianza": float(F.softmax(out_bin, dim=1)[0][0].item())
                }
                
            # --- FASE 2: ANALISTA (MULTICLASE + SOFT VOTING) ---
            probabilidades_acumuladas = torch.zeros(self.num_clases).to(self.device)
            
            for mod in mods_multi:
                out_multi = mod(self.x_nodos, edge_index, edge_index, edge_attr)
                probs = F.softmax(out_multi, dim=1)[0]
                probabilidades_acumuladas += probs
                
            # Promediamos y silenciamos "Benigno"
            probabilidades_finales = probabilidades_acumuladas / len(mods_multi)
            probabilidades_finales[self.id_benigno] = 0.0 
            
            clase_predicha_idx = torch.argmax(probabilidades_finales).item()
            confianza_final = float(probabilidades_finales[clase_predicha_idx].item())
            
            tactica_mitre = self.encoder_tactics.inverse_transform([clase_predicha_idx])[0]
            
        return {
            "es_ataque": True,
            "tactica_mitre": tactica_mitre,
            "confianza": confianza_final
        }