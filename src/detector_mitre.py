import json
import joblib
import numpy as np
import os
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

            assert hidden_channels % heads == 0, (
                f"hidden_channels ({hidden_channels}) "
                f"debe ser divisible por heads ({heads})"
            )

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
        
        src = edge_index_pred[0]
        dst = edge_index_pred[1]
        
        edge_features = torch.cat([x[src], x[dst], edge_attr_pred], dim=-1)
        return self.edge_classifier(edge_features)


# ==============================================================================
# 2. MOTOR DE INFERENCIA EN CASCADA (Streaming)
# ==============================================================================
class DetectorMITRE:
    def __init__(self, ruta_modelos=None, ruta_datos=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[*] Inicializando Detector MITRE en: {self.device}")

        dir_actual = os.path.dirname(os.path.abspath(__file__))
        
        if ruta_modelos is None:
            ruta_modelos = os.path.abspath(os.path.join(dir_actual, "..", "models"))
            
        if ruta_datos is None:
            ruta_datos = os.path.abspath(os.path.join(dir_actual, "..", "data", "processed"))

        # 1. Cargar Artefactos de Normalización y Diccionarios
        self.scaler_edges = joblib.load(os.path.join(ruta_modelos, "encoders", "scaler_edges.pkl"))
        self.encoder = joblib.load(os.path.join(ruta_modelos, "encoders", "encoder_tactics.pkl"))
        self.num_clases = len(self.encoder.classes_)
        
        # Se necesita saber la plantilla exacta de columnas para construir el tensor on-the-fly
        with open(os.path.join(ruta_datos, "columnas_modelo.json"), "r") as f:
            self.columnas_modelo = json.load(f)
            
        try:
            self.id_benigno = self.encoder.transform(['Benigno'])[0]
        except:
            self.id_benigno = 0
            
        # 2. Cargar Nodos Históricos y Mapeo
        node_df = pd.read_pickle(os.path.join(ruta_modelos, "encoders", "node_features_historicas.pkl"))
        self.mapeo_ips = {ip: i for i, ip in enumerate(node_df.index)}
        
        # OOV (Out-Of-Vocabulary): IP Nueva que nunca hemos visto
        self.id_unknown = len(self.mapeo_ips) 
        
        # Escalar nodos y añadir la fila de "IP Desconocida" (llena de 0s)
        scaler_nodes = joblib.load(f"{ruta_modelos}/encoders/scaler_nodes.pkl")
        x_nodos_np = scaler_nodes.transform(node_df.values)
        x_nodos_np = np.vstack([x_nodos_np, np.zeros(x_nodos_np.shape[1])])
        self.x_nodos = torch.tensor(x_nodos_np, dtype=torch.float).to(self.device)
        
        # 3. Reconstruir Topología Histórica Base (Backbone del GraphSAGE)

        df_hist = pd.read_parquet(os.path.join(ruta_datos, "dataset_historico_neo4j.parquet"), columns=['src_ip_zeek', 'dest_ip_zeek'])
        src_ids = df_hist['src_ip_zeek'].map(self.mapeo_ips).fillna(self.id_unknown).astype(int).values
        dst_ids = df_hist['dest_ip_zeek'].map(self.mapeo_ips).fillna(self.id_unknown).astype(int).values
        self.edge_index_historico = torch.tensor(np.vstack((src_ids, dst_ids)), dtype=torch.long).to(self.device)

        # 4. Cargar Umbrales Óptimos (Threshold Moving)
        try:
            with open(os.path.join(ruta_modelos, "config", "umbrales_optimos.json"), "r") as f:
                umbrales = json.load(f)
            self.umbral_binario = umbrales['binario']
            self.umbrales_multiclase = torch.tensor(umbrales['multiclase']).to(self.device)
        except:
            print("[!] Aviso: No se encontró umbrales_optimos.json, usando defaults.")
            self.umbral_binario = 0.50
            self.umbrales_multiclase = torch.ones(self.num_clases).to(self.device)

        # 5. Instanciar y Cargar los Modelos
        # (Aquí va la lógica repetitiva de cargar los pesos .pth en cada variable)
        self._cargar_modelos(ruta_modelos)

        print("CLASES DEL ENCODER:")
        for i, c in enumerate(self.encoder.classes_):
            print(i, c)


    def _cargar_modelos(self, ruta_modelos):
        """Función auxiliar para inicializar arquitecturas, cargar pesos y poner en modo eval()"""
        num_cols = len(self.columnas_modelo)
    
        # --- FASE 1: Binarios ---
        self.bin_web = AdvancedEdgeExpert(self.x_nodos.shape[1], num_cols, 128, 2, 'SAGE', 0.3).to(self.device)
        self.bin_web.load_state_dict(torch.load(os.path.join(ruta_modelos, "bin_web.pth"), map_location=self.device, weights_only=True))
        
        self.bin_infra = AdvancedEdgeExpert(self.x_nodos.shape[1], num_cols, 64, 2, 'SAGE', 0.2).to(self.device)
        self.bin_infra.load_state_dict(torch.load(os.path.join(ruta_modelos, "bin_infra.pth"), map_location=self.device, weights_only=True))
        
        self.bin_auth = AdvancedEdgeExpert(self.x_nodos.shape[1], num_cols, 128, 2, 'GAT', 0.4).to(self.device)
        self.bin_auth.load_state_dict(torch.load(os.path.join(ruta_modelos, "bin_auth.pth"), map_location=self.device, weights_only=True))
        
        self.bin_gen = AdvancedEdgeExpert(self.x_nodos.shape[1], num_cols, 256, 2, 'GAT', 0.5).to(self.device)
        self.bin_gen.load_state_dict(torch.load(os.path.join(ruta_modelos, "bin_gen.pth"), map_location=self.device, weights_only=True))

        # --- FASE 2: Multiclase ---
        self.multi_web_sage = AdvancedEdgeExpert(self.x_nodos.shape[1], num_cols, 128, self.num_clases, 'SAGE', 0.3).to(self.device)
        self.multi_web_sage.load_state_dict(torch.load(os.path.join(ruta_modelos, "multi_web_sage.pth"), map_location=self.device, weights_only=True))

        self.multi_infra_sage = AdvancedEdgeExpert(self.x_nodos.shape[1], num_cols, 64, self.num_clases, 'SAGE', 0.2).to(self.device)
        self.multi_infra_sage.load_state_dict(torch.load(os.path.join(ruta_modelos, "multi_infra_sage.pth"), map_location=self.device, weights_only=True))

        self.multi_infra_gat = AdvancedEdgeExpert(self.x_nodos.shape[1], num_cols, 64, self.num_clases, 'GAT', 0.2).to(self.device)
        self.multi_infra_gat.load_state_dict(torch.load(os.path.join(ruta_modelos, "multi_infra_gat.pth"), map_location=self.device, weights_only=True))

        self.multi_auth_sage = AdvancedEdgeExpert(self.x_nodos.shape[1], num_cols, 128, self.num_clases, 'SAGE', 0.4).to(self.device)
        self.multi_auth_sage.load_state_dict(torch.load(os.path.join(ruta_modelos, "multi_auth_sage.pth"), map_location=self.device, weights_only=True))

        self.multi_auth_gat = AdvancedEdgeExpert(self.x_nodos.shape[1], num_cols, 128, self.num_clases, 'GAT', 0.4).to(self.device)
        self.multi_auth_gat.load_state_dict(torch.load(os.path.join(ruta_modelos, "multi_auth_gat.pth"), map_location=self.device, weights_only=True))

        self.multi_gen_sage = AdvancedEdgeExpert(self.x_nodos.shape[1], num_cols, 256, self.num_clases, 'SAGE', 0.5).to(self.device)
        self.multi_gen_sage.load_state_dict(torch.load(os.path.join(ruta_modelos, "multi_gen_sage.pth"), map_location=self.device, weights_only=True))

        self.multi_gen_gat = AdvancedEdgeExpert(self.x_nodos.shape[1], num_cols, 256, self.num_clases, 'GAT', 0.5).to(self.device)
        self.multi_gen_gat.load_state_dict(torch.load(os.path.join(ruta_modelos, "multi_gen_gat.pth"), map_location=self.device, weights_only=True))

        # Congelar los pesos (Modo Inferencia) -> Vital por los BatchNorm1d
        modelos = [self.bin_web, self.bin_infra, self.bin_auth, self.bin_gen,
                   self.multi_web_sage, self.multi_infra_sage, self.multi_infra_gat,
                   self.multi_auth_sage, self.multi_auth_gat, self.multi_gen_sage, self.multi_gen_gat]
        for m in modelos:
            m.eval()


    def predecir_conexion(self, datos_json):
        """ Recibe un diccionario/JSON con la metadata de la conexión y devuelve la alerta. """
        
        # 1. Extraer identificadores y enrutamiento
        src_ip = datos_json.get('src_ip_zeek', 'unknown')
        dst_ip = datos_json.get('dest_ip_zeek', 'unknown')
        srv = datos_json.get('service', 'desconocido').lower()

        # 2. Resolver Nodos (Si la IP es nueva, toma el ID de self.id_unknown)
        src_id = self.mapeo_ips.get(src_ip, self.id_unknown)
        dst_id = self.mapeo_ips.get(dst_ip, self.id_unknown)

        # 3. Preparar Características (Edge Features on-the-fly)
        # Creamos una plantilla vacía para asegurar que el tensor tenga la misma forma que en el entreno
        df_edge = pd.DataFrame(0.0, index=[0], columns=self.columnas_modelo)

        # Insertar valores continuos
        cols_numericas = ['duration', 'orig_bytes', 'resp_bytes', 'orig_pkts', 'resp_pkts', 'missed_bytes', 'time_since_last_conn']
        for col in cols_numericas:
            if col in datos_json:
                df_edge.at[0, col] = float(datos_json[col])

        # Convertir estado de conexión a One-Hot dinámicamente
        estado = datos_json.get('conn_state', 'OTH')
        col_estado = f"state_{estado}"
        if col_estado in df_edge.columns:
            df_edge.at[0, col_estado] = 1.0
        else:
            if "state_OTH" in df_edge.columns:
                df_edge.at[0, "state_OTH"] = 1.0

        # Escalar
        edge_attr_np = self.scaler_edges.transform(df_edge.values)

        # 4. Generar Tensores para PyTorch Geometric
        edge_index_pred = torch.tensor([[src_id], [dst_id]], dtype=torch.long).to(self.device)
        edge_attr_pred = torch.tensor(edge_attr_np, dtype=torch.float).to(self.device)

        # 5. Enrutamiento del Comité Experto
        if srv in ['ssl', 'http']:
            m_bin, m_multi = self.bin_web, [self.multi_web_sage]
        elif srv in ['dns', 'ntp', 'dhcp']:
            m_bin, m_multi = self.bin_infra, [self.multi_infra_sage, self.multi_infra_gat]
        elif srv in ['smb', 'gssapi', 'ntlm', 'dce_rpc']:
            m_bin, m_multi = self.bin_auth, [self.multi_auth_sage, self.multi_auth_gat]
        else:
            m_bin, m_multi = self.bin_gen, [self.multi_gen_sage, self.multi_gen_gat]

        # 6. INFERENCIA EN CASCADA
        with torch.no_grad():
            
            # --- FASE 1: PORTERO ---
            # ATENCIÓN: Pasamos el grafo histórico como 'message passing' y la nueva conexión como 'predicción'
            out_bin = m_bin(self.x_nodos, self.edge_index_historico, edge_index_pred, edge_attr_pred)
            probs_bin = F.softmax(out_bin, dim=1)[0]
            
            prob_ataque = probs_bin[1].item() 

            # Filtrado Binario Seguro (Threshold Moving)
            if prob_ataque < self.umbral_binario:
                return {
                    "es_ataque": False, 
                    "tactic": "Benigno", 
                    "confianza": round(probs_bin[0].item(), 4)
                }

            # --- FASE 2: ANALISTA MULTICLASE (Soft Voting) ---
            probs_acum = torch.zeros(self.num_clases).to(self.device)
            for m in m_multi:
                out_multi = m(self.x_nodos, self.edge_index_historico, edge_index_pred, edge_attr_pred)
                probs_acum += F.softmax(out_multi, dim=1)[0]
            
            probs_final = probs_acum / len(m_multi)
            
            # Impedir matemáticamente que la fase 2 declare "Benigno"
            probs_final[self.id_benigno] = 0.0

            # --- APLICAR THRESHOLD MOVING MULTICLASE ---
            probs_ajustadas = probs_final / self.umbrales_multiclase
            tactic_idx = torch.argmax(probs_ajustadas).item()
            
            tactic_name = self.encoder.inverse_transform([tactic_idx])[0]
            
            return {
                "es_ataque": True,
                "tactic": tactic_name,
                "confianza": round(probs_final[tactic_idx].item(), 4)
            }