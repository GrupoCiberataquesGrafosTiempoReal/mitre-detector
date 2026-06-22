import json
import joblib
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv
import collections

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
# 2. MOTOR DE INFERENCIA INDUCTIVO EN CASCADA (Streaming Stateful)
# ==============================================================================
class DetectorMITRE:
    def __init__(self, max_window_size=10000, ruta_modelos=None, ruta_datos=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[*] Inicializando Detector MITRE en: {self.device}")

        dir_actual = os.path.dirname(os.path.abspath(__file__))
        if ruta_modelos is None:
            ruta_modelos = os.path.abspath(os.path.join(dir_actual, "..", "models"))
        if ruta_datos is None:
            ruta_datos = os.path.abspath(os.path.join(dir_actual, "..", "data", "processed"))

        # ---------------------------------------------------------
        # MEMORIA INDUCTIVA PARA STREAMING (Stateful)
        # ---------------------------------------------------------
        self.max_window_size = max_window_size
        
        # 1. Diccionario de mapeo de IPs a IDs dinámicos
        self.mapeo_ips = {}
        self.next_node_id = 0
        
        # 2. Diccionarios de estado para Node Features puras
        self.estado_nodos = {}
        
        # 3. Ventana deslizante de la topología (Message Passing base)
        self.ventana_aristas = collections.deque(maxlen=self.max_window_size)
        
        # 4. Memoria temporal para el 'time_since_last_conn'
        self.memoria_timestamps = {}

        # 5: Memoria de Aristas Comprometidas ---
        self.aristas_comprometidas = {}
        
        # 6. Tensor pre-asignado para embeddings de Nodos (Máx. 100k IPs simultáneas)
        # 6 columnas: out_bytes_mean, out_pkts_mean, out_degree, in_bytes_mean, in_pkts_mean, in_degree
        self.x_nodos_np = np.zeros((100000, 6), dtype=np.float32)

        # ---------------------------------------------------------
        # CARGA DE ARTEFACTOS Y UMBRALES
        # ---------------------------------------------------------
        self.scaler_edges = joblib.load(os.path.join(ruta_modelos, "encoders", "scaler_edges.pkl"))
        self.scaler_nodes = joblib.load(os.path.join(ruta_modelos, "encoders", "scaler_nodes.pkl"))
        self.encoder = joblib.load(os.path.join(ruta_modelos, "encoders", "encoder_tactics.pkl"))
        self.num_clases = len(self.encoder.classes_)
        
        with open(os.path.join(ruta_datos, "columnas_modelo.json"), "r") as f:
            self.columnas_modelo = json.load(f)
            
        try:
            self.id_benigno = self.encoder.transform(['Benigno'])[0]
        except:
            self.id_benigno = 0

        # Cargar Umbrales Óptimos
        try:
            with open(os.path.join(ruta_modelos, "config", "umbrales_optimos.json"), "r") as f:
                umbrales = json.load(f)
            self.umbral_binario = umbrales['binario']
            self.umbrales_multiclase = torch.tensor(umbrales['multiclase']).to(self.device)
        except:
            print("[!] Aviso: No se encontró umbrales_optimos.json, usando defaults.")
            self.umbral_binario = 0.50
            self.umbrales_multiclase = torch.ones(self.num_clases).to(self.device)

        # ---------------------------------------------------------
        # INICIALIZACIÓN DE REDES NEURONALES
        # ---------------------------------------------------------
        self._cargar_modelos(ruta_modelos)

        print("CLASES DEL ENCODER:")
        for i, c in enumerate(self.encoder.classes_):
            print(i, c)

    @staticmethod
    def _safe_float(value, default=0.0):
        try:
            value = float(value)
            if np.isnan(value):
                return default
            return value
        except (ValueError, TypeError):
            return default
        
    def _asegurar_capacidad_nodos(self):
        if self.next_node_id < self.x_nodos_np.shape[0]:
            return

        nuevo_tam = self.x_nodos_np.shape[0] * 2

        nuevo_buffer = np.zeros(
            (nuevo_tam, self.x_nodos_np.shape[1]),
            dtype=np.float32
        )

        nuevo_buffer[:self.x_nodos_np.shape[0]] = self.x_nodos_np

        self.x_nodos_np = nuevo_buffer

        print(f"[*] Buffer de nodos ampliado a {nuevo_tam}")


    def _cargar_modelos(self, ruta_modelos):
        """ Inicializa arquitecturas, carga pesos y congela en eval() """
        num_cols = len(self.columnas_modelo)
        num_node_features = 6 # Las 6 columnas estandarizadas de estadísticas de IP
    
        # --- FASE 1: Binarios ---
        self.bin_web = AdvancedEdgeExpert(num_node_features, num_cols, 128, 2, 'SAGE', 0.3).to(self.device)
        self.bin_web.load_state_dict(torch.load(os.path.join(ruta_modelos, "bin_web.pth"), map_location=self.device, weights_only=True))
        
        self.bin_infra = AdvancedEdgeExpert(num_node_features, num_cols, 64, 2, 'SAGE', 0.2).to(self.device)
        self.bin_infra.load_state_dict(torch.load(os.path.join(ruta_modelos, "bin_infra.pth"), map_location=self.device, weights_only=True))
        
        self.bin_auth = AdvancedEdgeExpert(num_node_features, num_cols, 128, 2, 'GAT', 0.4).to(self.device)
        self.bin_auth.load_state_dict(torch.load(os.path.join(ruta_modelos, "bin_auth.pth"), map_location=self.device, weights_only=True))
        
        self.bin_gen = AdvancedEdgeExpert(num_node_features, num_cols, 256, 2, 'GAT', 0.5).to(self.device)
        self.bin_gen.load_state_dict(torch.load(os.path.join(ruta_modelos, "bin_gen.pth"), map_location=self.device, weights_only=True))

        # --- FASE 2: Multiclase ---
        self.multi_web_sage = AdvancedEdgeExpert(num_node_features, num_cols, 128, self.num_clases, 'SAGE', 0.3).to(self.device)
        self.multi_web_sage.load_state_dict(torch.load(os.path.join(ruta_modelos, "multi_web_sage.pth"), map_location=self.device, weights_only=True))

        self.multi_infra_sage = AdvancedEdgeExpert(num_node_features, num_cols, 64, self.num_clases, 'SAGE', 0.2).to(self.device)
        self.multi_infra_sage.load_state_dict(torch.load(os.path.join(ruta_modelos, "multi_infra_sage.pth"), map_location=self.device, weights_only=True))

        self.multi_infra_gat = AdvancedEdgeExpert(num_node_features, num_cols, 64, self.num_clases, 'GAT', 0.2).to(self.device)
        self.multi_infra_gat.load_state_dict(torch.load(os.path.join(ruta_modelos, "multi_infra_gat.pth"), map_location=self.device, weights_only=True))

        self.multi_auth_sage = AdvancedEdgeExpert(num_node_features, num_cols, 128, self.num_clases, 'SAGE', 0.4).to(self.device)
        self.multi_auth_sage.load_state_dict(torch.load(os.path.join(ruta_modelos, "multi_auth_sage.pth"), map_location=self.device, weights_only=True))

        self.multi_auth_gat = AdvancedEdgeExpert(num_node_features, num_cols, 128, self.num_clases, 'GAT', 0.4).to(self.device)
        self.multi_auth_gat.load_state_dict(torch.load(os.path.join(ruta_modelos, "multi_auth_gat.pth"), map_location=self.device, weights_only=True))

        self.multi_gen_sage = AdvancedEdgeExpert(num_node_features, num_cols, 256, self.num_clases, 'SAGE', 0.5).to(self.device)
        self.multi_gen_sage.load_state_dict(torch.load(os.path.join(ruta_modelos, "multi_gen_sage.pth"), map_location=self.device, weights_only=True))

        self.multi_gen_gat = AdvancedEdgeExpert(num_node_features, num_cols, 256, self.num_clases, 'GAT', 0.5).to(self.device)
        self.multi_gen_gat.load_state_dict(torch.load(os.path.join(ruta_modelos, "multi_gen_gat.pth"), map_location=self.device, weights_only=True))

        modelos = [self.bin_web, self.bin_infra, self.bin_auth, self.bin_gen,
                   self.multi_web_sage, self.multi_infra_sage, self.multi_infra_gat,
                   self.multi_auth_sage, self.multi_auth_gat, self.multi_gen_sage, self.multi_gen_gat]
        for m in modelos:
            m.eval()


    def _actualizar_y_escalar_nodo(self, ip, bytes_val, pkts_val, es_origen=True):
        """ Actualiza los estados crudos en memoria y devuelve el vector escalado de 6 features """
        if ip not in self.estado_nodos:
            self.estado_nodos[ip] = {
                'out_bytes_sum': 0.0, 'out_count': 0.0, 'out_pkts_sum': 0.0,
                'in_bytes_sum': 0.0, 'in_count': 0.0, 'in_pkts_sum': 0.0
            }
            
        estado = self.estado_nodos[ip]
        
        # Mantenemos la lógica exacta usada en el entrenamiento
        if es_origen:
            estado['out_bytes_sum'] += bytes_val
            estado['out_pkts_sum'] += pkts_val
            estado['out_count'] += 1.0
        else:
            estado['in_bytes_sum'] += bytes_val
            estado['in_pkts_sum'] += pkts_val
            estado['in_count'] += 1.0
            
        # Calcular medias (protegiendo divisiones por cero)
        out_b_mean = estado['out_bytes_sum'] / max(1.0, estado['out_count'])
        out_p_mean = estado['out_pkts_sum'] / max(1.0, estado['out_count'])
        out_deg = estado['out_count']
        
        in_b_mean = estado['in_bytes_sum'] / max(1.0, estado['in_count'])
        in_p_mean = estado['in_pkts_sum'] / max(1.0, estado['in_count'])
        in_deg = estado['in_count']
        
        raw_features = np.array([[out_b_mean, out_p_mean, out_deg, in_b_mean, in_p_mean, in_deg]])
        
        # Aplicamos el scaler_nodes con el que entrenó el GraphSAGE
        scaled_features = self.scaler_nodes.transform(raw_features)
        return scaled_features[0]


    def predecir_conexion(self, datos_json):
        # 1. Extraer identificadores y limpiar NaNs (Blindaje para Logs de Zeek)
        src_ip = datos_json.get('src_ip_zeek', 'unknown')
        dst_ip = datos_json.get('dest_ip_zeek', 'unknown')
        srv = datos_json.get('service', 'desconocido').lower()
        
        orig_bytes = self._safe_float(datos_json.get('orig_bytes'))
        resp_bytes = self._safe_float(datos_json.get('resp_bytes'))
        orig_pkts = self._safe_float(datos_json.get('orig_pkts'))
        resp_pkts = self._safe_float(datos_json.get('resp_pkts'))

        # ---------------------------------------------------------
        # A. MAPEO DINÁMICO DE IPs
        # ---------------------------------------------------------
        for ip in [src_ip, dst_ip]:
            if ip not in self.mapeo_ips:
                self._asegurar_capacidad_nodos()
                self.mapeo_ips[ip] = self.next_node_id
                self.next_node_id += 1
                
        src_id = self.mapeo_ips[src_ip]
        dst_id = self.mapeo_ips[dst_ip]

        # ---------------------------------------------------------
        # B. ACTUALIZACIÓN DE NODE FEATURES (Inductivo)
        # ---------------------------------------------------------
        # Al origen le computamos los "orig_*" y al destino los "resp_*" (lógica de Zeek/Entreno)
        feat_src = self._actualizar_y_escalar_nodo(src_ip, orig_bytes, orig_pkts, es_origen=True)
        feat_dst = self._actualizar_y_escalar_nodo(dst_ip, resp_bytes, resp_pkts, es_origen=False)
        
        self.x_nodos_np[src_id, :] = feat_src
        self.x_nodos_np[dst_id, :] = feat_dst

        # ---------------------------------------------------------
        # C. ACTUALIZACIÓN DEL GRAFO DESLIZANTE
        # ---------------------------------------------------------
        self.ventana_aristas.append((src_id, dst_id))
        
        # Convertimos la ventana actual en el tensor de topología histórica viva
        aristas_activas = np.array(self.ventana_aristas).T
        edge_index_historico_vivo = torch.tensor(aristas_activas, dtype=torch.long).to(self.device)
        
        # Extraemos solo la porción de IPs activas para ahorrar memoria GPU
        x_nodos_vivo = torch.tensor(self.x_nodos_np[:self.next_node_id], dtype=torch.float).to(self.device)

        # ---------------------------------------------------------
        # D. EXTRACCIÓN Y CÁLCULO DE EDGE FEATURES
        # ---------------------------------------------------------
        clave_conn = f"{src_ip}-{dst_ip}"
        try:
            ts_actual = float(datos_json.get('ts', 0.0))
            if np.isnan(ts_actual): ts_actual = 0.0
        except:
            ts_actual = 0.0
        
        if clave_conn in self.memoria_timestamps:
            delta_time = ts_actual - self.memoria_timestamps[clave_conn]
        else:
            delta_time = 0.0
            
        self.memoria_timestamps[clave_conn] = ts_actual

        df_edge = pd.DataFrame(0.0, index=[0], columns=self.columnas_modelo)

        cols_numericas = ['duration', 'orig_bytes', 'resp_bytes', 'orig_pkts', 'resp_pkts', 'missed_bytes']
        for col in cols_numericas:
            val = datos_json.get(col, 0.0)
            try:
                val = float(val)
                if np.isnan(val): val = 0.0
            except:
                val = 0.0
            df_edge.at[0, col] = val

        if 'time_since_last_conn' in df_edge.columns:
            df_edge.at[0, 'time_since_last_conn'] = delta_time

        estado = datos_json.get('conn_state', 'OTH')
        col_estado = f"state_{estado}"
        if col_estado in df_edge.columns:
            df_edge.at[0, col_estado] = 1.0
        elif "state_OTH" in df_edge.columns:
            df_edge.at[0, "state_OTH"] = 1.0

        edge_attr_np = self.scaler_edges.transform(df_edge.values)

        edge_index_pred = torch.tensor([[src_id], [dst_id]], dtype=torch.long).to(self.device)
        edge_attr_pred = torch.tensor(edge_attr_np, dtype=torch.float).to(self.device)

        # ---------------------------------------------------------
        # E. ENRUTAMIENTO Y CASCADA DE INFERENCIA
        # ---------------------------------------------------------
        if srv in ['ssl', 'http']:
            m_bin, m_multi = self.bin_web, [self.multi_web_sage]
        elif srv in ['dns', 'ntp', 'dhcp']:
            m_bin, m_multi = self.bin_infra, [self.multi_infra_sage, self.multi_infra_gat]
        elif srv in ['smb', 'gssapi', 'ntlm', 'dce_rpc']:
            m_bin, m_multi = self.bin_auth, [self.multi_auth_sage, self.multi_auth_gat]
        else:
            m_bin, m_multi = self.bin_gen, [self.multi_gen_sage, self.multi_gen_gat]

        # Comprobar si esta conexión ocurre en un túnel ya comprometido recientemente (ej. últimos 5 mins)
        esta_comprometida = False
        if clave_conn in self.aristas_comprometidas:
            tiempo_desde_ataque = ts_actual - self.aristas_comprometidas[clave_conn]
            if tiempo_desde_ataque < 300.0:  # 5 minutos de ventana de sospecha
                esta_comprometida = True
            else:
                del self.aristas_comprometidas[clave_conn] # Expiró

        with torch.no_grad():
            # --- FASE 1: PORTERO ---
            out_bin = m_bin(x_nodos_vivo, edge_index_historico_vivo, edge_index_pred, edge_attr_pred)
            probs_bin = F.softmax(out_bin, dim=1)[0]
            prob_ataque = probs_bin[1].item() 

            # Si el portero dice Benigno (prob_ataque < umbral) PERO la arista es radiactiva,
            # forzamos el paso a la Fase 2 (Analista Multiclase)
            if prob_ataque < self.umbral_binario and not esta_comprometida:
                return {
                    "label_binary": False, 
                    "label_tactic": "Benigno", 
                    "confidence": round(probs_bin[0].item(), 4)
                }

            # --- FASE 2: ANALISTA MULTICLASE (Soft Voting) ---
            probs_acum = torch.zeros(self.num_clases).to(self.device)
            for m in m_multi:
                out_multi = m(x_nodos_vivo, edge_index_historico_vivo, edge_index_pred, edge_attr_pred)
                probs_acum += F.softmax(out_multi, dim=1)[0]
            
            probs_final = probs_acum / len(m_multi)
            probs_final[self.id_benigno] = 0.0

            probs_ajustadas = probs_final / self.umbrales_multiclase
            tactic_idx = torch.argmax(probs_ajustadas).item()
            tactic_name = self.encoder.inverse_transform([tactic_idx])[0]

            # --- ACTUALIZAR MEMORIA RADIACTIVA ---
            # Si hemos detectado un ataque, marcamos la arista para el futuro
            self.aristas_comprometidas[clave_conn] = ts_actual
            
            # Penalización visual de confianza si fue forzado por radiactividad
            confianza_final = round(probs_final[tactic_idx].item(), 4)
            if prob_ataque < self.umbral_binario and esta_comprometida:
                # El modelo multiclase está operando sobre tráfico que parece normal a nivel de red,
                # la confianza será más baja, pero forzamos la alerta por correlación temporal.
                confianza_final = round(probs_final[tactic_idx].item() * 0.8, 4)
            
            return {
                "label_binary": True,
                "label_tactic": tactic_name,
                "confidence": confianza_final
            }