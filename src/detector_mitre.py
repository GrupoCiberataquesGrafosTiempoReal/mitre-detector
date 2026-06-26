import json
import os
import collections
from typing import Any, Dict, List, Optional, Tuple

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
    def __init__(
        self,
        node_in_channels: int,
        edge_in_channels: int,
        hidden_channels: int,
        out_classes: int,
        conv_type: str = "SAGE",
        dropout_rate: float = 0.3,
    ):
        super(AdvancedEdgeExpert, self).__init__()
        self.conv_type = conv_type

        if conv_type == "SAGE":
            self.conv1 = SAGEConv(node_in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, hidden_channels)
            self.conv3 = SAGEConv(hidden_channels, hidden_channels)

        elif conv_type == "GAT":
            heads = 4
            assert hidden_channels % heads == 0, (
                f"hidden_channels ({hidden_channels}) "
                f"debe ser divisible por heads ({heads})"
            )
            self.conv1 = GATConv(node_in_channels, hidden_channels // heads, heads=heads)
            self.conv2 = GATConv(hidden_channels, hidden_channels // heads, heads=heads)
            self.conv3 = GATConv(hidden_channels, hidden_channels, heads=1)

        else:
            raise ValueError(f"conv_type no soportado: {conv_type}")

        clf_input_dim = (hidden_channels * 2) + edge_in_channels

        self.edge_classifier = nn.Sequential(
            nn.Linear(clf_input_dim, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, out_classes),
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
    """
    Detector MITRE en streaming con carga dinámica de modelos.

    Propiedades principales:
      - Reutiliza IDs de entrenamiento mediante mapeo_ips.pkl.
      - Arranca con la topología histórica exportada en edge_index_entrenamiento.pt.
      - Arranca con las node features históricas exportadas en x_nodos_entrenamiento.pt.
      - Añade nuevas IPs desde next_node_id sin alterar los IDs históricos.
      - Carga solo los modelos listados en config/model_manifest.json.
      - Si falta un experto multiclase para un dominio, puede enrutar al experto generalista.
      - Si no hay ningún experto multiclase disponible, devuelve Ataque_No_Clasificado.
    """

    DEFAULT_MODEL_SPECS = {
        "bin_sage": {
            "role": "binary",
            "domain": "global",
            "filename": "bin_sage.pth",
            "hidden_channels": 256,
            "out_classes": 2,
            "conv_type": "SAGE",
            "dropout_rate": 0.3,
        },
        "multi_web_sage": {
            "role": "multiclass",
            "domain": "web",
            "filename": "multi_web_sage.pth",
            "hidden_channels": 128,
            "out_classes": "num_clases",
            "conv_type": "SAGE",
            "dropout_rate": 0.3,
        },
        "multi_infra_sage": {
            "role": "multiclass",
            "domain": "infra",
            "filename": "multi_infra_sage.pth",
            "hidden_channels": 64,
            "out_classes": "num_clases",
            "conv_type": "SAGE",
            "dropout_rate": 0.2,
        },
        "multi_infra_gat": {
            "role": "multiclass",
            "domain": "infra",
            "filename": "multi_infra_gat.pth",
            "hidden_channels": 64,
            "out_classes": "num_clases",
            "conv_type": "GAT",
            "dropout_rate": 0.2,
        },
        "multi_auth_sage": {
            "role": "multiclass",
            "domain": "auth",
            "filename": "multi_auth_sage.pth",
            "hidden_channels": 128,
            "out_classes": "num_clases",
            "conv_type": "SAGE",
            "dropout_rate": 0.4,
        },
        "multi_auth_gat": {
            "role": "multiclass",
            "domain": "auth",
            "filename": "multi_auth_gat.pth",
            "hidden_channels": 128,
            "out_classes": "num_clases",
            "conv_type": "GAT",
            "dropout_rate": 0.4,
        },
        "multi_gen_sage": {
            "role": "multiclass",
            "domain": "gen",
            "filename": "multi_gen_sage.pth",
            "hidden_channels": 256,
            "out_classes": "num_clases",
            "conv_type": "SAGE",
            "dropout_rate": 0.5,
        },
        "multi_gen_gat": {
            "role": "multiclass",
            "domain": "gen",
            "filename": "multi_gen_gat.pth",
            "hidden_channels": 256,
            "out_classes": "num_clases",
            "conv_type": "GAT",
            "dropout_rate": 0.5,
        },
    }

    def __init__(
        self,
        max_window_size: int = 10000,
        ruta_modelos: Optional[str] = None,
        ruta_datos: Optional[str] = None,
        debug: bool = False,
        allow_multiclass_fallback: bool = True,
    ):
        # En Docker CPU, torch.cuda.is_available() será False y todo cargará en CPU.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.debug = debug
        self.allow_multiclass_fallback = allow_multiclass_fallback

        print(f"[*] Inicializando Detector MITRE en: {self.device}")

        dir_actual = os.path.dirname(os.path.abspath(__file__))
        if ruta_modelos is None:
            ruta_modelos = os.path.abspath(os.path.join(dir_actual, "..", "models"))
        if ruta_datos is None:
            ruta_datos = os.path.abspath(os.path.join(dir_actual, "..", "data", "processed"))

        self.ruta_modelos = ruta_modelos
        self.ruta_datos = ruta_datos
        self.max_window_size = max_window_size

        self.encoders_dir = os.path.join(ruta_modelos, "encoders")
        self.config_dir = os.path.join(ruta_modelos, "config")

        # ----------------------------------------------------------------------
        # 1. Artefactos de preprocesado y etiquetas
        # ----------------------------------------------------------------------
        self.scaler_edges = joblib.load(os.path.join(self.encoders_dir, "scaler_edges.pkl"))
        self.scaler_nodes = joblib.load(os.path.join(self.encoders_dir, "scaler_nodes.pkl"))
        self.encoder = joblib.load(os.path.join(self.encoders_dir, "encoder_tactics.pkl"))
        self.num_clases = len(self.encoder.classes_)

        columnas_path = os.path.join(ruta_datos, "columnas_modelo.json")
        with open(columnas_path, "r", encoding="utf-8") as f:
            self.columnas_modelo = json.load(f)

        self.id_benigno = int(self.encoder.transform(["Benigno"])[0])

        # ----------------------------------------------------------------------
        # 2. Grafo y nodos históricos de entrenamiento/validación
        # ----------------------------------------------------------------------
        self.mapeo_ips = joblib.load(os.path.join(self.encoders_dir, "mapeo_ips.pkl"))
        if not isinstance(self.mapeo_ips, dict):
            raise TypeError("mapeo_ips.pkl debe contener un diccionario {ip: id}.")

        self.next_node_id = (max(self.mapeo_ips.values()) + 1) if self.mapeo_ips else 0

        self.edge_index_base = self._torch_load(
            os.path.join(self.encoders_dir, "edge_index_entrenamiento.pt"),
            map_location=self.device,
        ).long().to(self.device).contiguous()

        if self.edge_index_base.dim() != 2 or self.edge_index_base.size(0) != 2:
            raise ValueError(
                "edge_index_entrenamiento.pt debe tener forma [2, num_aristas]. "
                f"Forma recibida: {tuple(self.edge_index_base.shape)}"
            )

        self.x_nodos_vivo = self._torch_load(
            os.path.join(self.encoders_dir, "x_nodos_entrenamiento.pt"),
            map_location=self.device,
        ).float().to(self.device).contiguous()

        if self.x_nodos_vivo.dim() != 2:
            raise ValueError(
                "x_nodos_entrenamiento.pt debe tener forma [num_nodos, num_features]. "
                f"Forma recibida: {tuple(self.x_nodos_vivo.shape)}"
            )

        self.num_node_features = int(self.x_nodos_vivo.size(1))
        self._asegurar_capacidad_nodos(max(self.next_node_id - 1, 0))

        # Ventana deslizante SOLO de aristas nuevas en inferencia.
        self.ventana_aristas_stream = collections.deque(maxlen=self.max_window_size)

        # ----------------------------------------------------------------------
        # 3. Estado histórico de nodos para actualización incremental
        # ----------------------------------------------------------------------
        self.estado_nodos: Dict[str, Dict[str, float]] = {}

        node_hist_path = os.path.join(self.encoders_dir, "node_features_historicas.pkl")
        if os.path.exists(node_hist_path):
            self.node_features_historicas = pd.read_pickle(node_hist_path)
        else:
            self.node_features_historicas = None
            print(
                "[!] Aviso: no se encontró node_features_historicas.pkl. "
                "Los estados de nodos conocidos se inicializarán desde cero."
            )

        # Memoria temporal para time_since_last_conn.
        self.memoria_timestamps: Dict[str, float] = {}

        # Memoria de aristas comprometidas.
        self.aristas_comprometidas: Dict[str, float] = {}

        # ----------------------------------------------------------------------
        # 4. Umbrales y manifiesto dinámico de modelos
        # ----------------------------------------------------------------------
        self._cargar_umbrales()
        self.model_manifest = self._cargar_model_manifest()

        # ----------------------------------------------------------------------
        # 5. Modelos
        # ----------------------------------------------------------------------
        self.bin_sage: Optional[AdvancedEdgeExpert] = None
        self.modelos_multiclase_por_dominio: Dict[str, List[Tuple[str, AdvancedEdgeExpert]]] = {
            "web": [],
            "infra": [],
            "auth": [],
            "gen": [],
        }
        self.modelos_cargados: Dict[str, AdvancedEdgeExpert] = {}

        self._cargar_modelos_dinamicos()

        print("[*] Artefactos cargados:")
        print(f"    - Nodos históricos: {self.x_nodos_vivo.size(0):,}")
        print(f"    - Aristas históricas base: {self.edge_index_base.size(1):,}")
        print(f"    - Siguiente ID de nodo: {self.next_node_id:,}")
        print(f"    - Umbral binario: {self.umbral_binario}")
        print("    - Modelos multiclase cargados por dominio:")
        for domain, models in self.modelos_multiclase_por_dominio.items():
            print(f"        {domain}: {[name for name, _ in models]}")

        if self.debug:
            print("CLASES DEL ENCODER:")
            for i, c in enumerate(self.encoder.classes_):
                print(i, c)

    # --------------------------------------------------------------------------
    # Helpers de carga y saneamiento
    # --------------------------------------------------------------------------
    @staticmethod
    def _torch_load(path: str, map_location=None):
        """
        Compatibilidad entre versiones de PyTorch:
        - weights_only=True cuando está disponible.
        - fallback sin weights_only en versiones antiguas.
        """
        try:
            return torch.load(path, map_location=map_location, weights_only=True)
        except TypeError:
            return torch.load(path, map_location=map_location)

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            value = float(value)
            if np.isnan(value) or np.isinf(value):
                return default
            return value
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _normalizar_ip(value: Any) -> str:
        if value is None:
            return "unknown"
        value = str(value).strip()
        return value if value else "unknown"

    @staticmethod
    def _normalizar_servicio(value: Any) -> str:
        if value is None:
            return "desconocido"
        value = str(value).strip().lower()
        return value if value else "desconocido"

    def _asegurar_capacidad_nodos(self, required_id: int):
        """Amplía x_nodos_vivo si el ID requerido excede la capacidad actual."""
        if required_id < self.x_nodos_vivo.size(0):
            return

        nuevo_tam = max(required_id + 1, self.x_nodos_vivo.size(0) * 2)
        padding = torch.zeros(
            (nuevo_tam - self.x_nodos_vivo.size(0), self.x_nodos_vivo.size(1)),
            dtype=self.x_nodos_vivo.dtype,
            device=self.device,
        )
        self.x_nodos_vivo = torch.cat([self.x_nodos_vivo, padding], dim=0)
        print(f"[*] Grafo ampliado a {nuevo_tam:,} nodos.")

    # --------------------------------------------------------------------------
    # Configuración: umbrales y manifiesto
    # --------------------------------------------------------------------------
    def _cargar_umbrales(self):
        try:
            with open(
                os.path.join(self.config_dir, "umbrales_optimos.json"),
                "r",
                encoding="utf-8",
            ) as f:
                umbrales = json.load(f)

            self.umbral_binario = float(umbrales.get("binario", 0.50))
            umbrales_multiclase = umbrales.get("multiclase", [1.0] * self.num_clases)

            if len(umbrales_multiclase) != self.num_clases:
                print(
                    "[!] Aviso: longitud de umbrales_multiclase no coincide con num_clases. "
                    "Usando vector de unos."
                )
                umbrales_multiclase = [1.0] * self.num_clases

            self.umbrales_multiclase = torch.tensor(
                umbrales_multiclase,
                dtype=torch.float,
                device=self.device,
            )

        except Exception as exc:
            print(f"[!] Aviso: no se encontró/leyó umbrales_optimos.json ({exc}). Usando defaults.")
            self.umbral_binario = 0.50
            self.umbrales_multiclase = torch.ones(self.num_clases, device=self.device)

    def _cargar_model_manifest(self) -> Dict[str, Any]:
        manifest_path = os.path.join(self.config_dir, "model_manifest.json")

        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            if self.debug:
                print(f"[*] Manifiesto de modelos cargado: {manifest_path}")
            return manifest

        # Modo legacy: si no hay manifiesto, carga los modelos que existan físicamente.
        print("[!] Aviso: no se encontró config/model_manifest.json. Usando detección legacy por ficheros .pth.")
        models = {}
        for name, spec in self.DEFAULT_MODEL_SPECS.items():
            path = os.path.join(self.ruta_modelos, spec["filename"])
            if os.path.exists(path):
                spec_copy = dict(spec)
                spec_copy["trained"] = True
                spec_copy["exported"] = True
                models[name] = spec_copy

        return {
            "version": 1,
            "fallback_multiclass_domain": "gen",
            "allow_multiclass_fallback": self.allow_multiclass_fallback,
            "models": models,
            "skipped_models": {},
        }

    # --------------------------------------------------------------------------
    # Carga dinámica de modelos
    # --------------------------------------------------------------------------
    def _crear_modelo_desde_spec(self, spec: Dict[str, Any]) -> AdvancedEdgeExpert:
        out_classes = spec.get("out_classes", self.num_clases)
        if out_classes == "num_clases":
            out_classes = self.num_clases

        return AdvancedEdgeExpert(
            node_in_channels=self.num_node_features,
            edge_in_channels=len(self.columnas_modelo),
            hidden_channels=int(spec["hidden_channels"]),
            out_classes=int(out_classes),
            conv_type=str(spec["conv_type"]),
            dropout_rate=float(spec.get("dropout_rate", 0.3)),
        ).to(self.device)

    def _cargar_modelos_dinamicos(self):
        models_cfg = self.model_manifest.get("models", {})

        # 1. Portero binario obligatorio.
        bin_spec = models_cfg.get("bin_sage")
        if not bin_spec:
            # Fallback legacy si existe el fichero.
            legacy_path = os.path.join(self.ruta_modelos, "bin_sage.pth")
            if os.path.exists(legacy_path):
                bin_spec = dict(self.DEFAULT_MODEL_SPECS["bin_sage"])
                bin_spec["trained"] = True
                bin_spec["exported"] = True
            else:
                raise FileNotFoundError(
                    "No se encontró bin_sage en model_manifest.json ni bin_sage.pth. "
                    "El portero binario es obligatorio."
                )

        self.bin_sage = self._crear_modelo_desde_spec(bin_spec)
        bin_filename = bin_spec.get("filename", "bin_sage.pth")
        self.bin_sage.load_state_dict(
            self._torch_load(os.path.join(self.ruta_modelos, bin_filename), map_location=self.device)
        )
        self.bin_sage.eval()
        self.modelos_cargados["bin_sage"] = self.bin_sage

        # 2. Modelos multiclase opcionales.
        for name, spec in models_cfg.items():
            if name == "bin_sage":
                continue

            if spec.get("role") != "multiclass":
                continue

            if not spec.get("trained", True) or not spec.get("exported", True):
                continue

            filename = spec.get("filename")
            if not filename:
                continue

            model_path = os.path.join(self.ruta_modelos, filename)
            if not os.path.exists(model_path):
                print(f"[!] Aviso: {name} aparece en el manifiesto, pero no existe {model_path}. Se omite.")
                continue

            model = self._crear_modelo_desde_spec(spec)
            model.load_state_dict(self._torch_load(model_path, map_location=self.device))
            model.eval()

            domain = str(spec.get("domain", "gen"))
            self.modelos_multiclase_por_dominio.setdefault(domain, []).append((name, model))
            self.modelos_cargados[name] = model

        # 3. Comprobación: debe haber al menos un modelo multiclase o la cascada no podrá clasificar tácticas.
        total_multi = sum(len(v) for v in self.modelos_multiclase_por_dominio.values())
        if total_multi == 0:
            print("[!] Aviso: no hay modelos multiclase cargados. Las alertas serán Ataque_No_Clasificado.")

    # --------------------------------------------------------------------------
    # Estado nodal incremental
    # --------------------------------------------------------------------------
    def _estado_historico_desde_node_df(self, ip: str) -> Dict[str, float]:
        """
        Inicializa el estado incremental de una IP conocida usando node_features_historicas.pkl.
        node_df contiene medias y grados. Reconstruimos sumas = media * grado.
        """
        estado = {
            "out_bytes_sum": 0.0,
            "out_count": 0.0,
            "out_pkts_sum": 0.0,
            "in_bytes_sum": 0.0,
            "in_count": 0.0,
            "in_pkts_sum": 0.0,
        }

        if self.node_features_historicas is None:
            return estado

        if ip not in self.node_features_historicas.index:
            return estado

        row = self.node_features_historicas.loc[ip]

        out_degree = self._safe_float(row.get("out_degree", 0.0))
        in_degree = self._safe_float(row.get("in_degree", 0.0))

        out_bytes_mean = self._safe_float(row.get("out_bytes_mean", 0.0))
        out_pkts_mean = self._safe_float(row.get("out_pkts_mean", 0.0))
        in_bytes_mean = self._safe_float(row.get("in_bytes_mean", 0.0))
        in_pkts_mean = self._safe_float(row.get("in_pkts_mean", 0.0))

        estado["out_count"] = out_degree
        estado["out_bytes_sum"] = out_bytes_mean * out_degree
        estado["out_pkts_sum"] = out_pkts_mean * out_degree

        estado["in_count"] = in_degree
        estado["in_bytes_sum"] = in_bytes_mean * in_degree
        estado["in_pkts_sum"] = in_pkts_mean * in_degree

        return estado

    def _actualizar_y_escalar_nodo(
        self,
        ip: str,
        bytes_val: float,
        pkts_val: float,
        es_origen: bool = True,
    ) -> np.ndarray:
        """
        Actualiza los estados crudos en memoria y devuelve el vector escalado de 6 features.
        """
        if ip not in self.estado_nodos:
            self.estado_nodos[ip] = self._estado_historico_desde_node_df(ip)

        estado = self.estado_nodos[ip]

        if es_origen:
            estado["out_bytes_sum"] += bytes_val
            estado["out_pkts_sum"] += pkts_val
            estado["out_count"] += 1.0
        else:
            estado["in_bytes_sum"] += bytes_val
            estado["in_pkts_sum"] += pkts_val
            estado["in_count"] += 1.0

        out_count = max(1.0, estado["out_count"])
        in_count = max(1.0, estado["in_count"])

        out_b_mean = estado["out_bytes_sum"] / out_count
        out_p_mean = estado["out_pkts_sum"] / out_count
        out_deg = estado["out_count"]

        in_b_mean = estado["in_bytes_sum"] / in_count
        in_p_mean = estado["in_pkts_sum"] / in_count
        in_deg = estado["in_count"]

        raw_features = np.array(
            [[out_b_mean, out_p_mean, out_deg, in_b_mean, in_p_mean, in_deg]],
            dtype=np.float32,
        )

        scaled_features = self.scaler_nodes.transform(raw_features).astype(np.float32)
        # Protección opcional contra outliers extremos:
        # scaled_features = np.clip(scaled_features, -10.0, 10.0)

        return scaled_features[0]

    # --------------------------------------------------------------------------
    # Construcción de grafo vivo
    # --------------------------------------------------------------------------
    def _construir_edge_index_historico_vivo(self) -> torch.Tensor:
        """
        Construye edge_index para Message Passing:
          - edge_index_base: topología histórica exportada desde entrenamiento/validación.
          - ventana_aristas_stream: aristas nuevas vistas desde que arrancó el detector.
        """
        if len(self.ventana_aristas_stream) == 0:
            return self.edge_index_base

        edge_index_stream = torch.tensor(
            list(self.ventana_aristas_stream),
            dtype=torch.long,
            device=self.device,
        ).t().contiguous()

        return torch.cat([self.edge_index_base, edge_index_stream], dim=1)

    # --------------------------------------------------------------------------
    # Edge features
    # --------------------------------------------------------------------------
    def _construir_edge_attr(self, datos_json: Dict[str, Any], delta_time: float) -> torch.Tensor:
        df_edge = pd.DataFrame(0.0, index=[0], columns=self.columnas_modelo)

        cols_numericas = [
            "duration",
            "orig_bytes",
            "resp_bytes",
            "orig_pkts",
            "resp_pkts",
            "missed_bytes",
        ]

        for col in cols_numericas:
            if col in df_edge.columns:
                df_edge.at[0, col] = self._safe_float(datos_json.get(col, 0.0))

        if "time_since_last_conn" in df_edge.columns:
            df_edge.at[0, "time_since_last_conn"] = delta_time

        estado = str(datos_json.get("conn_state", "OTH")).strip()
        if not estado:
            estado = "OTH"

        posibles_estados = [
            estado,
            estado.upper(),
            estado.lower(),
        ]

        estado_seteado = False
        for est in posibles_estados:
            col_estado = f"state_{est}"
            if col_estado in df_edge.columns:
                df_edge.at[0, col_estado] = 1.0
                estado_seteado = True
                break

        if not estado_seteado and "state_OTH" in df_edge.columns:
            df_edge.at[0, "state_OTH"] = 1.0

        edge_attr_np = self.scaler_edges.transform(df_edge.values).astype(np.float32)
        # Protección opcional contra outliers extremos:
        # edge_attr_np = np.clip(edge_attr_np, -10.0, 10.0)

        return torch.tensor(edge_attr_np, dtype=torch.float, device=self.device)

    # --------------------------------------------------------------------------
    # Enrutamiento de expertos
    # --------------------------------------------------------------------------
    def _dominio_por_servicio(self, srv: str) -> str:
        srv = self._normalizar_servicio(srv)
        tokens = {t.strip() for t in srv.split(",") if t.strip()}

        if srv in {"ssl", "http"} or tokens.intersection({"ssl", "http"}):
            return "web"

        if srv in {"dns", "ntp", "dhcp"} or tokens.intersection({"dns", "ntp", "dhcp"}):
            return "infra"

        if srv in {"smb", "gssapi", "ntlm", "dce_rpc"} or tokens.intersection(
            {"smb", "gssapi", "ntlm", "dce_rpc"}
        ):
            return "auth"

        return "gen"

    def _seleccionar_multiclase_por_servicio(self, srv: str) -> Tuple[str, List[Tuple[str, AdvancedEdgeExpert]], bool]:
        """
        Devuelve:
          - dominio solicitado
          - lista de modelos multiclase [(nombre, modelo)]
          - si se aplicó fallback a otro dominio
        """
        domain = self._dominio_por_servicio(srv)
        modelos = self.modelos_multiclase_por_dominio.get(domain, [])

        if modelos:
            return domain, modelos, False

        fallback_domain = str(self.model_manifest.get("fallback_multiclass_domain", "gen"))
        allow_fallback_manifest = bool(self.model_manifest.get("allow_multiclass_fallback", True))

        if self.allow_multiclass_fallback and allow_fallback_manifest and domain != fallback_domain:
            modelos_fallback = self.modelos_multiclase_por_dominio.get(fallback_domain, [])
            if modelos_fallback:
                if self.debug:
                    print(
                        f"[*] Fallback multiclase: dominio '{domain}' sin modelos; "
                        f"usando '{fallback_domain}'."
                    )
                return domain, modelos_fallback, True

        return domain, [], False

    # --------------------------------------------------------------------------
    # Inferencia principal
    # --------------------------------------------------------------------------
    def predecir_conexion(self, datos_json: Dict[str, Any]) -> Dict[str, Any]:
        # ----------------------------------------------------------------------
        # A. Extracción básica
        # ----------------------------------------------------------------------
        src_ip = self._normalizar_ip(datos_json.get("src_ip_zeek", "unknown"))
        dst_ip = self._normalizar_ip(datos_json.get("dest_ip_zeek", "unknown"))
        srv = self._normalizar_servicio(datos_json.get("service", "desconocido"))

        orig_bytes = self._safe_float(datos_json.get("orig_bytes", 0.0))
        resp_bytes = self._safe_float(datos_json.get("resp_bytes", 0.0))
        orig_pkts = self._safe_float(datos_json.get("orig_pkts", 0.0))
        resp_pkts = self._safe_float(datos_json.get("resp_pkts", 0.0))

        # ----------------------------------------------------------------------
        # B. IDs de nodos: reutiliza IDs históricos y asigna nuevos a OOV
        # ----------------------------------------------------------------------
        for ip in (src_ip, dst_ip):
            if ip not in self.mapeo_ips:
                self.mapeo_ips[ip] = self.next_node_id
                self.next_node_id += 1

        src_id = int(self.mapeo_ips[src_ip])
        dst_id = int(self.mapeo_ips[dst_ip])

        self._asegurar_capacidad_nodos(max(src_id, dst_id))

        # ----------------------------------------------------------------------
        # C. Actualización incremental de node features
        # ----------------------------------------------------------------------
        feat_src = self._actualizar_y_escalar_nodo(
            src_ip,
            orig_bytes,
            orig_pkts,
            es_origen=True,
        )
        feat_dst = self._actualizar_y_escalar_nodo(
            dst_ip,
            resp_bytes,
            resp_pkts,
            es_origen=False,
        )

        self.x_nodos_vivo[src_id, :] = torch.tensor(
            feat_src,
            dtype=torch.float,
            device=self.device,
        )
        self.x_nodos_vivo[dst_id, :] = torch.tensor(
            feat_dst,
            dtype=torch.float,
            device=self.device,
        )

        # ----------------------------------------------------------------------
        # D. Time delta y edge features
        # ----------------------------------------------------------------------
        clave_conn = f"{src_ip}-{dst_ip}"

        ts_actual = self._safe_float(datos_json.get("ts", 0.0))
        if clave_conn in self.memoria_timestamps:
            delta_time = ts_actual - self.memoria_timestamps[clave_conn]
            if delta_time < 0:
                delta_time = 0.0
        else:
            delta_time = 0.0

        self.memoria_timestamps[clave_conn] = ts_actual

        edge_attr_pred = self._construir_edge_attr(datos_json, delta_time)

        # ----------------------------------------------------------------------
        # E. Topología viva: base histórica + ventana streaming
        # ----------------------------------------------------------------------
        self.ventana_aristas_stream.append((src_id, dst_id))
        edge_index_historico_vivo = self._construir_edge_index_historico_vivo()

        edge_index_pred = torch.tensor(
            [[src_id], [dst_id]],
            dtype=torch.long,
            device=self.device,
        )

        # ----------------------------------------------------------------------
        # F. Selección dinámica de expertos
        # ----------------------------------------------------------------------
        domain, modelos_multiclase, used_fallback = self._seleccionar_multiclase_por_servicio(srv)

        # ----------------------------------------------------------------------
        # G. Memoria de aristas comprometidas
        # ----------------------------------------------------------------------
        esta_comprometida = False
        if clave_conn in self.aristas_comprometidas:
            tiempo_desde_ataque = ts_actual - self.aristas_comprometidas[clave_conn]
            if 0.0 <= tiempo_desde_ataque < 300.0:
                esta_comprometida = True
            else:
                del self.aristas_comprometidas[clave_conn]

        # ----------------------------------------------------------------------
        # H. Inferencia
        # ----------------------------------------------------------------------
        with torch.no_grad():
            out_bin = self.bin_sage(
                self.x_nodos_vivo,
                edge_index_historico_vivo,
                edge_index_pred,
                edge_attr_pred,
            )

            probs_bin = F.softmax(out_bin, dim=1)[0]
            prob_benigno = float(probs_bin[0].item())
            prob_ataque = float(probs_bin[1].item())

            if self.debug:
                logits_var = out_bin.var(dim=1).mean().item()
                logits_mean = out_bin.mean().item()
                print(f"Dominio servicio: {domain} | fallback: {used_fallback}")
                print(f"Modelos multiclase: {[name for name, _ in modelos_multiclase]}")
                print(f"Stats Nodo Origen ({src_ip}/{src_id}): {self.x_nodos_vivo[src_id].detach().cpu().numpy()}")
                print(f"Logits BIN: {out_bin.detach().cpu().numpy()}")
                print(f"Varianza media logits: {logits_var:.6f}")
                print(f"Media logits: {logits_mean:.6f}")
                print(f"Prob benigno: {prob_benigno:.4f} | Prob ataque: {prob_ataque:.4f}")

            # Si el portero dice Benigno y no hay memoria de compromiso reciente, cortamos cascada.
            if prob_ataque < self.umbral_binario and not esta_comprometida:
                return {
                    "label_binary": False,
                    "label_tactic": "Benigno",
                    "confidence": round(prob_benigno, 4),
                    "domain": domain,
                    "fallback_used": used_fallback,
                }

            # Si el portero alerta pero no existe ningún modelo multiclase válido.
            if len(modelos_multiclase) == 0:
                self.aristas_comprometidas[clave_conn] = ts_actual
                return {
                    "label_binary": True,
                    "label_tactic": "Ataque_No_Clasificado",
                    "confidence": round(prob_ataque, 4),
                    "domain": domain,
                    "fallback_used": used_fallback,
                    "reason": "no_multiclass_model_available",
                }

            # Fase 2: soft voting multiclase.
            probs_acum = torch.zeros(self.num_clases, dtype=torch.float, device=self.device)

            for _, modelo_multi in modelos_multiclase:
                out_multi = modelo_multi(
                    self.x_nodos_vivo,
                    edge_index_historico_vivo,
                    edge_index_pred,
                    edge_attr_pred,
                )
                probs_acum += F.softmax(out_multi, dim=1)[0]

            probs_final = probs_acum / len(modelos_multiclase)

            # En fase multiclase se fuerza a no emitir "Benigno".
            probs_final[self.id_benigno] = 0.0

            # Threshold moving: si umbral < 1, aumenta sensibilidad; si > 1, penaliza.
            probs_ajustadas = probs_final / torch.clamp(self.umbrales_multiclase, min=1e-6)

            tactic_idx = int(torch.argmax(probs_ajustadas).item())
            tactic_name = self.encoder.inverse_transform([tactic_idx])[0]

            self.aristas_comprometidas[clave_conn] = ts_actual

            confianza_final = float(probs_final[tactic_idx].item())
            if prob_ataque < self.umbral_binario and esta_comprometida:
                confianza_final *= 0.8

            return {
                "label_binary": True,
                "label_tactic": tactic_name,
                "confidence": round(confianza_final, 4),
                "domain": domain,
                "fallback_used": used_fallback,
                "models_used": [name for name, _ in modelos_multiclase],
            }
