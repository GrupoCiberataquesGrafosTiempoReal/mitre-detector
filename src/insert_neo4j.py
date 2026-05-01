import json
import numpy as np
from detector_mitre import DetectorMITRE as detector

# Al iniciar su script, carga las columnas que tú le exportaste
with open("../data/processed/columnas_modelo.json", "r") as f:
    columnas_esperadas = json.load(f)

# Las variables numéricas base
variables_base = ['duration', 'orig_bytes', 'resp_bytes', 'orig_pkts', 'resp_pkts', 'missed_bytes', 'time_since_last_conn']

def preparar_vector_para_ia(row_kafka):
    vector = []
    
    # 1. Añadir las variables numéricas (si viene nulo, poner 0.0)
    for col in variables_base:
        vector.append(float(row_kafka.get(col, 0.0) or 0.0))
    
    # 2. TRANSFORMACIÓN CRÍTICA: Construir el One-Hot Encoding a mano
    # El estado que viene en el paquete de Kafka
    estado_actual = str(row_kafka.get('conn_state', 'OTH'))
    
    # Recorremos el resto de columnas esperadas (que son las tipo 'state_S0', 'state_SF', etc.)
    columnas_dummies = columnas_esperadas[len(variables_base):]
    
    for col_dummy in columnas_dummies:
        # col_dummy es algo como "state_S0"
        estado_dummy = col_dummy.split('state_')[1] 
        if estado_actual == estado_dummy:
            vector.append(1.0)
        else:
            vector.append(0.0)
            
    return vector

# En su bucle de Kafka:
for mensaje in kafka:
    datos = mensaje.value
    
    # Transforma el paquete al vector matemático
    datos['vector_numerico'] = preparar_vector_para_ia(datos)
    
    # Le pregunta a tu modelo
    resultado_ia = detector.predecir_conexion(datos)
    
    # Inserción en Neo4j
    query = """
    MERGE (src:IP {address: $src_ip})
    MERGE (dst:IP {address: $dst_ip})
    CREATE (src)-[:CONECTA {
        service: $service,
        is_attack: $is_attack,
        mitre_tactic: $tactic,
        confidence: $conf,
        source: 'inferencia_ia',
        etiqueta_oculta_real: $real_label,  // Para poder comparar luego en la demo
        timestamp: $ts
    }]->(dst)
    """
    # Ejecuta el query con resultado_ia['es_ataque'], resultado_ia['tactica_mitre'], etc.


# # Lógica para el script de ingesta histórica
# for index, row in df_historico.iterrows():
#     query = """
#     MERGE (src:IP {address: $src_ip})
#     MERGE (dst:IP {address: $dst_ip})
#     CREATE (src)-[:CONECTA {
#         service: $service,
#         is_attack: $is_attack,
#         mitre_tactic: $tactic,
#         confidence: 1.0,
#         source: 'historico',
#         timestamp: $ts
#     }]->(dst)
#     """
#     neo4j_session.run(query, 
#                       src_ip=row['src_ip_zeek'], 
#                       dst_ip=row['dest_ip_zeek'],
#                       service=row['service'],
#                       is_attack=bool(row['is_attack']),
#                       tactic=row['label_tactic'],
#                       ts=row['ts'])
    



