import os
import pandas as pd
import time
import json
from detector_mitre import DetectorMITRE

# ==============================================================================
# CONFIGURACIÓN DE LA SIMULACIÓN
# ==============================================================================
dir_actual = os.path.dirname(os.path.abspath(__file__))
ruta_datos = os.path.abspath(os.path.join(dir_actual, "..", "data", "processed"))
RUTA_TEST = os.path.join(ruta_datos, "dataset_simulacion_kafka.parquet")
MUESTRAS_A_SIMULAR = 100000  # Cambiar valor probar con todo el dataset
MOSTRAR_BENIGNOS = True   # Cambiar a True para ver el log de cada conexión limpia

def iniciar_simulacion():
    print("======================================================")
    print(" INICIANDO SIMULADOR DE STREAMING KAFKA / ZEEK")
    print("======================================================\n")

    # 1. Inicializar el Detector (Esto carga los modelos en GPU/CPU y la RAM)
    print("[*] Levantando el Motor de Inferencia en Cascada...")
    start_init = time.time()
    detector = DetectorMITRE()
    print(f"[✓] Detector inicializado en {time.time() - start_init:.2f} segundos.\n")

    # 2. Cargar el dataset de prueba
    print(f"[*] Cargando dataset de simulación: {RUTA_TEST}")
    try:
        df_test = pd.read_parquet(RUTA_TEST)
        print(f"[✓] Dataset cargado. Total de conexiones disponibles: {len(df_test):,}\n")
    except FileNotFoundError:
        print(f"[X] Error: No se encuentra el archivo en {RUTA_TEST}")
        return

    # Para garantizar que probamos la detección, vamos a forzar que en estas 
    # primeras N muestras haya algunos ataques reales si el dataset está muy desbalanceado.
    # (Descomenta la siguiente línea si quieres probar solo con ataques puros)
    # df_test = df_test[df_test['is_attack'] == 1]

    print(f"[*] Comenzando ingesta de las primeras {MUESTRAS_A_SIMULAR} conexiones...")
    print("------------------------------------------------------")
    
    tiempos_inferencia = []
    alertas_generadas = 0

    # 3. Bucle de Simulación (Streaming de eventos)
    #for index, row in df_test.head(MUESTRAS_A_SIMULAR).iterrows():
    for index, row in df_test.iterrows():
        
        # Transformar la fila de Pandas a un Diccionario (Simula el JSON de Kafka)
        payload_json = row.to_dict()

        # Marcar inicio de tiempo de inferencia
        t_start = time.time()

        # ==========================================
        # LLAMADA AL MODELO GNN
        # ==========================================
        resultado = detector.predecir_conexion(payload_json)
        
        # Registrar latencia en milisegundos
        latencia_ms = (time.time() - t_start) * 1000
        tiempos_inferencia.append(latencia_ms)

        # 4. Gestión de la Alerta (Salida por consola)
        if resultado["es_ataque"]:
            alertas_generadas += 1
            src = payload_json.get('src_ip_zeek', 'Unknown')
            dst = payload_json.get('dest_ip_zeek', 'Unknown')
            srv = payload_json.get('service', 'desc')
            
            # Formateo de alerta crítica
            print(f"[🚨 ALERTA] {src} -> {dst} | Srv: {srv.upper()} "
                  f"| Táctica: {resultado['tactic']} (Conf: {resultado['confianza']:.2f}) "
                  f"| Latencia: {latencia_ms:.2f} ms")
        else:
            if MOSTRAR_BENIGNOS:
                print(f"[✓ OK] Conexión limpia | Latencia: {latencia_ms:.2f} ms")

    # ==============================================================================
    # MÉTRICAS FINALES PARA EL TFM
    # ==============================================================================
    print("\n======================================================")
    print(" RESUMEN DE RENDIMIENTO DE INFERENCIA")
    print("======================================================")
    print(f"Total conexiones procesadas : {MUESTRAS_A_SIMULAR}")
    print(f"Alertas MITRE emitidas      : {alertas_generadas}")
    print(f"Latencia Media por conexión : {sum(tiempos_inferencia) / len(tiempos_inferencia):.2f} ms")
    print(f"Latencia Mínima             : {min(tiempos_inferencia):.2f} ms")
    print(f"Latencia Máxima             : {max(tiempos_inferencia):.2f} ms")
    print("======================================================")

if __name__ == "__main__":
    iniciar_simulacion()