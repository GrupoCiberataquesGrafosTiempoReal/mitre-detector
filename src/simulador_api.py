import os
import pandas as pd
import time
import numpy as np
import requests # <--- Necesario para hacer las peticiones HTTP

# ==============================================================================
# CONFIGURACIÓN DE LA SIMULACIÓN
# ==============================================================================
API_URL = "http://host.docker.internal:8080/classify"

dir_actual = os.path.dirname(os.path.abspath(__file__))
ruta_datos = os.path.abspath(os.path.join(dir_actual, "..", "data", "processed"))
RUTA_TEST = os.path.join(ruta_datos, "dataset_simulacion_kafka.parquet")
MUESTRAS_A_SIMULAR = 1000  # Bajamos un poco para la prueba de red
SALTAR_N_FILAS = 0  
MOSTRAR_BENIGNOS = True   

def iniciar_simulacion_api():
    print("======================================================")
    print(" INICIANDO CLIENTE KAFKA -> API REST (DOCKER)")
    print(f" URL Destino: {API_URL}")
    print("======================================================\n")

    print(f"[*] Cargando dataset de simulación: {RUTA_TEST}")
    try:
        df_test = pd.read_parquet(RUTA_TEST)
        print(f"[✓] Dataset cargado. Total de conexiones disponibles: {len(df_test):,}\n")
    except FileNotFoundError:
        print(f"[X] Error: No se encuentra el archivo en {RUTA_TEST}")
        return

    # Limpiar NaNs de Pandas para que el JSON sea válido en HTTP
    df_test = df_test.replace({np.nan: None})

    print(f"[*] Comenzando envío de las primeras {MUESTRAS_A_SIMULAR} conexiones a la API...")
    print("------------------------------------------------------")
    
    tiempos_red_inferencia = []
    alertas_generadas = 0
    ultimo_ts = None
    fin_filas = SALTAR_N_FILAS + MUESTRAS_A_SIMULAR

    for index, row in df_test.iloc[SALTAR_N_FILAS : fin_filas].iterrows():
        payload_json = row.to_dict()
        
        # --- EXTRAER GROUND TRUTH (VALORES REALES) ---
        es_ataque_real = bool(payload_json.get('is_attack', False))
        tactica_real = payload_json.get('label_tactic', 'Benigno')

        # Simulación de tiempo real
        ts_actual = payload_json.get('ts') or 0.0
        if ultimo_ts is not None and ts_actual > ultimo_ts:
            delta_real = ts_actual - ultimo_ts
            time.sleep(min(delta_real, 0.5))
        ultimo_ts = ts_actual

        # --- PETICIÓN HTTP A LA API (Reemplaza a detector.predecir_conexion) ---
        t_start = time.time()
        try:
            # Enviamos el POST al contenedor Docker
            response = requests.post(API_URL, json=payload_json, timeout=2.0)
            
            if response.status_code == 200:
                resultado = response.json()
            else:
                print(f"[X] Error de la API HTTP {response.status_code}: {response.text}")
                continue
                
        except requests.exceptions.ConnectionError:
            print("[X] ERROR CRÍTICO: No se puede conectar a la API. ¿Está el contenedor Docker corriendo en el puerto 8080?")
            break
        except requests.exceptions.Timeout:
            print("[X] ERROR: Timeout. La API tardó demasiado en responder.")
            continue

        latencia_total_ms = (time.time() - t_start) * 1000
        tiempos_red_inferencia.append(latencia_total_ms)

        # --- GESTIÓN DE SALIDA CON COMPARACIÓN ---
        src = payload_json.get('src_ip_zeek', 'Unknown')
        dst = payload_json.get('dest_ip_zeek', 'Unknown')
        srv = payload_json.get('service', 'desc')

        if resultado["es_ataque"]:
            alertas_generadas += 1
            if not es_ataque_real:
                match_str = f"❌ FALSO POSITIVO (Real: {tactica_real})"
            elif tactica_real == resultado['tactic']:
                match_str = "✅ ACIERTO TÁCTICA"
            else:
                match_str = f"⚠️ ATAQUE DETECTADO, FALLO TÁCTICA (Real: {tactica_real})"
                
            print(f"[🚨 ALERTA] {src} -> {dst} | Srv: {srv.upper()} "
                  f"| Pred: {resultado['tactic']} (Conf: {resultado['confianza']:.2f}) "
                  f"| {match_str} | Latencia total: {latencia_total_ms:.2f} ms")
        else:
            if MOSTRAR_BENIGNOS:
                if not es_ataque_real:
                    match_str = f"✅ ACIERTO - Conf: {resultado['confianza']:.2f}"
                else:
                    match_str = f"❌ FALSO NEGATIVO (Era Ataque: {tactica_real}) - Conf: {resultado['confianza']:.2f}"
                    
                print(f"[✅ OK] {src} -> {dst} | Pred: Benigno | {match_str} | Latencia total: {latencia_total_ms:.2f} ms")

    print("\n======================================================")
    print(" RESUMEN DE RENDIMIENTO (RED + INFERENCIA)")
    print("======================================================")
    print(f"Total eventos enviados      : {len(tiempos_red_inferencia)}")
    print(f"Alertas MITRE recibidas     : {alertas_generadas}")
    if tiempos_red_inferencia:
        print(f"Latencia Media (Round-Trip) : {sum(tiempos_red_inferencia) / len(tiempos_red_inferencia):.2f} ms")
    print("======================================================")

if __name__ == "__main__":
    iniciar_simulacion_api()