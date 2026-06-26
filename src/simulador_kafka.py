import os
import pandas as pd
import time
import json
import numpy as np
from detector_mitre import DetectorMITRE

# ==============================================================================
# CONFIGURACIÓN DE LA SIMULACIÓN
# ==============================================================================
dir_actual = os.path.dirname(os.path.abspath(__file__))
ruta_datos = os.path.abspath(os.path.join(dir_actual, "..", "data", "processed"))
RUTA_TEST = os.path.join(ruta_datos, "dataset_simulacion_kafka.parquet")
MUESTRAS_A_SIMULAR = 100000
SALTAR_N_FILAS = 0#160000 
MOSTRAR_BENIGNOS = True   

def iniciar_simulacion():
    print("======================================================")
    print(" INICIANDO SIMULADOR DE STREAMING KAFKA / ZEEK")
    print("======================================================\n")

    print("[*] Levantando el Motor de Inferencia en Cascada...")
    start_init = time.time()
    detector = DetectorMITRE(max_window_size=50000)
    print(f"[✓] Detector inicializado en {time.time() - start_init:.2f} segundos.\n")

    print(f"[*] Cargando dataset de simulación: {RUTA_TEST}")
    try:
        df_test = pd.read_parquet(RUTA_TEST)
        print(f"[✓] Dataset cargado. Total de conexiones disponibles: {len(df_test):,}\n")
    except FileNotFoundError:
        print(f"[X] Error: No se encuentra el archivo en {RUTA_TEST}")
        return

    print(f"[*] Comenzando ingesta de las primeras {MUESTRAS_A_SIMULAR} conexiones...")
    print("------------------------------------------------------")
    
    tiempos_inferencia = []
    alertas_generadas = 0
    ultimo_ts = None

    fin_filas = SALTAR_N_FILAS + MUESTRAS_A_SIMULAR

    #primer_benigno_idx = df_test[df_test['is_attack'] == False].index[0]
    #print(f"Primer elemento benigno: {primer_benigno_idx}")

    for index, row in df_test.iloc[SALTAR_N_FILAS : fin_filas].iterrows():
    #for index, row in df_test.loc[primer_benigno_idx:].iterrows():
        payload_json = row.to_dict()
        
        # --- EXTRAER GROUND TRUTH (VALORES REALES) ---
        es_ataque_real = bool(payload_json.get('is_attack', 0.0))
        tactica_real = payload_json.get('label_tactic', 'Benigno')

        try:
            ts_actual = float(payload_json.get('ts', 0.0))
            if np.isnan(ts_actual): ts_actual = 0.0
        except:
            ts_actual = 0.0

        if ultimo_ts is not None and ts_actual > ultimo_ts:
            delta_real = ts_actual - ultimo_ts
            time.sleep(min(delta_real, 0.5))
            
        ultimo_ts = ts_actual

        t_start = time.time()
        resultado = detector.predecir_conexion(payload_json)
        latencia_ms = (time.time() - t_start) * 1000
        tiempos_inferencia.append(latencia_ms)

        # --- GESTIÓN DE SALIDA CON COMPARACIÓN ---
        src = payload_json.get('src_ip_zeek', 'Unknown')
        dst = payload_json.get('dest_ip_zeek', 'Unknown')
        srv = payload_json.get('service', 'desc')

        if resultado["label_binary"]:
            alertas_generadas += 1
            
            # Comprobar si hemos acertado el ataque y la táctica
            if not es_ataque_real:
                match_str = f"❌ FALSO POSITIVO (Real: {tactica_real})"
            elif tactica_real == resultado['label_tactic']:
                match_str = "✅ ACIERTO TÁCTICA"
            else:
                match_str = f"⚠️ ATAQUE DETECTADO, FALLO TÁCTICA (Real: {tactica_real})"
                
            # TEMPORAL:
            #if resultado['label_tactic'] != "Credential Access":
            print(f"[🚨 ALERTA] {src} -> {dst} | Srv: {srv.upper()} "
                f"| Pred: {resultado['label_tactic']} (Conf: {resultado['confidence']:.2f}) "
                f"| {match_str} | Lat: {latencia_ms:.2f} ms")
        else:
            if MOSTRAR_BENIGNOS:
                if not es_ataque_real:
                    match_str = f"✅ ACIERTO - Conf: {resultado['confidence']:.2f}"
                else:
                    match_str = f"❌ FALSO NEGATIVO (Era Ataque: {tactica_real}) - Conf: {resultado['confidence']:.2f}"
                    
                print(f"[✅ OK] {src} -> {dst} | Pred: Benigno | {match_str} | Lat: {latencia_ms:.2f} ms - Conf: {resultado['confidence']:.2f}")

    print("\n======================================================")
    print(" RESUMEN DE RENDIMIENTO DE INFERENCIA")
    print("======================================================")
    print(f"Total conexiones procesadas : {MUESTRAS_A_SIMULAR}")
    print(f"Alertas MITRE emitidas      : {alertas_generadas}")
    print(f"Latencia Media por conexión : {sum(tiempos_inferencia) / len(tiempos_inferencia):.2f} ms")
    print("======================================================")

if __name__ == "__main__":
    iniciar_simulacion()