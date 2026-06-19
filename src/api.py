import os
from fastapi import FastAPI, Request
import uvicorn
from detector_mitre import DetectorMITRE

# Variables de entorno para las rutas de los volúmenes
MODEL_PATH = os.getenv("MODEL_PATH", "/models")
DATA_PATH = os.getenv("DATA_PATH", "/data")

app = FastAPI(
    title="MITRE ATT&CK Detector API",
    description="API de inferencia mediante GNN para flujos de red Zeek",
    version="1.0.0"
)

# Instancia global del motor de inferencia
detector = None

@app.on_event("startup")
def startup_event():
    global detector
    print(f"[*] Inicializando motor de inferencia...")
    print(f"    - Ruta Modelos: {MODEL_PATH}")
    print(f"    - Ruta Datos: {DATA_PATH}")
    
    # Instanciar el detector apuntando a las rutas de los volúmenes
    detector = DetectorMITRE(ruta_modelos=MODEL_PATH, ruta_datos=DATA_PATH)
    print("[✓] Motor listo para recibir peticiones.")

@app.post("/classify")
async def classify_connection(request: Request):
    # Se recibe el JSON plano de la petición
    payload = await request.json()
    
    # Se pasa el JSON al detector
    resultado = detector.predecir_conexion(payload)
    
    return resultado

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)