import os
from fastapi import FastAPI
from pydantic import BaseModel, Field
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

# =======================
# DEFINICIÓN DEL ESQUEMAS
# =======================
class ZeekEvent(BaseModel):
    ts: float = Field(..., description="Timestamp epoch del evento", example=1711178855.790999)
    src_ip_zeek: str = Field(..., description="IP de origen (id.orig_h en Zeek)", example="192.168.1.100")
    dest_ip_zeek: str = Field(..., description="IP de destino (id.resp_h en Zeek)", example="10.0.0.5")
    service: str = Field(..., description="Protocolo de la capa de aplicación", example="http")
    duration: float = Field(0.0, description="Duración de la conexión", example=0.05)
    orig_bytes: int = Field(0, description="Bytes enviados por el origen", example=45)
    resp_bytes: int = Field(0, description="Bytes enviados por el destino", example=90)
    orig_pkts: int = Field(0, description="Paquetes enviados por el origen", example=1)
    resp_pkts: int = Field(0, description="Paquetes enviados por el destino", example=1)
    missed_bytes: int = Field(0, description="Bytes perdidos en la conexión", example=0)
    conn_state: str = Field(..., description="Estado de la conexión", example="SF")

class InferenceResponse(BaseModel):
    label_binary: bool = Field(..., description="Resultado de la Fase 1. Indica si el flujo es malicioso", example=False)
    label_tactic: str = Field(..., description="Resultado de la Fase 2. Táctica MITRE ATT&CK identificada o 'Benigno'", example="Benigno")
    confidence: float = Field(..., description="Nivel de confianza o probabilidad promedio devuelto por el ensemble", example=0.9852)

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

@app.post(
    "/classify", 
    summary="Clasificar evento de red Zeek",
    response_model=InferenceResponse
)
async def classify_connection(event: ZeekEvent):
    # Devuelve un diccionario nativo de Python validado y filtrado
    payload = event.model_dump()
    
    # Se pasa el JSON al detector
    resultado = detector.predecir_conexion(payload)
    
    return resultado

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)