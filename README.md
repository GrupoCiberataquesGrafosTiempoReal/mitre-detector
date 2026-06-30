# 🛡️ Sistema de Detección de Intrusiones (IDS) Basado en Grafos (GNN) y MITRE ATT&CK

Este repositorio contiene el código fuente, la experimentación y el motor de inferencia para un **clasificador jerárquico en cascada** capaz de detectar ciberataques en flujos de red Zeek en tiempo real, mapeándolos a tácticas de la matriz MITRE ATT&CK mediante Redes Neuronales de Grafos, principalmente GraphSAGE y GAT.

---

## 📂 Estructura del Repositorio

- **`data/`**: Contiene los datasets utilizados en el proyecto. Se divide en `raw/` para los archivos originales y `processed/` para los artefactos derivados, como el dataset histórico estratificado para Neo4j, el dataset de test estratificado para simulación Kafka/API y la plantilla `columnas_modelo.json`.
- **`models/`**: Directorio donde residen los pesos entrenados (`.pth`), la configuración de umbrales (`config/`), el manifiesto dinámico de modelos (`config/model_manifest.json`), los artefactos de normalización y estado histórico (`encoders/`) y los resultados experimentales de evaluación (`evaluation/`).
- **`notebooks/`**: Cuadernos interactivos para la exploración del tráfico, partición temporal estratificada, construcción del grafo, entrenamiento de los modelos predictivos y comparación experimental con modelos tabulares baseline.
- **`src/`**: Código fuente de producción.
  - `detector_mitre.py`: Núcleo del motor inductivo/stateful de inferencia.
  - `api.py`: Servicio web asíncrono basado en FastAPI para consumir las predicciones.
  - `insert_neo4j.py`: Código de pruebas para inserción de subgrafos en Neo4j.
  - `simulador_kafka.py`: Código de pruebas para evaluar predicciones con modelos preentrenados sobre eventos simulados.
  - `simulador_api.py`: Código de pruebas para validar predicciones contra la imagen Docker desplegada en localhost.
- **`Dockerfile` / `docker-compose.yml`**: Recetas de contenedorización para despliegue en CPU o CUDA.
- **`pyproject.toml`**: Dependencias de Python para Poetry.

---

## 🚀 Guía de Despliegue (Inferencia como Servicio)

El motor se expone a través de una API REST basada en FastAPI y encapsulada en Docker, permitiendo intercambiar modelos actualizando únicamente los volúmenes montados.

### 1. Preparación de Volúmenes

Antes de levantar el servicio, asegúrate de tener los modelos entrenados y los archivos de metadatos en tu máquina local. El `docker-compose.yml` montará estas dos carpetas en el contenedor:

- `./models`: Debe contener los archivos `.pth` exportados, las subcarpetas `encoders/` y `config/`, así como el manifiesto `config/model_manifest.json`.
- `./data`: Debe contener el archivo `columnas_modelo.json`.

Los modelos multiclase se cargan dinámicamente a partir del manifiesto. Por tanto, no es necesario que existan pesos para todos los dominios: los expertos que no hayan podido entrenarse por falta de muestras no se exportan y no se cargan en inferencia.

### 2. Construcción y Despliegue del Servicio

Puedes levantar la infraestructura de dos maneras, dependiendo de tu entorno:

**Opción A: Usando Docker Compose**

Esto utilizará la configuración en `docker-compose.yml`. En la raíz del repositorio, ejecuta:

```bash
docker compose up -d --build
```

> **Nota GPU (CUDA):** Si el servidor host dispone de una tarjeta NVIDIA y `nvidia-container-toolkit` instalado, puedes descomentar el bloque `deploy: resources:` en el `docker-compose.yml` para activar aceleración por hardware. También deberás ajustar `BUILD_TYPE` a `gpu`.

**Opción B: Usando Docker CLI manualmente**

Si prefieres gestionar el contenedor a mano, construye la imagen y levántala montando los volúmenes. Ajusta la ruta a los volúmenes si es necesario:

```bash
# Para la versión CPU:
docker build --no-cache -t mitre-detector:cpu .

# Para la versión compatible con CUDA:
docker build --no-cache --build-arg BUILD_TYPE=gpu -t mitre-detector:gpu .

docker run -d \
  --name mitre-detector \
  -p 8080:8080 \
  -v "$(pwd)/models:/models:ro" \
  -v "$(pwd)/data:/data:ro" \
  mitre-detector:cpu
```

Para usar la imagen GPU, sustituye `mitre-detector:cpu` por `mitre-detector:gpu` y asegúrate de ejecutar Docker con soporte NVIDIA según la configuración de tu entorno.

### 3. Consumir la API

El contenedor expone el servicio en el puerto `8080`. Puedes ver la documentación interactiva de Swagger UI y probar el endpoint enviando un JSON con estructura compatible con los registros Zeek:

- **Swagger UI:** `http://localhost:8080/docs`
- **Endpoint de predicción:** `POST http://localhost:8080/classify`

---

## 🧠 Arquitectura y Metodología del Sistema (GNN)

### 1. Clasificador Jerárquico en Cascada

Para abordar el reto inherente al análisis de red, donde el tráfico legítimo puede superar ampliamente a las anomalías, el sistema divide la inferencia en dos fases:

- **Fase 1 (Detección binaria):** Un modelo global basado en GraphSAGE discrimina entre tráfico benigno y tráfico malicioso. Actúa como filtro inicial antes de activar la clasificación táctica.
- **Fase 2 (Clasificación multiclase MITRE):** Recibe únicamente el tráfico catalogado como malicioso por la fase binaria. En esta etapa, el sistema enruta el evento hacia expertos multiclase especializados por dominio y, cuando no existe un experto específico válido, utiliza un experto global entrenado con todas las muestras maliciosas disponibles.

Esta arquitectura evita depender de modelos no entrenados: los expertos que carecen de muestras suficientes no se exportan y quedan registrados como omitidos en `model_manifest.json`.

### 2. Enrutador Topológico por Servicios

El tráfico se segmenta semánticamente antes de entrar a los expertos multiclase:

1. **Experto Web:** Tráfico `ssl` y `http`.
2. **Experto Infraestructura:** Servicios base como `dns`, `ntp` y `dhcp`.
3. **Experto Autenticación:** Protocolos de Windows e identidad como `smb`, `gssapi`, `ntlm` y `dce_rpc`.
4. **Experto Generalista:** Tráfico desconocido, anómalo o residual.
5. **Experto Global:** Modelo multiclase de respaldo entrenado con todos los ataques disponibles. Se utiliza cuando un dominio no dispone de experto específico entrenado o exportado.

### 3. Inferencia Inductiva y Correlación Temporal

El sistema opera en modo stateful sobre eventos de streaming, reutilizando el contexto histórico disponible y actualizando el estado de la red al vuelo:

- **Características de nodos y aristas:** Se calculan estadísticas bidireccionales de las IPs y variables propias de Zeek, incluyendo `conn_state`.
- **Ingeniería temporal:** La variable `time_since_last_conn` mide el tiempo transcurrido desde la última comunicación observada para el mismo par origen-destino, permitiendo capturar patrones temporales como beaconing o recurrencia anómala.
- **Topología histórica + ventana reciente:** El motor carga una topología histórica base de entrenamiento-validación (`edge_index_entrenamiento.pt`) y la combina con una ventana deslizante de nuevas aristas observadas durante la ejecución.
- **Memoria de aristas comprometidas:** Si el motor detecta una conexión maliciosa entre dos IPs, esa arista queda marcada temporalmente como comprometida. Durante una ventana breve, nuevas conexiones del mismo par pueden forzar el paso a la fase multiclase aunque el portero binario las considere benignas, aplicando una penalización de confianza para reflejar la incertidumbre.

### 4. Diversidad de Modelos (GraphSAGE + GAT) y Soft Voting

El núcleo predictivo combina arquitecturas de PyTorch Geometric:

- **GraphSAGE:** Captura patrones estructurales mediante agregación inductiva de vecindarios.
- **GAT (Atención):** Configurado con mecanismos de atención para ponderar conexiones relevantes dentro del vecindario.

En la Fase 2, cuando un dominio dispone de varios expertos válidos, se ejecuta un **Soft Voting Ensemble Intra-Etapa**, promediando las probabilidades de los modelos disponibles. La clase `Benigno` se fuerza a `0.0` en esta fase, ya que la decisión binaria previa ya ha determinado que el evento requiere clasificación táctica.

### 5. Mitigación del Desbalanceo y Control de Fuga de Datos

- **Partición temporal estratificada por clase:** Debido a la distribución temporal del dataset, no se aplica un corte cronológico global estricto. En su lugar, cada clase se ordena temporalmente y se divide en entrenamiento, validación y test, conservando el orden interno de cada táctica y garantizando representación de todas las clases.
- **Aumentación topológica en fase multiclase:** Para tácticas minoritarias, se duplican índices de predicción durante el cálculo de la pérdida, sin duplicar aristas en el tensor de Message Passing. Así se refuerza la señal supervisada sin alterar artificialmente la topología real del grafo.
- **Message Passing global y pérdida por experto:** Durante el entrenamiento, los modelos reciben contexto mediante el grafo histórico global de entrenamiento, mientras que la función de pérdida se restringe a las aristas del dominio correspondiente. Esto mantiene el contexto estructural común y permite especialización semántica por experto.
- **Focal Loss ponderada dinámicamente:** Se aplica una pérdida focal con pesos calculados según la frecuencia suavizada de las clases presentes, reduciendo el sesgo hacia clases mayoritarias y reforzando tácticas minoritarias.
- **Early Stopping guiado por Macro-F1:** La parada temprana se basa en Macro-F1 de validación, priorizando un equilibrio entre precisión y sensibilidad frente a la simple exactitud global.

---

## 📦 Artefactos principales exportados

El entrenamiento genera un paquete reproducible para inferencia:

- `models/*.pth`: Pesos de modelos realmente entrenados.
- `models/config/model_manifest.json`: Manifiesto de modelos exportados, omitidos y reglas de fallback.
- `models/config/umbrales_optimos.json`: Umbrales y coeficientes de sensibilidad para inferencia.
- `models/encoders/scaler_edges.pkl`: Escalador de características de arista.
- `models/encoders/scaler_nodes.pkl`: Escalador de características de nodo.
- `models/encoders/encoder_tactics.pkl`: Codificador de tácticas MITRE.
- `models/encoders/mapeo_ips.pkl`: Mapeo histórico de IPs a IDs de nodo.
- `models/encoders/x_nodos_entrenamiento.pt`: Características históricas de nodos.
- `models/encoders/edge_index_entrenamiento.pt`: Topología histórica base de entrenamiento-validación, excluyendo test.
- `data/processed/columnas_modelo.json`: Orden exacto de columnas esperado por el motor de inferencia.
- `models/evaluation/comparativa_gnn_vs_baselines.csv`: Tabla comparativa entre la GNN jerárquica y los modelos tabulares baseline.
- `models/evaluation/tiempos_entrenamiento_gnn.csv`: Tiempos de entrenamiento por experto GNN.
- `models/evaluation/tiempo_entrenamiento_total_gnn.json`: Tiempo total de entrenamiento, dispositivo utilizado, memoria máxima CUDA si aplica y metadatos de reproducibilidad.

---

## 🧪 Evaluación

La evaluación final se realiza sobre el subconjunto de test estratificado. El modelo GNN se evalúa de forma inductiva: las aristas de test se predicen utilizando la topología histórica disponible, pero no se incorporan al grafo base de Message Passing exportado.

Además de la evaluación del sistema GNN, el notebook incluye una comparación experimental con modelos tabulares baseline, principalmente **Random Forest** y **XGBoost**. Estos modelos reciben atributos de arista y características agregadas de los nodos origen/destino, pero no disponen de Message Passing ni de actualización topológica de embeddings. La comparación permite cuantificar si la arquitectura basada en grafos aporta valor frente a clasificadores tabulares fuertes entrenados sobre una representación equivalente sin contexto relacional.

Se reportan métricas globales y por clase, con especial atención a:

- **Accuracy global**, como referencia general.
- **Macro-F1 global**, para evitar que la clase mayoritaria o las tácticas dominantes oculten errores en clases minoritarias.
- **Macro-Recall global**.
- **Macro-Recall sobre clases de ataque**, excluyendo la clase `Benigno`.
- **MCC global**, como métrica robusta ante desbalanceo.
- **ROC-AUC binario**, calculado sobre la fase de detección Ataque/Benigno.
- **Latencia media por evento** y **throughput**, medidos en inferencia batch durante la evaluación experimental.

La comparación no presupone que la GNN supere a los modelos tabulares en todas las métricas. Su objetivo es evaluar el compromiso entre rendimiento predictivo, sensibilidad sobre tácticas de ataque, coste de inferencia y coherencia con el modelado mediante grafos.

---
