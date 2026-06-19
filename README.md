# 🛡️ Sistema de Detección de Intrusiones (IDS) Basado en Grafos (GNN) y MITRE ATT&CK

Este repositorio contiene el código fuente, la experimentación y el motor de inferencia para un **Clasificador Jerárquico en Cascada** capaz de detectar ciberataques en flujos de red (Zeek) en tiempo real, mapeándolos a las tácticas de la matriz MITRE ATT&CK utilizando Redes Neuronales de Grafos (GraphSAGE y GAT).

---

## 📂 Estructura del Repositorio

- **`data/`**: Contiene los datasets (`UWF-ZeekData24`). Se divide en `raw/` (archivos originales) y `processed/` (dataset histórico para Neo4j, dataset continuo de simulación para Kafka y la plantilla `columnas_modelo.json`).
- **`models/`**: Directorio donde residen los pesos pre-entrenados (`.pth`), la configuración de umbrales (`config/`) y los artefactos de normalización (`encoders/`).
- **`notebooks/`**: Cuadernos interactivos para la exploración del tráfico, partición temporal y el entrenamiento avanzado de los modelos predictivos.
- **`src/`**: Código fuente de producción.
  - `detector_mitre.py`: Core del motor inductivo.
  - `api.py`: Servicio web asíncrono (FastAPI) para consumir las predicciones.
  - `insert_neo4j.py`: Código de pruebas para inserción de subgrafos en Neo4j.
  - `simulador_kafka.py`: Código de pruebas para test de predicciones con modelos preentrenados.
  - `simulador_api.py`: Código de pruebas para test de predicciones contra imagen docker desplegada en localhost.
- **`Dockerfile` / `docker-compose.yml`**: Recetas de contenedorización listas para despliegue híbrido (CPU/CUDA).
- **`requirements.txt`**: Dependencias de Python.

---

## 🚀 Guía de Despliegue (Inferencia como Servicio)

El motor se expone a través de una API REST (FastAPI) encapsulada en Docker, permitiendo intercambiar modelos actualizando únicamente los volúmenes montados.

### 1. Preparación de Volúmenes

Antes de levantar el servicio, asegúrate de tener los modelos entrenados y los archivos de metadatos en tu máquina local. El `docker-compose.yml` montará estas dos carpetas en el contenedor:

- `./models`: Debe contener los archivos `.pth` y las subcarpetas `encoders/` y `config/`.
- `./data`: Debe contener el archivo `columnas_modelo.json`.

### 2. Levantar el Servicio

En la raíz del repositorio, ejecuta:

```bash
docker build --no-cache -t mitre-detector .

```

> **Nota GPU (CUDA):** Si el servidor _host_ dispone de una tarjeta NVIDIA y `nvidia-container-toolkit` instalado, puedes descomentar el bloque `deploy: resources:` en el `docker-compose.yml` para aceleración por hardware. En caso contrario, PyTorch operará en modo CPU automáticamente.

### 3. Consumir la API

El contenedor expone el servicio en el puerto `8080`. Puedes ver la documentación interactiva (Swagger UI) y probar el endpoint haciendo un POST con un JSON de Zeek dirigiéndote a:

- **Swagger UI:** [http://localhost:8080/docs](https://www.google.com/search?q=http://localhost:8080/docs)
- **Endpoint de predicción:** `POST http://localhost:8080/classify`

---

## 🧠 Arquitectura y Metodología del Sistema (GNN)

### 1. Clasificador Jerárquico en Cascada

Para abordar el reto inherente al análisis de red, donde el tráfico legítimo supera en órdenes de magnitud a las anomalías, el sistema divide la carga cognitiva:

- **Fase 1 (Detección Binaria):** Un comité entrenado exclusivamente para la discriminación binaria (Ataque vs. Benigno). Actúa como filtro de alta precisión.
- **Fase 2 (Clasificación multi-clase):** Recibe únicamente el tráfico catalogado como malicioso. Al eliminar el "ruido" benigno, estos modelos se especializan en diferenciar las sutiles huellas topológicas de las tácticas MITRE ATT&CK (ej. _Exfiltration_, _Privilege Escalation_).

### 2. Enrutador Topológico por Servicios

El tráfico se segmenta semánticamente antes de entrar a las redes neuronales:

1. **Experto Web:** Tráfico `ssl` y `http`.
2. **Experto Infraestructura:** Servicios base y UDP (`dns`, `ntp`, `dhcp`).
3. **Experto Autenticación:** Protocolos de Windows e identidad (`smb`, `gssapi`, `ntlm`, `dce_rpc`).
4. **Experto Generalista:** Tráfico desconocido, anómalo o residual.

### 3. Inferencia Inductiva y Correlación Temporal

El sistema opera en verdadero _streaming_ sin memorizar la topología estática, calculando características al vuelo:

- **Características de Nodos y Aristas:** Estadísticas bidireccionales y variables de Zeek (incluyendo el crítico `conn_state`).
- **Ingeniería Temporal:** La variable `time_since_last_conn` detecta patrones de _Beaconing_ (C2).
- **Cuarentena de Nodos (Correlación):** Si el motor GNN detecta un ataque volumétrico o anómalo (ej. _Reconnaissance_), el nodo origen y destino entran en "cuarentena". Las conexiones posteriores de esa IP sufren una bajada drástica en el umbral de tolerancia, permitiendo cazar tácticas críticas que de otro modo serían invisibles a nivel de NetFlow.

### 4. Diversidad de Modelos (GraphSAGE + GAT) y Soft Voting

El núcleo predictivo combina dos arquitecturas (PyTorch Geometric):

- **GraphSAGE:** Captura densidad volumétrica en vecindarios locales.
- **GAT (Atención):** Configurado con `heads=4`, pondera conexiones críticas camufladas entre tráfico benigno.

En la Fase 2, se ejecuta un **Soft Voting Ensemble Intra-Etapa**, promediando las probabilidades del modelo SAGE y GAT. La clase 'Benigno' se fuerza a `0.0`, obligando a emitir un veredicto.

### 5. Mitigación del Desbalanceo y Fuga Temporal (Data Leakage)

- **Corte Cronológico por Densidad de Ataques:** La separación Train/Val/Test evita la fuga de información temporal (Data Leakage). Se realiza un corte temporal continuo basado en los cuantiles de aparición de los ataques, garantizando que el Test Set simule el flujo futuro e ininterrumpido en un entorno real de Kafka. No se puede realizar una división estratificada normal debido a que es necesario conservar la continuidad temporal de los datos tanto en entrenamiento como en inferencia.
- **Augmentación Topológica (Oversampling):** En lugar de eliminar tráfico benigno (lo que destruiría la topología del grafo), se inyectan dinámicamente aristas duplicadas de las clases minoritarias durante el entrenamiento. Esto no afecta a la pureza del subconjunto de test.
- **Focal Loss Dinámica con Weight Clipping:** Se aplica un parámetro γ auto-ajustable basado en la distribución de las clases, penalizando severamente los errores en tácticas críticas sin provocar una explosión de los gradientes.
- **Early Stopping guiado por Macro-F1:** La parada temprana del entrenamiento no se basa en la reducción de la pérdida, sino en la maximización de la métrica Macro-F1 en validación, asegurando que la red no se limite a sobreajustarse a la clase mayoritaria.

---
