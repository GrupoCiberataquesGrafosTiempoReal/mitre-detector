### Repositorio para entrenamiento de comité de modelos predictivos mediante GraphSAGE y GAT

La carpeta **data** contiene todos los archivos de datos.
En **raw** está el dataset **UWF-ZeekData24** completo dividido en 7 archivos .parquet.
En **processed** se encuentra el dataset adaptado y dividio en dos partes:

- **dataset_historico_neo4j.parquet**: es el conjunto de entrenamiento de los modelos predictivos. Se insertarán en Neo4J directamente, sin desetiquetar/reetiquetar.
- **dataset_simulacion_kafka.parquet**: es el conjunto de test de los modelos predictivos. Se desetiquetarán y se hará predicciones con ellos.
- **columnas_modelo.json**: archivo necesario para mantener el orden de columnas al cargar los datos procesados. Esto es importante para utilizar los modelos predictivos previamente entrenados.

La carpeta **notebooks** contiene cuadernos con exploración del atributo _service_ de los elementos del dataset, subida de datos a Neo4J, y el código principal para el entrenamiento de los modelos predictivos.

La carpeta **models** contiene los modelos predictivos entrenados y listos para ser consumidos.
En la subcarpeta **encoders** se encuentran los codificadores necesarios para utilizar los modelos.

La carpeta **src** contiene los archivos **detector_mitre.py** y **insert_neo4j.py**, necesarios para poder procesar los datos de manera compatible con los modelos generados. Estas clases no son finales y pueden servir simplemente como guía para el código final de procesado, predicción e inserción en Neo4J.

La carpeta **experiments** contiene código variado de pruebas de entrenamiento de modelos anteriores.

---

# Arquitectura y Metodología del Sistema de Detección de Intrusiones basado en Grafos

## 1. Enfoque Arquitectónico: Clasificador Jerárquico en Cascada (Two-Stage Classifier)

Para abordar el reto inherente al análisis de tráfico de red, donde el tráfico legítimo supera en órdenes de magnitud a las anomalías, se ha diseñado una **arquitectura en cascada de dos fases**. En lugar de emplear un modelo monolítico, el sistema divide la carga cognitiva de la red neuronal:

- **Fase 1 (Detección Binaria - "El Portero"):** Un comité de modelos entrenado exclusivamente para la discriminación binaria (Ataque vs. Benigno). Actúa como un filtro de alta precisión que bloquea los falsos positivos del tráfico masivo.

- **Fase 2 (Clasificación Multiclase Forense - "El Analista"):** Un segundo comité que recibe únicamente el tráfico catalogado como malicioso por la Fase 1. Al haber eliminado el "ruido" de la clase mayoritaria (Benigno), estos modelos se especializan en diferenciar las sutiles huellas topológicas de las diferentes tácticas de la matriz **MITRE ATT\&CK** (ej. _Exfiltration_, _Persistence_, _Privilege Escalation_).

## 2. Enrutador Topológico por Servicios (Domain Routing)

El tráfico de red se segmenta semánticamente antes de entrar a las redes neuronales, enviándose a sub-modelos "expertos" según el protocolo de la capa de aplicación. Esta especialización evita la interferencia catastrófica (ej. el tráfico web tiene una topología diametralmente opuesta al tráfico DNS):

1. **Experto Web:** Analiza tráfico `ssl` y `http`.

2. **Experto Infraestructura:** Especializado en tráfico de servicios base y UDP (`dns`, `ntp`, `dhcp`).

3. **Experto Autenticación:** Crítico para detectar movimientos laterales. Analiza protocolos de Windows e identidad (`smb`, `gssapi`, `ntlm`, `dce_rpc`).

4. **Experto Generalista:** Un modelo de mayor capacidad neuronal destinado a procesar tráfico desconocido, anómalo o residual.

## 3. Ingeniería de Características (Feature Engineering)

El sistema utiliza un enfoque inductivo, permitiendo que el modelo generalice sobre direcciones IP no vistas durante el entrenamiento. Para ello, se extrajeron tres dimensiones de datos:

- **Características Inductivas de los Nodos (IPs):** En lugar de usar _embeddings_ estáticos que memorizan IPs, se extrajeron métricas de comportamiento histórico bidireccional (_in-degree_, _out-degree_, _in_bytes_mean_, _out_bytes_mean_, etc.).

- **Características Semánticas de las Aristas (Zeek Logs):** Se utilizaron las métricas crudas de las conexiones (`duration`, `orig_bytes`, `resp_bytes`, `orig_pkts`, `resp_pkts`, `missed_bytes`). A esto se sumó el **One-Hot Encoding del campo `conn_state`** (estado de la conexión), revelándose como una variable crítica para detectar tácticas escurridizas como _Privilege Escalation_.

- **Ingeniería Temporal (Beaconing):** Se calculó la variable `time_since_last_conn` (diferencia de tiempo entre conexiones de los mismos pares de IPs). Esta variable dotó al sistema de "memoria temporal", permitiendo identificar patrones topológicos de _Persistencia_ o llamadas automatizadas a servidores C2 (Command and Control).

## 4. Diversidad Arquitectónica de los Modelos (GNN)

El núcleo predictivo se construyó utilizando la librería PyTorch Geometric, combinando dos arquitecturas de Redes Neuronales de Grafos para lograr un aprendizaje complementario:

- **GraphSAGE (SAGEConv):** Ideal para capturar la densidad volumétrica y las características agregadas de vecindarios topológicos locales (ej. ataques de fuerza bruta o escaneos de red).

- **GAT (Graph Attention Networks):** Configurado con mecanismos de atención multi-cabezal (`heads=4`). A diferencia de SAGE, GAT aprende a ponderar qué conexiones vecinas son críticas, siendo excepcional para rastrear la aguja en el pajar (ej. una única conexión de robo de tickets Kerberos camuflada entre cientos de conexiones benignas).

Ambas arquitecturas se diseñaron con una **profundidad de 3 saltos convolucionales**, añadiendo capas de _Batch Normalization_ y _Dropout_ dinámico (0.2 a 0.5) para acelerar la convergencia y mitigar el sobreajuste.

## 5. Estrategias Avanzadas de Mitigación del Desbalanceo

El dataset presentaba un desbalanceo extremo (cientos de miles de conexiones benignas frente a un centenar de conexiones de _Exfiltration_). Se implementó un enfoque híbrido en la función de pérdida y el muestreo:

- **Undersampling selectivo:** Descarte aleatorio programado de un porcentaje de las clases masivas (Benigno y Credential Access) exclusivamente durante la fase de entrenamiento, permitiendo que la red "viera" las clases minoritarias con mayor frecuencia.

- **Focal Loss con Weight Clipping:** Se abandonó la entropía cruzada estándar (CrossEntropy) en favor de **Focal Loss** ($\gamma=2.0$). Esta función reduce dinámicamente la pérdida de los ejemplos fáciles (los que la red ya acierta con confianza) y focaliza el gradiente en las tácticas que el modelo clasifica erróneamente. Además, se aplicó un _clipping_ máximo a los pesos matemáticos (`max_weight = 12.0 - 20.0`) y un ajuste manual del peso de _Credential Access_ para evitar la explosión del gradiente y el efecto _Whac-A-Mole_.

## 6. Lógica de Inferencia en Producción (Soft Voting Ensemble)

En la fase de evaluación y producción (simulando la ingesta en Streaming desde Kafka), el pipeline sigue una lógica determinista robusta:

1. La conexión se enruta al modelo de la **Fase 1 (Binario)** correspondiente según su servicio. Si se clasifica como benigna, el proceso termina, asegurando un índice de falsos positivos virtualmente nulo en tráfico estándar.

2. Si se marca como ataque, la conexión se deriva a la **Fase 2 (Multiclase)**.

3. En la Fase 2, en lugar de consultar a un solo modelo, se ejecuta un **Soft Voting Ensemble Intra-Etapa**. Se consulta simultáneamente al modelo SAGE y al modelo GAT de ese servicio. Las probabilidades devueltas por ambas redes (tras aplicar una capa Softmax) se promedian.

4. Como medida adicional de seguridad forense, la probabilidad de la clase 'Benigno' en esta fase se fuerza matemáticamente a `0.0`, obligando al "Analista" a emitir siempre un veredicto basado en la matriz MITRE ATT\&CK. La etiqueta final se obtiene mediante el `argmax` de estas probabilidades promediadas.

---

### Notas para uso en la memoria

Este bloque sirve para cubrir prácticamente toda la sección metodológica del Capítulo 4 o 5 de la memoria. Explica **qué se hizo, por qué se hizo a nivel matemático y cuál es su implicación operativa en ciberseguridad**.

Luego, en el capítulo de Resultados, solo hay que pegar el _Classification Report_ y la Matriz de Confusión que se obtuvo en tu última prueba para demostrar que esta teoría funciona a la perfección.
