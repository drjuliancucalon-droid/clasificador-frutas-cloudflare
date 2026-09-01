# Documentación Técnica — Clasificador de Frutas y Verduras

**Actividad 3-1: Aplicaciones en la Nube y Servicios Especializados en Ciencia de Datos**
**Autor:** Julian Cucalon | **UAM** | **Agosto 2026**

---

## 1. Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USUARIO                                       │
│   (Frontend React + Vite en Cloudflare Pages)                       │
│   La inferencia del modelo corre AQUÍ, en el navegador,             │
│   con TensorFlow.js — ver sección 1.1.                              │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ imagen + predicción calculada
                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     API REST (Cloudflare Worker)                     │
│  Hono + TypeScript                                                    │
│  Valida la predicción recibida y la persiste. No calcula la          │
│  inferencia (ver 1.1).                                               │
│  Endpoints: /predict, /health, /history, /metrics                   │
└────────────┬───────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────┐
│   Cloudflare D1 (SQLite)│
│   - Historial           │
│   - Métricas            │
└─────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                      ENTRENAMIENTO (local / Colab)                     │
│  Python 3.13 · TensorFlow 2.21 · MobileNetV2 · TF2ONNX · TF.js       │
│                                                                        │
│  Dataset: Fruits-360 (4 clases: manzana, plátano, naranja, tomate)  │
│  Export: .keras (19.35 MB) + .onnx (8.93 MB) + TensorFlow.js (~9 MB) │
│  El export a TensorFlow.js (frontend/public/model/) es el que        │
│  efectivamente usa el sistema desplegado.                            │
└──────────────────────────────────────────────────────────────────────┘
```

### 1.1 Arquitectura de inferencia: por qué corre en el navegador

Cloudflare Workers, en el plan gratuito, limita el tiempo de CPU a **10 ms por invocación**. Un forward-pass de MobileNetV2 (incluso a 128×128, con la base congelada) toma del orden de decenas de milisegundos en WASM, muy por encima de ese límite — ejecutarlo dentro del Worker con `onnxruntime-web` no es viable sin pasar a un plan de pago, lo que rompería el objetivo de costo \$0 del proyecto.

Por eso el modelo entrenado se exportó también a **TensorFlow.js** (`scripts/convertir_tfjs.py`, `frontend/public/model/`), y la inferencia corre directamente en el navegador del usuario, sin límite de tiempo de CPU por invocación. El Worker deja de "simular" una predicción (como en una versión anterior de este proyecto) y pasa a:

1. Recibir la imagen y la predicción ya calculada por el cliente.
2. Validar que la clase esté entre las 4 conocidas y que la confianza esté en `[0,1]`.
3. Persistir el resultado en D1 y devolverlo (con `id` y `timestamp`) para historial y métricas.

Esto sigue satisfaciendo el requisito de la actividad: la API REST recibe imágenes y devuelve resultados de clasificación persistidos; lo que cambia es *dónde* ocurre el cómputo del modelo, una decisión de arquitectura razonada por las restricciones reales de la plataforma serverless elegida — un patrón común en despliegues edge de modelos ligeros.

## 2. Diagrama de Flujo de Datos

```
1. Usuario sube imagen JPG/PNG/WebP (máx 10MB)
       │
       ▼
2. Frontend React valida el archivo localmente
       │
       ▼
3. TensorFlow.js preprocesa la imagen (resize 128×128) y ejecuta el
   modelo MobileNetV2 entrenado — inferencia real, en el navegador
       │
       ▼
4. POST /predict (multipart/form-data: imagen + predicción calculada)
   → Cloudflare Worker
       │
       ▼
5. Worker:
   a. Valida formato y tamaño de la imagen → 400 si inválido
   b. Valida que la clase/confianza/probabilidades recibidas tengan
      forma válida → 400 si no
   c. Guarda en D1 (SQLite) → id autoincremental
   d. Devuelve JSON con predicción, confianza, tiempo de inferencia
       │
       ▼
6. Frontend muestra resultado (emoji + nombre + barra de confianza)
```

## 3. Endpoints de la API

### `GET /health`
Health check del sistema
```bash
curl https://clasificador-frutas-api.dr-juliancucalon.workers.dev/health
```
```json
{"status":"ok","version":"1.0.0","timestamp":"2026-08-31T19:00:00Z","database":"connected","model":"mobilenetv2-tfjs-client-v1"}
```

### `POST /predict`
Registra la clasificación de una imagen. La clase, confianza y
probabilidades normalmente las calcula el frontend con el modelo real
(TensorFlow.js); el ejemplo de abajo las pasa a mano solo para poder
probar el endpoint de forma aislada, sin navegador.
```bash
curl -X POST https://clasificador-frutas-api.dr-juliancucalon.workers.dev/predict \
  -F "image=@ruta/a/mi-manzana.jpg" \
  -F "clase_predicha=manzana" \
  -F "confianza=0.98" \
  -F 'probabilidades=[{"clase":"manzana","probabilidad":0.98},{"clase":"platano","probabilidad":0.01},{"clase":"naranja","probabilidad":0.005},{"clase":"tomate","probabilidad":0.005}]' \
  -F "tiempo_inferencia_ms=15"
```
```json
{
  "success": true,
  "prediccion": "manzana",
  "confianza": 0.98,
  "probabilidades": [
    {"clase": "manzana", "probabilidad": 0.98},
    {"clase": "platano", "probabilidad": 0.01},
    {"clase": "naranja", "probabilidad": 0.005},
    {"clase": "tomate", "probabilidad": 0.005}
  ],
  "tiempo_inferencia_ms": 15,
  "timestamp": "2026-08-31T19:00:00.000Z",
  "id": 1
}
```

### `GET /history`
Historial de predicciones (paginado)
```bash
curl https://clasificador-frutas-api.dr-juliancucalon.workers.dev/history?page=1&limit=20
```

### `GET /metrics`
Métricas de uso del sistema
```bash
curl https://clasificador-frutas-api.dr-juliancucalon.workers.dev/metrics
```
```json
{
  "total_predicciones": 0,
  "predicciones_hoy": 0,
  "distribucion_por_clase": [],
  "timestamp": "2026-08-31T19:00:00Z"
}
```

## 4. Métricas de Rendimiento del Modelo

### Métricas principales

| Métrica | Valor |
|---|---|
| Test Accuracy | 100% |
| Test Loss | 0.0036 |
| Tiempo de inferencia promedio (evaluación offline) | 7.25 ms/imagen |
| Tamaño del modelo (.keras) | 19.35 MB |
| Tamaño del modelo (.onnx) | 8.93 MB |
| Tamaño del modelo (TensorFlow.js, el que se sirve al navegador) | ~8.9 MB (3 shards) |
| Dataset | Fruits-360 (subset de 4 clases) |
| Total de imágenes de entrenamiento | 2,199 |
| Total de imágenes de prueba | 736 |

**Nota sobre el 100% de accuracy:** es creíble para este dataset (Fruits-360 usa fondo blanco uniforme y una sola fruta por imagen, 4 clases muy distintas entre sí), pero no debe interpretarse como una medida de qué tan bien generalizará el modelo a fotos reales tomadas por un usuario (fondos variados, más de un objeto, distinta iluminación). Es una limitación esperable del dataset, no un error del entrenamiento.

### Exportación del modelo
El modelo se exportó en 3 formatos:
- **Keras (.keras)**: formato nativo, para reentrenamiento y experimentación en Python.
- **ONNX (.onnx)**: para inferencia multiplataforma fuera del navegador (no se usa en el sistema desplegado; ver sección 1.1 sobre la decisión de inferencia).
- **TensorFlow.js**: el que efectivamente carga el frontend (`frontend/public/model/`) para clasificar en el navegador.

### Manejo de casos límite

- **Imagen no enviada**: se rechaza con error 400.
- **Formato no soportado**: se valida y se informa (solo JPEG, PNG, WebP).
- **Tamaño excesivo**: máximo 10MB.
- **Predicción/confianza con forma inválida**: el Worker la rechaza con 400 antes de guardarla (protege la base de datos de datos corruptos, sin importar el origen de la solicitud).

## 5. Requisitos de Despliegue

### Prerequisitos

```bash
# Herramientas
Node.js 18+
Python 3.11+
Docker 24+
Cuenta de Cloudflare

# Instalar dependencias
cd api && npm install
cd frontend && npm install
pip install tensorflow tf2onnx tensorflowjs scikit-learn
```

### Despliegue local con Docker
```bash
docker compose up --build
# API (Worker emulado con wrangler dev --local): http://localhost:8787
# Frontend (build de producción servido con nginx): http://localhost:8080
```

### Despliegue en Cloudflare
```bash
# API Worker
cd api && npx wrangler deploy

# Frontend Pages (la URL de la API queda fijada en frontend/vite.config.ts
# para el comando `build`, no depende de variables de entorno del shell)
cd frontend && npx vite build
npx wrangler pages deploy dist --project-name clasificador-frutas

# Base de datos D1
npx wrangler d1 create frutas-db
npx wrangler d1 execute frutas-db --remote --file api/src/schema.sql
```

### Reentrenar el modelo y regenerar el export a TensorFlow.js
```bash
pip install tensorflow tf2onnx tensorflowjs scikit-learn pillow numpy
python scripts/entrenar.py
python scripts/convertir_tfjs.py   # genera frontend/public/model/
```

## 6. Estructura del Repositorio

```
clasificador-frutas-cloudflare/
├── api/                      # Cloudflare Worker
│   ├── src/
│   │   ├── index.ts          # Código principal (5 endpoints)
│   │   └── schema.sql        # Esquema D1 SQLite
│   ├── package.json
│   └── wrangler.jsonc        # Configuración Worker + D1
├── frontend/                 # React + Vite
│   ├── public/
│   │   └── model/            # Modelo TensorFlow.js (usado en producción)
│   ├── src/
│   │   ├── App.tsx           # Componente principal (3 tabs + inferencia TF.js)
│   │   └── main.tsx          # Entry point
│   ├── index.html
│   └── vite.config.ts
├── models/                   # Modelos entrenados (.keras, .onnx) + métricas
│   ├── labels.json
│   └── metrics.json
├── scripts/
│   ├── entrenar.py           # Entrenamiento MobileNetV2
│   └── convertir_tfjs.py     # Conversión a TensorFlow.js
├── docker/
│   ├── Dockerfile.api        # Node + Wrangler (emula el Worker)
│   └── Dockerfile.frontend   # Build Vite + nginx
├── docker-compose.yml
├── docs/                     # Documentación técnica
├── README.md
└── MEMORY.md
```

## 7. URLs del Sistema

| Componente | URL |
|---|---|
| Frontend Web | https://1378fe71.clasificador-frutas.pages.dev |
| API REST | https://clasificador-frutas-api.dr-juliancucalon.workers.dev |
| Health Check | https://clasificador-frutas-api.dr-juliancucalon.workers.dev/health |
| Repositorio GitHub | https://github.com/drjuliancucalon-droid/clasificador-frutas-cloudflare |
