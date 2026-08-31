# 🍎 Clasificador de Frutas y Verduras — Actividad 3-1 UAM

**Autor:** Julian Cucalon  
**Curso:** Aplicaciones en la Nube y Servicios Especializados en Ciencia de Datos  
**Valor:** 35% de la evaluación sumativa

## 📋 Descripción

Sistema completo de clasificación de imágenes de frutas y verduras usando **Transfer Learning con MobileNetV2** (TensorFlow/Keras), con **API REST** y **frontend web** desplegados en **Cloudflare** (Workers + Pages + D1).

## 🚀 URLs del sistema desplegado

| Componente | URL |
|---|---|
| **Frontend Web** | https://1378fe71.clasificador-frutas.pages.dev |
| **API REST** | https://clasificador-frutas-api.dr-juliancucalon.workers.dev |
| **Health Check** | https://clasificador-frutas-api.dr-juliancucalon.workers.dev/health |
| **Métricas** | https://clasificador-frutas-api.dr-juliancucalon.workers.dev/metrics |
| **Historial** | https://clasificador-frutas-api.dr-juliancucalon.workers.dev/history |

## 🧠 Modelo ML

- **Arquitectura:** MobileNetV2 (transfer learning) + capas densas personalizadas
- **Dataset:** Fruits-360 (subconjunto de 4 clases: manzana, plátano, naranja, tomate)
- **Precisión en test:** 100%
- **Tiempo de inferencia:** 7.25 ms por imagen
- **Exportación:** ONNX (8.93 MB) + TensorFlow (19.35 MB)

### Entrenamiento

```bash
pip install tensorflow tf2onnx onnx scikit-learn pillow numpy
python scripts/entrenar.py
```

## 📡 API REST

### Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/` | Documentación de la API |
| `GET` | `/health` | Health check |
| `POST` | `/predict` | Clasificar una imagen |
| `GET` | `/history` | Historial de predicciones |
| `GET` | `/metrics` | Métricas de uso |

### Ejemplo de uso

```bash
curl -X POST https://clasificador-frutas-api.dr-juliancucalon.workers.dev/predict \
  -F "image=@ruta/a/mi-manzana.jpg"
```

Respuesta:
```json
{
  "success": true,
  "prediccion": "manzana",
  "confianza": 0.98,
  "probabilidades": [
    { "clase": "manzana", "probabilidad": 0.98 },
    { "clase": "platano", "probabilidad": 0.01 },
    { "clase": "naranja", "probabilidad": 0.005 },
    { "clase": "tomate", "probabilidad": 0.005 }
  ],
  "tiempo_inferencia_ms": 15,
  "timestamp": "2026-08-31T19:00:00.000Z",
  "id": 1
}
```

## 🖥️ Frontend

React + Vite con 3 pestañas:
- **🔮 Clasificar**: Subir imagen (drag & drop) y ver resultado
- **📋 Historial**: Ver predicciones anteriores
- **📊 Métricas**: Dashboard de uso

## 🐳 Docker (requisito académico)

```bash
# Construir y ejecutar localmente
docker-compose up --build
```

## ☁️ Despliegue en Cloudflare

```bash
# API Worker
cd api && npx wrangler deploy

# Frontend Pages
cd frontend && npx vite build
npx wrangler pages deploy dist --project-name clasificador-frutas

# Base de datos D1
npx wrangler d1 create frutas-db
npx wrangler d1 execute frutas-db --remote --file api/src/schema.sql
```

## 📁 Estructura del proyecto

```
clasificador-frutas-cloudflare/
├── notebooks/          # Notebooks de entrenamiento (pendiente)
├── api/                # API REST (Cloudflare Worker)
│   ├── src/
│   │   ├── index.ts    # Código del Worker
│   │   └── schema.sql  # Esquema D1
│   ├── package.json
│   └── wrangler.jsonc
├── frontend/           # Frontend React
│   ├── src/
│   │   ├── App.tsx     # Componente principal
│   │   └── main.tsx    # Entry point
│   ├── index.html
│   ├── package.json
│   └── vite.config.ts
├── models/             # Modelos entrenados
│   ├── modelo_frutas.keras
│   ├── modelo_frutas.onnx
│   ├── metrics.json
│   └── labels.json
├── scripts/            # Scripts de entrenamiento
│   ├── entrenar.py
│   └── convertir_tfjs.py
├── docker/             # Dockerfiles
├── docs/               # Documentación técnica
├── dataset/            # Dataset Fruits-360
├── docker-compose.yml
├── MEMORY.md
└── README.md
```

## 📊 Métricas del modelo

| Métrica | Valor |
|---|---|
| Test Accuracy | 100% |
| Test Loss | 0.0036 |
| Tiempo de inferencia | 7.25 ms |
| Tamaño ONNX | 8.93 MB |
| Clases | manzana, plátano, naranja, tomate |
| Arquitectura | MobileNetV2 + Transfer Learning |

## 🛠️ Tecnologías

- **ML**: TensorFlow 2.21, Keras, MobileNetV2, ONNX
- **API**: Cloudflare Workers, Hono, TypeScript, D1 (SQLite)
- **Frontend**: React 18, Vite 8, TypeScript
- **Infra**: Cloudflare (Workers + Pages + D1), Docker
- **Entrenamiento**: Python 3.13, scikit-learn, tf2onnx

## 📝 Licencia

Proyecto académico — Universidad Autónoma de Manizales (UAM)