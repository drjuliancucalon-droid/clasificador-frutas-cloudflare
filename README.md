# 🍎 Clasificador de Frutas y Verduras — Actividad 3-1 UAM

**Autor:** Julian Cucalon
**Curso:** Aplicaciones en la Nube y Servicios Especializados en Ciencia de Datos
**Valor:** 35% de la evaluación sumativa

## 📋 Descripción

Sistema completo de clasificación de imágenes de frutas y verduras usando **Transfer Learning con MobileNetV2** (TensorFlow/Keras), con **API REST** (Cloudflare Worker) y **frontend web** (React + Cloudflare Pages), desplegado en Cloudflare con costo \$0.

La inferencia del modelo corre en el navegador con **TensorFlow.js** — no en el Worker — porque el plan gratuito de Cloudflare Workers limita el tiempo de CPU a 10ms por invocación, insuficiente para un forward-pass de MobileNetV2. Ver `docs/documentacion-tecnica.md`, sección 1.1, para el detalle de esta decisión.

## 🚀 URLs del sistema desplegado

| Componente | URL |
|---|---|
| **Frontend Web** | https://1378fe71.clasificador-frutas.pages.dev |
| **API REST** | https://clasificador-frutas-api.dr-juliancucalon.workers.dev |
| **Health Check** | https://clasificador-frutas-api.dr-juliancucalon.workers.dev/health |
| **Métricas** | https://clasificador-frutas-api.dr-juliancucalon.workers.dev/metrics |
| **Historial** | https://clasificador-frutas-api.dr-juliancucalon.workers.dev/history |

## 🧠 Modelo ML

- **Arquitectura:** MobileNetV2 pre-entrenado en ImageNet (transfer learning) + capas densas personalizadas
- **Dataset:** Fruits-360 (subconjunto de 4 clases: manzana, plátano, naranja, tomate)
- **Precisión en test:** 100% (ver nota sobre generalización en `docs/documentacion-tecnica.md`, sección 4)
- **Tiempo de inferencia (evaluación offline):** 7.25 ms/imagen
- **Exportación:** `.keras` (19.35 MB), `.onnx` (8.93 MB) y **TensorFlow.js** (~8.9 MB — el que efectivamente usa el sistema desplegado)

### Entrenamiento y exportación

```bash
pip install tensorflow tf2onnx tensorflowjs scikit-learn pillow numpy
python scripts/entrenar.py
python scripts/convertir_tfjs.py
```

## 📡 API REST

### Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| `GET /` | - | Documentación de la API |
| `GET /health` | Health check | Estado del sistema |
| `POST /predict` | Registrar clasificación | Recibe imagen + predicción (calculada en el navegador), valida y persiste |
| `GET /history` | Historial | Predicciones anteriores (paginado) |
| `GET /metrics` | Métricas | Dashboard de uso |

**Stack:** Cloudflare Workers + Hono (TypeScript) + D1 (SQLite)

## 🖥️ Frontend

**Stack:** React 18 + Vite 5 + TypeScript + TensorFlow.js

Funcionalidades:
- Drag & drop para subir imágenes
- Clasificación real en el navegador (TensorFlow.js + modelo MobileNetV2 entrenado)
- Visualización de resultados con emoji + barra de confianza
- Historial de predicciones
- Dashboard de métricas de uso

## 🐳 Docker

```bash
docker compose up --build
# API (Worker emulado localmente): http://localhost:8787
# Frontend: http://localhost:8080
```

## 🌐 URLs del Sistema

| Componente | URL |
|---|---|
| **Frontend** | https://1378fe71.clasificador-frutas.pages.dev |
| **API REST** | https://clasificador-frutas-api.dr-juliancucalon.workers.dev |
| **Health Check** | https://clasificador-frutas-api.dr-juliancucalon.workers.dev/health |
| **Repositorio GitHub** | https://github.com/drjuliancucalon-droid/clasificador-frutas-cloudflare |

---

## 📁 Estructura de la Carpeta de Entrega

```
Entrega_Final_Actividad3-1/
├── README.md                          ← Este archivo
├── notebooks/
│   └── entrenar.py                    ← Script de entrenamiento
├── api/
│   ├── package.json
│   ├── wrangler.jsonc                 ← Config Cloudflare Worker
│   └── src/
│       ├── index.ts                   ← Código de la API (5 endpoints)
│       └── schema.sql                 ← Esquema D1 (SQLite)
├── frontend/
│   ├── public/
│   │   └── model/                     ← Modelo TensorFlow.js (usado en producción)
│   ├── index.html
│   ├── package.json
│   ├── vite.config.ts
│   └── src/
│       ├── main.tsx                   ← Entry point React
│       └── App.tsx                    ← Componente principal (3 tabs + inferencia TF.js)
├── docker/
│   ├── Dockerfile.api                 ← Node + Wrangler (emula el Worker)
│   ├── Dockerfile.frontend            ← Build Vite + nginx
│   └── docker-compose.yml
├── docs/
│   ├── documentacion-tecnica.md       ← Documentación técnica completa
│   └── screenshots/                   ← Evidencia de pruebas
├── assets/
│   ├── metrics.json                   ← Métricas del modelo
│   └── labels.json                    ← Etiquetas de clases
└── .gitignore
```

---

## 📋 Estado de cumplimiento (verificado, no autoevaluado)

Ver `Informe_Validacion_Actividad3-1.docx` en la raíz del proyecto para la auditoría completa con evidencia. Estado tras aplicar las correcciones de esa auditoría:

| Criterio | % | Estado |
|---|---|---|
| Funcionalidad del modelo (precisión, edge cases, tiempo) | 25% | Modelo real conectado end-to-end (TensorFlow.js); pendiente de verificación final tras redeploy |
| Implementación técnica (calidad código, arquitectura) | 25% | TypeScript, Hono, React, validaciones cliente/servidor |
| Despliegue y operación (cloud, Docker, monitoreo) | 25% | Cloudflare + Docker (Node/Wrangler) + D1 + métricas; pendiente de verificación final tras redeploy |
| Documentación y entregables (claridad, completitud) | 25% | README, docs, diagramas, ejemplos, checklist honesto |

## 📝 Notas para el Estudiante

1. **Modelos grandes** (`.keras` y `.onnx`) no se incluyen en GitHub por tamaño; el modelo TensorFlow.js en `frontend/public/model/` sí se incluye (es el que usa el sistema en producción).
2. Para **re-entrenar**: `python scripts/entrenar.py` y luego `python scripts/convertir_tfjs.py`.
3. Para **re-desplegar la API**: `npx wrangler deploy` desde `api/`.
4. Para **reconstruir el frontend**: `npx vite build` desde `frontend/` (la URL de la API en producción está fijada en `vite.config.ts`, no depende de variables de entorno del shell).
