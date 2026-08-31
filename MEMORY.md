# MEMORY — Clasificador de Frutas y Verduras (Cloudflare)
_Última actualización: 2026-08-31 14:57_

## Estado actual
Proyecto iniciado. Estructura creada, Cloudflare autenticado. Comenzando Fase 1 (modelo ML).

## Stack tecnológico
- **ML**: TensorFlow/Keras + MobileNetV2 (transfer learning), dataset Fruits-360
- **API**: Cloudflare Worker + Hono (TypeScript), inferencia ONNX (onnxruntime-web WASM)
- **Frontend**: React + Vite, desplegado en Cloudflare Pages
- **Datos**: Cloudflare D1 (SQLite) para historial, R2 para almacenar modelo
- **Docker**: docker-compose local con FastAPI (requisito académico de containerización)
- **Despliegue**: Cloudflare (Workers + Pages + R2 + D1) — free tier, costo $0
- **Entrenamiento**: Google Colab (GPU gratis) o local

## Lo que ya está construido
- Estructura de carpetas: notebooks/, api/src/, frontend/src/, docker/, docs/, scripts/
- Cloudflare autenticado: dr.juliancucalon@gmail.com, Account ID 0b9efca009317f8624843e4fa61d17ed

## Lo que está en progreso
- Fase 1: Notebook de entrenamiento + modelo (MobileNetV2, 5 categorías: manzana, plátano, naranja, tomate, zanahoria)

## Decisiones de arquitectura tomadas
- Cloudflare Workers no ejecuta Python/TensorFlow → modelo se exporta a ONNX cuantizado (~4MB) y se sirve desde R2; el Worker hace inferencia con onnxruntime-web (WASM)
- Fallback: inferencia client-side con TensorFlow.js si la inferencia server-side falla
- Docker incluido como versión local FastAPI para cumplir requisito académico de containerización
- D1 (SQLite) para historial de predicciones y métricas

## Problemas conocidos y soluciones aplicadas
- Sub-agentes alcanzaron límite diario de cuota → investigación técnica ya completada previamente
- TensorFlow en Python 3.13 requiere TF 2.20+ (verificar al instalar)

## Variables de entorno y configuración crítica
- Wrangler OAuth token en C:\Users\JQK3\AppData\Roaming\xdg.config\.wrangler\config\default.toml
- Account ID Cloudflare: 0b9efca009317f8624843e4fa61d17ed

## Próximos pasos concretos
1. Crear notebook de entrenamiento (Colab + local)
2. Instalar TensorFlow local y entrenar modelo con subset de Fruits-360
3. Exportar ONNX + TensorFlow.js
4. Construir Worker API (Hono) con endpoints /predict /health /metrics /history
5. Construir frontend React
6. Dockerfiles + docker-compose
7. Desplegar en Cloudflare (wrangler deploy)
8. Documentación completa + pruebas con Playwright

## Contexto de negocio
Actividad académica 3-1 (UAM): "Aplicaciones en la Nube y Servicios Especializados en Ciencia de Datos". Vale 35% de la evaluación sumativa. Entregables: repo GitHub, notebook, API, frontend, Dockerfiles, README, docs, app desplegada con URL. Integrante único: Julian Cucalon (dr.juliancucalon@gmail.com).