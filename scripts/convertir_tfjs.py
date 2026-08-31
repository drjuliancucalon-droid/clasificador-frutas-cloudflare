"""
Exporta el modelo Keras a formato TensorFlow.js para inferencia en navegador.
"""
import os
import tensorflowjs as tfjs

BASE = os.path.join(os.path.dirname(__file__), "..")
MODELO_KERAS = os.path.join(BASE, "models", "modelo_frutas.keras")
OUTPUT_DIR = os.path.join(BASE, "frontend", "public", "modelo_tfjs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

tfjs.converters.save_keras_model(
    MODELO_KERAS,
    OUTPUT_DIR
)

print(f"✅ Modelo TFJS exportado a: {OUTPUT_DIR}")
print(f"   Archivos: {os.listdir(OUTPUT_DIR)}")