"""
Exporta el modelo Keras entrenado (proyecto/models/modelo_frutas.keras) a
formato TensorFlow.js, para que el frontend lo cargue y ejecute la
inferencia directamente en el navegador (ver docs/documentacion-tecnica.md,
sección 1.1, sobre por qué la inferencia no corre en el Worker).

Requiere: pip install tensorflow tensorflowjs tf_keras
Salida: frontend/public/model/ (model.json + shards .bin)

Ejecutar como script normal:  python scripts/convertir_tfjs.py

--------------------------------------------------------------------------
NOTA IMPORTANTE — dos incompatibilidades reales que este script resuelve
--------------------------------------------------------------------------
1. El paquete `tensorflowjs` (4.x en PyPI) importa, sin condicionarlo al uso
   real, `tensorflow_decision_forests`, que a su vez arrastra una versión de
   protobuf incompatible con la que TensorFlow exige. `import tensorflowjs`
   falla con un VersionError salvo que se "stubee" ese módulo (y jax/flax,
   con el mismo problema) — no se usan para exportar un modelo Keras normal.

2. El modelo se entrena y guarda con Keras 3 (formato nativo `.keras`), cuyo
   esquema de configuración cambió (p. ej. InputLayer usa "batch_shape" en
   vez de "batch_input_shape", y el grafo de nodos del modelo funcional
   tiene otra forma). @tensorflow/tfjs-layers, del lado JavaScript, todavía
   espera el esquema de Keras 2 y no puede cargar un model.json generado
   directamente desde un modelo Keras 3.

Por eso el proceso corre en DOS pasos, cada uno en su propio proceso de
Python (la variable TF_USE_LEGACY_KERAS solo se lee de forma confiable al
arrancar el intérprete, no sirve cambiarla a mitad de ejecución):

  Paso 1 (Keras 3): carga el modelo original, reconstruye la misma
  arquitectura con `tf_keras` (Keras 2, ya viene como dependencia de
  tensorflowjs), copia los pesos capa por capa, verifica numéricamente que
  las predicciones coincidan, y guarda un .h5 intermedio.

  Paso 2 (TF_USE_LEGACY_KERAS=1, o sea tf.keras = Keras 2 real): carga ese
  .h5 y lo exporta a TensorFlow.js — con el esquema de Keras 2, compatible
  con tfjs-layers.
"""
import os
import subprocess
import sys

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MODELO_KERAS3 = os.path.join(BASE, "models", "modelo_frutas.keras")
OUTPUT_DIR = os.path.join(BASE, "frontend", "public", "model")
H5_TMP = os.path.join(BASE, "models", "_tmp_modelo_keras2.h5")
IMG_SIZE = (128, 128)
NUM_CLASSES = 4

STEP1_REBUILD = f"""
import numpy as np
import keras as keras3
import tf_keras

print("Cargando modelo original (Keras 3): {MODELO_KERAS3}")
src_model = keras3.models.load_model(r"{MODELO_KERAS3}")

print("Reconstruyendo la misma arquitectura con tf_keras (Keras 2)...")
base_model = tf_keras.applications.MobileNetV2(
    input_shape=({IMG_SIZE[0]}, {IMG_SIZE[1]}, 3), include_top=False, weights=None,
)
inputs = tf_keras.Input(shape=({IMG_SIZE[0]}, {IMG_SIZE[1]}, 3), name="input")
x = tf_keras.layers.Rescaling(1.0 / 127.5, offset=-1.0)(inputs)
x = base_model(x, training=False)
x = tf_keras.layers.GlobalAveragePooling2D()(x)
x = tf_keras.layers.Dropout(0.3)(x)
outputs = tf_keras.layers.Dense({NUM_CLASSES}, activation="softmax", name="output")(x)
dst_model = tf_keras.Model(inputs, outputs, name="clasificador_frutas")

assert len(src_model.layers) == len(dst_model.layers), (
    "La arquitectura cambió: revisar este script junto con entrenar.py"
)
for src_layer, dst_layer in zip(src_model.layers, dst_model.layers):
    w = src_layer.get_weights()
    if w:
        dst_layer.set_weights(w)

np.random.seed(42)
x_test = np.random.rand(1, {IMG_SIZE[0]}, {IMG_SIZE[1]}, 3).astype("float32")
p_src = src_model.predict(x_test, verbose=0)
p_dst = dst_model.predict(x_test, verbose=0)
max_diff = float(np.max(np.abs(p_src - p_dst)))
print(f"Verificacion numerica (mismo input): diff maxima = {{max_diff:.2e}}")
assert max_diff < 1e-4, "El modelo reconstruido no coincide con el original -- no se exporta."

dst_model.save(r"{H5_TMP}", save_format="h5")
print("OK: guardado intermedio en {H5_TMP}")
"""

STEP2_EXPORT = f"""
import os, sys, types

class _AnyAttr(types.ModuleType):
    __path__ = []
    def __getattr__(self, name):
        child = _AnyAttr(self.__name__ + "." + name)
        sys.modules[child.__name__] = child
        setattr(self, name, child)
        return child
    def __call__(self, *a, **k):
        return None

def stub_tree(name):
    m = _AnyAttr(name)
    sys.modules[name] = m
    return m

# tensorflowjs importa incondicionalmente estos paquetes al cargarse, aunque
# no se usen para exportar un modelo Keras estandar; sin esto, el import
# falla por un conflicto de version de protobuf.
stub_tree("tensorflow_decision_forests")
stub_tree("jax")
stub_tree("flax")
for p in ["jax.experimental", "jax.experimental.jax2tf", "jax.numpy", "jax.tree_util"]:
    stub_tree(p)

import tensorflowjs as tfjs
import tensorflow as tf  # con TF_USE_LEGACY_KERAS=1, tf.keras = Keras 2 real

print("tf.keras backend:", tf.keras.__name__)
model = tf.keras.models.load_model(r"{H5_TMP}")

os.makedirs(r"{OUTPUT_DIR}", exist_ok=True)
print("Exportando a TensorFlow.js:", r"{OUTPUT_DIR}")
tfjs.converters.save_keras_model(model, r"{OUTPUT_DIR}")
os.remove(r"{H5_TMP}")
print("OK")
"""


def run_step(label, code, extra_env=None):
    print("\n=== " + label + " ===")
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    result = subprocess.run([sys.executable, "-c", code], env=env)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main():
    if not os.path.exists(MODELO_KERAS3):
        sys.exit(f"No se encontró el modelo entrenado: {MODELO_KERAS3}\nEjecuta primero scripts/entrenar.py")

    run_step("Paso 1/2: reconstrucción con Keras 2 (tf_keras)", STEP1_REBUILD)
    run_step("Paso 2/2: exportación a TensorFlow.js", STEP2_EXPORT, extra_env={"TF_USE_LEGACY_KERAS": "1"})

    print(f"\n✅ Modelo TensorFlow.js exportado a: {OUTPUT_DIR}")
    print(f"   Archivos: {os.listdir(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
