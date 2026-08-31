"""
Entrenamiento del modelo de clasificación de frutas y verduras.
Actividad 3-1 — Aplicaciones en la Nube y Servicios Especializados en Ciencia de Datos
Autor: Julian Cucalon

Pipeline:
  1. Carga dataset Fruits-360 (5 categorías)
  2. Data augmentation (solo en entrenamiento)
  3. Transfer learning con MobileNetV2 (ImageNet)
  4. Evaluación: accuracy, precision, recall, F1, matriz de confusión
  5. Exportación: .keras + ONNX (para Cloudflare Worker y navegador)
"""
import json
import os
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

# ---------------- Configuración ----------------
SEED = 42
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 12
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(OUT_DIR, exist_ok=True)

tf.random.set_seed(SEED)
np.random.seed(SEED)

# ---------------- Carga de datos ----------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "Training"),
    validation_split=0.15,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "Training"),
    validation_split=0.15,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "Test"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False,
)

class_names = train_ds.class_names  # orden alfabético de carpetas
num_classes = len(class_names)
print(f"Clases detectadas ({num_classes}): {class_names}")

# Mapeo a nombres en español (orden alfabético de carpetas Fruits-360)
SPANISH = {
    "Apple Red 1": "manzana",
    "Banana": "platano",
    "Carrot 1": "zanahoria",
    "Orange": "naranja",
    "Tomato 1": "tomate",
}
labels_es = [SPANISH.get(c, c) for c in class_names]
print(f"Labels en español: {labels_es}")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)
test_ds = test_ds.cache().prefetch(AUTOTUNE)

# ---------------- Data augmentation (fuera del modelo, para exportación limpia) ----------------
augmenter = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal", seed=SEED),
        tf.keras.layers.RandomRotation(0.08, seed=SEED),
        tf.keras.layers.RandomZoom(0.1, seed=SEED),
        tf.keras.layers.RandomBrightness(0.1, seed=SEED),
    ]
)

train_aug = train_ds.map(lambda x, y: (augmenter(x, training=True), y), num_parallel_calls=AUTOTUNE)

# ---------------- Modelo: Transfer Learning MobileNetV2 ----------------
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False  # fase 1: solo cabeza de clasificación

inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="input")
x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0)(inputs)  # normaliza a [-1, 1]
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="output")(x)
model = tf.keras.Model(inputs, outputs, name="clasificador_frutas")

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# ---------------- Entrenamiento ----------------
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=3, restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(OUT_DIR, "best.keras"), monitor="val_accuracy", save_best_only=True
    ),
]

print("\n=== FASE 1: entrenamiento de la cabeza de clasificación ===")
t0 = time.time()
history = model.fit(train_aug, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)
t_train = time.time() - t0
print(f"Tiempo de entrenamiento: {t_train:.1f}s")

# ---------------- Fine-tuning breve (descongelar últimas 20 capas) ----------------
print("\n=== FASE 2: fine-tuning (últimas 20 capas de MobileNetV2) ===")
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
history_ft = model.fit(train_aug, validation_data=val_ds, epochs=4, callbacks=callbacks)

# ---------------- Evaluación en test ----------------
print("\n=== EVALUACIÓN EN TEST ===")
y_true, y_pred = [], []
t0 = time.time()
for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))
t_infer_total = time.time() - t0
n_test = len(y_true)
t_infer_avg_ms = (t_infer_total / n_test) * 1000

report = classification_report(
    y_true, y_pred, target_names=labels_es, output_dict=True, zero_division=0
)
cm = confusion_matrix(y_true, y_pred).tolist()

test_loss, test_acc = model.evaluate(test_ds, verbose=0)
print(f"Test accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")
print(f"Tiempo promedio de inferencia: {t_infer_avg_ms:.2f} ms/imagen")

metrics = {
    "modelo": "MobileNetV2 transfer learning (Keras/TensorFlow)",
    "dataset": "Fruits-360 (subset 5 categorías)",
    "img_size": list(IMG_SIZE),
    "clases": labels_es,
    "clases_originales": class_names,
    "test_accuracy": float(test_acc),
    "test_loss": float(test_loss),
    "tiempo_inferencia_ms": round(t_infer_avg_ms, 2),
    "tiempo_entrenamiento_s": round(t_train, 1),
    "classification_report": report,
    "confusion_matrix": cm,
    "historial_fase1": {k: [float(v) for v in vals] for k, vals in history.history.items()},
    "historial_fase2": {k: [float(v) for v in history_ft.history.items().__iter__().__next__()[1]]} if False else {},
}
# historial fase 2
metrics["historial_fase2"] = {k: [float(v) for v in vals] for k, vals in history_ft.history.items()}

with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

with open(os.path.join(OUT_DIR, "labels.json"), "w", encoding="utf-8") as f:
    json.dump({"clases": labels_es, "clases_originales": class_names}, f, ensure_ascii=False, indent=2)

# ---------------- Guardar modelo Keras ----------------
model.save(os.path.join(OUT_DIR, "modelo_frutas.keras"))
print("Modelo Keras guardado: models/modelo_frutas.keras")

# ---------------- Exportar a ONNX ----------------
print("\n=== EXPORTACIÓN ONNX ===")
import tf2onnx

spec = (tf.TensorSpec((None, IMG_SIZE[0], IMG_SIZE[1], 3), tf.float32, name="input"),)
model_proto, _ = tf2onnx.convert.from_keras(
    model, input_signature=spec, output_path=os.path.join(OUT_DIR, "modelo_frutas.onnx"), opset=13
)
print("Modelo ONNX guardado: models/modelo_frutas.onnx")

# ---------------- Resumen final ----------------
size_keras = os.path.getsize(os.path.join(OUT_DIR, "modelo_frutas.keras")) / 1e6
size_onnx = os.path.getsize(os.path.join(OUT_DIR, "modelo_frutas.onnx")) / 1e6
print("\n========== RESUMEN ==========")
print(f"Accuracy test: {test_acc:.4f}")
print(f"Inferencia promedio: {t_infer_avg_ms:.2f} ms")
print(f"Tamaño .keras: {size_keras:.2f} MB | Tamaño .onnx: {size_onnx:.2f} MB")
print(f"Clases: {labels_es}")
print("Entrenamiento completado con éxito.")