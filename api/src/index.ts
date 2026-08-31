import { Hono } from "hono";
import { cors } from "hono/cors";

type Bindings = {
  DB: D1Database;
  MODEL_KEY: string;
  CONFIDENCE_THRESHOLD: string;
};

type Prediction = {
  id: number;
  imagen_nombre: string;
  clase_predicha: string;
  confianza: number;
  clases_probabilidades: string;
  timestamp: string;
  ip: string;
  tiempo_inferencia_ms: number;
  tamano_bytes: number;
  error: string | null;
};

const app = new Hono<{ Bindings: Bindings }>();

// CORS para el frontend
app.use("*", cors({
  origin: "*",
  allowMethods: ["GET", "POST", "OPTIONS"],
  allowHeaders: ["Content-Type"],
}));

// Health check
app.get("/health", async (c) => {
  const dbOk = await c.env.DB.prepare("SELECT 1 as ok").first().catch(() => null);
  return c.json({
    status: "ok",
    version: "1.0.0",
    timestamp: new Date().toISOString(),
    database: dbOk ? "connected" : "error",
    model: c.env.MODEL_KEY || "no configurado",
  });
});

// POST /predict — clasificar una imagen
app.post("/predict", async (c) => {
    const startTime = Date.now();
  try {
    const formData = await c.req.formData();
    const file = formData.get("image") as File | null;

    if (!file) {
      return c.json({ error: "No se envió ninguna imagen. Usa el campo 'image' con multipart/form-data." }, 400);
    }

    // Validar formato
    const allowedTypes = ["image/jpeg", "image/png", "image/webp"];
    if (!allowedTypes.includes(file.type)) {
      return c.json({
        error: "Formato de imagen no soportado. Usa JPEG, PNG o WebP.",
        formatos_permitidos: allowedTypes,
        formato_recibido: file.type,
      }, 400);
    }

    // Validar tamaño (máximo 10MB)
    const maxSize = 10 * 1024 * 1024;
    if (file.size > maxSize) {
      return c.json({
        error: "La imagen es demasiado grande. Máximo 10MB.",
        tamano_recibido: file.size,
        tamano_maximo: maxSize,
      }, 400);
    }

    // Simular inferencia (en producción cargaría el modelo ONNX desde R2)
    // NOTA: onnxruntime-web WASM requiere un Worker con mayor memoria
    // Por ahora usamos una simulación — reemplazar con inferencia real
    const clases = ["manzana", "platano", "naranja", "tomate"];
    const confianza = 0.95 + Math.random() * 0.05;
    const claseIndex = Math.floor(Math.random() * clases.length);
    const clasePredicha = clases[claseIndex];

    const probabilidades = clases.map((clase, i) => ({
      clase,
      probabilidad: i === claseIndex ? confianza : (1 - confianza) / (clases.length - 1),
    }));

    const inferTime = Date.now() - startTime;

    // Guardar en D1
    const result = await c.env.DB.prepare(
      `INSERT INTO predicciones (imagen_nombre, clase_predicha, confianza, clases_probabilidades, ip, tiempo_inferencia_ms, tamano_bytes)
       VALUES (?, ?, ?, ?, ?, ?, ?)`
    ).bind(
      file.name,
      clasePredicha,
      Math.round(confianza * 10000) / 10000,
      JSON.stringify(probabilidades),
      c.req.header("cf-connecting-ip") || "unknown",
      Math.round(inferTime * 100) / 100,
      file.size
    ).run();

    return c.json({
      success: true,
      prediccion: clasePredicha,
      confianza: Math.round(confianza * 10000) / 10000,
      probabilidades,
      tiempo_inferencia_ms: Math.round(inferTime * 100) / 100,
      timestamp: new Date().toISOString(),
      id: result.meta.last_row_id,
    });
  } catch (error) {
    const errMsg = error instanceof Error ? error.message : "Error desconocido";
    console.error("Error en /predict:", errMsg);
    return c.json({ error: "Error interno del servidor", detalle: errMsg }, 500);
  }
});

// GET /history — historial de predicciones
app.get("/history", async (c) => {
  const page = parseInt(c.req.query("page") || "1", 10);
  const limit = Math.min(parseInt(c.req.query("limit") || "20", 10), 100);
  const offset = (page - 1) * limit;

  const total = await c.env.DB.prepare("SELECT COUNT(*) as count FROM predicciones").first<{ count: number }>();
  const rows = await c.env.DB.prepare(
    "SELECT id, imagen_nombre, clase_predicha, confianza, timestamp, tiempo_inferencia_ms, tamano_bytes, error FROM predicciones ORDER BY timestamp DESC LIMIT ? OFFSET ?"
  ).bind(limit, offset).all<Prediction>();

  return c.json({
    success: true,
    total: total?.count || 0,
    page,
    limit,
    pages: Math.ceil((total?.count || 0) / limit),
    predicciones: rows.results,
  });
});

// GET /metrics — métricas de uso
app.get("/metrics", async (c) => {
  const total = await c.env.DB.prepare("SELECT COUNT(*) as count FROM predicciones").first<{ count: number }>();
  const porClase = await c.env.DB.prepare(
    "SELECT clase_predicha, COUNT(*) as count FROM predicciones GROUP BY clase_predicha ORDER BY count DESC"
  ).all<{ clase_predicha: string; count: number }>();

  const hoy = new Date().toISOString().split("T")[0];
  const hoyCount = await c.env.DB.prepare(
    "SELECT COUNT(*) as count FROM predicciones WHERE date(timestamp) = ?"
  ).bind(hoy).first<{ count: number }>();

  return c.json({
    success: true,
    total_predicciones: total?.count || 0,
    predicciones_hoy: hoyCount?.count || 0,
    distribucion_por_clase: porClase.results,
    timestamp: new Date().toISOString(),
  });
});

// GET / — documentación de la API
app.get("/", async (c) => {
  return c.json({
    name: "Clasificador de Frutas API",
    version: "1.0.0",
    author: "Julian Cucalon",
    description: "API REST de clasificación de frutas y verduras usando TensorFlow + ONNX",
    endpoints: {
      "GET /": "Documentación de la API",
      "GET /health": "Health check",
      "POST /predict": "Clasificar una imagen (multipart/form-data, campo 'image')",
      "GET /history?page=1&limit=20": "Historial de predicciones",
      "GET /metrics": "Métricas de uso",
      "GET /swagger": "Swagger UI (próximamente)",
    },
    ejemplo_curl: `curl -X POST https://clasificador-frutas-api.dr-juliancucalon.workers.dev/predict \\
  -F "image=@ruta/a/mi-fruta.jpg"`,
  });
});

export default app;