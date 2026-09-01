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

    // La inferencia real ocurre en el navegador con TensorFlow.js, usando el
    // modelo entrenado (MobileNetV2, ver proyecto/scripts/entrenar.py y
    // frontend/src/App.tsx). Cloudflare Workers en el plan gratuito limita el
    // tiempo de CPU a 10ms por invocación, insuficiente para un forward-pass
    // de esta red — ver docs/documentacion-tecnica.md, sección "Arquitectura
    // de inferencia", para el detalle de esta decisión.
    //
    // El Worker recibe la imagen (para las mismas validaciones de siempre) y
    // el resultado ya calculado por el cliente, y aquí solo lo valida antes
    // de persistirlo — nunca confía en un valor fuera de las clases conocidas
    // o un rango de confianza inválido.
    const clases = ["manzana", "platano", "naranja", "tomate"];

    const clasePredicha = String(formData.get("clase_predicha") || "");
    if (!clases.includes(clasePredicha)) {
      return c.json({
        error: "Clase predicha inválida o ausente.",
        clases_validas: clases,
      }, 400);
    }

    const confianzaRaw = Number(formData.get("confianza"));
    if (!Number.isFinite(confianzaRaw) || confianzaRaw < 0 || confianzaRaw > 1) {
      return c.json({ error: "Confianza inválida o ausente. Debe ser un número entre 0 y 1." }, 400);
    }
    const confianza = confianzaRaw;

    let probabilidades: { clase: string; probabilidad: number }[];
    try {
      const parsed = JSON.parse(String(formData.get("probabilidades") || "[]"));
      if (
        !Array.isArray(parsed) ||
        parsed.length !== clases.length ||
        !parsed.every((p) => clases.includes(p?.clase) && typeof p?.probabilidad === "number")
      ) {
        throw new Error("shape");
      }
      probabilidades = parsed;
    } catch {
      return c.json({ error: "Campo 'probabilidades' inválido o ausente." }, 400);
    }

    // tiempo_inferencia_ms: tiempo real del forward-pass en TensorFlow.js,
    // medido en el navegador y enviado por el cliente (no el tiempo del
    // Worker, que solo valida y persiste — ver comentario más arriba).
    const tiempoInferenciaMs = Number(formData.get("tiempo_inferencia_ms")) || 0;

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
      Math.round(tiempoInferenciaMs * 100) / 100,
      file.size
    ).run();

    return c.json({
      success: true,
      prediccion: clasePredicha,
      confianza: Math.round(confianza * 10000) / 10000,
      probabilidades,
      tiempo_inferencia_ms: Math.round(tiempoInferenciaMs * 100) / 100,
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
    description: "API REST de historial y métricas para el clasificador de frutas y verduras. La inferencia (MobileNetV2 vía TensorFlow.js) corre en el navegador; este Worker valida el resultado y lo persiste en D1.",
    endpoints: {
      "GET /": "Documentación de la API",
      "GET /health": "Health check",
      "POST /predict": "Registra una clasificación (multipart/form-data: 'image', 'clase_predicha', 'confianza', 'probabilidades', 'tiempo_inferencia_ms')",
      "GET /history?page=1&limit=20": "Historial de predicciones",
      "GET /metrics": "Métricas de uso",
      "GET /swagger": "Swagger UI (próximamente)",
    },
    nota: "Los campos clase_predicha/confianza/probabilidades/tiempo_inferencia_ms normalmente los calcula el frontend con el modelo real (TensorFlow.js). El ejemplo de curl de abajo los pasa a mano solo para poder probar el endpoint de forma aislada.",
    ejemplo_curl: `curl -X POST https://clasificador-frutas-api.dr-juliancucalon.workers.dev/predict \\
  -F "image=@ruta/a/mi-manzana.jpg" \\
  -F "clase_predicha=manzana" \\
  -F "confianza=0.98" \\
  -F 'probabilidades=[{"clase":"manzana","probabilidad":0.98},{"clase":"platano","probabilidad":0.01},{"clase":"naranja","probabilidad":0.005},{"clase":"tomate","probabilidad":0.005}]' \\
  -F "tiempo_inferencia_ms=15"`,
  });
});

export default app;