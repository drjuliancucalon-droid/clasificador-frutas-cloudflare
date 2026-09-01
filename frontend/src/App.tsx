import React, { useState, useCallback, useRef } from "react";
import * as tf from "@tensorflow/tfjs";
import "./App.css";

const API_BASE = import.meta.env.VITE_API_URL || "https://clasificador-frutas-api.dr-juliancucalon.workers.dev";
const CLASES = ["manzana", "platano", "naranja", "tomate"];
// Mismo orden que labels.json / metrics.json (orden alfabético de las carpetas
// originales del dataset: Apple Red 1, Banana, Orange, Tomato 1).
const MODEL_URL = "/model/model.json";
const IMG_SIZE = 128;

// La inferencia corre en el navegador con TensorFlow.js (no en el Worker):
// Cloudflare Workers en el plan gratuito limita el tiempo de CPU a 10ms por
// invocación, insuficiente para un forward-pass de MobileNetV2. El modelo
// entrenado (proyecto/models/modelo_frutas.keras) se exportó a este formato
// con scripts/convertir_tfjs.py. Ver docs/documentacion-tecnica.md, sección
// "Arquitectura de inferencia".
let modelPromise: Promise<tf.LayersModel> | null = null;
function getModel(): Promise<tf.LayersModel> {
  if (!modelPromise) {
    modelPromise = tf.loadLayersModel(MODEL_URL);
  }
  return modelPromise;
}

async function preprocessImage(file: File): Promise<tf.Tensor4D> {
  const bitmap = await createImageBitmap(file);
  const canvas = document.createElement("canvas");
  canvas.width = IMG_SIZE;
  canvas.height = IMG_SIZE;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("No se pudo preparar la imagen (canvas no disponible)");
  ctx.drawImage(bitmap, 0, 0, IMG_SIZE, IMG_SIZE);
  // El modelo incluye su propia capa Rescaling ([-1,1]); aquí se entregan
  // valores de píxel crudos (0-255), igual que en el entrenamiento.
  return tf.tidy(() => tf.browser.fromPixels(canvas).toFloat().expandDims(0) as tf.Tensor4D);
}

type LocalPrediction = {
  clasePredicha: string;
  confianza: number;
  probabilidades: { clase: string; probabilidad: number }[];
  tiempoInferenciaMs: number;
};

async function classifyLocally(file: File): Promise<LocalPrediction> {
  const model = await getModel();
  const input = await preprocessImage(file);
  const t0 = performance.now();
  const output = model.predict(input) as tf.Tensor;
  const probs = Array.from(await output.data());
  const tiempoInferenciaMs = performance.now() - t0;
  tf.dispose([input, output]);

  const claseIndex = probs.indexOf(Math.max(...probs));
  return {
    clasePredicha: CLASES[claseIndex],
    confianza: probs[claseIndex],
    probabilidades: CLASES.map((clase, i) => ({ clase, probabilidad: probs[i] })),
    tiempoInferenciaMs,
  };
}

type PredictionResult = {
  success: boolean;
  prediccion: string;
  confianza: number;
  probabilidades: { clase: string; probabilidad: number }[];
  tiempo_inferencia_ms: number;
  timestamp: string;
  id: number;
};

type HistoryItem = {
  id: number;
  imagen_nombre: string;
  clase_predicha: string;
  confianza: number;
  timestamp: string;
  tiempo_inferencia_ms: number;
};

export default function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [metrics, setMetrics] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<"predict" | "history" | "metrics">("predict");
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback((file: File) => {
    if (!file.type.startsWith("image/")) {
      setError("Solo se permiten imágenes (JPEG, PNG, WebP)");
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      setError("La imagen es demasiado grande (máx 10MB)");
      return;
    }
    setSelectedFile(file);
    setPreview(URL.createObjectURL(file));
    setResult(null);
    setError(null);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }, [handleFile]);

  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
  }, []);

  const handlePredict = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);
    try {
      // 1) Inferencia real en el navegador (TensorFlow.js + modelo entrenado)
      const local = await classifyLocally(selectedFile);

      // 2) Se envía la imagen + la predicción calculada al Worker, que valida,
      //    persiste en D1 y devuelve el registro (id, timestamp) para historial/métricas.
      const formData = new FormData();
      formData.append("image", selectedFile);
      formData.append("clase_predicha", local.clasePredicha);
      formData.append("confianza", String(local.confianza));
      formData.append("probabilidades", JSON.stringify(local.probabilidades));
      formData.append("tiempo_inferencia_ms", String(local.tiempoInferenciaMs));

      try {
        const res = await fetch(`${API_BASE}/predict`, { method: "POST", body: formData });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Error al registrar la predicción");
        setResult(data);
      } catch (apiErr) {
        // La clasificación ya ocurrió en el navegador; si solo falla el
        // registro remoto, igual se muestra el resultado real al usuario.
        setResult({
          success: true,
          prediccion: local.clasePredicha,
          confianza: local.confianza,
          probabilidades: local.probabilidades,
          tiempo_inferencia_ms: local.tiempoInferenciaMs,
          timestamp: new Date().toISOString(),
          id: 0,
        });
        setError("Clasificado localmente, pero no se pudo guardar en el historial (API no disponible)");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Error al clasificar la imagen");
    } finally {
      setLoading(false);
    }
  };

  const loadHistory = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/history`);
      const data = await res.json();
      setHistory(data.predicciones || []);
    } catch (err) {
      setError("Error al cargar historial");
    } finally {
      setLoading(false);
    }
  };

  const loadMetrics = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/metrics`);
      const data = await res.json();
      setMetrics(data);
    } catch (err) {
      setError("Error al cargar métricas");
    } finally {
      setLoading(false);
    }
  };

  const getEmoji = (clase: string) => {
    const map: Record<string, string> = {
      manzana: "🍎", platano: "🍌", naranja: "🍊", tomate: "🍅"
    };
    return map[clase] || "❓";
  };

  // Color de acento por clase, para que el resultado se sienta "vivo" y no
  // siempre del mismo tono azul sin importar la fruta detectada.
  const getAccent = (clase: string) => {
    const map: Record<string, { bg: string; border: string; text: string; bar: string }> = {
      manzana: { bg: "#fef2f2", border: "#fecaca", text: "#b91c1c", bar: "linear-gradient(90deg, #f87171, #dc2626)" },
      platano: { bg: "#fffbeb", border: "#fde68a", text: "#b45309", bar: "linear-gradient(90deg, #fbbf24, #d97706)" },
      naranja: { bg: "#fff7ed", border: "#fed7aa", text: "#c2410c", bar: "linear-gradient(90deg, #fb923c, #ea580c)" },
      tomate: { bg: "#fdf2f8", border: "#fbcfe8", text: "#be185d", bar: "linear-gradient(90deg, #fb7185, #e11d48)" },
    };
    return map[clase] || { bg: "#f0f9ff", border: "#bae6fd", text: "#1a1a2e", bar: "linear-gradient(90deg, #22c55e, #16a34a)" };
  };

  const handleClear = (e: React.MouseEvent) => {
    e.stopPropagation();
    setSelectedFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  return (
    <div style={{
      maxWidth: 800, margin: "0 auto", padding: 20,
      fontFamily: "'Segoe UI', system-ui, sans-serif", color: "#1a1a2e",
      background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
      minHeight: "100vh"
    }}>
      <div style={{ background: "white", borderRadius: 16, padding: 24, boxShadow: "0 10px 40px rgba(0,0,0,0.2)" }}>
        <h1 style={{ fontSize: 28, margin: "0 0 4px", textAlign: "center" }}>
          🍎 Clasificador de Frutas
        </h1>
        <p style={{ textAlign: "center", color: "#666", margin: "0 0 20px", fontSize: 14 }}>
          Julian Cucalon · Actividad 3-1 UAM
        </p>

        {/* Tabs */}
        <div style={{ display: "flex", gap: 8, marginBottom: 20 }}>
          {(["predict", "history", "metrics"] as const).map((t) => (
            <button key={t} onClick={() => { setTab(t); if (t === "history") loadHistory(); if (t === "metrics") loadMetrics(); }}
              className={`tab-btn tab-${t}${tab === t ? " active" : ""}`}>
              {t === "predict" ? "🔮 Clasificar" : t === "history" ? "📋 Historial" : "📊 Métricas"}
            </button>
          ))}
        </div>

        {error && (
          <div style={{ background: "#fee", color: "#c33", padding: "8px 12px", borderRadius: 8, marginBottom: 12, fontSize: 13 }}>
            ❌ {error}
          </div>
        )}

        {tab === "predict" && (
          <>
            {/* Drop zone */}
            <div
              onDrop={handleDrop}
              onDragOver={(e) => e.preventDefault()}
              onDragEnter={handleDragEnter}
              onDragLeave={handleDragLeave}
              onClick={() => fileInputRef.current?.click()}
              className={`upload-zone${preview ? " has-preview" : ""}${dragActive ? " drag-active" : ""}`}
            >
              <input ref={fileInputRef} type="file" accept="image/*" hidden onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])} />
              {preview ? (
                <>
                  <button className="remove-file-btn" onClick={handleClear} title="Quitar imagen" aria-label="Quitar imagen">✕</button>
                  <img src={preview} alt="Preview" style={{ maxHeight: 200, maxWidth: "100%", borderRadius: 8, display: "block", margin: "0 auto" }} />
                  {selectedFile && (
                    <div className="file-chip">
                      📎 <span>{selectedFile.name}</span>
                    </div>
                  )}
                </>
              ) : (
                <div>
                  <div className="upload-icon">{dragActive ? "📥" : "📸"}</div>
                  <p style={{ margin: 0, color: "#4b5563" }}>
                    {dragActive ? "¡Suelta la imagen aquí!" : (<>Arrastra una imagen aquí o <span className="browse-hint">haz clic para seleccionar</span></>)}
                  </p>
                  <p style={{ margin: "4px 0 0", fontSize: 12, color: "#999" }}>JPEG, PNG o WebP · Máx 10MB</p>
                </div>
              )}
            </div>

            {selectedFile && (
              <button onClick={handlePredict} disabled={loading} className={`classify-btn${!loading && !result ? " ready" : ""}`}>
                {loading ? (<><span className="spinner" /> Clasificando...</>) : "🔍 Clasificar Imagen"}
              </button>
            )}

            {result && (() => {
              const accent = getAccent(result.prediccion);
              return (
                <div className="result-card" style={{ background: accent.bg, borderColor: accent.border }}>
                  <h3 style={{ margin: "0 0 12px", fontSize: 18, color: accent.text }}>Resultado:</h3>
                  <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 12 }}>
                    <span style={{ fontSize: 48 }}>{getEmoji(result.prediccion)}</span>
                    <div>
                      <div style={{ fontSize: 24, fontWeight: 700, color: accent.text, textTransform: "capitalize" }}>
                        {result.prediccion}
                      </div>
                      <div style={{ fontSize: 14, color: "#666" }}>
                        Confianza: {(result.confianza * 100).toFixed(1)}%
                      </div>
                    </div>
                  </div>
                  <div style={{ fontSize: 12, color: "#999" }}>
                    ⏱ {result.tiempo_inferencia_ms.toFixed(0)}ms · ID: {result.id}
                  </div>
                  {/* Barra de confianza */}
                  <div style={{ marginTop: 8, height: 8, background: "rgba(0,0,0,0.08)", borderRadius: 4, overflow: "hidden" }}>
                    <div style={{ height: "100%", width: `${result.confianza * 100}%`, background: accent.bar, borderRadius: 4, transition: "width 0.6s ease" }} />
                  </div>
                </div>
              );
            })()}
          </>
        )}

        {tab === "history" && (
          <div>
            <h3 style={{ margin: "0 0 12px" }}>📋 Historial de Predicciones</h3>
            {loading ? (
              <p style={{ color: "#666" }}>Cargando...</p>
            ) : history.length === 0 ? (
              <p style={{ color: "#999" }}>No hay predicciones aún. ¡Clasifica tu primera imagen!</p>
            ) : (
              <div style={{ maxHeight: 400, overflowY: "auto" }}>
                {history.map((h) => (
                  <div key={h.id} className="history-row" style={{ display: "flex", alignItems: "center", gap: 12, padding: "8px 10px", borderBottom: "1px solid #eee" }}>
                    <span style={{ fontSize: 24 }}>{getEmoji(h.clase_predicha)}</span>
                    <div style={{ flex: 1 }}>
                      <div style={{ fontWeight: 600, fontSize: 14, textTransform: "capitalize" }}>{h.clase_predicha}</div>
                      <div style={{ fontSize: 12, color: "#666" }}>{(h.confianza * 100).toFixed(1)}% · {h.imagen_nombre}</div>
                    </div>
                    <div style={{ fontSize: 11, color: "#999" }}>{new Date(h.timestamp).toLocaleString()}</div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {tab === "metrics" && (
          <div>
            <h3 style={{ margin: "0 0 12px" }}>📊 Métricas de Uso</h3>
            {loading ? (
              <p style={{ color: "#666" }}>Cargando...</p>
            ) : metrics ? (
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                <div className="metric-card" style={{ background: "linear-gradient(135deg, #eff6ff, #dbeafe)" }}>
                  <div style={{ fontSize: 32, fontWeight: 700, color: "#2563eb" }}>{metrics.total_predicciones}</div>
                  <div style={{ fontSize: 12, color: "#666" }}>Total Predicciones</div>
                </div>
                <div className="metric-card" style={{ background: "linear-gradient(135deg, #fffbeb, #fef3c7)" }}>
                  <div style={{ fontSize: 32, fontWeight: 700, color: "#d97706" }}>{metrics.predicciones_hoy}</div>
                  <div style={{ fontSize: 12, color: "#666" }}>Predicciones Hoy</div>
                </div>
                <div style={{ gridColumn: "1 / -1" }}>
                  <h4 style={{ margin: "8px 0" }}>Distribución por Clase:</h4>
                  {metrics.distribucion_por_clase?.map((d: any) => {
                    const accent = getAccent(d.clase_predicha);
                    return (
                      <div key={d.clase_predicha} className="history-row" style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6, padding: "6px 10px" }}>
                        <span style={{ fontSize: 20 }}>{getEmoji(d.clase_predicha)}</span>
                        <span style={{ flex: 1, fontSize: 14, textTransform: "capitalize" }}>{d.clase_predicha}</span>
                        <span style={{ fontWeight: 700, fontSize: 14, color: accent.text }}>{d.count}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            ) : (
              <p style={{ color: "#999" }}>No hay métricas disponibles.</p>
            )}
          </div>
        )}

        <div style={{ marginTop: 20, paddingTop: 12, borderTop: "1px solid #eee", fontSize: 11, color: "#999", textAlign: "center" }}>
          API: {API_BASE} · <a href={`${API_BASE}/health`} target="_blank" style={{ color: "#667eea" }}>Health</a> · <a href={`${API_BASE}/`} target="_blank" style={{ color: "#667eea" }}>Docs</a>
        </div>
      </div>
      <div style={{ textAlign: "center", marginTop: 12, fontSize: 12, color: "rgba(255,255,255,0.7)" }}>
        UAM · Aplicaciones en la Nube · Actividad 3-1
      </div>
    </div>
  );
}