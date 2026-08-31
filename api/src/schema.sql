-- Esquema de base de datos D1 para el clasificador de frutas
CREATE TABLE IF NOT EXISTS predicciones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    imagen_nombre TEXT NOT NULL,
    clase_predicha TEXT NOT NULL,
    confianza REAL NOT NULL,
    clases_probabilidades TEXT NOT NULL,  -- JSON
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    ip TEXT,
    tiempo_inferencia_ms REAL,
    tamano_bytes INTEGER,
    error TEXT
);

CREATE INDEX IF NOT EXISTS idx_predicciones_timestamp ON predicciones(timestamp);
CREATE INDEX IF NOT EXISTS idx_predicciones_clase ON predicciones(clase_predicha);