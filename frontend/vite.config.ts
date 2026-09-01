import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

// La URL de la API en producción se fija aquí explícitamente (en vez de
// depender solo de un archivo .env o de una variable de entorno del shell)
// porque un VITE_API_URL=http://localhost:8787 exportado durante desarrollo
// local terminó filtrándose a un build de producción anterior, dejando el
// frontend desplegado incapaz de contactar la API real. Con `command ===
// "build"` esto no puede volver a pasar por accidente.
const PRODUCTION_API_URL = "https://clasificador-frutas-api.dr-juliancucalon.workers.dev";

// Excepción única y explícita: el build DENTRO de Docker (Dockerfile.frontend)
// sí debe apuntar al contenedor "api" local, expuesto en el host como
// http://localhost:8787 (ver docker-compose.yml, "8787:8787") — si no, el
// contenedor "api" queda corriendo sin que nadie lo use. Se activa solo con
// la variable DOCKER_LOCAL_BUILD (puesta únicamente en Dockerfile.frontend),
// nunca con VITE_API_URL directamente — así un VITE_API_URL suelto en el
// shell del desarrollador no puede volver a filtrarse a producción.
const isDockerLocalBuild = process.env.DOCKER_LOCAL_BUILD === "true";

export default defineConfig(({ command }) => ({
  root: ".",
  plugins: [react()],
  define:
    command === "build"
      ? {
          "import.meta.env.VITE_API_URL": JSON.stringify(
            isDockerLocalBuild ? "http://localhost:8787" : PRODUCTION_API_URL
          ),
        }
      : {},
  build: {
    outDir: "dist",
  },
  server: {
    port: 5173,
    host: true,
  },
}));