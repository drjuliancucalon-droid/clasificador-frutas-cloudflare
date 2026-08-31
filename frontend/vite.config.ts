import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  root: ".",
  plugins: [react()],
  build: {
    outDir: "dist",
  },
  server: {
    port: 5173,
    host: true,
  },
});