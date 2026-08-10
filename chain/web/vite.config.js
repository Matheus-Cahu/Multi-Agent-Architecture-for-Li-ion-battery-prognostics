import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// A pagina roda em :5173 e a API em :3000. O proxy evita CORS no dev:
// o fetch usa caminho relativo ("/reports") e o Vite encaminha para a API.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/reports": { target: process.env.API_URL || "http://localhost:3000", changeOrigin: true }
    }
  }
});
