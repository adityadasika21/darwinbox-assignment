import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: true,
    // A tunnel presents an arbitrary hostname; Vite rejects unknown hosts by default.
    allowedHosts: true,
    // The API is same-origin in dev, so SSE and uploads need no CORS dance.
    proxy: { "/api": { target: "http://127.0.0.1:8000", changeOrigin: true } },
  },
});
