import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    setupFiles: "./src/test/setup.tsx",
    globals: true
  },
  server: {
    port: 5173,
    proxy: {
      "/api": "http://localhost:8000"
    }
  }
});
