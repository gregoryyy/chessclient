import { defineConfig } from "vite";

export default defineConfig({
  server: {
    // Enable cross-origin isolation so WASM threads/SAB can work
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp"
    }
  },
  // Static files in /public are served as-is; nothing else required
});
