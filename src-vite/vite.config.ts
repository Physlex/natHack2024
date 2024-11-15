import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

/**
 * Defines the configuration details. Allows linting due to `defineConfig`, but is
 * effectively just some json.
 */
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "../src-django/src/static",
    emptyOutDir: true,
    manifest: true,
    rollupOptions: {
      input: "src/main.tsx"
    }
  }
})