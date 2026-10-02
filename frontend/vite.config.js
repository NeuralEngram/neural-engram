import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Built files are served by FastAPI at /ui (see api/main.py).
// During development, `npm run dev` proxies API calls to FastAPI on :8000.
export default defineConfig({
  plugins: [react()],
  base: '/ui/',
  server: {
    port: 5173,
    proxy: {
      '/app': 'http://127.0.0.1:8000',
    },
  },
})
