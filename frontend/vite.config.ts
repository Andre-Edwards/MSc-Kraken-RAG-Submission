import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const localApiProxy = {
  '/api': 'http://127.0.0.1:8000',
};

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: localApiProxy,
  },
  preview: {
    host: '127.0.0.1',
    port: 4173,
    proxy: localApiProxy,
  },
});
