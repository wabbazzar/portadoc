import { defineConfig } from 'vite';

export default defineConfig({
  base: './',
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
  optimizeDeps: {
    exclude: ['pdfjs-dist'],
  },
  server: {
    port: 5173,
    open: false,
  },
});
