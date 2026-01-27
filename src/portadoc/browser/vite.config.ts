import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
  // Use /portadoc/ for GitHub Pages, ./ for local dev
  base: process.env.NODE_ENV === 'production' ? '/portadoc/' : './',
  build: {
    outDir: 'dist',
    sourcemap: true,
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
      },
    },
  },
  optimizeDeps: {
    exclude: ['pdfjs-dist'],
  },
  server: {
    port: 5173,
    open: false,
  },
  publicDir: 'public',
});
