import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      // so you can import from 'shared-theme' instead of relative paths
      'shared-theme': path.resolve(__dirname, 'src/shared-theme'),
      dashboard:    path.resolve(__dirname, 'src/dashboard'),
    },
    extensions: ['.tsx', '.ts', '.jsx', '.js', '.json'],
  },
  server: {
    proxy: {
      '/api': 'http://localhost:5000'
    }
  }
});
