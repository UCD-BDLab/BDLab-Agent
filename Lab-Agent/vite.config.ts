import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      'shared-theme': path.resolve(__dirname, 'src/shared-theme'),
      'Lab-Agent': path.resolve(__dirname, 'src/Lab-Agent'),
    },
    extensions: ['.tsx', '.ts', '.jsx', '.js', '.json'],
  },
  server: {
    proxy: {
      '/api': 'http://localhost:5000'
    }
  }
});
