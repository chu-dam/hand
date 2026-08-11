import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  build: {
    rolldownOptions: {
      output: {
        codeSplitting: {
          maxSize: 450 * 1024,
          groups: [
            {
              name: "three",
              test: /node_modules[\\/](three|urdf-loader)[\\/]/,
            },
            {
              name: "react",
              test: /node_modules[\\/](react|react-dom)[\\/]/,
            },
            {
              name: "ros",
              test: /node_modules[\\/](roslib|eventemitter3|cbor-js)[\\/]/,
            },
          ],
        },
      },
    },
  },
  server: {
    host: "0.0.0.0",
    port: 8080,
    strictPort: true,
    proxy: {
      "/api": "http://127.0.0.1:8081",
    },
  },
  preview: {
    host: "0.0.0.0",
    port: 8080,
    strictPort: true,
  },
});
