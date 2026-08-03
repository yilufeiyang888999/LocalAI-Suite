import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';

export default defineConfig({
  plugins: [vue()],
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
    host: '127.0.0.1',
    watch: {
      // The Tauri dev run writes to src-tauri\target\debug\deps\*.exe while
      // Vite is also trying to watch the same tree, which races and trips
      // Windows file locking (EBUSY). Skipping the target dir keeps the
      // frontend watcher focused on what it actually needs to reload.
      ignored: ['**/src-tauri/target/**']
    }
  },
  envPrefix: ['VITE_', 'TAURI_'],
  build: {
    target: process.env.TAURI_ENV_PLATFORM === 'windows' ? 'chrome105' : 'safari13',
    minify: !process.env.TAURI_ENV_DEBUG,
    sourcemap: !!process.env.TAURI_ENV_DEBUG
  }
});
