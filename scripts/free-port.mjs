// Frees a TCP port before the dev server starts, so a leftover vite/node
// process from a previous run never blocks `npm run dev` / `npm run tauri:dev`.
// No dependencies; cross-platform. Usage: node scripts/free-port.mjs [port]
import { execSync } from 'node:child_process';

const port = process.argv[2] || '1420';

function pidsOnWindows() {
  const out = execSync('netstat -ano -p tcp', { encoding: 'utf8' });
  const pids = new Set();
  for (const line of out.split('\n')) {
    // Local address column looks like `127.0.0.1:1420`; only match LISTENING.
    if (line.includes(`:${port} `) && /LISTENING/i.test(line)) {
      const pid = line.trim().split(/\s+/).pop();
      if (pid && pid !== '0') pids.add(pid);
    }
  }
  return [...pids];
}

function pidsOnUnix() {
  try {
    const out = execSync(`lsof -ti tcp:${port}`, { encoding: 'utf8' });
    return out.split('\n').map((s) => s.trim()).filter(Boolean);
  } catch {
    return [];
  }
}

try {
  const pids = process.platform === 'win32' ? pidsOnWindows() : pidsOnUnix();
  for (const pid of pids) {
    try {
      if (process.platform === 'win32') {
        execSync(`taskkill /PID ${pid} /F`, { stdio: 'ignore' });
      } else {
        execSync(`kill -9 ${pid}`, { stdio: 'ignore' });
      }
      console.log(`[free-port] released port ${port} (pid ${pid})`);
    } catch {
      // Process may have already exited; ignore.
    }
  }
} catch {
  // netstat/lsof unavailable — skip silently, vite will still report a clear error.
}
