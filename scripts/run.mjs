import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const mode = process.argv[2] === "start" ? "start" : "dev";
const python = process.env.GPU_TRACKER_PYTHON ||
  (existsSync(join(root, ".venv", "bin", "python")) ? join(root, ".venv", "bin", "python") : "python3");
const next = join(root, "node_modules", "next", "dist", "bin", "next");
const env = { ...process.env, PYTHONPATH: join(root, "src") };
const api = spawn(python, [
  "-m", "gpu_avail_tracker", "--api-only",
  "--env-file", process.env.GPU_TRACKER_ENV_FILE || join(root, ".env"),
  "--api-port", process.env.GPU_TRACKER_API_PORT || "8000",
], { cwd: root, env, stdio: "inherit" });
const web = spawn(process.execPath, [
  next, mode, "-H", "127.0.0.1", "-p", process.env.WEB_PORT || "3000",
], { cwd: root, env, stdio: "inherit" });

let closing = false;
function stop(code = 0) {
  if (closing) return;
  closing = true;
  api.kill("SIGTERM");
  web.kill("SIGTERM");
  process.exitCode = code;
}
api.on("error", (error) => { console.error(error); stop(1); });
web.on("error", (error) => { console.error(error); stop(1); });
api.on("exit", (code) => stop(code || 1));
web.on("exit", (code) => stop(code || 0));
process.on("SIGINT", () => stop());
process.on("SIGTERM", () => stop());
