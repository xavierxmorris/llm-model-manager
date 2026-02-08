#!/usr/bin/env python3
"""
LLM Model Manager - Spin up/down different local LLM models on one GPU.

Usage:
    python llm_manager.py list                  # List available models
    python llm_manager.py start <model-id>      # Start a model server
    python llm_manager.py stop                  # Stop the running server
    python llm_manager.py switch <model-id>     # Stop current + start new
    python llm_manager.py status                # Show what's running
    python llm_manager.py test                  # Quick health + completions test
    python llm_manager.py add                   # Interactive: add a new model

Designed for: NVIDIA RTX PRO 4500 Blackwell 32GB + llama.cpp
"""

import json
import subprocess
import sys
import os
import time
import signal
import argparse
import urllib.request
import urllib.error
import textwrap
from pathlib import Path

# ─── Config ──────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
CONFIG_FILE = SCRIPT_DIR / "models.json"
PID_FILE = SCRIPT_DIR / ".llm-server.pid"
ACTIVE_MODEL_FILE = SCRIPT_DIR / ".llm-active-model"

# ─── Helpers ─────────────────────────────────────────────────────────────────

def load_config():
    """Load the models.json configuration."""
    if not CONFIG_FILE.exists():
        print(f"[ERROR] Config file not found: {CONFIG_FILE}")
        print(f"  Create one with 'llm_manager.py add' or copy the template.")
        sys.exit(1)
    with open(CONFIG_FILE, "r") as f:
        return json.load(f)


def save_config(config):
    """Save config back to models.json."""
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


def get_server_exe(config):
    """Get the llama-server executable path."""
    base = Path(config["defaults"]["llama_cpp_path"])
    exe = base / "llama-server.exe"
    if not exe.exists():
        # Try without .exe for Linux/Mac
        exe = base / "llama-server"
    if not exe.exists():
        print(f"[ERROR] llama-server not found at: {base}")
        sys.exit(1)
    return str(exe)


def is_server_running():
    """Check if a server process is running via PID file."""
    if not PID_FILE.exists():
        return False, None
    pid = int(PID_FILE.read_text().strip())
    try:
        # On Windows, this checks if process exists
        os.kill(pid, 0)
        return True, pid
    except (OSError, ProcessLookupError):
        # Stale PID file
        PID_FILE.unlink(missing_ok=True)
        ACTIVE_MODEL_FILE.unlink(missing_ok=True)
        return False, None


def is_port_in_use(port):
    """Check if a port is already in use."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def health_check(port, timeout=5):
    """Check if the server is healthy."""
    try:
        url = f"http://127.0.0.1:{port}/health"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            return data.get("status") == "ok"
    except Exception:
        return False


def get_active_model():
    """Get the currently active model ID."""
    if ACTIVE_MODEL_FILE.exists():
        return ACTIVE_MODEL_FILE.read_text().strip()
    return None


def format_size(gb):
    """Format size nicely."""
    if gb >= 1:
        return f"{gb:.1f} GB"
    return f"{gb * 1024:.0f} MB"


# ─── Commands ────────────────────────────────────────────────────────────────

def cmd_list(config):
    """List all available models."""
    models = config["models"]
    active = get_active_model()
    running, _ = is_server_running()

    print()
    print("  ╔══════════════════════════════════════════════════════════════════╗")
    print("  ║                    Available LLM Models                         ║")
    print("  ╠══════════════════════════════════════════════════════════════════╣")

    for model_id, m in models.items():
        is_active = running and active == model_id
        status = " ● RUNNING" if is_active else ""
        exists = Path(m["model_path"]).exists()
        file_status = "✓" if exists else "✗ NOT DOWNLOADED"

        print(f"  ║                                                                  ║")
        print(f"  ║  [{model_id}]{status}")
        print(f"  ║    {m['name']}")
        print(f"  ║    {m['description']}")
        print(f"  ║    Arch: {m.get('architecture', 'dense')}  |  Quant: {m['quant']}  |  Size: {format_size(m['size_gb'])}")
        print(f"  ║    File: {file_status}  {m['model_path']}")
        if m.get('mmproj_path'):
            mm_exists = Path(m['mmproj_path']).exists()
            mm_status = '✓' if mm_exists else '✗ NOT DOWNLOADED'
            print(f"  ║    Vision: {mm_status}  {m['mmproj_path']}")

    print(f"  ║                                                                  ║")
    print("  ╚══════════════════════════════════════════════════════════════════╝")
    print()

    if not running:
        print("  No model currently running. Use: llm_manager.py start <model-id>")
    print()


def cmd_start(config, model_id, wait=True):
    """Start a model server."""
    models = config["models"]
    defaults = config["defaults"]

    if model_id not in models:
        print(f"[ERROR] Unknown model: '{model_id}'")
        print(f"  Available: {', '.join(models.keys())}")
        sys.exit(1)

    model = models[model_id]
    model_path = model["model_path"]

    # Check model file exists
    if not Path(model_path).exists():
        print(f"[ERROR] Model file not found: {model_path}")
        print(f"  Download it first before starting.")
        sys.exit(1)

    # Check if something is already running
    running, old_pid = is_server_running()
    if running:
        old_model = get_active_model() or "unknown"
        print(f"[INFO] Server already running (PID {old_pid}, model: {old_model})")
        print(f"  Use 'switch {model_id}' to swap, or 'stop' first.")
        sys.exit(1)

    port = defaults.get("port", 8080)
    if is_port_in_use(port):
        print(f"[WARN] Port {port} is already in use. Attempting to continue...")

    # Build command
    server_exe = get_server_exe(config)
    cmd = [server_exe, "-m", model_path]

    # Add multimodal projector if configured
    mmproj = model.get("mmproj_path")
    if mmproj:
        if not Path(mmproj).exists():
            print(f"[WARN] mmproj file not found: {mmproj}")
            print(f"  Multimodal/vision features will not be available.")
        else:
            cmd.extend(["--mmproj", mmproj])
            print(f"  Vision:  mmproj loaded ({Path(mmproj).name})")

    # Add model-specific args
    for arg, val in model.get("server_args", {}).items():
        cmd.extend([arg, str(val)])

    # Add defaults
    cmd.extend(["--host", defaults.get("host", "0.0.0.0")])
    cmd.extend(["--port", str(port)])

    # Set environment
    env = os.environ.copy()
    for k, v in defaults.get("env", {}).items():
        env[k] = v

    print(f"[START] Launching: {model['name']}")
    print(f"  Model:  {model_path}")
    print(f"  Size:   {format_size(model['size_gb'])}")
    print(f"  Server: http://127.0.0.1:{port}")
    print(f"  Cmd:    {' '.join(cmd[:6])}...")
    print()

    # Launch server as detached background process
    if sys.platform == "win32":
        # Windows: CREATE_NEW_PROCESS_GROUP + DETACHED_PROCESS
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        DETACHED_PROCESS = 0x00000008
        flags = CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS

        log_file = SCRIPT_DIR / "server.log"
        with open(log_file, "w") as log:
            proc = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                creationflags=flags,
            )
    else:
        # Unix: use preexec_fn to detach
        log_file = SCRIPT_DIR / "server.log"
        with open(log_file, "w") as log:
            proc = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                preexec_fn=os.setsid,
            )

    # Save PID and active model
    PID_FILE.write_text(str(proc.pid))
    ACTIVE_MODEL_FILE.write_text(model_id)

    print(f"[OK] Server started (PID: {proc.pid})")
    print(f"  Log: {log_file}")

    if wait:
        # Wait for server to become healthy
        print(f"  Waiting for server to load model", end="", flush=True)
        for i in range(120):  # up to 2 minutes
            time.sleep(1)
            print(".", end="", flush=True)

            # Check if process died
            if proc.poll() is not None:
                print(f"\n[ERROR] Server exited with code {proc.returncode}")
                print(f"  Check log: {log_file}")
                PID_FILE.unlink(missing_ok=True)
                ACTIVE_MODEL_FILE.unlink(missing_ok=True)
                sys.exit(1)

            if health_check(port):
                print(f"\n[READY] Server is up! http://127.0.0.1:{port}")
                print(f"  OpenAI-compatible API: http://127.0.0.1:{port}/v1/chat/completions")
                return True

        print(f"\n[WARN] Server didn't respond within 120s. It may still be loading.")
        print(f"  Check: curl http://127.0.0.1:{port}/health")

    return True


def cmd_stop(config, quiet=False):
    """Stop the running server."""
    running, pid = is_server_running()

    if not running:
        if not quiet:
            print("[INFO] No server is currently running.")
        return True

    model_id = get_active_model() or "unknown"
    if not quiet:
        print(f"[STOP] Stopping server (PID: {pid}, model: {model_id})")

    try:
        if sys.platform == "win32":
            # Windows: use taskkill for reliable termination
            subprocess.run(
                ["taskkill", "/F", "/PID", str(pid)],
                capture_output=True,
                timeout=10,
            )
        else:
            os.kill(pid, signal.SIGTERM)
            # Wait for graceful shutdown
            for _ in range(10):
                time.sleep(0.5)
                try:
                    os.kill(pid, 0)
                except OSError:
                    break
            else:
                os.kill(pid, signal.SIGKILL)
    except Exception as e:
        if not quiet:
            print(f"  [WARN] Kill signal sent (process may have already exited): {e}")

    # Cleanup state files
    PID_FILE.unlink(missing_ok=True)
    ACTIVE_MODEL_FILE.unlink(missing_ok=True)

    # Wait for port to free up
    port = config["defaults"].get("port", 8080)
    for _ in range(20):
        if not is_port_in_use(port):
            break
        time.sleep(0.5)

    if not quiet:
        print(f"[OK] Server stopped.")
    return True


def cmd_switch(config, model_id):
    """Stop current model and start a new one."""
    models = config["models"]

    if model_id not in models:
        print(f"[ERROR] Unknown model: '{model_id}'")
        print(f"  Available: {', '.join(models.keys())}")
        sys.exit(1)

    current = get_active_model()
    running, _ = is_server_running()

    if running and current == model_id:
        print(f"[INFO] Model '{model_id}' is already running!")
        return

    target = models[model_id]
    print(f"╔══════════════════════════════════════════════════════╗")
    print(f"║  Switching models                                    ║")
    if current and running:
        print(f"║  FROM: {models.get(current, {}).get('name', current)[:45]:<45} ║")
    print(f"║  TO:   {target['name'][:45]:<45} ║")
    print(f"╚══════════════════════════════════════════════════════╝")
    print()

    if running:
        cmd_stop(config)
        # Brief pause to ensure GPU memory is freed
        print("[INFO] Waiting for GPU memory to free...")
        time.sleep(3)

    cmd_start(config, model_id)


def cmd_status(config):
    """Show current server status."""
    running, pid = is_server_running()
    port = config["defaults"].get("port", 8080)

    print()
    if not running:
        print("  Status: STOPPED")
        print("  No model server is currently running.")
        print()
        return

    model_id = get_active_model() or "unknown"
    model = config["models"].get(model_id, {})
    healthy = health_check(port)

    print(f"  Status:  {'● HEALTHY' if healthy else '⟳ LOADING/UNHEALTHY'}")
    print(f"  PID:     {pid}")
    print(f"  Model:   {model.get('name', model_id)}")
    print(f"  Arch:    {model.get('architecture', 'unknown')}")
    print(f"  Quant:   {model.get('quant', 'unknown')}")
    print(f"  Size:    {format_size(model.get('size_gb', 0))}")
    print(f"  API:     http://127.0.0.1:{port}/v1/chat/completions")
    print(f"  Health:  http://127.0.0.1:{port}/health")
    print()


def cmd_test(config):
    """Quick test of the running server."""
    port = config["defaults"].get("port", 8080)

    running, _ = is_server_running()
    if not running:
        print("[ERROR] No server is running. Start one first.")
        sys.exit(1)

    print("[TEST] Checking server health...")
    if not health_check(port):
        print("  ✗ Server is not healthy / still loading")
        sys.exit(1)
    print("  ✓ Server is healthy")

    print("[TEST] Sending completion request...")
    try:
        url = f"http://127.0.0.1:{port}/v1/chat/completions"
        payload = json.dumps({
            "model": "local",
            "messages": [{"role": "user", "content": "Say hello in exactly 5 words."}],
            "max_tokens": 50,
            "temperature": 0.7,
        }).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            msg = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            print(f"  ✓ Response: {msg.strip()}")
            print(f"  ✓ Tokens - prompt: {usage.get('prompt_tokens', '?')}, "
                  f"completion: {usage.get('completion_tokens', '?')}")
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        sys.exit(1)

    print("[OK] All tests passed!")


def cmd_add(config):
    """Interactive: add a new model to the config."""
    print()
    print("  Add a new model to the manager")
    print("  ─────────────────────────────")
    print()

    model_id = input("  Model ID (short name, e.g. 'llama3-8b'): ").strip()
    if not model_id:
        print("  [Cancelled]")
        return
    if model_id in config["models"]:
        print(f"  [ERROR] Model '{model_id}' already exists.")
        return

    name = input("  Display name: ").strip()
    description = input("  Description: ").strip()
    architecture = input("  Architecture (dense/MoE/etc) [dense]: ").strip() or "dense"
    params = input("  Parameter count (e.g. '32B'): ").strip()
    quant = input("  Quantization (e.g. 'Q5_K_M'): ").strip()
    size_gb = float(input("  File size in GB: ").strip())
    model_path = input("  Full path to .gguf file: ").strip()

    print()
    print("  Server arguments (press Enter to use defaults):")
    ngl = input("    -ngl (GPU layers) [99]: ").strip() or "99"
    ctx = input("    -c (context size) [8192]: ").strip() or "8192"
    slots = input("    -np (parallel slots) [2]: ").strip() or "2"

    config["models"][model_id] = {
        "name": name,
        "description": description,
        "architecture": architecture,
        "params": params,
        "quant": quant,
        "size_gb": size_gb,
        "model_path": model_path,
        "server_args": {
            "-ngl": ngl,
            "-fa": "on",
            "-c": ctx,
            "--cache-type-k": "q8_0",
            "--cache-type-v": "q4_0",
            "-np": slots,
        },
    }

    save_config(config)
    print(f"\n  [OK] Model '{model_id}' added to {CONFIG_FILE}")
    print(f"  Start it with: python llm_manager.py start {model_id}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LLM Model Manager - Spin up/down local LLM models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          python llm_manager.py list                    List all models
          python llm_manager.py start qwen2.5-32b      Start Qwen 2.5 32B
          python llm_manager.py switch qwen3-next-80b   Switch to Qwen3-Next
          python llm_manager.py stop                    Stop running server
          python llm_manager.py status                  Check what's running
          python llm_manager.py test                    Test the running server
          python llm_manager.py add                     Add a new model
        """),
    )
    parser.add_argument(
        "command",
        choices=["list", "start", "stop", "switch", "status", "test", "add"],
        help="Command to execute",
    )
    parser.add_argument(
        "model_id",
        nargs="?",
        help="Model ID (required for start/switch)",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Don't wait for server to become healthy (start/switch)",
    )

    args = parser.parse_args()
    config = load_config()

    if args.command == "list":
        cmd_list(config)
    elif args.command == "start":
        if not args.model_id:
            print("[ERROR] Please specify a model ID.")
            print(f"  Available: {', '.join(config['models'].keys())}")
            sys.exit(1)
        cmd_start(config, args.model_id, wait=not args.no_wait)
    elif args.command == "stop":
        cmd_stop(config)
    elif args.command == "switch":
        if not args.model_id:
            print("[ERROR] Please specify a model ID to switch to.")
            print(f"  Available: {', '.join(config['models'].keys())}")
            sys.exit(1)
        cmd_switch(config, args.model_id)
    elif args.command == "status":
        cmd_status(config)
    elif args.command == "test":
        cmd_test(config)
    elif args.command == "add":
        cmd_add(config)


if __name__ == "__main__":
    main()
