# LLM Model Manager

> Spin up, switch, and manage multiple local LLM models on a single GPU with one command.

Built for **llama.cpp** servers on **NVIDIA RTX PRO 4500 Blackwell (32GB VRAM)** — but works on any GPU + llama.cpp setup.

## Why?

When you have multiple GGUF models locally (different sizes, quants, architectures), you need to:
1. Stop the current server
2. Wait for GPU memory to free
3. Rebuild the command with the right flags
4. Start the new server
5. Wait for it to load

**This tool does all of that in one command.**

## Quick Start

```bash
# List available models
python llm_manager.py list

# Start a model
python llm_manager.py start qwen2.5-32b

# Switch to another model (auto-stops the current one)
python llm_manager.py switch qwen3-next-80b

# Check status
python llm_manager.py status

# Stop the server
python llm_manager.py stop
```

Or use the batch wrapper:
```cmd
llm list
llm start qwen2.5-32b
llm switch qwen3-next-80b
llm stop
```

## Commands

| Command | Description |
|---------|-------------|
| `list` | Show all configured models with status |
| `start <model-id>` | Start a model server (fails if one is already running) |
| `stop` | Stop the currently running server |
| `switch <model-id>` | Stop current → free GPU → start new model |
| `status` | Show running model, PID, health, API endpoint |
| `test` | Quick health check + send a test completion |
| `add` | Interactive wizard to add a new model |

## Configuration

Models are defined in `models.json`:

```json
{
  "defaults": {
    "llama_cpp_path": "D:\\LLM\\llama-cpp",
    "host": "0.0.0.0",
    "port": 8080,
    "env": {
      "BLACKWELL_NATIVE_FP4": "1"
    }
  },
  "models": {
    "qwen2.5-32b": {
      "name": "Qwen 2.5 32B Instruct (Q5_K_M)",
      "description": "Dense 32B model, excellent for coding",
      "architecture": "dense",
      "params": "32B",
      "quant": "Q5_K_M",
      "size_gb": 21.66,
      "model_path": "C:\\llm\\models\\qwen2.5-32b-instruct-q5_k_m-00001-of-00006.gguf",
      "server_args": {
        "-ngl": "99",
        "-fa": "on",
        "-c": "16384",
        "--cache-type-k": "q8_0",
        "--cache-type-v": "q4_0",
        "-np": "2"
      }
    }
  }
}
```

### Adding a New Model

**Option 1: Interactive wizard**
```bash
python llm_manager.py add
```

**Option 2: Edit `models.json` directly**

Add a new entry under `"models"`:
```json
"my-model": {
  "name": "My Model Display Name",
  "description": "Short description",
  "architecture": "dense",
  "params": "7B",
  "quant": "Q4_K_M",
  "size_gb": 4.5,
  "model_path": "D:\\LLM\\models\\my-model.gguf",
  "server_args": {
    "-ngl": "99",
    "-fa": "on",
    "-c": "8192",
    "--cache-type-k": "q8_0",
    "--cache-type-v": "q4_0",
    "-np": "2"
  }
}
```

## Preconfigured Models

| Model ID | Model | Type | Quant | Size | VRAM Fit |
|----------|-------|------|-------|------|----------|
| `qwen2.5-32b` | Qwen 2.5 32B Instruct | Dense 32B | Q5_K_M | 21.7 GB | ✅ 32GB |
| `qwen3-next-80b` | Qwen3-Next-80B-A3B | MoE 80B (3B active) | UD-Q2_K_XL | 30.1 GB | ✅ 32GB |

## Architecture

```
llm-manager/
├── llm_manager.py     # Main Python script (zero dependencies)
├── llm.bat            # Windows batch wrapper
├── models.json        # Model configuration
├── .llm-server.pid    # Auto-created: tracks running server PID
├── .llm-active-model  # Auto-created: tracks which model is active
└── server.log         # Auto-created: server stdout/stderr
```

### How It Works

1. **`start`**: Launches `llama-server` as a detached background process, saves PID, waits for health check
2. **`stop`**: Reads PID file, sends kill signal, waits for port to free, cleans up state
3. **`switch`**: Runs stop → 3s GPU cooldown → start (ensures VRAM is fully released)
4. **`status`**: Checks PID liveness + HTTP health endpoint

### OpenAI-Compatible API

Once a model is running, use the standard OpenAI API format:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

Works with any OpenAI-compatible client library:
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

## Hardware Tested

- **GPU**: NVIDIA RTX PRO 4500 Blackwell (32GB VRAM)
- **RAM**: 62GB DDR
- **CPU**: 16 threads, Zen4 (AVX512)
- **OS**: Windows 11
- **llama.cpp**: b7966 (CUDA 13.1)

## Requirements

- Python 3.10+ (no pip dependencies needed)
- [llama.cpp](https://github.com/ggml-org/llama.cpp/releases) with CUDA support
- GGUF model files

## License

MIT
