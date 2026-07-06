#!/usr/bin/env bash
# rlvr-tito stack manager: vLLM + TITO proxy + training driver on one GPU.
#
# Encodes the operational lessons from bringing this up on a 24 GB RTX 3090:
#
#   1. `pkill -f vllm` DOES NOT clear the GPU. vLLM spawns EngineCore worker
#      processes, and the proxy holds CUDA context via /dev/nvidiactl (which
#      `lsof /dev/nvidia0` misses). Only `fuser -k /dev/nvidia*` reliably
#      kills everything holding GPU memory.
#   2. Startup order is vLLM -> (fully ready) -> proxy+trainer -> driver.
#      vLLM's init (torch.compile + CUDA-graph profiling) spikes ABOVE its
#      steady-state --gpu-memory-utilization budget; loading the trainer
#      model during that window OOMs one or both processes.
#   3. "Ready" means /v1/models lists the model AND GPU usage exceeds the
#      model's weight size. A stale vLLM from a previous run will happily
#      answer /health while the new one is crashing behind it.
#
# Usage:
#   MODEL=Qwen/Qwen3.5-4B scripts/stack.sh start
#   scripts/stack.sh status
#   scripts/stack.sh stop
#   scripts/stack.sh train --rounds 30 --tasks-per-round 4
#
# Env overrides:
#   MODEL          served + trained model        (default Qwen/Qwen3.5-4B)
#   GPU_UTIL       vLLM --gpu-memory-utilization (default 0.50)
#   GROUP_SIZE     GRPO group size               (default 8)
#   ENFORCE_EAGER  1 = skip torch.compile/CUDA graphs. Slower decode but
#                  removes the init memory spike and ~5 min of startup;
#                  use on cards where vLLM init OOMs (default 1)
#   TOOL_PARSER    vLLM --tool-call-parser (default qwen3_xml — the parser
#                  for Qwen3/3.5 tool-call output; use hermes for Qwen2.5)
#   VLLM_PORT      (default 8000)   PROXY_PORT (default 9000)
#   LOG_DIR        (default /workspace/logs)
#   SECRETS_FILE   env file with ANTHROPIC_API_KEY (default /workspace/secrets.env)
#   REPO_DIR       rlvr-tito checkout (default /workspace/rlvr-tito)
#   TAU2_PATH      tau2-bench checkout (default /workspace/tau2-bench)
#   MIN_MODEL_MB   GPU MiB that proves weights are loaded (default 8000, ~4B bf16)

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
GPU_UTIL="${GPU_UTIL:-0.50}"
GROUP_SIZE="${GROUP_SIZE:-8}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
TOOL_PARSER="${TOOL_PARSER:-qwen3_xml}"
VLLM_PORT="${VLLM_PORT:-8000}"
PROXY_PORT="${PROXY_PORT:-9000}"
LOG_DIR="${LOG_DIR:-/workspace/logs}"
SECRETS_FILE="${SECRETS_FILE:-/workspace/secrets.env}"
REPO_DIR="${REPO_DIR:-/workspace/rlvr-tito}"
TAU2_PATH="${TAU2_PATH:-/workspace/tau2-bench}"
MIN_MODEL_MB="${MIN_MODEL_MB:-8000}"

mkdir -p "$LOG_DIR"

gpu_used_mb() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1
}

vllm_serving_model() {
    curl -sf "http://localhost:${VLLM_PORT}/v1/models" 2>/dev/null \
        | python3 -c "import sys,json; print(any(m['id']=='$MODEL' for m in json.load(sys.stdin).get('data',[])))" 2>/dev/null \
        | grep -q True
}

cmd_stop() {
    echo "Killing every process holding an nvidia device..."
    fuser -k /dev/nvidia* 2>/dev/null || true
    pkill -9 -f 'vllm serve' 2>/dev/null || true
    pkill -9 -f 'uvicorn rlvr_tito' 2>/dev/null || true
    pkill -9 -f 'tau2_retail_train' 2>/dev/null || true
    sleep 5
    local used
    used=$(gpu_used_mb)
    if [ "$used" -gt 500 ]; then
        echo "WARNING: GPU still shows ${used} MiB after kill. Offenders:"
        nvidia-smi --query-compute-apps=pid,used_memory --format=csv
        return 1
    fi
    echo "GPU clear (${used} MiB used)."
}

cmd_start() {
    # Refuse to start on a dirty GPU — a half-dead previous stack answering
    # health checks is how we lost an hour to phantom-readiness.
    local used
    used=$(gpu_used_mb)
    if [ "$used" -gt 500 ]; then
        echo "GPU not clean (${used} MiB in use). Run 'stack.sh stop' first."
        exit 1
    fi

    local eager_flag=""
    [ "$ENFORCE_EAGER" = "1" ] && eager_flag="--enforce-eager"

    echo "[1/3] Starting vLLM ($MODEL, gpu-util $GPU_UTIL ${eager_flag:+, eager})..."
    # setsid + </dev/null: fully detach from the calling session, or an ssh
    # invocation of this script never returns (backgrounded children keep the
    # session's stdin open and the remote shell hangs after the script ends).
    # --enable-auto-tool-choice + parser: tau2 agents send tool_choice="auto";
    # vLLM 400s every request without these. qwen3_xml parses Qwen3/3.5
    # tool-call output (hermes is for Qwen2.5-era JSON-in-<tool_call> format).
    VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 setsid nohup vllm serve "$MODEL" \
        --enable-lora --max-lora-rank 32 --port "$VLLM_PORT" \
        --gpu-memory-utilization "$GPU_UTIL" $eager_flag \
        --enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER" \
        > "$LOG_DIR/vllm.log" 2>&1 < /dev/null &
    local vllm_pid=$!

    # Ready = model listed in /v1/models AND weights visibly on the GPU.
    # /health alone is not sufficient: it can be answered by a stale server,
    # and it goes live before KV-cache allocation finishes.
    echo "     Waiting for vLLM (up to 15 min)..."
    local ok=0
    for i in $(seq 1 90); do
        if ! kill -0 "$vllm_pid" 2>/dev/null; then
            echo "vLLM process died during init. Last log lines:"
            tail -30 "$LOG_DIR/vllm.log"
            exit 1
        fi
        if vllm_serving_model && [ "$(gpu_used_mb)" -gt "$MIN_MODEL_MB" ]; then
            ok=1; break
        fi
        sleep 10
    done
    if [ "$ok" != "1" ]; then
        echo "vLLM never became ready. Last log lines:"
        tail -30 "$LOG_DIR/vllm.log"
        exit 1
    fi
    echo "     vLLM ready ($(gpu_used_mb) MiB on GPU)."

    echo "[2/3] Starting proxy + trainer..."
    (cd "$REPO_DIR" && \
        TRAIN=1 VLLM_URL="http://localhost:${VLLM_PORT}" \
        MODEL="$MODEL" GROUP_SIZE="$GROUP_SIZE" \
        METRICS_PATH="$LOG_DIR/trainer_metrics.jsonl" \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        setsid nohup uvicorn rlvr_tito.proxy:app --host 0.0.0.0 --port "$PROXY_PORT" \
        > "$LOG_DIR/proxy.log" 2>&1 < /dev/null &)

    echo "     Waiting for proxy..."
    for i in $(seq 1 30); do
        if curl -sf "http://localhost:${PROXY_PORT}/stats" > /dev/null 2>&1; then
            break
        fi
        sleep 5
    done
    curl -sf "http://localhost:${PROXY_PORT}/stats" > /dev/null \
        || { echo "proxy never came up:"; tail -30 "$LOG_DIR/proxy.log"; exit 1; }

    echo "[3/3] Stack up. GPU: $(gpu_used_mb) MiB used."
    echo "     (Trainer model loads lazily inside the proxy after its own"
    echo "      vLLM-readiness gate; watch $LOG_DIR/proxy.log and GPU usage.)"
    echo "Start training with: scripts/stack.sh train --rounds 30"
}

cmd_train() {
    [ -f "$SECRETS_FILE" ] || { echo "missing $SECRETS_FILE (needs ANTHROPIC_API_KEY)"; exit 1; }
    curl -sf "http://localhost:${PROXY_PORT}/stats" > /dev/null \
        || { echo "proxy not up — run 'stack.sh start' first"; exit 1; }
    # set -a exports everything the file defines; plain `source` would set
    # shell-local vars that never reach the python child process.
    set -a; # shellcheck disable=SC1090
    source "$SECRETS_FILE"; set +a
    echo "Starting training driver (log: $LOG_DIR/train.log)..."
    TAU2_PATH="$TAU2_PATH" PROXY_URL="http://localhost:${PROXY_PORT}" MODEL="$MODEL" \
        setsid nohup python "$REPO_DIR/examples/tau2_retail_train.py" \
        --metrics-path "$LOG_DIR/rollout_metrics.jsonl" "$@" \
        > "$LOG_DIR/train.log" 2>&1 < /dev/null &
    echo "Driver PID $!. Follow with: tail -f $LOG_DIR/train.log"
}

cmd_status() {
    echo "── GPU ──"
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader || true
    echo "── vLLM (:$VLLM_PORT) ──"
    if vllm_serving_model; then echo "serving $MODEL"; else echo "NOT serving $MODEL"; fi
    echo "── proxy (:$PROXY_PORT) ──"
    curl -sf "http://localhost:${PROXY_PORT}/stats" 2>/dev/null || echo "unreachable"
    echo
    echo "── recent trainer metrics ──"
    tail -3 "$LOG_DIR/trainer_metrics.jsonl" 2>/dev/null || echo "(none yet)"
    echo "── recent training log ──"
    tail -5 "$LOG_DIR/train.log" 2>/dev/null || echo "(none yet)"
}

case "${1:-}" in
    start)  cmd_start ;;
    stop)   cmd_stop ;;
    status) cmd_status ;;
    train)  shift; cmd_train "$@" ;;
    *) echo "usage: stack.sh {start|stop|status|train [driver args]}"; exit 1 ;;
esac
