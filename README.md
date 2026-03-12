# Quick Start: ToolOrchestra + Nemotron + Cosmos


## 1. Make sure Cosmos is already running

Before anything else, confirm the Cosmos container is already up and healthy.

## 2. Launch the Nemotron container

If the Nemotron container is not already running, start it with:
```
docker run --runtime nvidia --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env "HF_TOKEN=$HF_TOKEN" \
  -p 8001:8000 \
  --ipc=host \
  vllm/vllm-openai:latest \
  nvidia/Nemotron-Orchestrator-8B \
  --gpu-memory-utilization 0.20 \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```
Check that it is healthy with:
```
curl http://localhost:8001/v1/models
```

### 3. Activate the working environment

The working environment already exists on Shreyans’ Spark.
```
conda activate toolorchestra-local
export REPO_PATH="$PWD"
export HF_HOME="$HOME/.cache/huggingface"
```
If you ever need to recreate it from scratch:
```
conda create -n toolorchestra-local python=3.10 -y
conda activate toolorchestra-local
pip install -U pip setuptools wheel
# Lighter dependencies only
pip install requests openai pyyaml tqdm
pip install transformers anthropic

export REPO_PATH="$PWD"
export HF_HOME="$HOME/.cache/huggingface"
```

### 4. Example run: 2-round tool-use case

This example uses the local orchestrator runner and tool registry:
```
python evaluation/eval_orchestrator_local.py \
  --task "Follow this protocol exactly: (1) First tool call: ask only for a provisional triage category from the overall scene. Do not ask about bleeding, posture, or movement. (2) Second tool call: ask only whether there is visible severe bleeding, abnormal posture, or lack of movement, and whether those findings change the provisional triage category. You must make these as two distinct tool calls. Only after the second tool call may you produce the final answer." \
  --video "/home/darpa_triage_vlm/vlm_workspace/data/share/datasets/year3_labeling/RGB_Snippets/snippets_proj225664_task245750644_ann85399073_20260201T204055Z/casualty_16/casualty_16_f1-1589/casualty_16_f1-1589_fps50_bbox_overlaid.mp4" \
  --reasoning \
  --fps 2.0 \
  --max-rounds 5 \
  --verbose
```

Files used
- evaluation/eval_orchestrator_local.py — local orchestration runner
- evaluation/tools.json — exposed tool definitions
- evaluation/tool_handlers.py — Python implementations for tool execution