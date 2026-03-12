# evaluation/eval_cosmos_local.py

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import requests
from pathlib import Path

REPO_PATH = os.getenv("REPO_PATH")
repo_root = Path(REPO_PATH) if REPO_PATH else Path(__file__).resolve().parents[1]

if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

from LLM_CALL import get_llm_response
from evaluation.tool_handlers import dispatch_tool


NEMOTRON_MODEL = "nvidia/Nemotron-Orchestrator-8B"
COSMOS_URL = "http://localhost:9900/api/infer"

## Example call:
# python evaluation/eval_cosmos_local.py   --task "Assess the injuries visible on this casualty. What is the appropriate triage category (Immediate, Delayed, Minor, Expectant)?"   --video "/home/darpa_triage_vlm/vlm_workspace/data/share/datasets/year3_labeling/RGB_Snippets/snippets_proj225664_task245750644_ann85399073_20260201T204055Z/casualty_16/casualty_16_f1-1589/casualty_16_f1-1589_fps50_bbox_overlaid.mp4"   --cosmos-reasoning   --cosmos-fps 2.0   --verbose


# ToolOrchestra-style local model config for your already-running Nemotron vLLM server.
LOCAL_MODEL_CONFIG = [
    {
        "ip_addr": "localhost",
        "port": 8001,
    }
]

# Loading in the available tools from the tools.json file (same directorty as this file)
TOOLS_PATH = Path(__file__).resolve().parent / "tools.json"

with open(TOOLS_PATH, "r") as f:
    TOOLS = json.load(f)


SYSTEM_PROMPT = (
    "You are an orchestration model with access to one tool, cosmos_infer. "
    "Use cosmos_infer only when the user's request requires visual analysis of the provided "
    "image or video content. If the task can be answered from text or general knowledge alone, "
    "answer directly without calling the tool. "
    "When tool results are returned, use them to produce the final user-facing answer."
)


def response_to_message_text(resp: Any) -> str:
    """
    Convert ToolOrchestra/OpenAI raw response into assistant text if present.
    """
    try:
        return resp.choices[0].message.content or ""
    except Exception:
        return ""


def response_to_tool_calls(resp: Any) -> List[Dict[str, Any]]:
    """
    Extract tool_calls from OpenAI-style response object if present.
    """
    try:
        tool_calls = resp.choices[0].message.tool_calls
        return tool_calls or []
    except Exception:
        return []


def tool_call_to_dict(tc: Any) -> Dict[str, Any]:
    """
    Normalize OpenAI SDK tool call objects into a plain dict.
    """
    if isinstance(tc, dict):
        return tc

    function_obj = getattr(tc, "function", None)
    return {
        "id": getattr(tc, "id", None),
        "type": getattr(tc, "type", "function"),
        "function": {
            "name": getattr(function_obj, "name", None),
            "arguments": getattr(function_obj, "arguments", "{}"),
        },
    }


def build_user_message(task: str, videos: List[str], images: List[str]) -> str:
    parts = [f"User task: {task}", ""]
    if videos:
        parts.append("Available video paths:")
        parts.extend([f"- {v}" for v in videos])
        parts.append("")
    if images:
        parts.append("Available image paths:")
        parts.extend([f"- {i}" for i in images])
        parts.append("")
    if not videos and not images:
        parts.append("No media paths were provided.")
        parts.append("")
    parts.append(
        "Use the provided paths exactly as given if you call cosmos_infer."
    )
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--video", action="append", default=[])
    parser.add_argument("--image", action="append", default=[])
    parser.add_argument("--cosmos-reasoning", action="store_true")
    parser.add_argument("--cosmos-fps", type=float, default=2.0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_message(
                task=args.task,
                videos=args.video,
                images=args.image,
            ),
        },
    ]

    print("\n=== STEP 1: Ask Nemotron via ToolOrchestra LLM_CALL ===\n")
    first_response = get_llm_response(
        model=NEMOTRON_MODEL,
        messages=messages,
        return_raw_response=True,
        tools=TOOLS,
        model_type="vllm",
        max_length=1024,
        temperature=0.0,
        model_config=LOCAL_MODEL_CONFIG,
    )

    if args.verbose:
        print("Raw first response object:")
        print(first_response)
        print()

    first_text = response_to_message_text(first_response)
    first_tool_calls = response_to_tool_calls(first_response)

    if first_tool_calls:
        tc = tool_call_to_dict(first_tool_calls[0])
        fn = tc["function"]["name"]
        fn_args = json.loads(tc["function"]["arguments"])

        print("=== STEP 2: Nemotron requested tool ===\n")
        print(json.dumps(tc, indent=2))
        print()

        tool_arguments = {
            "prompt": fn_args.get("prompt", args.task),
            "videos": fn_args.get("videos", args.video),
            "images": fn_args.get("images", args.image),
            "reasoning": bool(fn_args.get("reasoning", args.cosmos_reasoning)),
            "fps": float(fn_args.get("fps", args.cosmos_fps)),
            "max_tokens": fn_args.get("max_tokens"),
        }

        tool_result = dispatch_tool(fn, tool_arguments)

        if args.verbose:
            print("Raw tool result:")
            print(json.dumps(tool_result, indent=2))
            print()

        messages.append(
            {
                "role": "assistant",
                "content": first_text,
                "tool_calls": [tc],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": json.dumps(tool_result),
            }
        )

        print("=== STEP 3: Ask Nemotron for final answer ===\n")
        second_response = get_llm_response(
            model=NEMOTRON_MODEL,
            messages=messages,
            return_raw_response=True,
            tools=TOOLS,
            model_type="vllm",
            max_length=512,
            temperature=0.0,
            model_config=LOCAL_MODEL_CONFIG,
        )

        if args.verbose:
            print("Raw second response object:")
            print(second_response)
            print()

        final_text = response_to_message_text(second_response)
        print("=== FINAL ANSWER ===\n")
        print(final_text.strip())
        return

    print("=== FINAL ANSWER (direct, no tool call) ===\n")
    print(first_text.strip())


if __name__ == "__main__":
    main()