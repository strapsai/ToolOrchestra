# evaluation/tool_handlers.py

import json
from typing import Any, Dict, Optional

import requests


def cosmos_infer_handler(arguments: Dict[str, Any]) -> Dict[str, Any]:
    COSMOS_URL = "http://localhost:9900/api/infer"

    payload: Dict[str, Any] = {
        "prompt": arguments["prompt"],
        "reasoning": bool(arguments.get("reasoning", False)),
        "fps": float(arguments.get("fps", 2.0)),
    }

    videos = arguments.get("videos")
    images = arguments.get("images")
    max_tokens = arguments.get("max_tokens")

    if videos:
        payload["videos"] = videos
    if images:
        payload["images"] = images
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    resp = requests.post(COSMOS_URL, json=payload, timeout=1200)
    resp.raise_for_status()
    return resp.json()


TOOL_REGISTRY = {
    "cosmos_infer": cosmos_infer_handler,
}


def dispatch_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    if tool_name not in TOOL_REGISTRY:
        raise ValueError(f"Unknown tool: {tool_name}")
    return TOOL_REGISTRY[tool_name](arguments)