# evaluation/tool_handlers.py

import json
from typing import Any, Dict, Optional

import requests
from evaluation.medical_triage_utils import (
    TEMPORAL_LABELS,
    build_static_prompt,
    build_temporal_prompt,
    classify_labels,
    get_label_keys,
    parse_vlm_response,
    validate_labels,
)

COSMOS_URL = "http://localhost:9900/api/infer"

def cosmos_infer_handler(arguments: Dict[str, Any]) -> Dict[str, Any]:
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

# New Tool Handler for Static (i.e. Images) Medical Assessment
def medical_static_assessment_handler(arguments: Dict[str, Any]) -> Dict[str, Any]:
    sensor_name = arguments["sensor_name"]
    images = arguments["images"]

    all_label_keys = get_label_keys(sensor_name)
    static_keys, _ = classify_labels(all_label_keys, TEMPORAL_LABELS)

    prompt = build_static_prompt(static_keys)

    payload: Dict[str, Any] = {
        "prompt": prompt,
        "images": images,
        "reasoning": False,
    }

    resp = requests.post(COSMOS_URL, json=payload, timeout=1200)
    resp.raise_for_status()
    vlm_response = resp.json()

    parsed = parse_vlm_response(vlm_response.get("content", ""))
    errors = validate_labels(parsed, static_keys)

    if errors:
        raise ValueError(
            "medical_static_assessment returned invalid labels:\n"
            + "\n".join(errors)
        )

    return {
        "tool": "medical_static_assessment",
        "sensor_name": sensor_name,
        "label_subset": "static",
        "expected_keys": static_keys,
        "labels": parsed,
        "usage": vlm_response.get("usage", {}),
        "duration_s": vlm_response.get("duration_s"),
    }

# New Tool Handler for Temporal (i.e. Videos) Medical Assessment
def medical_temporal_assessment_handler(arguments: Dict[str, Any]) -> Dict[str, Any]:
    sensor_name = arguments["sensor_name"]
    videos = arguments["videos"]
    reasoning = bool(arguments.get("reasoning", False))
    fps = float(arguments.get("fps", 2.0))

    all_label_keys = get_label_keys(sensor_name)
    _, temporal_keys = classify_labels(all_label_keys, TEMPORAL_LABELS)

    prompt = build_temporal_prompt(temporal_keys)

    payload: Dict[str, Any] = {
        "prompt": prompt,
        "videos": videos,
        "reasoning": reasoning,
        "fps": fps,
    }

    resp = requests.post(COSMOS_URL, json=payload, timeout=1200)
    resp.raise_for_status()
    vlm_response = resp.json()

    parsed = parse_vlm_response(vlm_response.get("content", ""))
    errors = validate_labels(parsed, temporal_keys)

    if errors:
        raise ValueError(
            "medical_temporal_assessment returned invalid labels:\n"
            + "\n".join(errors)
        )

    return {
        "tool": "medical_temporal_assessment",
        "sensor_name": sensor_name,
        "label_subset": "temporal",
        "expected_keys": temporal_keys,
        "labels": parsed,
        "usage": vlm_response.get("usage", {}),
        "duration_s": vlm_response.get("duration_s"),
    }


TOOL_REGISTRY = {
    "cosmos_infer": cosmos_infer_handler,
    "medical_static_assessment": medical_static_assessment_handler,
    "medical_temporal_assessment": medical_temporal_assessment_handler,
}


def dispatch_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    if tool_name not in TOOL_REGISTRY:
        available_tools = ", ".join(sorted(TOOL_REGISTRY.keys()))
        raise ValueError(
            f"Unknown tool requested: '{tool_name}'. "
            f"Available tools: [{available_tools}]. "
            f"Arguments: {arguments}"
        )
    return TOOL_REGISTRY[tool_name](arguments)