# Added and adapted by ALX-CODE in 2026 for plaintext-to-LingBot JSON integration.
# It interoperates with Robbyant/LingBot-Video prompt structures at baseline
# commit a2bb04b78edd848500dc27a26e035a95442ae186. See ../NOTICE.md.
from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any


CAMERA_FIELDS = (
    "color",
    "frame_size",
    "shot_type_angle",
    "lens_size",
    "composition",
    "lighting",
    "lighting_type",
)

ELEMENT_FIELDS = (
    "name",
    "description",
    "actions",
    "location",
    "relative_size",
    "shape_and_color",
    "texture",
    "appearance_details",
    "relationship",
    "orientation",
    "pose",
    "expression",
    "clothing",
    "gender",
    "skin_tone_and_texture",
    "number_of_objects",
)


def build_expand_prompt(plain_prompt: str, duration_seconds: float) -> str:
    duration = float(duration_seconds)
    return f"""You are the first stage of the LingBot-Video prompt rewriter.

Turn the USER PROMPT into one concise, natural English video caption of 400-900
characters for a {duration:g}-second clip. Output only one prose paragraph.

Requirements:
- Preserve every explicit subject, count, identity, color, position, action,
  direction, camera instruction, visual style, and negative constraint.
- Lead with the scene and main action. Add plausible visual detail only when it
  cannot contradict the user.
- Give every real action or camera change an explicit time range in seconds.
  Cover the chronology from 0.0s through at most {duration:g}s without overlap,
  reversal, teleportation, or impossible motion.
- Keep subject identities, left/right directions, clothing, and object state
  consistent through the clip.
- Describe camera framing, angle, movement/stability, lighting, and atmosphere.
- Ignore audio, dialogue delivery, music, and sound effects; this is visual-only.
- Do not use headings, bullets, alternatives, explanations, or JSON.

USER PROMPT:
{plain_prompt.strip()}
"""


def build_json_prompt(detailed_caption: str, duration_seconds: float) -> str:
    duration = float(duration_seconds)
    schema = {
        "comprehensive_description": {
            "scene_content_description": "detailed visual scene; no camera movement",
            "camera_movement_description": "camera behavior, angle, framing and stability",
        },
        "prominent_elements": [
            {
                "name": "short element label",
                "description": "specific visual description",
                "actions": [
                    {
                        "timestamp": f"[0.0s - {duration:g}s]",
                        "action": "observable action with speed/direction",
                    }
                ],
                "location": "viewer-relative frame position",
                "relative_size": "small|medium|large|dominant",
                "shape_and_color": "shape and dominant colors",
                "texture": "visible surface texture",
                "appearance_details": "patterns, markings and exact visible text",
                "relationship": "spatial relationship to other elements",
                "orientation": "viewer-relative orientation",
                "pose": "human pose or empty string",
                "expression": "human expression or empty string",
                "clothing": "human clothing or empty string",
                "gender": "apparent human gender or empty string",
                "skin_tone_and_texture": "human skin appearance or empty string",
                "number_of_objects": "exact count or several|many|numerous",
            }
        ],
        "camera_info": {
            "color": "Warm|Cool|Mixed|Saturated|Desaturated|Black and White",
            "frame_size": "Extreme Wide|Wide|Medium Wide|Medium|Medium Close Up|Close Up|Extreme Close Up",
            "shot_type_angle": "High angle|Low angle|Dutch angle|Overhead|Aerial|Eye level",
            "lens_size": "Ultra Wide / Fisheye|Wide|Medium|Long Lens|Telephoto",
            "composition": "Center|Balanced|Symmetrical|Left heavy|Right heavy|Short side",
            "lighting": "Hard light|Soft light|High contrast|Low contrast|Side light|Top light|Underlight|Backlight|Edge light|Silhouette",
            "lighting_type": "Daylight|Sunny|Overcast|Moonlight|Artificial light|Practical light|Tungsten|Fluorescent|Firelight|Mixed light",
        },
        "world_knowledge": [],
    }
    return f"""You are the second stage of the LingBot-Video prompt rewriter.

Convert the DETAILED CAPTION into exactly one valid JSON object matching this
schema. Output minified raw JSON on one line with no insignificant whitespace:
no Markdown fence, preamble, analysis, or comments.

SCHEMA EXAMPLE:
{json.dumps(schema, ensure_ascii=False, separators=(",", ":"))}

Hard rules:
- Preserve every subject, count, identity, action, order, direction, timestamp,
  spatial relation, color, camera instruction, and negative constraint.
- Every prominent subject/object gets one prominent_elements entry. Every stated
  action gets its own action entry; never summarize a sequence.
- All timestamps must be within 0.0s and {duration:g}s and maintain continuity.
- Fill every listed field. Use an empty string for non-applicable human fields,
  never "unknown" or "N/A". Keep world_knowledge as a JSON array.
- Directions and locations are from the viewer's perspective. Camera metadata
  must be internally consistent with the prose and scene scale.
- Describe only visible information. Ignore all audio instructions.

DETAILED CAPTION:
{detailed_caption.strip()}
"""


def build_repair_prompt(
    raw_output: str,
    validation_errors: list[str],
    detailed_caption: str,
    duration_seconds: float,
) -> str:
    errors = "\n".join(f"- {item}" for item in validation_errors)
    return f"""Repair the candidate LingBot-Video JSON below. Output one complete,
valid raw JSON object only. Preserve the detailed caption exactly, fill every
required schema field, and keep every timestamp within 0.0s-{duration_seconds:g}s.

VALIDATION ERRORS:
{errors}

DETAILED CAPTION:
{detailed_caption.strip()}

CANDIDATE OUTPUT:
{raw_output.strip()}
"""


def extract_json_object(raw_output: str) -> dict[str, Any]:
    text = (raw_output or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("Qwen did not return a valid JSON object")


def normalize_generated_prompt(
    payload: dict[str, Any],
    duration_seconds: float,
) -> tuple[str, list[str]]:
    duration = float(duration_seconds)
    # Qwen occasionally preserves the meaning of this free-form field while
    # choosing a scalar or object instead of the schema's array container.
    # Coerce only the container shape here; semantic/schema errors elsewhere
    # remain strict and still use the normal repair path.
    normalized_payload = deepcopy(payload)
    caption = normalized_payload.get("caption", normalized_payload)
    if isinstance(caption, dict):
        world_knowledge = caption.get("world_knowledge")
        items = world_knowledge if isinstance(world_knowledge, list) else [world_knowledge]
        normalized_world_knowledge: list[str] = []
        for item in items:
            if item is None:
                continue
            if isinstance(item, dict):
                for key, value in item.items():
                    key_text = str(key).strip()
                    if isinstance(value, str):
                        value_text = value.strip()
                    else:
                        value_text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                    text = f"{key_text}: {value_text}" if key_text and value_text else key_text or value_text
                    if text:
                        normalized_world_knowledge.append(text)
                continue
            if isinstance(item, str):
                text = item.strip()
            else:
                text = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
            if text:
                normalized_world_knowledge.append(text)
        caption["world_knowledge"] = normalized_world_knowledge
    errors = validate_caption(caption, duration)
    if errors:
        return "", errors
    normalized = {
        "caption": caption,
        "duration": int(duration) if duration.is_integer() else duration,
    }
    return json.dumps(normalized, ensure_ascii=False, indent=2), []


def repair_missing_element_actions(
    payload: dict[str, Any],
    duration_seconds: float,
    detailed_caption: str = "",
) -> tuple[dict[str, Any], int]:
    """Fill only missing/empty action arrays; leave every other validation error strict."""
    repaired = deepcopy(payload)
    caption = repaired.get("caption", repaired)
    if not isinstance(caption, dict):
        return repaired, 0
    elements = caption.get("prominent_elements")
    if not isinstance(elements, list):
        return repaired, 0

    comprehensive = caption.get("comprehensive_description")
    scene = ""
    if isinstance(comprehensive, dict):
        value = comprehensive.get("scene_content_description")
        if isinstance(value, str):
            scene = value.strip()
    count = 0
    for element in elements:
        if not isinstance(element, dict):
            continue
        actions = element.get("actions")
        if isinstance(actions, list) and actions:
            continue
        name = element.get("name")
        name = name.strip() if isinstance(name, str) and name.strip() else "The visible element"
        description = element.get("description")
        description = description.strip() if isinstance(description, str) else ""
        basis = description or scene or detailed_caption.strip()
        if basis:
            action_text = (
                f"{name} remains visually coherent with its description ({basis}) while following "
                "only the physically plausible motion implied for this element"
            )
        else:
            action_text = (
                f"{name} remains visually consistent and follows only the physically plausible "
                "motion implied for this element"
            )
        element["actions"] = [
            {
                "timestamp": f"[0.0s - {float(duration_seconds):g}s]",
                "action": action_text,
            }
        ]
        count += 1
    return repaired, count


def validate_caption(caption: Any, duration_seconds: float) -> list[str]:
    errors: list[str] = []
    if not isinstance(caption, dict):
        return ["caption must be a JSON object"]

    comprehensive = caption.get("comprehensive_description")
    if not isinstance(comprehensive, dict):
        errors.append("comprehensive_description must be an object")
    else:
        scene = comprehensive.get("scene_content_description")
        camera = comprehensive.get("camera_movement_description")
        if not isinstance(scene, str) or not scene.strip():
            errors.append("scene_content_description must be a non-empty string")
        if not isinstance(camera, str) or not camera.strip():
            errors.append("camera_movement_description must be a non-empty string")

    camera_info = caption.get("camera_info")
    if not isinstance(camera_info, dict):
        errors.append("camera_info must be an object")
    else:
        for field in CAMERA_FIELDS:
            if not isinstance(camera_info.get(field), str) or not camera_info[field].strip():
                errors.append(f"camera_info.{field} must be a non-empty string")

    world_knowledge = caption.get("world_knowledge")
    if not isinstance(world_knowledge, list):
        errors.append("world_knowledge must be an array")
    elif any(not isinstance(item, str) or not item.strip() for item in world_knowledge):
        errors.append("world_knowledge entries must be non-empty strings")

    elements = caption.get("prominent_elements")
    if not isinstance(elements, list) or not elements:
        errors.append("prominent_elements must be a non-empty array")
        return errors

    duration = float(duration_seconds)
    for index, element in enumerate(elements):
        prefix = f"prominent_elements[{index}]"
        if not isinstance(element, dict):
            errors.append(f"{prefix} must be an object")
            continue
        for field in ELEMENT_FIELDS:
            if field not in element:
                errors.append(f"{prefix}.{field} is required")
        for field in ("name", "description"):
            if not isinstance(element.get(field), str) or not element[field].strip():
                errors.append(f"{prefix}.{field} must be a non-empty string")
        actions = element.get("actions")
        if not isinstance(actions, list) or not actions:
            errors.append(f"{prefix}.actions must be a non-empty array")
            continue
        previous_end: float | None = None
        for action_index, action in enumerate(actions):
            action_prefix = f"{prefix}.actions[{action_index}]"
            if not isinstance(action, dict):
                errors.append(f"{action_prefix} must be an object")
                continue
            timestamp = action.get("timestamp")
            if not isinstance(timestamp, str):
                errors.append(f"{action_prefix}.timestamp must be a string")
            else:
                match = re.fullmatch(
                    r"\[\s*(\d+(?:\.\d+)?)\s*s\s*-\s*(\d+(?:\.\d+)?)\s*s\s*\]",
                    timestamp,
                )
                if match is None:
                    errors.append(
                        f"{action_prefix}.timestamp must use '[start_s - end_s]' interval syntax"
                    )
                else:
                    start, end = (float(value) for value in match.groups())
                    if start > end:
                        errors.append(f"{action_prefix}.timestamp start must not exceed end")
                    if start < 0.0 or end > duration + 1e-6:
                        errors.append(f"{action_prefix}.timestamp exceeds the {duration:g}s duration")
                    if previous_end is not None and start < previous_end - 1e-6:
                        errors.append(f"{action_prefix}.timestamp overlaps the previous action")
                    previous_end = max(previous_end or 0.0, end)
            action_text = action.get("action")
            if not isinstance(action_text, str) or not action_text.strip():
                errors.append(f"{action_prefix}.action must be a non-empty string")
    return errors
