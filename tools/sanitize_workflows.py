"""Remove local paths, preview history, and private prompt history from workflows."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = "lingbot-video-dense-1.3b"
GENERIC_T2V = (
    "A cinematic shot of a red fox walking through a misty forest at sunrise. "
    "Natural motion, stable anatomy, realistic lighting, and one continuous camera move."
)
GENERIC_TI2V = (
    "Preserve the supplied first frame exactly, then animate the subject with smooth, "
    "physically plausible motion and stable identity in one continuous shot."
)
GENERIC_FLF = (
    "Create one continuous, physically plausible transition from the supplied first frame "
    "to the supplied last frame. Preserve identity, geometry, lighting, and scene continuity."
)
PUBLICATION_DEFAULTS = {
    "width": 640,
    "height": 352,
    "fps": 20.0,
    "duration_seconds": 3.0,
    "num_frames": 61,
    "steps": 28,
    "cfg": 3.0,
    "shift": 3.0,
    "seed": 42,
}
TI2V_INPUT_WARNING = (
    "⚠ INPUT IMAGE REQUIRED BEFORE QUEUEING\n\n"
    "The placeholder `select_input_image_1.png` is intentionally not included. "
    "Choose your first frame in Load Image before pressing Queue, or ComfyUI will report "
    "a missing-input error.\n\n"
)
FLF_INPUT_WARNING = (
    "⚠ TWO INPUT IMAGES REQUIRED BEFORE QUEUEING\n\n"
    "The placeholder files `select_input_image_1.png` and `select_input_image_2.png` are "
    "intentionally not included. Choose both endpoint images before pressing Queue, or "
    "ComfyUI will report a missing-input error.\n\n"
)


def _reorder_lazy_switch_inputs(graph: dict, node: dict) -> None:
    """Match the current enabled/original/processed schema without changing links."""
    inputs = node.get("inputs")
    if not isinstance(inputs, list):
        return
    by_name = {entry.get("name"): (index, entry) for index, entry in enumerate(inputs)}
    expected = ("enabled", "original", "processed")
    if any(name not in by_name for name in expected):
        raise ValueError(f"LingBotLazyImageSwitch {node.get('id')} has an unknown input schema")

    old_to_new = {by_name[name][0]: index for index, name in enumerate(expected)}
    node["inputs"] = [by_name[name][1] for name in expected]
    node_id = node.get("id")
    for link in graph.get("links", []):
        if len(link) >= 5 and link[3] == node_id and link[4] in old_to_new:
            link[4] = old_to_new[link[4]]


def _normalize_publication_defaults(graph: dict) -> None:
    """Reset release workflows to a deterministic, 16 GB-friendly first run."""
    defaults = PUBLICATION_DEFAULTS
    node_types = {node.get("type") for node in graph.get("nodes", [])}
    is_ti2v = "LingBotTI2VSampler" in node_types
    is_flf = "LingBotFLFSampler" in node_types

    for node in graph.get("nodes", []):
        node_type = node.get("type", "")
        widgets = node.get("widgets_values")
        if node_type == "LingBotPromptSettings" and isinstance(widgets, list):
            node["widgets_values"] = [
                defaults["width"],
                defaults["height"],
                defaults["duration_seconds"],
            ]
        elif node_type == "LingBotGenerationSettings" and isinstance(widgets, list):
            node["widgets_values"] = [
                defaults["width"],
                defaults["height"],
                defaults["fps"],
                defaults["duration_seconds"],
                defaults["steps"],
                defaults["cfg"],
                defaults["shift"],
                defaults["seed"],
                "fixed",
            ]
        elif node_type in {"LingBotSampler", "LingBotTI2VSampler", "LingBotFLFSampler"}:
            node["widgets_values"] = [
                defaults["width"],
                defaults["height"],
                defaults["num_frames"],
                defaults["steps"],
                defaults["cfg"],
                defaults["shift"],
                defaults["seed"],
                "fixed",
                "sequential_sage",
            ]
        elif node_type == "LingBotPostProcessSettings" and isinstance(widgets, list):
            node["widgets_values"] = [defaults["fps"], False, 2.0, False]
        elif node_type == "FlashVSRNode" and isinstance(widgets, list):
            node["widgets_values"] = [
                "FlashVSR-v1.1",
                "tiny",
                2,
                True,
                True,
                False,
                defaults["seed"],
                "fixed",
            ]
        elif node_type == "LingBotPromptEncode" and isinstance(widgets, list):
            if len(widgets) > 2:
                widgets[2] = defaults["duration_seconds"]
        elif node_type in {"LingBotTI2VPromptEncode", "LingBotFLFPromptEncode"} and isinstance(
            widgets, list
        ):
            if len(widgets) > 2:
                widgets[2] = defaults["duration_seconds"]
            if len(widgets) > 5:
                widgets[4:6] = [defaults["width"], defaults["height"]]
        elif node_type == "LingBotLazyImageSwitch":
            _reorder_lazy_switch_inputs(graph, node)
            node["widgets_values"] = [False]

    warning = TI2V_INPUT_WARNING if is_ti2v else FLF_INPUT_WARNING if is_flf else None
    if warning:
        notes = [node for node in graph.get("nodes", []) if node.get("type") == "MarkdownNote"]
        quick_start = next(
            (node for node in notes if "QUICK START" in str(node.get("widgets_values", [""])[0]).upper()),
            None,
        )
        if quick_start is None:
            raise ValueError("Image-conditioned publication workflow has no quick-start note")
        body = str(quick_start.get("widgets_values", [""])[0])
        if not body.startswith("⚠"):
            quick_start["widgets_values"] = [warning + body]
        quick_start["title"] = (
            "⚠ Two Input Images Required" if is_flf else "⚠ Input Image Required"
        )
        quick_start["color"] = "#6b2424"
        quick_start["bgcolor"] = "#351414"


def sanitize(path: Path) -> None:
    graph = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(graph, dict) and "nodes" not in graph:
        for node in graph.values():
            if not isinstance(node, dict):
                continue
            node_type = node.get("class_type", "")
            inputs = node.get("inputs", {})
            if node_type == "LingBotModelLoader":
                inputs["model_dir"] = MODEL_DIR
                inputs["transformer_subfolder"] = "transformer_fp8_dense"
                inputs["group_offload"] = False
        rendered = json.dumps(graph, indent=2, ensure_ascii=False) + "\n"
        path.write_text(rendered, encoding="utf-8")
        return

    image_index = 0
    for node in graph.get("nodes", []):
        node.pop("outputs", None)
        node_type = node.get("type", "")
        widgets = node.get("widgets_values")
        if node_type == "LingBotModelLoader" and isinstance(widgets, list):
            widgets[0] = MODEL_DIR
            widgets[1] = "transformer_fp8_dense"
            widgets[2] = False
        elif node_type == "LingBotPromptEncode" and isinstance(widgets, list):
            widgets[0] = GENERIC_T2V
            if len(widgets) > 1:
                widgets[1] = ""
        elif node_type == "LingBotTI2VPromptEncode" and isinstance(widgets, list):
            widgets[0] = GENERIC_TI2V
            if len(widgets) > 1:
                widgets[1] = ""
        elif node_type == "LingBotFLFPromptEncode" and isinstance(widgets, list):
            widgets[0] = GENERIC_FLF
            if len(widgets) > 1:
                widgets[1] = ""
        elif node_type == "LoadImage" and isinstance(widgets, list):
            image_index += 1
            widgets[0] = f"select_input_image_{image_index}.png"
        elif node_type == "VHS_VideoCombine" and isinstance(widgets, dict):
            widgets["filename_prefix"] = "LingBotFP8/video"
            widgets["videopreview"] = {"hidden": False, "paused": False}

    if path.parent.name == "workflows":
        _normalize_publication_defaults(graph)

    rendered = json.dumps(graph, indent=2, ensure_ascii=False) + "\n"
    private_path = re.search(r"[A-Za-z]:\\\\Users\\\\[^\\\\]+", rendered, re.I)
    unix_home = re.search(r'/(?:home|Users)/[^/\"]+', rendered, re.I)
    if private_path or unix_home:
        raise ValueError(f"Private path remained in {path}")
    path.write_text(rendered, encoding="utf-8")


def main() -> None:
    for directory in (ROOT / "workflows", ROOT / "example_workflows"):
        for path in sorted(directory.glob("*.json")):
            sanitize(path)
            print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
