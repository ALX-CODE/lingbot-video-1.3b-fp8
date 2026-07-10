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
