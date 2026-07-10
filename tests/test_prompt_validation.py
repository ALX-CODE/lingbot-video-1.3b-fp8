from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "lingbot_prompt_rewriter_test", ROOT / "lingbot_video" / "prompt_rewriter.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _caption() -> dict:
    return {
        "comprehensive_description": {
            "scene_content_description": "A subject walks through a stable scene.",
            "camera_movement_description": "The camera tracks backward smoothly.",
        },
        "prominent_elements": [
            {
                "name": "subject",
                "description": "A consistently depicted walking subject.",
                "actions": [
                    {"timestamp": "[0.0s - 1.5s]", "action": "takes one balanced step"},
                    {"timestamp": "[1.5s - 3.5s]", "action": "continues walking"},
                ],
                "location": "foreground",
                "relative_size": "large",
                "shape_and_color": "human silhouette",
                "texture": "natural",
                "appearance_details": "stable identity",
                "relationship": "tracked by the camera",
                "orientation": "forward",
                "pose": "walking",
                "expression": "focused",
                "clothing": "coat",
                "gender": "adult",
                "skin_tone_and_texture": "natural",
                "number_of_objects": "1",
            }
        ],
        "camera_info": {
            "color": "Mixed",
            "frame_size": "Medium",
            "shot_type_angle": "Eye level",
            "lens_size": "Normal",
            "composition": "Centered",
            "lighting": "Natural",
            "lighting_type": "Mixed light",
        },
        "world_knowledge": ["Walking motion obeys gravity and stable ground contact."],
    }


class PromptValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_module()

    def test_decimal_duration_is_preserved(self):
        text, errors = self.module.normalize_generated_prompt({"caption": _caption()}, 3.5)
        self.assertEqual(errors, [])
        self.assertEqual(json.loads(text)["duration"], 3.5)

    def test_reversed_and_overlapping_intervals_are_rejected(self):
        caption = _caption()
        caption["prominent_elements"][0]["actions"] = [
            {"timestamp": "[2.0s - 1.0s]", "action": "moves"},
            {"timestamp": "[0.5s - 2.5s]", "action": "moves again"},
        ]
        errors = self.module.validate_caption(caption, 3.5)
        self.assertTrue(any("start must not exceed end" in error for error in errors))
        self.assertTrue(any("overlaps the previous action" in error for error in errors))

    def test_world_knowledge_and_camera_movement_require_content(self):
        caption = _caption()
        caption["world_knowledge"] = [""]
        caption["comprehensive_description"]["camera_movement_description"] = ""
        errors = self.module.validate_caption(caption, 3.5)
        self.assertTrue(any("world_knowledge entries" in error for error in errors))
        self.assertTrue(any("camera_movement_description" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
