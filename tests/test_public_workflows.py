from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / "workflows"
PUBLIC_WORKFLOWS = (
    "lingbot_fp8_t2v.json",
    "lingbot_fp8_ti2v.json",
    "lingbot_fp8_flf_experimental.json",
)


def load_workflow(name: str) -> dict:
    return json.loads((WORKFLOWS / name).read_text(encoding="utf-8"))


def nodes_by_type(graph: dict, node_type: str) -> list[dict]:
    return [node for node in graph["nodes"] if node.get("type") == node_type]


class PublicWorkflowTests(unittest.TestCase):
    def test_all_published_workflows_parse_and_use_safe_defaults(self) -> None:
        for name in PUBLIC_WORKFLOWS:
            with self.subTest(workflow=name):
                graph = load_workflow(name)
                self.assertIsInstance(graph["nodes"], list)
                settings = nodes_by_type(graph, "LingBotGenerationSettings")
                self.assertEqual(len(settings), 1)
                self.assertEqual(
                    settings[0]["widgets_values"],
                    [640, 352, 20.0, 3.0, 28, 3.0, 3.0, 42, "fixed"],
                )
                prompt_settings = nodes_by_type(graph, "LingBotPromptSettings")
                if prompt_settings:
                    self.assertEqual(prompt_settings[0]["widgets_values"], [640, 352, 3.0])

                samplers = [
                    node
                    for node in graph["nodes"]
                    if node.get("type")
                    in {"LingBotSampler", "LingBotTI2VSampler", "LingBotFLFSampler"}
                ]
                self.assertEqual(len(samplers), 1)
                self.assertEqual(
                    samplers[0]["widgets_values"],
                    [640, 352, 61, 28, 3.0, 3.0, 42, "fixed", "sequential_sage"],
                )

    def test_linked_generation_values_reach_the_sampler(self) -> None:
        expected_outputs = [640, 352, 61, 20.0, 28, 3.0, 3.0, 42, 3.0]
        for name in PUBLIC_WORKFLOWS:
            with self.subTest(workflow=name):
                graph = load_workflow(name)
                settings = nodes_by_type(graph, "LingBotGenerationSettings")[0]
                sampler = next(
                    node for node in graph["nodes"] if node.get("type", "").endswith("Sampler")
                )
                target_links = {
                    link[4]: link for link in graph["links"] if link[3] == sampler["id"]
                }
                for slot in range(2, 9):
                    self.assertEqual(target_links[slot][1], settings["id"])
                    source_slot = target_links[slot][2]
                    self.assertEqual(
                        expected_outputs[source_slot], sampler["widgets_values"][slot - 2]
                    )

                for prompt_settings in nodes_by_type(graph, "LingBotPromptSettings"):
                    prompt_links = {
                        link[0]: link
                        for link in graph["links"]
                        if link[1] == prompt_settings["id"] and link[3] == settings["id"]
                    }
                    linked_inputs = {
                        entry["name"]: prompt_links[entry["link"]][2]
                        for entry in settings.get("inputs", [])
                        if entry.get("link") in prompt_links
                    }
                    self.assertEqual(
                        linked_inputs,
                        {"width": 0, "height": 1, "duration_seconds": 2},
                    )

    def test_postprocessing_is_disabled_and_ordered_flash_then_rife(self) -> None:
        for name in PUBLIC_WORKFLOWS:
            with self.subTest(workflow=name):
                graph = load_workflow(name)
                post = nodes_by_type(graph, "LingBotPostProcessSettings")[0]
                self.assertEqual(post["widgets_values"], [20.0, False, 2.0, False])
                switches = nodes_by_type(graph, "LingBotLazyImageSwitch")
                flash = nodes_by_type(graph, "FlashVSRNode")[0]
                rife = nodes_by_type(graph, "RIFEInterpolation")[0]
                links = graph["links"]
                first_switch = next(
                    switch
                    for switch in switches
                    if any(link[1] == flash["id"] and link[3] == switch["id"] for link in links)
                )
                second_switch = next(
                    switch
                    for switch in switches
                    if any(link[1] == rife["id"] and link[3] == switch["id"] for link in links)
                )
                self.assertTrue(
                    any(link[1] == first_switch["id"] and link[3] == rife["id"] for link in links)
                )
                self.assertTrue(
                    any(link[1] == rife["id"] and link[3] == second_switch["id"] for link in links)
                )

    def test_lazy_switches_match_current_schema_and_link_slots(self) -> None:
        for name in PUBLIC_WORKFLOWS:
            graph = load_workflow(name)
            for switch in nodes_by_type(graph, "LingBotLazyImageSwitch"):
                with self.subTest(workflow=name, node=switch["id"]):
                    self.assertEqual(
                        [entry["name"] for entry in switch["inputs"]],
                        ["enabled", "original", "processed"],
                    )
                    for slot, entry in enumerate(switch["inputs"]):
                        link_id = entry.get("link")
                        if link_id is not None:
                            link = next(link for link in graph["links"] if link[0] == link_id)
                            self.assertEqual((link[3], link[4]), (switch["id"], slot))

    def test_image_workflows_have_placeholders_and_prominent_warnings(self) -> None:
        cases = {
            "lingbot_fp8_ti2v.json": ["select_input_image_1.png"],
            "lingbot_fp8_flf_experimental.json": [
                "select_input_image_1.png",
                "select_input_image_2.png",
            ],
        }
        for name, placeholders in cases.items():
            with self.subTest(workflow=name):
                graph = load_workflow(name)
                images = [node["widgets_values"][0] for node in nodes_by_type(graph, "LoadImage")]
                self.assertEqual(images, placeholders)
                warnings = [
                    node
                    for node in nodes_by_type(graph, "MarkdownNote")
                    if str(node.get("title", "")).startswith("⚠")
                ]
                self.assertEqual(len(warnings), 1)
                warning_text = warnings[0]["widgets_values"][0]
                self.assertIn("BEFORE QUEUEING", warning_text)
                for placeholder in placeholders:
                    self.assertIn(placeholder, warning_text)

    def test_sanitizer_normalizes_a_publication_copy_idempotently(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "sanitize_workflows", ROOT / "tools" / "sanitize_workflows.py"
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        source = WORKFLOWS / "lingbot_fp8_ti2v.json"
        with tempfile.TemporaryDirectory() as temp_dir:
            workflow_dir = Path(temp_dir) / "workflows"
            workflow_dir.mkdir()
            target = workflow_dir / source.name
            graph = load_workflow(source.name)
            nodes_by_type(graph, "LingBotGenerationSettings")[0]["widgets_values"] = [
                1024,
                1024,
                30,
                8,
                60,
                5,
                5,
                999,
                "randomize",
            ]
            target.write_text(json.dumps(graph), encoding="utf-8")
            module.sanitize(target)
            first = target.read_text(encoding="utf-8")
            module.sanitize(target)
            self.assertEqual(target.read_text(encoding="utf-8"), first)
            normalized = json.loads(first)
            self.assertEqual(
                nodes_by_type(normalized, "LingBotGenerationSettings")[0]["widgets_values"],
                [640, 352, 20.0, 3.0, 28, 3.0, 3.0, 42, "fixed"],
            )


if __name__ == "__main__":
    unittest.main()
