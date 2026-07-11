from __future__ import annotations

import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path

import torch


PACK_ROOT = Path(__file__).resolve().parents[1]
_default_comfy_root = PACK_ROOT.parents[1]
COMFY_ROOT = Path(os.environ.get("COMFYUI_ROOT", _default_comfy_root)).resolve()
COMFY_AVAILABLE = (COMFY_ROOT / "comfy").is_dir()
if COMFY_AVAILABLE and str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))


def _load_pack():
    if not COMFY_AVAILABLE:
        raise unittest.SkipTest(
            "ComfyUI is not available; set COMFYUI_ROOT to run integration tests"
        )
    name = "_comfyui_lingbotvideo_test"
    if name in sys.modules and hasattr(sys.modules[name], "NODE_CLASS_MAPPINGS"):
        return sys.modules[name]
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(
        name,
        PACK_ROOT / "__init__.py",
        submodule_search_locations=[str(PACK_ROOT)],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


class PackTests(unittest.TestCase):
    def test_node_registration_and_defaults(self):
        pack = _load_pack()
        self.assertEqual(
            set(pack.NODE_CLASS_MAPPINGS),
            {
                "LingBotModelLoader",
                "LingBotTextEncode",
                "LingBotPromptEncode",
                "LingBotTI2VPromptEncode",
                "LingBotFLFPromptEncode",
                "LingBotPromptPreview",
                "LingBotPromptSettings",
                "LingBotGenerationSettings",
                "LingBotPostProcessSettings",
                "LingBotLazyImageSwitch",
                "LingBotLatentSink",
                "LingBotSampler",
                "LingBotTI2VSampler",
                "LingBotFLFSampler",
                "LingBotVAEDecode",
            },
        )
        sampler_inputs = pack.NODE_CLASS_MAPPINGS["LingBotSampler"].INPUT_TYPES()[
            "required"
        ]
        self.assertEqual(sampler_inputs["width"][1]["default"], 640)
        self.assertEqual(sampler_inputs["height"][1]["default"], 352)
        self.assertEqual(sampler_inputs["num_frames"][1]["default"], 73)
        self.assertEqual(sampler_inputs["steps"][1]["default"], 40)
        self.assertEqual(sampler_inputs["cfg"][1]["default"], 3.0)
        loader_inputs = pack.NODE_CLASS_MAPPINGS["LingBotModelLoader"].INPUT_TYPES()[
            "required"
        ]
        self.assertEqual(
            loader_inputs["model_dir"][0][0], "lingbot-video-dense-1.3b"
        )
        self.assertEqual(
            loader_inputs["transformer_subfolder"][0][0], "transformer_fp8_dense"
        )
        self.assertIs(loader_inputs["group_offload"][1]["default"], False)
        prompt_options = pack.NODE_CLASS_MAPPINGS["LingBotTextEncode"].INPUT_TYPES()[
            "required"
        ]["prompt"][1]
        self.assertIs(prompt_options["dynamicPrompts"], False)
        rewrite_inputs = pack.NODE_CLASS_MAPPINGS["LingBotPromptEncode"].INPUT_TYPES()[
            "required"
        ]
        self.assertIs(rewrite_inputs["plaintext_prompt"][1]["dynamicPrompts"], False)
        self.assertIs(
            rewrite_inputs["structured_json_override"][1]["dynamicPrompts"], False
        )
        settings_inputs = pack.NODE_CLASS_MAPPINGS["LingBotGenerationSettings"].INPUT_TYPES()[
            "required"
        ]
        self.assertEqual(settings_inputs["fps"][1]["default"], 20.0)
        self.assertEqual(settings_inputs["duration_seconds"][1]["default"], 3.0)
        prompt_settings = pack.NODE_CLASS_MAPPINGS["LingBotPromptSettings"].INPUT_TYPES()[
            "required"
        ]
        self.assertEqual(prompt_settings["width"][1]["default"], 640)
        self.assertEqual(prompt_settings["height"][1]["default"], 352)
        self.assertEqual(prompt_settings["duration_seconds"][1]["default"], 3.0)
        for node_name in (
            "LingBotModelLoader",
            "LingBotTextEncode",
            "LingBotPromptEncode",
            "LingBotTI2VPromptEncode",
            "LingBotFLFPromptEncode",
            "LingBotSampler",
            "LingBotTI2VSampler",
            "LingBotFLFSampler",
            "LingBotVAEDecode",
        ):
            self.assertEqual(
                pack.NODE_CLASS_MAPPINGS[node_name].INPUT_TYPES()["hidden"]["unique_id"],
                "UNIQUE_ID",
            )

        ti2v_inputs = pack.NODE_CLASS_MAPPINGS["LingBotTI2VPromptEncode"].INPUT_TYPES()[
            "required"
        ]
        self.assertIn("first_frame", ti2v_inputs)
        self.assertIs(ti2v_inputs["plaintext_prompt"][1]["dynamicPrompts"], False)
        ti2v_sampler = pack.NODE_CLASS_MAPPINGS["LingBotTI2VSampler"]
        self.assertEqual(
            ti2v_sampler.INPUT_TYPES()["optional"]["cfg_execution"][1]["default"],
            "sequential_sage",
        )
        flf_inputs = pack.NODE_CLASS_MAPPINGS["LingBotFLFPromptEncode"].INPUT_TYPES()[
            "required"
        ]
        self.assertIn("first_frame", flf_inputs)
        self.assertIn("last_frame", flf_inputs)

    def test_ti2v_first_frame_crop_matches_requested_geometry(self):
        pack = _load_pack()
        image = torch.zeros((1, 100, 200, 3), dtype=torch.float32)
        image[:, :, :100, 0] = 1.0
        pixel = pack.nodes._prepare_ti2v_first_frame(image, 352, 640)
        self.assertEqual(tuple(pixel.shape), (1, 3, 1, 352, 640))
        self.assertGreaterEqual(float(pixel.min()), 0.0)
        self.assertLessEqual(float(pixel.max()), 1.0)
        grounded = pack.nodes._ground_rewrite_to_first_frame("continue motion", True)
        self.assertIn("exact first frame", grounded)
        self.assertIn("continue motion", grounded)
        self.assertEqual(
            pack.nodes._ground_rewrite_to_first_frame("continue motion", False),
            "continue motion",
        )
        endpoints = pack.nodes._ground_rewrite_to_visual_frames("bridge motion", 2)
        self.assertIn("exact first", endpoints)
        self.assertIn("exact last", endpoints)
        self.assertIn("bridge motion", endpoints)

    def test_flf_endpoint_conditioning_pins_only_first_and_last_latents(self):
        pack = _load_pack()
        pipeline = sys.modules[
            "_comfyui_lingbotvideo_test.lingbot_video.pipeline_lingbot_video"
        ].LingBotVideoPipeline
        latents = torch.zeros((1, 2, 5, 2, 2), dtype=torch.float32)
        first = torch.ones((1, 2, 1, 2, 2), dtype=torch.float32)
        last = torch.full((1, 2, 1, 2, 2), 2.0, dtype=torch.float32)
        actual = pipeline._apply_endpoint_conditioning(latents, first, last)
        self.assertTrue(torch.all(actual[:, :, 0:1] == 1))
        self.assertTrue(torch.all(actual[:, :, 1:-1] == 0))
        self.assertTrue(torch.all(actual[:, :, -1:] == 2))
        with self.assertRaisesRegex(ValueError, "exactly one"):
            pipeline._apply_endpoint_conditioning(
                torch.zeros_like(latents),
                first,
                torch.zeros((1, 2, 2, 2, 2)),
            )

    def test_visual_prompt_template_has_one_token_per_ordered_image(self):
        pack = _load_pack()

        class CaptureProcessor:
            def __init__(self):
                self.text = None

            def __call__(self, **kwargs):
                self.text = kwargs["text"]
                return kwargs

        pipe = pack.nodes._ComfyLingBotPipeline(
            transformer=None,
            vae=None,
            text_encoder=None,
            processor=None,
            scheduler=None,
        )
        processor = CaptureProcessor()
        pipe.processor = processor
        pipe._build_prompt_inputs("transition", images=["first", "last"])
        self.assertEqual(processor.text[0].count("<|image_pad|>"), 2)

    def test_experiment_settings_validate_and_return_typed_values(self):
        pack = _load_pack()
        prompt_settings = pack.NODE_CLASS_MAPPINGS["LingBotPromptSettings"]()
        self.assertEqual(prompt_settings.settings(640, 352, 3), (640, 352, 3.0))
        with self.assertRaisesRegex(ValueError, "Duration"):
            prompt_settings.settings(640, 352, 0)
        settings = pack.NODE_CLASS_MAPPINGS["LingBotGenerationSettings"]()
        self.assertEqual(
            settings.settings(640, 352, 20, 3, 40, 3, 3, 42),
            (640, 352, 61, 20.0, 40, 3.0, 3.0, 42, 3.0),
        )
        self.assertEqual(
            settings.settings(640, 352, 24, 3, 40, 3, 3, 42)[2],
            73,
        )
        with self.assertRaisesRegex(ValueError, "Duration"):
            settings.settings(640, 352, 20, 0, 40, 3, 3, 42)

    def test_prompt_rewrite_cache_is_independent_and_image_sensitive(self):
        pack = _load_pack()
        pack.nodes._PROMPT_REWRITE_CACHE.clear()
        image = torch.zeros((1, 3, 1, 16, 16), dtype=torch.float32)
        base = pack.nodes._prompt_rewrite_cache_key(
            Path("model"), "move forward", 3.0, image, None
        )
        same = pack.nodes._prompt_rewrite_cache_key(
            Path("model"), "move forward", 3.0, image.clone(), None
        )
        changed = image.clone()
        changed[..., 0, 0] = 1.0
        different = pack.nodes._prompt_rewrite_cache_key(
            Path("model"), "move forward", 3.0, changed, None
        )
        self.assertEqual(base, same)
        self.assertNotEqual(base, different)
        pack.nodes._cache_prompt_rewrite(base, '{"caption":{}}', "expanded")
        self.assertEqual(
            pack.nodes._get_cached_prompt_rewrite(base),
            ('{"caption":{}}', "expanded"),
        )

    def test_json_stopping_criteria_ignores_braces_inside_strings(self):
        pack = _load_pack()

        class Processor:
            fragments = {
                1: "{",
                2: '"text":"a } brace and an escaped \\" quote",',
                3: '"nested":{',
                4: '"value":1}',
                5: "}",
            }

            def decode(self, token_ids, **_):
                return "".join(self.fragments[token_id] for token_id in token_ids)

        criteria = pack.nodes._BalancedJSONObjectStoppingCriteria(Processor(), input_length=1)
        generated = [99]
        for token_id in (1, 2, 3, 4):
            generated.append(token_id)
            self.assertFalse(bool(criteria(torch.tensor([generated]), None)[0]))
        generated.append(5)
        self.assertTrue(bool(criteria(torch.tensor([generated]), None)[0]))

    def test_postprocess_settings_and_lazy_switch(self):
        pack = _load_pack()
        settings = pack.NODE_CLASS_MAPPINGS["LingBotPostProcessSettings"]()
        self.assertEqual(
            settings.settings(20, False, 2, False),
            (False, 40.0, False, 20.0),
        )
        self.assertEqual(
            settings.settings(20, True, 2, True),
            (True, 40.0, True, 40.0),
        )
        switch = pack.NODE_CLASS_MAPPINGS["LingBotLazyImageSwitch"]
        self.assertEqual(switch.check_lazy_status(False), ["original"])
        self.assertEqual(switch.check_lazy_status(True), ["processed"])
        inputs = switch.INPUT_TYPES()["required"]
        self.assertIs(inputs["original"][1]["lazy"], True)
        self.assertIs(inputs["processed"][1]["lazy"], True)

    def test_int8_rowwise_dequantization(self):
        _load_pack()
        module_name = "_comfyui_lingbotvideo_test.lingbot_video.transformer_lingbot_video"
        grouped_experts = sys.modules[module_name].LingBotVideoGroupedExperts(
            num_experts=2,
            hidden_size=3,
            intermediate_size=4,
            quant="int8_rowwise",
        )
        with torch.no_grad():
            grouped_experts.w1.fill_(2)
            grouped_experts.w1_scale.fill_(0.25)
        actual = grouped_experts.dequant("w1", torch.bfloat16)
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertEqual(actual.shape, (2, 4, 3))
        self.assertTrue(torch.all(actual == torch.tensor(0.5, dtype=torch.bfloat16)))

    def test_transformer_dtype_preserves_int8_and_casts_dense(self):
        pack = _load_pack()
        self.assertIsNone(pack.nodes._transformer_load_dtype(is_int8=True))
        self.assertIs(pack.nodes._transformer_load_dtype(is_int8=False), torch.bfloat16)

    def test_dense_int8_linear_dequantizes_rowwise(self):
        _load_pack()
        module_name = "_comfyui_lingbotvideo_test.lingbot_video.transformer_lingbot_video"
        mod = sys.modules[module_name]
        linear = mod.LingBotVideoInt8Linear(3, 2, bias=True)
        with torch.no_grad():
            linear.weight.copy_(torch.tensor([[2, -4, 6], [1, 2, 3]], dtype=torch.int8))
            linear.weight_scale.copy_(torch.tensor([0.5, 0.25]))
            linear.bias.zero_()
        # Regression: the first implementation derived the block compute dtype
        # from the INT8 storage dtype and reached addmm with Char activations.
        actual = linear(torch.ones(1, 3, dtype=torch.int8))
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertTrue(torch.allclose(actual.float(), torch.tensor([[2.0, 1.5]])))
        self.assertIs(mod._linear_compute_dtype(linear), torch.bfloat16)

    def test_dense_int8_config_replaces_only_block_linears(self):
        _load_pack()
        module_name = "_comfyui_lingbotvideo_test.lingbot_video.transformer_lingbot_video"
        mod = sys.modules[module_name]
        model = mod.LingBotVideoTransformer3DModel(
            hidden_size=8,
            num_attention_heads=1,
            axes_dims=(2, 2, 4),
            axes_lens=(8, 8, 8),
            depth=1,
            intermediate_size=16,
            text_dim=8,
            freq_dim=8,
            dense_quant="int8_rowwise",
        )
        self.assertIsInstance(model.blocks[0].attn.to_q, mod.LingBotVideoInt8Linear)
        self.assertIsInstance(model.blocks[0].ffn.up_proj, mod.LingBotVideoInt8Linear)
        self.assertIsInstance(model.patch_embedder, torch.nn.Linear)

    def test_dense_fp8_config_replaces_only_fused_speed_linears(self):
        _load_pack()
        module_name = "_comfyui_lingbotvideo_test.lingbot_video.transformer_lingbot_video"
        mod = sys.modules[module_name]
        model = mod.LingBotVideoTransformer3DModel(
            hidden_size=8,
            num_attention_heads=1,
            axes_dims=(2, 2, 4),
            axes_lens=(8, 8, 8),
            depth=1,
            intermediate_size=16,
            text_dim=8,
            freq_dim=8,
            dense_fp8="e4m3fn_tensorwise_fused",
        )
        self.assertIsInstance(model.blocks[0].attn.to_q, mod.LingBotVideoFP8Linear)
        self.assertIsInstance(model.blocks[0].ffn.gate_proj, mod.LingBotVideoFP8Linear)
        self.assertIsInstance(model.blocks[0].attn.to_out, torch.nn.Linear)
        self.assertIsInstance(model.blocks[0].ffn.down_proj, torch.nn.Linear)
        model.to(dtype=torch.bfloat16)
        self.assertEqual(model.blocks[0].attn.to_q.weight.dtype, torch.float8_e4m3fn)
        self.assertEqual(model.blocks[0].attn.to_q.weight_scale.dtype, torch.float32)
        self.assertEqual(model.blocks[0].attn.to_out.weight.dtype, torch.bfloat16)

    def test_structured_prompt_normalization_matches_reference_runner(self):
        pack = _load_pack()
        prompt = json.dumps(
            {
                "caption": {
                    "comprehensive_description": {
                        "scene_content_description": "A woman in a bright apartment.",
                        "camera_movement_description": "A stationary eye-level medium shot.",
                    }
                },
                "duration": 3,
            }
        )
        self.assertEqual(
            pack.nodes._normalize_structured_prompt(prompt),
            '{"comprehensive_description":{"scene_content_description":"A woman in a bright apartment.","camera_movement_description":"A stationary eye-level medium shot."}}',
        )
        with self.assertRaisesRegex(ValueError, "structured JSON"):
            pack.nodes._normalize_structured_prompt("a plain text prompt")
        with self.assertRaisesRegex(ValueError, "stripped the prompt's JSON braces"):
            pack.nodes._normalize_structured_prompt('"caption":"comprehensive_description"')

    def test_plaintext_rewriter_extracts_and_validates_complete_json(self):
        _load_pack()
        rewriter = sys.modules[
            "_comfyui_lingbotvideo_test.lingbot_video.prompt_rewriter"
        ]
        mapping_prompt = rewriter.build_json_prompt("A woman walks forward.", 3.0)
        self.assertIn("Output minified raw JSON on one line", mapping_prompt)
        schema_example = mapping_prompt.split("SCHEMA EXAMPLE:\n", 1)[1].split(
            "\n\nHard rules:", 1
        )[0]
        self.assertNotIn("\n", schema_example)
        element = {
            "name": "woman",
            "description": "A woman in a blue coat",
            "actions": [{"timestamp": "[0.0s - 3.0s]", "action": "walks forward"}],
            "location": "center frame",
            "relative_size": "large",
            "shape_and_color": "human silhouette, blue coat",
            "texture": "woven fabric",
            "appearance_details": "dark hair",
            "relationship": "in front of the windows",
            "orientation": "facing camera",
            "pose": "walking",
            "expression": "calm",
            "clothing": "blue coat",
            "gender": "woman",
            "skin_tone_and_texture": "natural skin",
            "number_of_objects": "1",
        }
        caption = {
            "comprehensive_description": {
                "scene_content_description": "A woman walks through an apartment.",
                "camera_movement_description": "Stable eye-level medium shot.",
            },
            "prominent_elements": [element],
            "camera_info": {field: "test" for field in rewriter.CAMERA_FIELDS},
            "world_knowledge": [],
        }
        parsed = rewriter.extract_json_object(
            "```json\n" + json.dumps({"caption": caption}) + "\n```"
        )
        structured, errors = rewriter.normalize_generated_prompt(parsed, 3.0)
        self.assertEqual(errors, [])
        self.assertEqual(json.loads(structured)["duration"], 3)
        for generated_value, expected in (
            ("Rain makes pavement reflective.", ["Rain makes pavement reflective."]),
            ({"rain": "creates wet reflections"}, ["rain: creates wet reflections"]),
            (None, []),
        ):
            world_variant = json.loads(json.dumps(parsed))
            world_variant["caption"]["world_knowledge"] = generated_value
            normalized_variant, variant_errors = rewriter.normalize_generated_prompt(
                world_variant,
                3.0,
            )
            self.assertEqual(variant_errors, [])
            self.assertEqual(
                json.loads(normalized_variant)["caption"]["world_knowledge"],
                expected,
            )
            self.assertEqual(
                world_variant["caption"]["world_knowledge"],
                generated_value,
            )
        missing_world = json.loads(json.dumps(parsed))
        del missing_world["caption"]["world_knowledge"]
        normalized_missing, missing_errors = rewriter.normalize_generated_prompt(
            missing_world,
            3.0,
        )
        self.assertEqual(missing_errors, [])
        self.assertEqual(
            json.loads(normalized_missing)["caption"]["world_knowledge"],
            [],
        )
        missing_actions = json.loads(json.dumps(parsed))
        missing_actions["caption"]["prominent_elements"][0]["actions"] = []
        repaired, repair_count = rewriter.repair_missing_element_actions(
            missing_actions,
            3.0,
            "The woman walks forward from 0.0s through 3.0s.",
        )
        repaired_json, repaired_errors = rewriter.normalize_generated_prompt(repaired, 3.0)
        self.assertEqual(repair_count, 1)
        self.assertEqual(repaired_errors, [])
        repaired_action = json.loads(repaired_json)["caption"]["prominent_elements"][0][
            "actions"
        ][0]
        self.assertEqual(repaired_action["timestamp"], "[0.0s - 3s]")
        self.assertIn("woman", repaired_action["action"].lower())
        element["actions"][0]["timestamp"] = "[0.0s - 4.0s]"
        self.assertIn("exceeds", " ".join(rewriter.validate_caption(caption, 3.0)))

    def test_prompt_preview_returns_generated_text_for_comfyui(self):
        pack = _load_pack()
        result = pack.NODE_CLASS_MAPPINGS["LingBotPromptPreview"]().preview(
            '{"caption":{}}', "expanded prose"
        )
        self.assertIn("expanded prose", result["ui"]["text"][0])
        self.assertEqual(result["result"], ('{"caption":{}}',))

    def test_api_workflow_chain_and_reference_settings(self):
        graph = json.loads(
            (PACK_ROOT / "example_workflows" / "lingbot_dense_t2v.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            graph["1"]["inputs"]["transformer_subfolder"], "transformer_fp8_dense"
        )
        self.assertIs(graph["1"]["inputs"]["group_offload"], False)
        self.assertEqual(graph["2"]["inputs"]["model"], ["1", 0])
        self.assertEqual(graph["3"]["inputs"]["conditioning"], ["2", 0])
        self.assertEqual(graph["3"]["inputs"]["width"], 640)
        self.assertEqual(graph["3"]["inputs"]["height"], 352)
        self.assertEqual(graph["3"]["inputs"]["num_frames"], 73)
        self.assertEqual(graph["3"]["inputs"]["steps"], 40)
        self.assertEqual(graph["3"]["inputs"]["cfg"], 3.0)
        self.assertEqual(graph["4"]["inputs"]["latents"], ["3", 0])
        self.assertEqual(graph["5"]["class_type"], "VHS_VideoCombine")
        self.assertEqual(graph["5"]["inputs"]["images"], ["4", 0])

    def test_ui_experiment_workflow_chain_and_controls(self):
        graph = json.loads(
            (PACK_ROOT / "example_workflows" / "lingbot_dense_experiment_ui.json").read_text(
                encoding="utf-8"
            )
        )
        nodes = {node["id"]: node for node in graph["nodes"]}
        self.assertEqual(graph["version"], 0.4)
        self.assertEqual(nodes[2]["type"], "LingBotPromptEncode")
        self.assertEqual(nodes[15]["type"], "LingBotPromptPreview")
        self.assertEqual(nodes[16]["type"], "LingBotPromptSettings")
        self.assertEqual(nodes[3]["type"], "LingBotGenerationSettings")
        self.assertEqual(nodes[3]["widgets_values"], [640, 352, 20.0, 3.0, 40, 3.0, 3.0, 42])
        self.assertEqual(nodes[4]["title"], "3. Denoise — Live Step / Elapsed / ETA")
        self.assertEqual(nodes[6]["widgets_values"]["format"], "video/h264-mp4")
        self.assertEqual(nodes[10]["type"], "LingBotPostProcessSettings")
        self.assertEqual(nodes[10]["widgets_values"], [20.0, False, 2.0, False])
        self.assertEqual(nodes[11]["type"], "RIFEInterpolation")
        self.assertEqual(nodes[13]["type"], "FlashVSRNode")
        self.assertEqual(nodes[13]["widgets_values"][0:3], ["FlashVSR-v1.1", "tiny", 2])
        self.assertEqual(nodes[12]["type"], "LingBotLazyImageSwitch")
        self.assertEqual(nodes[14]["type"], "LingBotLazyImageSwitch")
        self.assertIn([25, 14, 0, 6, 0, "IMAGE"], graph["links"])
        self.assertIn([26, 10, 3, 6, 4, "FLOAT"], graph["links"])

        links = {link[0]: link for link in graph["links"]}
        for node in graph["nodes"]:
            for output in node.get("outputs", []):
                for link_id in output.get("links") or []:
                    self.assertIn(link_id, links)
                    self.assertEqual(links[link_id][1], node["id"])
            for input_slot, input_data in enumerate(node.get("inputs", [])):
                link_id = input_data.get("link")
                if link_id is not None:
                    self.assertIn(link_id, links)
                    self.assertEqual(links[link_id][3:5], [node["id"], input_slot])

    def test_fast_workflow_exposes_cfg_modes_with_verified_sage_default(self):
        baseline = json.loads(
            (PACK_ROOT / "example_workflows" / "lingbot_dense_experiment_ui.json").read_text(
                encoding="utf-8"
            )
        )
        fast = json.loads(
            (PACK_ROOT / "example_workflows" / "lingbot_dense_experiment_fast_ui.json").read_text(
                encoding="utf-8"
            )
        )
        baseline_sampler = next(node for node in baseline["nodes"] if node["id"] == 4)
        fast_sampler = next(node for node in fast["nodes"] if node["id"] == 4)
        self.assertNotIn("cfg_execution", [item["name"] for item in baseline_sampler["inputs"]])
        self.assertEqual(fast_sampler["inputs"][-1]["name"], "cfg_execution")
        self.assertEqual(fast_sampler["widgets_values"][-1], "sequential_sage")
        for graph in (baseline, fast):
            prompt = next(node for node in graph["nodes"] if node["id"] == 2)
            preview = next(node for node in graph["nodes"] if node["id"] == 15)
            prompt_settings = next(node for node in graph["nodes"] if node["id"] == 16)
            self.assertEqual(prompt["type"], "LingBotPromptEncode")
            self.assertEqual(preview["type"], "LingBotPromptPreview")
            self.assertEqual(prompt_settings["type"], "LingBotPromptSettings")
            post = next(node for node in graph["nodes"] if node["id"] == 10)
            self.assertEqual(post["widgets_values"], [20.0, False, 2.0, False])
            links = {link[0]: link for link in graph["links"]}
            self.assertEqual((links[13][1], links[13][3]), (5, 13))
            self.assertEqual((links[20][1], links[20][3]), (13, 12))
            self.assertEqual((links[21][1], links[21][3]), (12, 11))
            self.assertEqual((links[24][1], links[24][3]), (11, 14))
            self.assertEqual((links[25][1], links[25][3]), (14, 6))

    def test_ti2v_workflow_uses_one_image_for_qwen_and_fixed_frame_sampling(self):
        graph = json.loads(
            (
                PACK_ROOT
                / "example_workflows"
                / "lingbot_dense_ti2v_experiment_ui.json"
            ).read_text(encoding="utf-8")
        )
        nodes = {node["id"]: node for node in graph["nodes"]}
        self.assertEqual(nodes[16]["type"], "LoadImage")
        self.assertEqual(nodes[2]["type"], "LingBotTI2VPromptEncode")
        self.assertEqual(nodes[4]["type"], "LingBotTI2VSampler")
        self.assertEqual(nodes[17]["type"], "PreviewImage")
        self.assertEqual(nodes[18]["type"], "LingBotPromptSettings")
        self.assertEqual(nodes[4]["inputs"][-1]["name"], "cfg_execution")
        self.assertEqual(nodes[4]["widgets_values"][-1], "sequential_sage")
        self.assertIn("First Frame Fixed", nodes[4]["title"])
        self.assertIn("Exact Center Crop", nodes[17]["title"])
        links = {link[0]: link for link in graph["links"]}
        self.assertEqual(links[30][1:5], [16, 0, 2, 1])
        self.assertEqual(links[27][1:5], [18, 2, 2, 2])
        self.assertEqual(links[31][1:5], [18, 0, 2, 3])
        self.assertEqual(links[32][1:5], [18, 1, 2, 4])
        self.assertEqual(links[33][1:5], [2, 3, 17, 0])
        self.assertEqual(links[34][1:5], [18, 0, 3, 0])
        self.assertEqual(links[35][1:5], [18, 1, 3, 1])
        self.assertEqual(links[36][1:5], [18, 2, 3, 2])
        self.assertEqual((links[13][1], links[13][3]), (5, 13))
        self.assertEqual((links[21][1], links[21][3]), (12, 11))

    def test_experimental_flf_workflow_routes_two_ordered_endpoint_images(self):
        graph = json.loads(
            (
                PACK_ROOT
                / "example_workflows"
                / "lingbot_dense_flf_experimental_ui.json"
            ).read_text(encoding="utf-8")
        )
        nodes = {node["id"]: node for node in graph["nodes"]}
        self.assertEqual(nodes[16]["type"], "LoadImage")
        self.assertEqual(nodes[18]["type"], "LoadImage")
        self.assertEqual(nodes[2]["type"], "LingBotFLFPromptEncode")
        self.assertEqual(nodes[4]["type"], "LingBotFLFSampler")
        self.assertEqual(nodes[20]["type"], "LingBotPromptSettings")
        self.assertEqual(nodes[4]["widgets_values"][-1], "sequential_sage")
        self.assertIn("EXPERIMENTAL", nodes[4]["title"])
        links = {link[0]: link for link in graph["links"]}
        self.assertEqual(links[30][1:5], [16, 0, 2, 1])
        self.assertEqual(links[34][1:5], [18, 0, 2, 2])
        self.assertEqual(links[27][1:5], [20, 2, 2, 3])
        self.assertEqual(links[31][1:5], [20, 0, 2, 4])
        self.assertEqual(links[32][1:5], [20, 1, 2, 5])
        self.assertEqual(links[33][1:5], [2, 3, 17, 0])
        self.assertEqual(links[35][1:5], [2, 4, 19, 0])
        self.assertEqual(links[36][1:5], [20, 0, 3, 0])
        self.assertEqual(links[37][1:5], [20, 1, 3, 1])
        self.assertEqual(links[38][1:5], [20, 2, 3, 2])

    def test_progress_duration_format(self):
        pack = _load_pack()
        self.assertEqual(pack.nodes._format_duration(0.2), "0s")
        self.assertEqual(pack.nodes._format_duration(65.2), "1m 05s")
        self.assertEqual(pack.nodes._format_duration(3661), "1h 01m 01s")

    def test_scheduler_uses_diffusers_scheduler_config_name(self):
        pack = _load_pack()
        model_dir = PACK_ROOT / "models" / "lingbot_video" / "lingbot-video-dense-1.3b"
        if model_dir.is_dir():
            scheduler = pack.nodes._component_dir(
                model_dir,
                "scheduler",
                ("scheduler_config.json",),
            )
            self.assertTrue((scheduler / "scheduler_config.json").is_file())

    def test_pipeline_component_introspection_keeps_base_signature(self):
        pack = _load_pack()
        pipe = pack.nodes._ComfyLingBotPipeline(
            transformer=None,
            vae=None,
            text_encoder=None,
            processor=None,
            scheduler=None,
        ).set_execution_device(torch.device("cuda"))
        self.assertEqual(
            set(pipe.components),
            {"transformer", "vae", "text_encoder", "processor", "scheduler"},
        )
        self.assertEqual(pipe._execution_device, torch.device("cuda"))


if __name__ == "__main__":
    unittest.main()
