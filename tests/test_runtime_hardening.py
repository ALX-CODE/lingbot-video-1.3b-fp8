from __future__ import annotations

import ast
import hashlib
import importlib.util
import tempfile
import unittest
import urllib.error
from pathlib import Path
from typing import Any
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_downloader():
    spec = importlib.util.spec_from_file_location("lingbot_download_fp8", ROOT / "tools" / "download_fp8.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_preflight():
    tree = ast.parse((ROOT / "nodes.py").read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_preflight_fp8_runtime"
    )
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "nodes.py", "exec"), namespace)
    return namespace["_preflight_fp8_runtime"]


class _Response:
    def __init__(self, payload: bytes, status: int = 200):
        self.payload = payload
        self.status = status
        self.headers = {"Content-Length": str(len(payload))}
        self.offset = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, size: int) -> bytes:
        chunk = self.payload[self.offset : self.offset + size]
        self.offset += len(chunk)
        return chunk


class DownloaderHardeningTests(unittest.TestCase):
    def setUp(self):
        self.module = _load_downloader()

    def test_release_url_is_version_pinned(self):
        self.assertEqual(self.module.RELEASE_VERSION, "v1.0.2")
        self.assertIn("/releases/download/v1.0.2", self.module.BASE_URL)
        self.assertNotIn("latest", self.module.BASE_URL)

    def test_valid_complete_partial_is_promoted_without_network(self):
        payload = b"complete checkpoint"
        self.module.ASSETS = {"config.json": hashlib.sha256(payload).hexdigest()}
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            (destination / "config.json.part").write_bytes(payload)
            with mock.patch.object(self.module.urllib.request, "urlopen", side_effect=AssertionError("network used")):
                self.module.download("config.json", destination)
            self.assertEqual((destination / "config.json").read_bytes(), payload)
            self.assertFalse((destination / "config.json.part").exists())

    def test_http_416_restarts_from_zero(self):
        payload = b"fresh asset"
        self.module.ASSETS = {"config.json": hashlib.sha256(payload).hexdigest()}
        error = urllib.error.HTTPError("url", 416, "range", {}, None)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            (destination / "config.json.part").write_bytes(b"stale full partial")
            with mock.patch.object(
                self.module.urllib.request,
                "urlopen",
                side_effect=[error, _Response(payload)],
            ) as urlopen:
                self.module.download("config.json", destination)
            self.assertEqual(urlopen.call_count, 2)
            self.assertEqual((destination / "config.json").read_bytes(), payload)

    def test_checksum_mismatch_discards_partial(self):
        expected = hashlib.sha256(b"expected").hexdigest()
        self.module.ASSETS = {"config.json": expected}
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            with mock.patch.object(self.module.urllib.request, "urlopen", return_value=_Response(b"corrupt")):
                with self.assertRaisesRegex(RuntimeError, "discarded corrupt partial"):
                    self.module.download("config.json", destination)
            self.assertFalse((destination / "config.json.part").exists())


class FP8PreflightTests(unittest.TestCase):
    def setUp(self):
        self.preflight = _load_preflight()

    def test_non_cuda_device_has_actionable_fallback(self):
        with self.assertRaisesRegex(RuntimeError, "BF16 'transformer'"):
            self.preflight(torch.device("cpu"))

    def test_missing_scaled_mm_fails_before_cuda_probe(self):
        with mock.patch.object(torch, "_scaled_mm", None):
            with self.assertRaisesRegex(RuntimeError, "torch._scaled_mm"):
                self.preflight(torch.device("cuda"))

    def test_probe_failure_reports_gpu_capability_and_fallback(self):
        original_zeros = torch.zeros
        original_ones = torch.ones

        def fake_zeros(shape, **_kwargs):
            return original_zeros(shape)

        def fake_ones(shape, **_kwargs):
            return original_ones(shape)

        with (
            mock.patch.object(torch.cuda, "get_device_capability", return_value=(12, 0)),
            mock.patch.object(torch.cuda, "get_device_name", return_value="Test GPU"),
            mock.patch.object(torch, "zeros", side_effect=fake_zeros),
            mock.patch.object(torch, "ones", side_effect=fake_ones),
            mock.patch.object(torch, "_scaled_mm", side_effect=RuntimeError("kernel unsupported")),
        ):
            with self.assertRaisesRegex(RuntimeError, "Test GPU.*12.0.*BF16 'transformer'"):
                self.preflight(torch.device("cuda"))

    def test_probe_uses_transposed_weight_layout(self):
        original_zeros = torch.zeros
        original_ones = torch.ones

        def fake_zeros(shape, **kwargs):
            kwargs.pop("device", None)
            return original_zeros(shape, **kwargs)

        def fake_ones(shape, **kwargs):
            kwargs.pop("device", None)
            return original_ones(shape, **kwargs)

        def fake_scaled_mm(lhs, rhs, *_args, **_kwargs):
            self.assertEqual(lhs.stride(), (16, 1))
            self.assertEqual(rhs.stride(), (1, 16))
            return original_zeros((16, 16), dtype=torch.bfloat16)

        with (
            mock.patch.object(torch.cuda, "get_device_capability", return_value=(12, 0)),
            mock.patch.object(torch.cuda, "get_device_name", return_value="Test GPU"),
            mock.patch.object(torch.cuda, "synchronize"),
            mock.patch.object(torch, "zeros", side_effect=fake_zeros),
            mock.patch.object(torch, "ones", side_effect=fake_ones),
            mock.patch.object(torch, "_scaled_mm", side_effect=fake_scaled_mm),
        ):
            self.preflight(torch.device("cuda"))


class PublicLoaderScopeTests(unittest.TestCase):
    def test_loader_exposes_only_dense_fp8_and_bf16_choices(self):
        tree = ast.parse((ROOT / "nodes.py").read_text(encoding="utf-8"))
        loader = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LingBotModelLoader"
        )
        input_types = next(
            node for node in loader.body if isinstance(node, ast.FunctionDef) and node.name == "INPUT_TYPES"
        )
        input_types.decorator_list = []
        namespace = {"Any": Any, "_discover_model_dirs": lambda: ["model"]}
        exec(compile(ast.Module(body=[input_types], type_ignores=[]), "nodes.py", "exec"), namespace)
        inputs = namespace["INPUT_TYPES"](None)
        self.assertEqual(
            inputs["required"]["transformer_subfolder"][0],
            ["transformer_fp8_dense", "transformer"],
        )
        self.assertEqual(
            inputs["required"]["group_offload"][1]["tooltip"],
            "Leave off for the dense 1.3B FP8 or BF16 transformer.",
        )

    def test_qwen_loads_do_not_enable_remote_code(self):
        source = (ROOT / "nodes.py").read_text(encoding="utf-8")
        self.assertNotIn("trust_remote_code", source)


if __name__ == "__main__":
    unittest.main()
