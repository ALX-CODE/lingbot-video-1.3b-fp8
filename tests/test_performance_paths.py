from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch


PACK_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PACK_ROOT.parents[1]
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))


def _load_transformer_module():
    package_name = "_comfyui_lingbotvideo_performance_test"
    module_name = f"{package_name}.lingbot_video.transformer_lingbot_video"
    if module_name in sys.modules:
        return sys.modules[module_name]

    package_spec = importlib.util.spec_from_file_location(
        package_name,
        PACK_ROOT / "__init__.py",
        submodule_search_locations=[str(PACK_ROOT)],
    )
    assert package_spec is not None and package_spec.loader is not None
    package = importlib.util.module_from_spec(package_spec)
    sys.modules[package_name] = package
    package_spec.loader.exec_module(package)
    return sys.modules[module_name]


class AttentionBackendTests(unittest.TestCase):
    def test_native_padded_cfg_keeps_2d_mask_and_explicit_backend(self):
        module = _load_transformer_module()
        attention = module.LingBotVideoAttention(4, 1, 1e-6, False, True)
        x = torch.randn(2, 5, 4)
        rotary = torch.ones(2, 5, 2, dtype=torch.complex64)
        mask = torch.tensor(
            [[True, True, True, True, False], [True, True, True, False, False]]
        )
        captured = {}

        def fake_dispatch(q, k, v, **kwargs):
            captured.update(kwargs)
            return v

        with mock.patch.object(module, "dispatch_attention_fn", side_effect=fake_dispatch):
            attention(x, rotary, mask, attention_backend="native")

        self.assertEqual(captured["backend"], "native")
        self.assertIs(captured["attn_mask"], mask)
        self.assertEqual(captured["attn_mask"].shape, (2, 5))
        self.assertEqual(captured["attn_mask"].dtype, torch.bool)

    def test_transformer_builds_per_sample_joint_mask_for_native_cfg(self):
        module = _load_transformer_module()
        captured = {}

        class CaptureBlock(torch.nn.Module):
            def forward(self, x, temb6, rotary, attention_mask, **kwargs):
                captured["mask"] = attention_mask
                captured["backend"] = kwargs["attention_backend"]
                return x

        model = module.LingBotVideoTransformer3DModel(
            patch_size=(1, 1, 1),
            in_channels=1,
            out_channels=1,
            hidden_size=6,
            num_attention_heads=1,
            depth=0,
            intermediate_size=12,
            text_dim=4,
            freq_dim=4,
            axes_dims=(2, 2, 2),
            axes_lens=(16, 16, 16),
        )
        model.blocks = torch.nn.ModuleList([CaptureBlock()])
        model(
            torch.randn(2, 1, 1, 1, 1),
            torch.tensor([500.0, 500.0]),
            torch.randn(2, 3, 4),
            encoder_attention_mask=torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool),
            padded_batch=True,
            attention_backend="native",
            return_dict=False,
        )

        self.assertEqual(captured["backend"], "native")
        self.assertEqual(captured["mask"].shape, (2, 4))
        self.assertTrue(torch.equal(captured["mask"][:, 0], torch.ones(2, dtype=torch.bool)))
        self.assertTrue(
            torch.equal(
                captured["mask"][:, 1:],
                torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool),
            )
        )

    def test_masked_sage_routes_to_varlen_but_unmasked_sage_stays_sage(self):
        module = _load_transformer_module()
        mask = torch.ones(1, 7, dtype=torch.bool)
        self.assertEqual(module._resolve_attention_backend("sage", mask), "sage_varlen")
        self.assertEqual(module._resolve_attention_backend("sage", None), "sage")
        self.assertEqual(module._resolve_attention_backend("native", mask), "native")


class TimestepModulationTests(unittest.TestCase):
    def test_project_once_then_broadcast_matches_tokenwise_projection(self):
        module = _load_transformer_module()
        torch.manual_seed(7)
        batch, sequence, hidden = 2, 11, 8
        projection = torch.nn.Sequential(torch.nn.SiLU(), torch.nn.Linear(hidden, 6 * hidden))
        timestep_embedding = torch.randn(batch, hidden)

        old = projection(
            timestep_embedding.unsqueeze(1).expand(batch, sequence, hidden).reshape(-1, hidden)
        ).reshape(batch, sequence, -1)
        per_sample = projection(timestep_embedding)
        new = module._expand_per_sample_conditioning(per_sample, sequence).expand_as(old)

        # GEMM launch shapes differ (B rows versus B*S identical rows), so the
        # last fp32 bit may differ even though the operation is algebraically
        # identical and remains entirely on the fp32-sensitive path.
        torch.testing.assert_close(new, old, rtol=2e-5, atol=2e-7)

    def test_packed_projection_preserves_sample_token_mapping_and_padding(self):
        module = _load_transformer_module()
        torch.manual_seed(11)
        hidden = 8
        lengths = [3, 5]
        padding = 2
        projection = torch.nn.Sequential(torch.nn.SiLU(), torch.nn.Linear(hidden, 6 * hidden))
        timestep_embedding = torch.randn(2, hidden)
        old_input = torch.cat(
            [
                timestep_embedding[i : i + 1].unsqueeze(1).expand(1, length, -1)
                for i, length in enumerate(lengths)
            ],
            dim=1,
        )
        old = projection(old_input.reshape(-1, hidden)).reshape(1, sum(lengths), -1)
        old = torch.cat([old, torch.zeros(1, padding, old.shape[-1])], dim=1)

        new = module._expand_per_sample_conditioning(
            projection(timestep_embedding),
            sum(lengths) + padding,
            lengths,
            padding_size=padding,
        )
        torch.testing.assert_close(new, old, rtol=2e-5, atol=2e-7)


if __name__ == "__main__":
    unittest.main()
