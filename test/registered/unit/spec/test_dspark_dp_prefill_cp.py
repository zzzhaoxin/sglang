"""CPU coverage of the TP8/DP4/prefill-CP2 DSpark communication contract."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.arg_groups import speculative_hook as hook
from sglang.srt.layers.cp.cp_decode_attn_tp import CpDecodeAttnTpContext
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models import dspark
from sglang.srt.models.deepseek_v4_dspark import DeepseekV4ForCausalLMDSpark
from sglang.srt.speculative.ragged_verify import RaggedVerifyMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _cfg(**changes):
    values = dict(
        device="cuda",
        enable_prefill_cp=True,
        cp_strategy="interleave",
        enable_cp_decode_attn_tp=True,
        tp_size=8,
        attn_dp_size=4,
        attn_cp_size=2,
        pp_size=1,
        moe_a2a_backend="none",
    )
    values.update(changes)
    return SimpleNamespace(**values)


class TestDSparkDpPrefillCp(unittest.TestCase):
    def test_narrow_layout_guard(self):
        hf = SimpleNamespace(
            model_type="deepseek_v41", architectures=["DeepseekV4ForCausalLM"]
        )
        with patch(
            "sglang.srt.speculative.ragged_verify.read_ragged_verify_mode",
            return_value=RaggedVerifyMode.STATIC,
        ):
            hook._validate_dspark_dp_prefill_cp(_cfg(), hf)
            for changes in (
                dict(device="npu"),
                dict(enable_prefill_cp=False),
                dict(cp_strategy="zigzag"),
                dict(enable_cp_decode_attn_tp=False),
                dict(tp_size=16),
                dict(attn_dp_size=2),
                dict(attn_cp_size=4),
                dict(pp_size=2),
                dict(moe_a2a_backend="deepep"),
            ):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    hook._validate_dspark_dp_prefill_cp(_cfg(**changes), hf)
            with self.assertRaises(ValueError):
                hook._validate_dspark_dp_prefill_cp(
                    _cfg(),
                    SimpleNamespace(
                        model_type="qwen3", architectures=["Qwen3ForCausalLM"]
                    ),
                )
        with patch(
            "sglang.srt.speculative.ragged_verify.read_ragged_verify_mode",
            return_value=RaggedVerifyMode.COMPACT,
        ):
            with self.assertRaisesRegex(ValueError, "static"):
                hook._validate_dspark_dp_prefill_cp(_cfg(), hf)

    def test_vocab_gather_follows_head_not_full_tp(self):
        # Different DP groups can have unequal row counts. Full TP gather is
        # invalid even if cropping happens to hide its extra vocab partitions.
        local = torch.arange(10).reshape(2, 5)
        attn_group = SimpleNamespace(all_gather=Mock(return_value=local))
        head = SimpleNamespace(use_attn_tp_group=True, org_vocab_size=4)
        with (
            patch.object(
                dspark,
                "get_parallel",
                return_value=SimpleNamespace(attn_tp_group=attn_group),
            ),
            patch.object(
                dspark,
                "tensor_model_parallel_all_gather",
                side_effect=AssertionError("cross-DP gather"),
            ),
        ):
            torch.testing.assert_close(
                dspark.gather_and_crop_vocab(local, head), local[:, :4]
            )
            attn_group.all_gather.assert_called_once_with(local, dim=-1)

    def test_non_dp_head_keeps_tp_gather(self):
        local = torch.ones(3, 4)
        full = torch.cat([local, local * 2], dim=-1)
        with patch.object(
            dspark, "tensor_model_parallel_all_gather", return_value=full
        ) as gather:
            result = dspark.gather_and_crop_vocab(
                local, SimpleNamespace(org_vocab_size=7)
            )
        gather.assert_called_once_with(local, dim=-1)
        torch.testing.assert_close(result, full[:, :7])

    def test_decisions_sync_cp_partners_not_other_dp_requests(self):
        from sglang.srt.speculative.dspark_components import dspark_worker_v2 as worker

        cp_group, default_group = object(), object()
        with patch.object(worker, "get_dp_tp_group", return_value=default_group):
            parallel = SimpleNamespace(
                attn_dp_enabled=True, attn_cp_size=2, attn_cp_group=cp_group
            )
            self.assertIs(worker._dspark_tp_sync_group(parallel), cp_group)
            parallel.attn_cp_size = 1
            self.assertIs(worker._dspark_tp_sync_group(parallel), default_group)
            parallel.attn_dp_enabled = False
            parallel.attn_cp_size = 2
            self.assertIs(worker._dspark_tp_sync_group(parallel), default_group)

    def test_cp2_decode_slices_and_restores_attention(self):
        # Static CP topology does not imply token-sharded draft/verify rows.
        # Both CP partners process the same rows, each computing half the heads.
        import sglang.srt.layers.cp.cp_decode_attn_tp as cp_tp

        full_weight = torch.arange(32, dtype=torch.float32).reshape(8, 4)
        x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        partials = []
        for rank in range(2):
            parallel = SimpleNamespace(
                enable_cp_decode_attn_tp=True, attn_cp_size=2, attn_cp_rank=rank
            )
            with patch.object(cp_tp, "get_parallel", return_value=parallel):
                ctx = CpDecodeAttnTpContext()
            obj = SimpleNamespace(weight=full_weight.clone())
            for mode in (
                ForwardMode.DECODE,
                ForwardMode.TARGET_VERIFY,
                ForwardMode.DRAFT_EXTEND_V2,
            ):
                fb = SimpleNamespace(forward_mode=mode)
                with (
                    patch.object(cp_tp, "is_cp_active", return_value=False),
                    patch.object(cp_tp, "dsa_use_prefill_cp", return_value=False),
                    ctx.maybe_use_decode_attn_tp(
                        fb, [], tensor_attrs=[(obj, "weight", 0)]
                    ),
                ):
                    self.assertTrue(ctx.use_decode_attn_tp)
                    torch.testing.assert_close(obj.weight, full_weight.chunk(2)[rank])
                torch.testing.assert_close(obj.weight, full_weight)
            partials.append(x @ full_weight.chunk(2)[rank].T)
            with (
                patch.object(cp_tp, "is_cp_active", return_value=True),
                ctx.maybe_use_decode_attn_tp(
                    SimpleNamespace(), [], tensor_attrs=[(obj, "weight", 0)]
                ),
            ):
                self.assertFalse(ctx.use_decode_attn_tp)
                torch.testing.assert_close(obj.weight, full_weight)
        torch.testing.assert_close(torch.cat(partials, dim=-1), x @ full_weight.T)

    def test_draft_rejects_active_cp_before_embedding(self):
        with patch(
            "sglang.srt.models.deepseek_v4_dspark.is_cp_active", return_value=True
        ):
            with self.assertRaisesRegex(ValueError, "active context parallelism"):
                DeepseekV4ForCausalLMDSpark.forward(
                    SimpleNamespace(), None, None, SimpleNamespace()
                )

    def test_decode_row_linear_reduces_cp_not_dp_or_static_attn_tp(self):
        from sglang.srt.layers import linear

        x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        cp_group = SimpleNamespace(all_reduce=Mock(side_effect=lambda t: t * 2))
        forbidden = SimpleNamespace(
            all_reduce=Mock(side_effect=AssertionError("wrong group"))
        )
        layer = SimpleNamespace(
            tp_group=object(),
            input_is_parallel=True,
            skip_bias_add=False,
            bias=None,
            quant_method=SimpleNamespace(apply=lambda *a, **kw: x),
            use_decode_attn_tp=True,
            use_dp_attention_reduce=True,
            reduce_results=False,
        )
        with (
            patch.object(linear, "get_group_rank_size", return_value=(0, 1)),
            patch.object(
                linear, "get_forward", return_value=SimpleNamespace(sp_active=False)
            ),
            patch.object(linear, "should_skip_mlp_all_reduce", return_value=False),
            patch.object(
                linear,
                "get_parallel",
                return_value=SimpleNamespace(
                    attn_cp_group=cp_group, attn_tp_group=forbidden, tp_group=forbidden
                ),
            ),
            patch.object(
                linear, "use_symmetric_memory", return_value=nullcontext()
            ) as alloc,
        ):
            output, _ = linear.RowParallelLinear.forward(layer, x)
            torch.testing.assert_close(output, x * 2)
            cp_group.all_reduce.assert_called_once_with(x)
            alloc.assert_called_once_with(cp_group)
            cp_group.all_reduce.reset_mock()
            output, _ = linear.RowParallelLinear.forward(layer, x, skip_all_reduce=True)
            torch.testing.assert_close(output, x)
            cp_group.all_reduce.assert_not_called()


if __name__ == "__main__":
    unittest.main()
