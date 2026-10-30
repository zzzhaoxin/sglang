"""CPU regression tests for bounded replay with TP8 / DP4 / CP2."""

import tempfile
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.layers import dp_attention as dpa
from sglang.srt.arg_groups import deepseek_v4_hook as hook
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
from sglang.srt.layers.cp.interleave import cp_sequence_to_interleave_order
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor import forward_batch_info as fbi
from sglang.srt.managers import scheduler
from sglang.srt.models.deepseek_common import decoder_swa_bounded_replay as replay
from sglang.srt.models import deepseek_v4 as model
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


def _batch():
    return SimpleNamespace(
        global_num_tokens_cpu=[24] * 4,
        global_num_tokens_gpu=torch.tensor([24] * 4),
        global_num_tokens_padded_cpu=[24] * 4,
        global_num_tokens_for_logprob_cpu=[21] * 4,
        global_num_tokens_for_logprob_gpu=torch.tensor([21] * 4),
        global_dp_buffer_len=96,
        dp_padding_mode=dpa.DpPaddingMode.MAX_LEN,
        dp_local_start_pos=None,
        dp_local_num_tokens=None,
        global_num_token_non_padded=torch.tensor(21),
        global_num_token_non_padded_cpu=21,
        num_token_non_padded=torch.tensor(21),
        attn_cp_metadata=None,
    )


def _distributed_worker(rank, rendezvous):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=8)
    try:
        dp_rank, cp_rank = divmod(rank, 2)
        cp_groups = [dist.new_group([2 * i, 2 * i + 1]) for i in range(4)]
        parallel = SimpleNamespace(
            tp_size=8,
            attn_dp_rank=dp_rank,
            attn_cp_rank=cp_rank,
            attn_cp_size=2,
            attn_tp_rank=0,
            tp_group=SimpleNamespace(cpu_group=dist.group.WORLD),
        )
        # Unequal tails, including one CP shard with no logical tokens.
        logical = [7, 3, 1, 0][dp_rank]
        shard = (logical + 1) // 2
        cp = SimpleNamespace(
            max_rank_len=[shard] * 2,
            per_rank_actual_token=[shard] * 2,
            per_rank_logical_token=[(logical + 1) // 2, logical // 2],
            total_seq_lens=logical,
        )
        fb = _batch()
        fb.attn_cp_metadata = cp
        if logical == 0:
            fb.global_num_token_non_padded_cpu = 0
        with (
            patch.object(replay, "get_parallel", return_value=parallel),
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
            patch.object(dpa, "get_parallel", return_value=parallel),
        ):
            counts = replay.late_layer_dp_counts(
                fb, cp.per_rank_logical_token[cp_rank], cp_metadata=cp
            )
            assert counts == [8, 4, 2, 0], counts
            saved = replay.enter_late_layer_dp(fb, counts)
            assert fb.global_num_token_non_padded_cpu == logical
            assert fb.num_token_non_padded.item() == cp.per_rank_logical_token[cp_rank]
            assert dpa.get_local_dp_buffer_len() == shard * 2
            # A DP slot is CP rank-major, padded; every CP partner holds all rows.
            local = torch.arange(shard * 2, dtype=torch.float32).view(-1, 1)
            local += dp_rank * 100
            gathered = torch.empty(sum(counts), 1)

            def all_reduce(x):
                dist.all_reduce(x)
                return x

            with (
                patch.object(
                    dpa,
                    "get_dp_local_info",
                    return_value=(sum(counts[:dp_rank]), counts[dp_rank]),
                ),
                patch.object(dpa, "world_dp_gather_enabled", return_value=False),
                patch.object(
                    dpa, "tensor_model_parallel_all_reduce", side_effect=all_reduce
                ),
                patch.object(dpa, "memcpy_func", dpa.memcpy_cpu),
                patch.object(dpa, "_note_dp_gather_in_prefill_graph"),
            ):
                dpa.dp_gather_replicate(gathered, local, fb)
                expected = torch.cat(
                    [
                        torch.arange(n, dtype=torch.float32) + i * 100
                        for i, n in enumerate(counts)
                    ]
                ).view(-1, 1)
                torch.testing.assert_close(gathered, expected)
                restored = torch.empty_like(local)
                dpa.dp_scatter(restored, gathered, fb)
                torch.testing.assert_close(restored, local)
            replay.exit_late_layer_dp(fb, saved)
            assert fb.global_num_tokens_cpu == [24] * 4
            assert fb.global_dp_buffer_len == 96

            # Real Gloo exchange for the tail=None short-prefill case. Only
            # DP1 has CP metadata; every idle peer must still refresh.
            fb = _batch()
            fb.global_num_tokens_padded_cpu = [0, 24, 0, 0]
            cp = (
                SimpleNamespace(max_rank_len=[12, 12]) if dp_rank == 1 else None
            )
            counts = replay.late_layer_dp_counts(fb, 12 if cp else 0, cp)
            assert counts == [0, 24, 0, 0], (rank, counts)
            # 23 logical rows: CP0 owns evens, CP1 owns odds + a zero pad.
            # Exercise actual CP gather and DP gather/scatter on Gloo, not an
            # invented replicated input. The global MoE layout is rank-major.
            fb.dp_local_start_pos = fb.dp_local_num_tokens = None
            local = torch.arange(cp_rank, 23, 2, dtype=torch.float32).view(-1, 1)
            if cp_rank == 1:
                local = torch.cat([local, torch.zeros(1, 1)])
            cp_full = torch.empty(24, 1) if dp_rank == 1 else torch.empty(0, 1)
            if dp_rank == 1:
                dist.all_gather_into_tensor(cp_full, local, group=cp_groups[dp_rank])
            full = torch.empty(24, 1)
            expected = torch.cat([
                torch.arange(0, 23, 2), torch.arange(1, 23, 2), torch.tensor([0])
            ]).float().view(-1, 1)
            with (
                patch.object(dpa, "get_dp_local_info", return_value=(0 if dp_rank <= 1 else 24, counts[dp_rank])),
                patch.object(dpa, "world_dp_gather_enabled", return_value=False),
                patch.object(dpa, "tensor_model_parallel_all_reduce", side_effect=all_reduce),
                patch.object(dpa, "memcpy_func", dpa.memcpy_cpu),
                patch.object(dpa, "_note_dp_gather_in_prefill_graph"),
            ):
                dpa.dp_gather_replicate(full, cp_full, fb)
                torch.testing.assert_close(full, expected)
                restored = torch.empty_like(cp_full)
                dpa.dp_scatter(restored, full, fb)
                if dp_rank == 1:
                    torch.testing.assert_close(restored.chunk(2)[cp_rank], local)
    finally:
        dist.destroy_process_group()


class TestBoundedReplayDpCp(unittest.TestCase):
    def test_early_cp_slots_include_physical_padding(self):
        # Stop immediately after real prepare_mlp_sync_batch sizes its buffers;
        # no kernels or device tensors are needed to exercise that geometry.
        fb = _batch()
        fb.global_num_tokens_cpu = [0, 23, 25, 1]
        fb.batch_size = 1
        fb.is_extend_in_batch = True
        fb.can_run_dp_prefill_cuda_graph = False
        fb.dp_spec_prefill_coordination_applied = False
        # global_num_tokens_cpu carries four DP slots, so attn_dp_size is 4.
        # The CP slot padding is attention-DP only: without DP there is no
        # gather whose width every peer (idle included) has to agree on, and
        # padding there changes a batch shape the CP prefill already agreed on.
        parallel = SimpleNamespace(attn_cp_size=2, attn_tp_size=1, attn_dp_size=4)
        runner = SimpleNamespace(
            model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="deepseek_v41")),
            attn_tp_sequence_sharded=lambda n: False,
        )
        class Sized(Exception):
            pass

        with (
            patch.object(fbi, "get_parallel", return_value=parallel),
            patch.object(fbi, "_mega_moe_materializes_idle_rank", return_value=False),
            patch.object(fbi, "_elastic_should_preserve_local_token_counts", return_value=False),
            patch.object(fbi, "get_exec", return_value=SimpleNamespace(graph=SimpleNamespace(cuda_graph_config=SimpleNamespace(prefill=None)))),
            patch.object(dpa.DpPaddingMode, "get_dp_padding_mode", return_value=dpa.DpPaddingMode.SUM_LEN),
            patch("sglang.srt.layers.cp.utils.is_mla_cp_enabled", return_value=True),
            patch("sglang.srt.layers.cp.padding.get_cp_padding_align_size", return_value=2),
            patch("sglang.srt.layers.moe.utils.get_moe_a2a_backend", return_value=SimpleNamespace(is_none=lambda: True)),
            patch.object(fbi, "dp_slot_in", return_value=1),
            patch.object(fbi, "set_dp_buffer_len_from_batch", side_effect=Sized),
            self.assertRaises(Sized),
        ):
            fbi.ForwardBatch.prepare_mlp_sync_batch(fb, runner)
        self.assertEqual(fb.global_num_tokens_padded_cpu, [0, 24, 28, 4])
        self.assertEqual(fb.global_dp_buffer_len, 56)

    def test_overlap_prefill_boundaries_use_global_vote(self):
        parallel = SimpleNamespace(attn_cp_size=2)
        cfg = SimpleNamespace(features=SimpleNamespace(enable_decoder_swa_bounded_replay=True))
        owner = SimpleNamespace(require_mlp_sync=True, result_queue=[object()])
        for current, previous, expected in [(True, False, True), (False, True, True), (False, False, False)]:
            # Both active and IDLE local modes carry the same global vote.
            for mode in [ForwardMode.EXTEND, ForwardMode.IDLE]:
                batch = SimpleNamespace(is_extend_in_batch=current, forward_mode=mode, spec_algorithm=SimpleNamespace(is_none=lambda: True))
                last = SimpleNamespace(is_extend_in_batch=previous, forward_mode=mode)
                with (
                    patch.object(scheduler, "get_parallel", return_value=parallel),
                    patch.object(scheduler, "get_exec", return_value=cfg),
                    patch.object(scheduler.envs.SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP, "get", return_value=False),
                ):
                    self.assertEqual(scheduler.Scheduler.is_disable_overlap_for_batch(owner, batch, last), expected)

    def test_validates_effective_cp2_before_cp_normalization(self):
        cfg = SimpleNamespace(
            tp_size=8,
            attn_dp_size=4,
            dp_size=1,
            attn_cp_size=8,
            cp_strategy="interleave",
            enable_prefill_cp=True,
            enable_decoder_swa_bounded_replay=True,
            enable_encoder_swa_bounded_replay=False,
            cuda_graph_config=SimpleNamespace(
                prefill=SimpleNamespace(backend=Backend.DISABLED, max_seq_len=None)
            ),
            dsv4_attn_backend="flashmla",
            speculative_algorithm="DSPARK",
            enable_hisparse=False,
            enable_two_batch_overlap=False,
            pp_size=1,
            disaggregation_mode="null",
            moe_a2a_backend="none",
            ep_size=4,
            moe_dense_tp_size=1,
        )
        hf = SimpleNamespace(model_type="deepseek_v41", vision_n_layers=0)
        with (
            patch.object(hook, "resolving_view", return_value=cfg),
            patch.object(
                hook, "model_config_of", return_value=SimpleNamespace(hf_config=hf)
            ),
            patch.object(
                hook,
                "get_platform",
                return_value=SimpleNamespace(is_cuda=True, is_hip=False),
            ),
            patch.object(hook, "is_npu", return_value=False),
            patch.object(hook, "declare_resolution") as resolution,
            patch(
                "sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate.is_unified_kv_triton",
                return_value=False,
            ),
        ):
            hook.validate_deepseek_v41_features(cfg)
            hook.validate_deepseek_v4_cp(cfg)
            self.assertTrue(
                any(
                    call.kwargs.get("attn_cp_size") == 2
                    for call in resolution.call_args_list
                )
            )

    def test_tp8_dp4_cp2_collectives(self):
        with tempfile.TemporaryDirectory() as tmp:
            mp.start_processes(
                _distributed_worker,
                args=(f"file://{tmp}/gloo",),
                nprocs=8,
                join=True,
                start_method="fork",
            )

    def test_tail_original_cp_ownership_and_padding(self):
        # Two tails with odd starts and an uneven owner distribution. Reinterleaving
        # the compact sequence would assign different IDs to the hidden rows.
        indices = torch.tensor([3, 4, 5, 9, 10])
        ids = torch.arange(12) + 100
        packed = None
        for rank in range(2):
            parallel = SimpleNamespace(attn_cp_rank=rank, attn_cp_size=2)
            fb = SimpleNamespace(
                batch_size=2,
                positions=torch.arange(12),
                forward_mode=ForwardMode.EXTEND,
            )
            with patch(
                "sglang.srt.layers.attention.deepseek_v4_backend.get_parallel",
                return_value=parallel,
            ):
                layout = DeepseekV4AttnBackend._late_layer_tail_cp_layout(
                    None, fb, indices, torch.tensor([3, 2])
                )
            cp = layout["cp_metadata"]
            fb.attn_cp_metadata = cp
            with patch(
                "sglang.srt.layers.cp.interleave.get_parallel", return_value=parallel
            ):
                packed = cp_sequence_to_interleave_order(ids[indices], fb, 6)
            expected_local = ids[rank::2][layout["local_token_indices"]]
            padded_local = torch.cat(
                [expected_local, ids.new_zeros(layout["pad_rows"])]
            )
            torch.testing.assert_close(packed.chunk(2)[rank], padded_local)
            torch.testing.assert_close(packed[cp.gather_index], ids[indices])
        torch.testing.assert_close(packed, torch.tensor([104, 110, 0, 103, 105, 109]))

    def test_single_dp_slot_still_uses_full_cp_physical_rows(self):
        fb = _batch()
        fb.global_num_tokens_padded_cpu = [24]
        cp = SimpleNamespace(max_rank_len=[3, 3])
        with patch.object(
            torch.cuda, "is_current_stream_capturing", return_value=False
        ):
            self.assertEqual(replay.late_layer_dp_counts(fb, 2, cp), [6])

    def test_unchanged_non_cp_counts_skip_handoff(self):
        fb = _batch()
        fb.global_num_tokens_padded_cpu = [24]
        with patch.object(
            torch.cuda, "is_current_stream_capturing", return_value=False
        ):
            self.assertIsNone(replay.late_layer_dp_counts(fb, 24))

    def test_short_prefill_and_idle_peers_agree_on_handoff(self):
        # No tail: the short active prompt keeps its early physical slot. The
        # idle peers must nevertheless make the SAME handoff decision as CP.
        early = [0, 24, 0, 0]
        decisions = []
        for rank in range(8):
            dp_rank, cp_rank = divmod(rank, 2)
            fb = _batch()
            fb.global_num_tokens_padded_cpu = early
            cp = SimpleNamespace(max_rank_len=[12, 12]) if dp_rank == 1 else None
            parallel = SimpleNamespace(
                tp_size=8,
                attn_cp_size=2,
                tp_group=SimpleNamespace(cpu_group=None),
            )

            def gather(dst, src, **kwargs):
                # Accommodate the refresh bit alongside each physical count.
                values = torch.tensor(
                    [[n, int(i == 1)] for i, n in enumerate(early) for _ in range(2)]
                )
                dst.copy_(values[:, : src.numel()].reshape_as(dst))

            with (
                patch.object(replay, "get_parallel", return_value=parallel),
                patch.object(replay, "all_gather_single", side_effect=gather),
                patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
            ):
                decisions.append(replay.late_layer_dp_counts(fb, 12 if cp else 0, cp))
        self.assertEqual(decisions, [early] * 8)

    def test_moe_cp_dp_gather_combine_are_symmetric(self):
        # Exercise the actual decoder method with an identity expert. Each DP
        # slot must be distinct before the TP sum, then select DP before CP.
        for dp_rank in range(4):
            for cp_rank in range(2):
                fb = _batch()
                fb.num_token_non_padded = None
                rows = [8, 4, 2, 0]
                full = torch.arange(sum(rows), dtype=torch.float32).view(-1, 1)
                start = sum(rows[:dp_rank])
                cp_full = full[start : start + rows[dp_rank]].clone()
                parallel = SimpleNamespace(
                    attn_dp_size=4,
                    attn_cp_size=2,
                    attn_cp_rank=cp_rank,
                    tp_group=SimpleNamespace(all_reduce=lambda x: x * 8),
                )
                layer = SimpleNamespace(
                    mlp=lambda x, *args, **kwargs: x / 8,
                    config=SimpleNamespace(model_type="deepseek_v41"),
                )

                def gather(dst, src, batch):
                    torch.testing.assert_close(src, cp_full)
                    dst.copy_(full)

                def scatter(dst, src, batch):
                    torch.testing.assert_close(src, full)
                    dst.copy_(src[start : start + rows[dp_rank]])

                with (
                    patch.object(model, "get_parallel", return_value=parallel),
                    patch.object(model, "is_cp_active", return_value=True),
                    patch.object(
                        model,
                        "get_moe_a2a_backend",
                        return_value=SimpleNamespace(is_none=lambda: True),
                    ),
                    patch.object(
                        model, "attn_cp_interleave_gather", return_value=cp_full
                    ),
                    # The handoff derives its slot width from the batch, not from
                    # the local tensor, so pin both to this DP rank's row count.
                    patch.object(
                        model,
                        "_cp_moe_slot_rows",
                        side_effect=lambda batch: rows[dp_rank],
                    ),
                    patch.object(
                        model, "sum_post_experts_output", side_effect=lambda x: x * 8
                    ),
                    patch.object(
                        model,
                        "get_global_dp_buffer",
                        side_effect=lambda _: torch.empty_like(full),
                    ),
                    patch.object(model, "dp_gather_replicate", side_effect=gather),
                    patch.object(model, "dp_scatter", side_effect=scatter),
                    patch(
                        "sglang.srt.layers.cp.interleave.get_parallel",
                        return_value=parallel,
                    ),
                    patch.object(
                        model,
                        "get_forward",
                        return_value=SimpleNamespace(scoped=lambda **kw: nullcontext()),
                    ),
                ):
                    output = model.DeepseekV4DecoderLayer._run_moe_ffn_dp_sync(
                        layer,
                        cp_full.chunk(2)[cp_rank],
                        fb,
                        input_ids=None,
                        input_ids_global=None,
                    )
                torch.testing.assert_close(output, cp_full.chunk(2)[cp_rank])


if __name__ == "__main__":
    unittest.main()
