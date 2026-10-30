"""CPU regression: CP+DPA MoE row identity for the early (pre-tail) handoff.

The early handoff runs the MoE on a DP-gathered buffer whose rows must line up
with two independently produced tensors:

* the hash-routing token IDs (built from the prompt), and
* the CP-gathered hidden states (built from the rank-local shard),

one DP slot per attention-DP rank, each slot a rank-major CP buffer
``[cp0 rows | cp1 rows | ...]``. This test builds synthetic per-DP/per-CP token
IDs with unequal lengths and checks, for every ``(dp, cp, local row)``, that the
gathered IDs and the gathered hidden rows hold the same token, and that the DP
slot offsets tile the gathered buffer exactly.

No ``torch.distributed`` and no GPU: the CP all-gather / DP gather / DP scatter
are the plain tensor copies those collectives perform.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.cp import interleave as il
from sglang.srt.layers.cp import padding as cppadding
from sglang.srt.layers.cp.interleave import (
    InterleaveContextParallelMetadata,
    InterleaveCPStrategy,
    cp_fit_slot_rows,
    cp_interleave_local_input_ids,
    cp_sequence_to_interleave_order,
    cp_take_slot_rows,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

CP_SIZE = 2
# Unequal prompts per attention-DP rank: includes an odd length (CP padding),
# a length below the CP size, and one that pads a whole CP block.
EXTEND_LENS = [23, 11, 5, 1]


def _parallel(cp_rank: int, cp_size: int = CP_SIZE):
    return SimpleNamespace(attn_cp_size=cp_size, attn_cp_rank=cp_rank)


def _metadata(total: int) -> InterleaveContextParallelMetadata:
    """Real interleave metadata, including the physical shard padding."""
    strategy = InterleaveCPStrategy(cp_size=CP_SIZE)
    metadata = strategy.build_metadata(
        num_tokens=total, seqs_len=[total], extend_seqs_len=[total]
    )
    with patch.object(cppadding, "get_parallel", return_value=_parallel(0)):
        cppadding.pad_logical_token_to_physical(metadata)
    return metadata


def _batch(prompt: torch.Tensor, metadata: InterleaveContextParallelMetadata):
    return SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        attn_cp_metadata=metadata,
        input_ids=prompt,
    )


def _local_rows(token_ids: torch.Tensor, metadata, cp_rank: int) -> torch.Tensor:
    """This CP rank's shard, in the interleave order the gather concatenates.

    Every real row holds its own token ID, so row identity is checkable from the
    value; shard padding is zero, exactly as ``shard_hidden_states`` pads.
    """
    logical = metadata.per_rank_logical_token[cp_rank]
    physical = metadata.per_rank_actual_token[cp_rank]
    shard = token_ids[cp_rank::CP_SIZE][:logical]
    assert shard.shape[0] == logical
    out = token_ids.new_zeros((physical,))
    out[:logical] = shard
    return out


def _cp_gather(locals_) -> torch.Tensor:
    """What ``attn_cp_interleave_gather`` returns: rank-major, one block per rank."""
    assert len(locals_) == CP_SIZE
    return torch.cat(list(locals_))


def _dp_gather(local_slots, slot_rows):
    """``dp_gather_replicate``, zero-filled, with only CP rank 0 writing."""
    offsets = [sum(slot_rows[:d]) for d in range(len(slot_rows))]
    out = torch.full((sum(slot_rows),), -1, dtype=local_slots[0].dtype)
    for d, local in enumerate(local_slots):
        out[offsets[d] : offsets[d] + slot_rows[d]] = local[: slot_rows[d]]
    return out, offsets


class TestCpDpMoeRowIdentity(unittest.TestCase):
    def setUp(self):
        self.cases = []
        for dp_rank, total in enumerate(EXTEND_LENS):
            prompt = torch.arange(total, dtype=torch.int64) + 1000 * (dp_rank + 1)
            metadata = _metadata(total)
            batch = _batch(prompt, metadata)
            # Slot width this DP rank contributes: the padded host count that
            # ``prepare_mlp_sync_batch`` publishes for interleave CP.
            slot_rows = sum(metadata.per_rank_actual_token)
            locals_ = []
            slot_ids = []
            for cp_rank in range(CP_SIZE):
                with patch.object(
                    il, "get_parallel", return_value=_parallel(cp_rank)
                ):
                    local = _local_rows(prompt, metadata, cp_rank)
                    locals_.append(local)
                    slot_ids.append(
                        cp_interleave_local_input_ids(batch, local.shape[0], slot_rows)
                    )
            with patch.object(il, "get_parallel", return_value=_parallel(0)):
                slot_hidden = cp_fit_slot_rows(_cp_gather(locals_), batch, slot_rows)
            self.cases.append(
                dict(
                    dp_rank=dp_rank,
                    prompt=prompt,
                    metadata=metadata,
                    slot_rows=slot_rows,
                    locals=locals_,
                    slot_ids=slot_ids,
                    slot_hidden=slot_hidden,
                )
            )

    def test_gathered_ids_match_hidden_rows_for_every_row(self):
        """Every (dp, cp, local row) resolves to the same token in both."""

        for case in self.cases:
            metadata, slot_rows = case["metadata"], case["slot_rows"]
            block = slot_rows // CP_SIZE
            self.assertEqual(case["slot_hidden"].shape[0], slot_rows)
            for cp_rank in range(CP_SIZE):
                ids = case["slot_ids"][cp_rank]
                self.assertEqual(ids.shape[0], slot_rows)
                # The two DP-gathered tensors are row-for-row identical.
                torch.testing.assert_close(ids, case["slot_hidden"], rtol=0, atol=0)
                logical = metadata.per_rank_logical_token[cp_rank]
                for row in range(block):
                    self.assertEqual(
                        ids[cp_rank * block + row].item(),
                        case["slot_hidden"][cp_rank * block + row].item(),
                        f"dp={case['dp_rank']} cp={cp_rank} row={row}",
                    )
                # This CP rank's real rows are its interleave shard, in order.
                torch.testing.assert_close(
                    case["slot_hidden"][cp_rank * block : cp_rank * block + logical],
                    case["prompt"][cp_rank::CP_SIZE][:logical],
                    rtol=0,
                    atol=0,
                )
                # Its block padding stays zero, so pad rows cannot hash as
                # token 0 and cannot route as a real token.
                torch.testing.assert_close(
                    case["slot_hidden"][
                        cp_rank * block + logical : (cp_rank + 1) * block
                    ],
                    torch.zeros(block - logical, dtype=torch.int64),
                    rtol=0,
                    atol=0,
                )

    def test_slot_offsets_tile_the_global_buffer(self):
        """Each (dp, cp, local row) lands at its own slot offset, exactly once."""

        slot_rows = [case["slot_rows"] for case in self.cases]
        gathered, offsets = _dp_gather(
            [case["slot_hidden"] for case in self.cases], slot_rows
        )
        self.assertEqual(sum(slot_rows), gathered.numel())
        self.assertEqual(offsets, [0, 24, 36, 44])

        seen = torch.zeros(gathered.numel(), dtype=torch.bool)
        for case, dp_offset in zip(self.cases, offsets):
            metadata = case["metadata"]
            block = case["slot_rows"] // CP_SIZE
            self.assertEqual(dp_offset, offsets[case["dp_rank"]])
            for cp_rank in range(CP_SIZE):
                logical = metadata.per_rank_logical_token[cp_rank]
                slot_block = dp_offset + cp_rank * block
                torch.testing.assert_close(
                    gathered[slot_block : slot_block + logical],
                    case["prompt"][cp_rank::CP_SIZE][:logical],
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    gathered[slot_block + logical : slot_block + block],
                    torch.zeros(block - logical, dtype=torch.int64),
                    rtol=0,
                    atol=0,
                )
                seen[slot_block : slot_block + block] = True
        self.assertTrue(
            seen.all().item(), "every gathered row belongs to exactly one slot block"
        )

    def test_narrow_slot_keeps_every_cp_block(self):
        """A slot narrower than the gather must trim per block, not at the head.

        This is the captured-prefill-bucket geometry: the local CP buffer is
        wider than the DP slot the batch publishes. A head slice keeps only CP
        rank 0's rows and silently drops the other ranks' real rows, which is
        what ``memcpy``'s ``min(slot, local)`` truncation used to do.
        """

        total = 23
        prompt = torch.arange(total, dtype=torch.int64) + 1
        metadata = _metadata(total)
        batch = _batch(prompt, metadata)
        slot_rows = sum(metadata.per_rank_actual_token)  # 24
        wide_rows = slot_rows + 2 * CP_SIZE  # a wider captured block
        locals_ = []
        for cp_rank in range(CP_SIZE):
            local = torch.zeros(wide_rows)
            local[: metadata.per_rank_logical_token[cp_rank]] = prompt[cp_rank::CP_SIZE]
            locals_.append(local)
        gathered = _cp_gather(locals_)
        with patch.object(il, "get_parallel", return_value=_parallel(0)):
            fitted = cp_fit_slot_rows(gathered, batch, slot_rows)
        head_slice = gathered[:slot_rows]
        block = slot_rows // CP_SIZE
        for cp_rank in range(CP_SIZE):
            expected = locals_[cp_rank][:block]
            torch.testing.assert_close(
                fitted[cp_rank * block : (cp_rank + 1) * block],
                expected,
                rtol=0,
                atol=0,
            )
            if cp_rank:
                self.assertFalse(
                    torch.equal(
                        head_slice[cp_rank * block : (cp_rank + 1) * block],
                        expected,
                    ),
                    "a head slice must not be mistaken for the correct fit",
                )

    def test_wide_slot_pads_every_cp_block(self):
        """A slot wider than the gather pads each block, keeping rows aligned."""

        total = 5
        prompt = torch.arange(total, dtype=torch.int64) + 7
        metadata = _metadata(total)
        batch = _batch(prompt, metadata)
        gathered_rows = sum(metadata.per_rank_actual_token)  # 8
        slot_rows = gathered_rows + 2 * CP_SIZE
        locals_ = []
        for cp_rank in range(CP_SIZE):
            local = torch.zeros(gathered_rows // CP_SIZE, dtype=prompt.dtype)
            local[: metadata.per_rank_logical_token[cp_rank]] = prompt[cp_rank::CP_SIZE]
            locals_.append(local)
        with patch.object(il, "get_parallel", return_value=_parallel(0)):
            fitted = cp_fit_slot_rows(_cp_gather(locals_), batch, slot_rows)
        self.assertEqual(fitted.shape[0], slot_rows)
        block = slot_rows // CP_SIZE
        for cp_rank in range(CP_SIZE):
            log = metadata.per_rank_logical_token[cp_rank]
            torch.testing.assert_close(
                fitted[cp_rank * block : cp_rank * block + log],
                prompt[cp_rank::CP_SIZE][:log],
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                fitted[cp_rank * block + log : (cp_rank + 1) * block],
                torch.zeros(block - log, dtype=fitted.dtype),
                rtol=0,
                atol=0,
            )
            self.assertEqual(fitted.dtype, prompt.dtype)
    def test_slot_narrower_than_a_logical_shard_fails_loudly(self):
        """Silently truncating real rows must be impossible."""

        total = 23
        prompt = torch.arange(total, dtype=torch.int64) + 1
        metadata = _metadata(total)
        batch = _batch(prompt, metadata)
        gathered = torch.zeros(sum(metadata.per_rank_actual_token))
        with patch.object(il, "get_parallel", return_value=_parallel(0)):
            with self.assertRaises(ValueError):
                # 2 rows per CP block cannot hold the 12/11-row logical shards.
                cp_fit_slot_rows(gathered, batch, 2 * CP_SIZE)

    def test_take_slot_rows_round_trip(self):
        """The scattered slot comes back as this rank's rows, padding zeroed."""

        for case in self.cases:
            metadata = case["metadata"]
            for cp_rank in range(CP_SIZE):
                with patch.object(
                    il, "get_parallel", return_value=_parallel(cp_rank)
                ):
                    back = cp_take_slot_rows(
                        case["slot_ids"][cp_rank].clone(),
                        _batch(case["prompt"], metadata),
                        metadata.per_rank_actual_token[cp_rank],
                    )
                logical = metadata.per_rank_logical_token[cp_rank]
                self.assertEqual(
                    back.shape[0], metadata.per_rank_actual_token[cp_rank]
                )
                torch.testing.assert_close(
                    back[:logical],
                    case["prompt"][cp_rank::CP_SIZE][:logical],
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    back[logical:],
                    torch.zeros(back.numel() - logical, dtype=back.dtype),
                    rtol=0,
                    atol=0,
                )

    def test_full_prompt_fallback_matches_input_ids_global(self):
        """The prefill-graph fallback (no input_ids_global) uses the same map."""

        for case in self.cases:
            metadata, slot_rows = case["metadata"], case["slot_rows"]
            batch = _batch(case["prompt"], metadata)
            with patch.object(il, "get_parallel", return_value=_parallel(0)):
                fallback = cp_interleave_local_input_ids(
                    batch, metadata.per_rank_actual_token[0], slot_rows
                )
                prebuilt = cp_sequence_to_interleave_order(
                    case["prompt"], batch, slot_rows
                )
                batch.input_ids_global = prebuilt
                preferred = cp_interleave_local_input_ids(
                    batch, metadata.per_rank_actual_token[0], slot_rows
                )
            torch.testing.assert_close(fallback, prebuilt, rtol=0, atol=0)
            torch.testing.assert_close(preferred, prebuilt, rtol=0, atol=0)
            # And the fallback equals the CP-shard gather for this rank.
            torch.testing.assert_close(fallback, case["slot_hidden"], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
