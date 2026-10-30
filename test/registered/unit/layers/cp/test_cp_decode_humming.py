"""CPU regressions using the installed Humming schema/meta/padding API.

Only the CUDA layout transform is replaced. Numerical checks decode canonical
FP8 words; they do not claim to exercise a CUDA Humming GEMM or graph replay.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from humming.layer import HummingMethod
from humming.schema import HummingInputSchema, HummingWeightSchema
from humming.schema.fp8 import Fp8WeightSchema

from sglang.srt.layers.cp import cp_decode_attn_tp as cp
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.quantization.humming import HummingLinearMethod
from sglang.srt.layers.quantization.humming import _lazy_import_humming


def context(rank):
    with patch.object(
        cp,
        "get_parallel",
        return_value=SimpleNamespace(
            enable_cp_decode_attn_tp=True, attn_cp_size=2, attn_cp_rank=rank
        ),
    ):
        return cp.CpDecodeAttnTpContext()


def linear(cls, n=768, k=768, schema=None, bias=False):
    layer = cls.__new__(cls)
    torch.nn.Module.__init__(layer)
    layer.output_partition_sizes = [n]
    layer.output_partition_sizes_sum = n
    layer.output_size_per_partition = n
    layer.input_size_per_partition = k
    layer.param_dtype = torch.bfloat16
    layer.with_bias = bias
    layer.use_decode_attn_tp = False
    layer.register_buffer("locks", torch.zeros(1024, dtype=torch.int32))
    schema = schema or HummingWeightSchema(
        b_dtype="float8e4m3", weight_scale_group_size=128
    )
    attrs = schema.get_tensors_attrs(n, k, torch.bfloat16, has_bias=bias)
    for name, spec in attrs.items():
        shape = spec["shape"]
        if name == "weight":
            # Distinct N/K values expose a wrong rank, axis or packed-word cut.
            values = (torch.arange(n * k).reshape(n, k) % 29 - 14).float()
            data = values.to(torch.float8_e4m3fn).view(torch.int32)
            # Other schemas only exercise slicing/validation, not numerics.
            if tuple(data.shape) != tuple(shape):
                data = torch.arange(torch.tensor(shape).prod()).reshape(shape).int()
        else:
            data = (torch.arange(torch.tensor(shape).prod()).reshape(shape) % 7 + 1).to(
                spec["dtype"]
            )
        layer.register_parameter(name, torch.nn.Parameter(data, requires_grad=False))
    return layer, schema


class TestCpDecodeHumming(unittest.TestCase):
    def setUp(self):
        _lazy_import_humming()
        self.capability = patch.object(
            torch.cuda, "get_device_capability", return_value=(9, 0)
        )
        self.capability.start()
        self.addCleanup(self.capability.stop)
        self.input_schema = HummingInputSchema(a_dtype="bfloat16")
        self.canonical = []

        def transform(layer):
            # Exercise REAL Humming validation and independent padding. Use an
            # intentionally non-sliceable layout surrogate for the CUDA pack.
            meta = layer.humming_metas[""]
            tensors = dict(layer.state_dict())
            self.canonical.append((meta, {k: v.clone() for k, v in tensors.items()}))
            padded = HummingMethod.check_and_pad_tensors(meta, tensors)
            for name, tensor in padded.items():
                if name == "weight":
                    tensor = tensor.t().contiguous()
                setattr(layer, name, torch.nn.Parameter(tensor, requires_grad=False))

        self.transform = patch.object(
            HummingMethod, "transform_humming_layer", side_effect=transform
        )
        self.transform.start()
        self.addCleanup(self.transform.stop)

    def prepare(self, ctx, layer, schema):
        ctx.prepare_humming_decode_shard(layer, schema, self.input_schema)

    def test_column_and_row_rank_numerics_padding(self):
        torch.manual_seed(17)
        for cls in (ColumnParallelLinear, RowParallelLinear):
            pieces = []
            for rank in range(2):
                with self.subTest(cls=cls, rank=rank):
                    ctx = context(rank)
                    layer, schema = linear(cls)
                    full = layer.weight.view(torch.float8_e4m3fn).float()
                    full *= layer.weight_scale.float().repeat_interleave(128, 1)
                    x = torch.randn(3, full.shape[1])
                    self.prepare(ctx, layer, schema)
                    meta, raw = self.canonical[-1]
                    local = raw["weight"].view(torch.float8_e4m3fn).float()
                    local *= raw["weight_scale"].float().repeat_interleave(128, 1)
                    if cls is ColumnParallelLinear:
                        expected = full.chunk(2, 0)[rank]
                        out = x @ local.t()
                        torch.testing.assert_close(
                            out, (x @ full.t()).chunk(2, 1)[rank]
                        )
                        self.assertEqual((meta.shape_n, meta.pad_shape_n), (512, 128))
                    else:
                        expected = full.chunk(2, 1)[rank]
                        out = x.chunk(2, 1)[rank] @ local.t()
                        self.assertEqual((meta.shape_k, meta.pad_shape_k), (384, 0))
                    torch.testing.assert_close(local, expected)
                    pieces.append(local)
                    # Preparation must not mutate the full canonical layer.
                    self.assertEqual(tuple(layer.weight.shape), (768, 192))
            reconstructed = torch.cat(pieces, 0 if cls is ColumnParallelLinear else 1)
            torch.testing.assert_close(reconstructed, full)
            if cls is RowParallelLinear:
                reduced = sum(x.chunk(2, 1)[r] @ pieces[r].t() for r in range(2))
                torch.testing.assert_close(reduced, x @ full.t(), rtol=1e-5, atol=1e-3)

    def test_complete_state_restore_exception_and_stable_pointers(self):
        ctx = context(1)
        layers = []
        for cls in (ColumnParallelLinear, RowParallelLinear):
            layer, schema = linear(cls, bias=True)
            self.prepare(ctx, layer, schema)
            HummingMethod.prepare_layer_meta(
                layer,
                768,
                768,
                schema,
                self.input_schema,
                pad_n_to_multiple=256,
                pad_k_to_multiple=128,
                torch_dtype=torch.bfloat16,
                has_bias=True,
            )
            HummingMethod.transform_humming_layer(layer)
            layers.append(layer)
        # Real wo_a is unquantized, uses plain BMM and must remain generic.
        wo_a, _ = linear(ColumnParallelLinear)
        wo_a.weight = torch.nn.Parameter(torch.randn(768, 768), requires_grad=False)
        del wo_a.weight_scale
        layers.insert(1, wo_a)
        attn = SimpleNamespace(tp_q_head_num=16)
        originals = [dict(l._parameters) for l in layers]
        metas = [getattr(l, "humming_metas", None) for l in layers]
        ptrs = []
        with patch.object(
            ctx,
            "set_decode_attn_tp",
            side_effect=lambda _: setattr(ctx, "use_decode_attn_tp", True),
        ):
            for iteration in range(4):
                with self.assertRaisesRegex(RuntimeError, "body"):
                    with ctx.maybe_use_decode_attn_tp(None, layers, radix_attn=attn):
                        current = []
                        for l in (layers[0], layers[2]):
                            state = l._cp_decode_humming_state
                            for name, value in state.items():
                                self.assertIs(getattr(l, name), value)
                            current.append(
                                tuple(
                                    v.data_ptr()
                                    for v in state.values()
                                    if isinstance(v, torch.Tensor)
                                )
                            )
                        if iteration:
                            self.assertEqual(current, ptrs)
                        ptrs = current
                        self.assertEqual(attn.tp_q_head_num, 8)
                        self.assertTrue(layers[2].use_decode_attn_tp)
                        self.assertEqual(wo_a.weight.shape[0], 384)
                        raise RuntimeError("body")
                self.assertEqual(attn.tp_q_head_num, 16)
                self.assertFalse(layers[2].use_decode_attn_tp)
                for l, params, meta in zip(layers, originals, metas):
                    for name, param in params.items():
                        self.assertIs(getattr(l, name), param)
                    self.assertIs(getattr(l, "humming_metas", None), meta)
                    self.assertEqual(l.input_size_per_partition, 768)
                    self.assertEqual(l.output_size_per_partition, 768)
        self.assertEqual(wo_a.weight.shape, (768, 768))

    def test_scale_layouts_and_zero_points(self):
        for scale_type, group_n in (
            ("tensor", 0),
            ("channel", 0),
            ("group", 0),
            ("block", 128),
        ):
            for cls in (ColumnParallelLinear, RowParallelLinear):
                with self.subTest(scale_type=scale_type, cls=cls):
                    schema = HummingWeightSchema(
                        b_dtype="float8e4m3",
                        weight_scale_type=scale_type,
                        weight_scale_group_size=(
                            128 if scale_type in ("group", "block") else 0
                        ),
                        weight_scale_group_size_n=group_n,
                    )
                    layer, _ = linear(cls, schema=schema)
                    self.prepare(context(1), layer, schema)
                    scale = self.canonical[-1][1]["weight_scale"]
                    original = layer.weight_scale
                    if cls is ColumnParallelLinear and scale_type != "tensor":
                        expected = original.chunk(2, 0)[1]
                    elif cls is RowParallelLinear and scale_type in ("group", "block"):
                        expected = original.chunk(2, 1)[1]
                    else:
                        expected = original
                    torch.testing.assert_close(scale, expected)
        for cls in (ColumnParallelLinear, RowParallelLinear):
            schema = HummingWeightSchema(
                b_dtype="uint4",
                weight_scale_group_size=128,
                has_zero_point=True,
                weight_scale_2_type="channel",
            )
            layer, _ = linear(cls, schema=schema)
            self.prepare(context(1), layer, schema)
            raw = self.canonical[-1][1]
            axis = 0 if cls is ColumnParallelLinear else 1
            torch.testing.assert_close(
                raw["zero_point"], layer.zero_point.chunk(2, axis)[1]
            )
            expected = (
                layer.weight_scale_2.chunk(2, 0)[1]
                if axis == 0
                else layer.weight_scale_2
            )
            torch.testing.assert_close(raw["weight_scale_2"], expected)

    def test_reject_split_scale_block_before_transform(self):
        layer, schema = linear(RowParallelLinear, k=384)
        with self.assertRaisesRegex(ValueError, "scale block"):
            self.prepare(context(0), layer, schema)
        self.assertEqual(len(self.canonical), 0)

    def test_missing_preparation_fails_closed(self):
        ctx = context(0)
        layer, _ = linear(ColumnParallelLinear)
        layer.humming_metas = {"": object()}
        ptr = layer.weight.data_ptr()
        with patch.object(
            ctx,
            "set_decode_attn_tp",
            side_effect=lambda _: setattr(ctx, "use_decode_attn_tp", True),
        ):
            with self.assertRaisesRegex(RuntimeError, "not prepared"):
                with ctx.maybe_use_decode_attn_tp(None, [layer]):
                    self.fail("unreachable")
        self.assertEqual(layer.weight.data_ptr(), ptr)

    def test_row_k_padding_is_local_not_sliced_full_padding(self):
        schema = HummingWeightSchema(b_dtype="float8e4m3", weight_scale_group_size=32)
        layer, _ = linear(RowParallelLinear, k=320, schema=schema)
        self.prepare(context(1), layer, schema)
        meta, raw = self.canonical[-1]
        self.assertEqual((meta.shape_k, meta.pad_shape_k), (256, 96))
        self.assertEqual(raw["weight"].shape, (768, 40))
        self.assertEqual(raw["weight_scale"].shape, (768, 5))
        torch.testing.assert_close(raw["weight"], layer.weight.chunk(2, 1)[1])
        torch.testing.assert_close(
            raw["weight_scale"], layer.weight_scale.chunk(2, 1)[1]
        )

    def test_prefill_preserves_full_state_and_disabled_skips_prepare(self):
        ctx = context(0)
        layer, schema = linear(ColumnParallelLinear)
        layer.humming_metas = {"": object()}
        weight, metas = layer.weight, layer.humming_metas
        # Both CP-prefill detectors must bypass decode state activation.
        for active, dsa in ((True, False), (False, True)):
            with patch.object(cp, "is_cp_active", return_value=active), patch.object(
                cp, "dsa_use_prefill_cp", return_value=dsa
            ):
                with ctx.maybe_use_decode_attn_tp(None, [layer]):
                    self.assertIs(layer.weight, weight)
                    self.assertIs(layer.humming_metas, metas)
                    self.assertEqual(layer.output_size_per_partition, 768)
        ctx.decode_tp_size = None
        self.prepare(ctx, layer, schema)
        self.assertFalse(hasattr(layer, "_cp_decode_humming_state"))

    def test_process_loading_hook_real_fp8_conversion(self):
        for cls in (ColumnParallelLinear, RowParallelLinear):
            layer, _ = linear(cls)
            layer._cp_decode_attn_tp_shard = True
            layer.is_fallback = False
            layer.weight = torch.nn.Parameter(
                layer.weight.view(torch.float8_e4m3fn), requires_grad=False
            )
            del layer.weight_scale
            method = HummingLinearMethod.__new__(HummingLinearMethod)
            # Current checkpoint uses 32x32 FP8 scales, expanded along N by
            # convert_humming, not sliced as if they were packed GEMM scales.
            layer.weight_scale_inv = torch.nn.Parameter(
                torch.ones(24, 24), requires_grad=False
            )
            method.weight_schema = Fp8WeightSchema(weight_block_size=(32, 32))
            from humming.schema.fp8 import Fp8InputSchema

            method.input_schema = Fp8InputSchema()
            method.force_weight_schema = None
            with patch.object(cp, "get_cp_decode_attn_tp_ctx", return_value=context(1)):
                method.process_weights_after_loading(layer)
            self.assertIn("humming_metas", layer._cp_decode_humming_state)
            self.assertEqual(layer.humming_metas[""].shape_n, 768)
            local_meta = layer._cp_decode_humming_state["humming_metas"][""]
            self.assertEqual(
                (
                    local_meta.shape_n
                    if cls is ColumnParallelLinear
                    else local_meta.shape_k
                ),
                512 if cls is ColumnParallelLinear else 384,
            )


if __name__ == "__main__":
    unittest.main()
