import itertools
import random
import unittest
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from sglang.srt.layers.moe.ep_moe.kernels import (
    grouped_gemm_triton,
    post_reorder_triton_kernel,
    pre_reorder_triton_kernel,
    run_moe_ep_preproess,
    silu_and_mul_triton_kernel,
)
from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE, use_deep_gemm
from sglang.srt.layers.moe.topk import select_experts
from sglang.test.test_utils import CustomTestCase


# For test
def ep_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    renormalize: bool,
    # ep config
    num_experts: int = 256,
    fp8_dtype: torch.types = torch.float8_e4m3fn,
    num_experts_per_partition: int = 128,
    start_expert_id: int = 0,
    end_expert_id: int = 127,
    use_grouped_topk: bool = False,
    num_expert_group: Optional[int] = None,
    topk_group: Optional[int] = None,
    custom_routing_function: Optional[Callable] = None,
    use_fp8_w8a8: bool = False,
    w1_scale_inv: Optional[torch.Tensor] = None,
    w2_scale_inv: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
):
    use_blockwise_fp8 = block_shape is not None
    topk_weights, topk_ids = select_experts(
        hidden_states=hidden_states,
        router_logits=router_logits,
        top_k=top_k,
        use_grouped_topk=use_grouped_topk,
        renormalize=renormalize,
        topk_group=topk_group,
        num_expert_group=num_expert_group,
        # correction_bias=correction_bias, #skip this in test
        custom_routing_function=custom_routing_function,
    )

    reorder_topk_ids, src2dst, seg_indptr = run_moe_ep_preproess(topk_ids, num_experts)

    gateup_input = torch.empty(
        (int(hidden_states.shape[0] * top_k), hidden_states.shape[1]),
        device=hidden_states.device,
        dtype=(
            fp8_dtype
            if (use_fp8_w8a8 and not use_blockwise_fp8)
            else hidden_states.dtype
        ),
    )

    if use_fp8_w8a8 and not use_blockwise_fp8:
        max_value = (
            torch.max(hidden_states).repeat(num_experts_per_partition).to(torch.float32)
        )
        w1_input_scale = max_value / torch.finfo(fp8_dtype).max
    else:
        w1_input_scale = None

    # PreReorder
    pre_reorder_triton_kernel[(hidden_states.shape[0],)](
        hidden_states,
        gateup_input,
        src2dst,
        topk_ids,
        w1_input_scale,
        start_expert_id,
        end_expert_id,
        top_k,
        hidden_states.shape[1],
        BLOCK_SIZE=512,
    )

    seg_indptr_cur_rank = seg_indptr[start_expert_id : end_expert_id + 2]
    weight_indices_cur_rank = torch.arange(
        0,
        num_experts_per_partition,
        device=hidden_states.device,
        dtype=torch.int64,
    )

    # GroupGemm-0
    gateup_output = torch.empty(
        gateup_input.shape[0],
        w1.shape[1],
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    gateup_output = grouped_gemm_triton(
        a=gateup_input,
        b=w1,
        c=gateup_output,
        batch_size=num_experts_per_partition,
        weight_column_major=True,
        seg_indptr=seg_indptr_cur_rank,
        weight_indices=weight_indices_cur_rank,
        use_fp8_w8a8=use_fp8_w8a8,
        scale_a=w1_input_scale,
        scale_b=w1_scale_inv,
        block_shape=block_shape,
    )

    # Act
    down_input = torch.empty(
        gateup_output.shape[0],
        gateup_output.shape[1] // 2,
        device=gateup_output.device,
        dtype=(
            fp8_dtype
            if (use_fp8_w8a8 and not use_blockwise_fp8)
            else hidden_states.dtype
        ),
    )
    if use_fp8_w8a8 and not use_blockwise_fp8:
        w2_input_scale = torch.ones(
            num_experts_per_partition,
            dtype=torch.float32,
            device=hidden_states.device,
        )
    else:
        w2_input_scale = None

    silu_and_mul_triton_kernel[(gateup_output.shape[0],)](
        gateup_output,
        down_input,
        gateup_output.shape[1],
        reorder_topk_ids,
        w2_input_scale,
        start_expert_id,
        end_expert_id,
        BLOCK_SIZE=512,
    )

    # GroupGemm-1
    down_output = torch.empty(
        down_input.shape[0],
        w2.shape[1],
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    down_output = grouped_gemm_triton(
        a=down_input,
        b=w2,
        c=down_output,
        batch_size=num_experts_per_partition,
        weight_column_major=True,
        seg_indptr=seg_indptr_cur_rank,
        weight_indices=weight_indices_cur_rank,
        use_fp8_w8a8=use_fp8_w8a8,
        scale_a=w2_input_scale,
        scale_b=w2_scale_inv,
        block_shape=block_shape,
    )

    # PostReorder
    output = torch.empty_like(hidden_states)
    post_reorder_triton_kernel[(hidden_states.size(0),)](
        down_output,
        output,
        src2dst,
        topk_ids,
        topk_weights,
        start_expert_id,
        end_expert_id,
        top_k,
        hidden_states.size(1),
        BLOCK_SIZE=512,
    )
    return output


# test util
def block_dequant(
    x_q_block: torch.Tensor,
    x_s: torch.Tensor,
    block_size: List[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """This function converts block-wise quantization to tensor-wise quantization.
    The inputs are block-wise quantization tensor `x_q_block`, block-wise quantization scale
    and the block size.
    The outputs are tensor-wise quantization tensor and tensor-wise quantization scale.
    Note only float8 is supported for now.
    """

    # process 3D tensor
    if x_q_block.dim() == 3:
        batch_size = x_q_block.size(0)
        return torch.stack(
            [block_dequant(x_q_block[b], x_s[b], block_size) for b in range(batch_size)]
        )

    block_n, block_k = block_size[0], block_size[1]
    n, k = x_q_block.shape
    n_tiles = (n + block_n - 1) // block_n
    k_tiles = (k + block_k - 1) // block_k
    assert n_tiles == x_s.shape[0]
    assert k_tiles == x_s.shape[1]

    x_dq_block = x_q_block.to(torch.float32)

    x_dq_block_tiles = [
        [
            x_dq_block[
                j * block_n : min((j + 1) * block_n, n),
                i * block_k : min((i + 1) * block_k, k),
            ]
            for i in range(k_tiles)
        ]
        for j in range(n_tiles)
    ]

    for i in range(k_tiles):
        for j in range(n_tiles):
            x_dq_block_tiles[j][i][:, :] = x_dq_block_tiles[j][i] * x_s[j][i]

    return x_dq_block


class TestW8A8BlockFP8EPMoE(CustomTestCase):
    DTYPES = [torch.half, torch.bfloat16]
    M = [1, 222, 1024, 2048]
    N = [128, 1024, 2048]
    K = [256, 4096, 5120]
    E = [8, 16]
    ep_size = [2, 4]
    TOP_KS = [2, 4]
    BLOCK_SIZE = [[128, 128]]
    SEEDS = [0]

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _w8a8_block_fp8_ep_moe(
        self, M, N, K, E, ep_size, topk, block_size, dtype, seed
    ):
        torch.manual_seed(seed)
        random.seed(seed)
        # NOTE(HandH1998): to avoid overflow when out_dtype = torch.half
        factor_for_scale = 1e-2
        fp8_info = torch.finfo(torch.float8_e4m3fn)
        fp8_max, fp8_min = fp8_info.max, fp8_info.min

        a = torch.randn((M, K), dtype=dtype) / 10

        w1_fp32 = (torch.rand((E, 2 * N, K), dtype=dtype) - 0.5) * 2 * fp8_max
        w1 = w1_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

        w2_fp32 = (torch.rand((E, K, N), dtype=dtype) - 0.5) * 2 * fp8_max
        w2 = w2_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

        block_n, block_k = block_size[0], block_size[1]
        n_tiles_w1 = (2 * N + block_n - 1) // block_n
        n_tiles_w2 = (K + block_n - 1) // block_n
        k_tiles_w1 = (K + block_k - 1) // block_k
        k_tiles_w2 = (N + block_k - 1) // block_k

        w1_s = (
            torch.rand((E, n_tiles_w1, k_tiles_w1), dtype=torch.float32)
            * factor_for_scale
        )
        w2_s = (
            torch.rand((E, n_tiles_w2, k_tiles_w2), dtype=torch.float32)
            * factor_for_scale
        )

        w1_ref = block_dequant(w1, w1_s, block_size).to(dtype)
        w2_ref = block_dequant(w2, w2_s, block_size).to(dtype)

        score = torch.randn((M, E), dtype=dtype)
        num_experts_per_partition = E // ep_size
        cur_rank = random.randint(0, ep_size - 1)
        start_id = cur_rank * num_experts_per_partition
        end_id = start_id + num_experts_per_partition - 1

        with torch.inference_mode():
            out = ep_moe(
                hidden_states=a,
                w1=w1,
                w2=w2,
                router_logits=score,
                top_k=topk,
                renormalize=False,
                use_fp8_w8a8=True,
                w1_scale_inv=w1_s,
                w2_scale_inv=w2_s,
                block_shape=block_size,
                num_experts=E,
                num_experts_per_partition=num_experts_per_partition,
                start_expert_id=start_id,
                end_expert_id=end_id,
            )
            ref_out = ep_moe(
                hidden_states=a,
                w1=w1_ref,
                w2=w2_ref,
                router_logits=score,
                top_k=topk,
                renormalize=False,
                use_fp8_w8a8=False,
                w1_scale_inv=None,
                w2_scale_inv=None,
                block_shape=None,
                num_experts=E,
                num_experts_per_partition=num_experts_per_partition,
                start_expert_id=start_id,
                end_expert_id=end_id,
            )
        self.assertTrue(
            torch.mean(torch.abs(out.to(torch.float32) - ref_out.to(torch.float32)))
            / (torch.mean(torch.abs(ref_out.to(torch.float32))) + 1e-6)
            < 0.06
        )

    def test_w8a8_block_fp8_ep_moe(self):
        for params in itertools.product(
            self.M,
            self.N,
            self.K,
            self.E,
            self.ep_size,
            self.TOP_KS,
            self.BLOCK_SIZE,
            self.DTYPES,
            self.SEEDS,
        ):
            with self.subTest(
                M=params[0],
                N=params[1],
                K=params[2],
                E=params[3],
                ep_size=params[4],
                topk=params[5],
                block_size=params[6],
                dtype=params[7],
                seed=params[8],
            ):
                self._w8a8_block_fp8_ep_moe(*params)
            torch.cuda.empty_cache()


class TestDeepEPMoEContiguous(CustomTestCase):
    """Test the contiguous forward method of DeepEPMoE."""
    
    # Test with smaller sizes for regular testing
    DTYPES = [torch.bfloat16]
    M = [64, 128, 512]
    N = [128, 256]
    K = [256, 4096]
    E = [8, 24]
    TOP_KS = [2, 4]
    BLOCK_SIZE = [[128, 128]]
    SEEDS = [0]
    
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        if not use_deep_gemm:
            raise unittest.SkipTest("deep_gemm is not available")
        torch.set_default_device("cuda")
    
    def _test_deepep_contiguous(self, M, N, K, E, topk, block_size, dtype, seed):
        """Test DeepEPMoE contiguous forward against normal forward."""
        torch.manual_seed(seed)
        
        # Set up scaling factor to avoid overflow
        factor_for_scale = 1e-2
        fp8_info = torch.finfo(torch.float8_e4m3fn)
        fp8_max, fp8_min = fp8_info.max, fp8_info.min
        
        # Create input tensor
        hidden_states = torch.randn((M, K), dtype=dtype) / 10
        
        # Create expert weights
        w13_fp32 = (torch.rand((E, 2 * N, K), dtype=torch.float32) - 0.5) * 2 * fp8_max
        w13 = w13_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
        
        w2_fp32 = (torch.rand((E, K, N), dtype=torch.float32) - 0.5) * 2 * fp8_max
        w2 = w2_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
        
        # Create block-wise scales
        block_n, block_k = block_size[0], block_size[1]
        n_tiles_w13 = (2 * N + block_n - 1) // block_n
        n_tiles_w2 = (K + block_n - 1) // block_n
        k_tiles_w13 = (K + block_k - 1) // block_k
        k_tiles_w2 = (N + block_k - 1) // block_k
        
        w13_scale = torch.rand((E, n_tiles_w13, k_tiles_w13), dtype=torch.float32) * factor_for_scale
        w2_scale = torch.rand((E, n_tiles_w2, k_tiles_w2), dtype=torch.float32) * factor_for_scale
        
        # Create router logits
        router_logits = torch.randn((M, E), dtype=dtype)
        
        # Create DeepEPMoE instance
        from sglang.srt.layers.quantization.fp8 import Fp8Config
        from sglang.srt.utils import DeepEPMode
        
        quant_config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            weight_block_size=block_size,
            activation_scheme="dynamic"
        )
        
        moe = DeepEPMoE(
            num_experts=E,
            top_k=topk,
            hidden_size=K,
            intermediate_size=N,
            params_dtype=dtype,
            quant_config=quant_config,
            activation="silu",
            deepep_mode=DeepEPMode.low_latency
        )
        
        # Set weights manually
        moe.w13_weight.data = w13
        moe.w2_weight.data = w2
        moe.w13_weight_scale_inv.data = w13_scale
        moe.w2_weight_scale_inv.data = w2_scale
        
        # Get topk experts
        topk_weights, topk_ids = select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=topk,
            renormalize=False
        )
        
        # Prepare inputs for normal forward
        reorder_topk_ids, src2dst, seg_indptr = run_moe_ep_preproess(topk_ids, E)
        
        # Prepare inputs for contiguous forward
        from sglang.srt.layers.quantization.fp8_kernel import per_token_group_quant_fp8
        hidden_states_fp8, scale = per_token_group_quant_fp8(hidden_states, block_k)
        
        with torch.inference_mode():
            # Run normal forward
            normal_output = moe.forward_normal(
                hidden_states=hidden_states_fp8,
                reorder_topk_ids=reorder_topk_ids,
                seg_indptr=seg_indptr
            )
            
            # Run contiguous forward
            contiguous_output = moe.forward_deepgemm_contiguous(
                hidden_states_fp8=(hidden_states_fp8, scale),
                reorder_topk_ids=reorder_topk_ids,
                seg_indptr=seg_indptr
            )
        
        # Compare outputs
        normal_output = normal_output.to(torch.float32)
        contiguous_output = contiguous_output.to(torch.float32)
        
        # Calculate relative error
        rel_error = torch.mean(torch.abs(contiguous_output - normal_output)) / (torch.mean(torch.abs(normal_output)) + 1e-6)
        
        # Assert that the relative error is small
        self.assertTrue(rel_error < 0.05, f"Relative error {rel_error} is too large")
    
    def test_deepep_contiguous(self):
        """Run tests with different parameters."""
        for params in itertools.product(
            self.M,
            self.N,
            self.K,
            self.E,
            self.TOP_KS,
            self.BLOCK_SIZE,
            self.DTYPES,
            self.SEEDS,
        ):
            with self.subTest(
                M=params[0],
                N=params[1],
                K=params[2],
                E=params[3],
                topk=params[4],
                block_size=params[5],
                dtype=params[6],
                seed=params[7],
            ):
                self._test_deepep_contiguous(*params)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main(verbosity=2)
