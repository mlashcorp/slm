"""
Minimal FP8 training for NanoChat — tensorwise dynamic scaling only.

Based on Karpathy's nanochat implementation:
https://github.com/karpathy/nanochat/blob/master/nanochat/fp8.py

This is a drop-in replacement for torchao's Float8Linear (~2000 lines)
with ~150 lines. We only need the "tensorwise" recipe (one scalar scale per tensor).

How FP8 Training Works
======================

A standard Linear layer does one matmul in forward and two in backward:
    forward:  output = input @ weight.T
    backward:  grad_input = grad_output @ weight
               grad_weight = grad_output.T @ input

FP8 training wraps each of these three matmuls with:
1. Compute scale = FP8_MAX / max(|tensor|) for each operand
2. Quantize: fp8_tensor = clamp(tensor * scale, -FP8_MAX, FP8_MAX).to(fp8)
3. Matmul via torch._scaled_mm (cuBLAS FP8 kernel, ~2x faster than bf16)
4. Dequantize: _scaled_mm handles this internally using the inverse scales

FP8 Dtype Choice
================

There are two FP8 formats:
- float8_e4m3fn: 4-bit exponent, 3-bit mantissa, range [-448, 448]
  Higher precision (more mantissa bits), used for input and weight.
- float8_e5m2: 5-bit exponent, 2-bit mantissa, range [-57344, 57344]
  Wider range (more exponent bits), used for gradients which can be large.

Hardware Requirements
=====================

FP8 tensor cores require:
- H100+ (Hopper) or RTX 5090 (Blackwell) for native FP8 support
- torch._scaled_mm kernel for cuBLAS FP8 matmul

RTX 5090 (Blackwell) supports FP8 natively.

Usage
=====

    from src.training.fp8 import Float8Linear, convert_to_float8_training

    # Convert model to FP8
    def fp8_filter(mod, fqn):
        if not isinstance(mod, nn.Linear):
            return False
        if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
            return False
        if min(mod.in_features, mod.out_features) < 128:
            return False
        return True

    convert_to_float8_training(model, module_filter_fn=fp8_filter)
"""
import torch
import torch.nn as nn
from typing import Optional, Callable

# Avoid division by zero
EPS = 1e-12

# Compute dtype (bf16 on modern GPUs)
COMPUTE_DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float32


@torch.no_grad()
def _to_fp8(x: torch.Tensor, fp8_dtype: torch.dtype) -> tuple:
    """Dynamically quantize a tensor to FP8 using tensorwise scaling.

    "Tensorwise" means one scalar scale for the entire tensor (as opposed
    to "rowwise" which computes a separate scale per row).

    Args:
        x: Input tensor (any shape)
        fp8_dtype: Either torch.float8_e4m3fn or torch.float8_e5m2

    Returns:
        (fp8_data, inverse_scale) tuple for use with torch._scaled_mm
    """
    fp8_max = torch.finfo(fp8_dtype).max

    # Compute max absolute value across entire tensor
    amax = x.float().abs().max()

    # Scale maps [0, amax] -> [0, fp8_max]
    # Use float64 for consistent numerics between compile and eager
    scale = fp8_max / amax.double().clamp(min=EPS)
    scale = scale.float()

    # Quantize: scale into FP8 range, saturate, cast
    x_scaled = x.float() * scale
    x_clamped = x_scaled.clamp(-fp8_max, fp8_max)
    x_fp8 = x_clamped.to(fp8_dtype)

    # _scaled_mm expects the *inverse* scale
    inv_scale = scale.reciprocal()

    return x_fp8, inv_scale


def _to_col_major(x: torch.Tensor) -> torch.Tensor:
    """Rearrange a 2D tensor's memory to column-major layout.

    torch._scaled_mm requires its second operand in column-major layout.
    """
    return x.t().contiguous().t()


@torch._dynamo.allow_in_graph
class _Float8Matmul(torch.autograd.Function):
    """Custom autograd for the three FP8 GEMMs of a Linear layer.

    Forward: quantizes input and weight to FP8, saves for backward.

    Backward: two GEMMs:
    1. grad_input = grad_output @ weight
    2. grad_weight = grad_output.T @ input
    """

    @staticmethod
    def forward(ctx, input_2d: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # Quantize both operands to e4m3 (higher precision format)
        input_fp8, input_inv = _to_fp8(input_2d, torch.float8_e4m3fn)
        weight_fp8, weight_inv = _to_fp8(weight, torch.float8_e4m3fn)

        # Save for backward
        ctx.save_for_backward(input_fp8, input_inv, weight_fp8, weight_inv)

        # output = input @ weight.T
        # input_fp8 is [B, K] contiguous = row-major (good for first arg)
        # weight_fp8 is [N, K] contiguous, so weight_fp8.t() is [K, N] column-major
        output = torch._scaled_mm(
            input_fp8,
            weight_fp8.t(),
            scale_a=input_inv,
            scale_b=weight_inv,
            out_dtype=input_2d.dtype,
            use_fast_accum=True,  # Faster, slightly less accurate
        )

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        in_fp8, in_inv, w_fp8, w_inv = ctx.saved_tensors

        # GEMM 1: grad_input = grad_output @ weight
        # Shapes: [B, N] @ [N, K] -> [B, K]
        # Gradients use e5m2 (wider range), weights use e4m3
        go_fp8, go_inv = _to_fp8(grad_output, torch.float8_e5m2)

        # go_fp8 is row-major, w_fp8 needs column-major
        w_col = _to_col_major(w_fp8)

        grad_input = torch._scaled_mm(
            go_fp8,
            w_col,
            scale_a=go_inv,
            scale_b=w_inv,
            out_dtype=grad_output.dtype,
            use_fast_accum=False,  # More precise gradients
        )

        # GEMM 2: grad_weight = grad_output.T @ input
        # Shapes: [N, B] @ [B, K] -> [N, K]
        go_T = go_fp8.t().contiguous()  # Row-major
        in_col = _to_col_major(in_fp8)  # Column-major

        grad_weight = torch._scaled_mm(
            go_T,
            in_col,
            scale_a=go_inv,
            scale_b=in_inv,
            out_dtype=grad_output.dtype,
            use_fast_accum=False,
        )

        return grad_input, grad_weight


class Float8Linear(nn.Linear):
    """Drop-in nn.Linear replacement that does FP8 compute.

    Weights and biases remain in their original precision (e.g., fp32/bf16).
    Only the matmul is performed in FP8 via the _Float8Matmul autograd function.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Cast input to compute dtype
        input = input.to(COMPUTE_DTYPE)

        # _scaled_mm only works on 2D tensors
        orig_shape = input.shape
        input_2d = input.reshape(-1, orig_shape[-1])

        output = _Float8Matmul.apply(input_2d, self.weight)

        output = output.reshape(*orig_shape[:-1], output.shape[-1])

        if self.bias is not None:
            output = output + self.bias.to(output.dtype)

        return output

    @classmethod
    def from_float(cls, mod: nn.Linear) -> "Float8Linear":
        """Create Float8Linear from nn.Linear, sharing the same weight and bias.

        Uses meta device to avoid allocating a temporary weight tensor.
        """
        with torch.device("meta"):
            new_mod = cls(mod.in_features, mod.out_features, bias=mod.bias is not None)
            new_mod.weight = mod.weight
            new_mod.bias = mod.bias
        return new_mod


class Float8LinearConfig:
    """Minimal config for FP8 linear layers."""

    @staticmethod
    def from_recipe_name(recipe_name: str) -> "Float8LinearConfig":
        if recipe_name != "tensorwise":
            raise ValueError(
                f"Only 'tensorwise' recipe is supported, got '{recipe_name}'. "
                f"Rowwise/axiswise recipes require the full torchao library."
            )
        return Float8LinearConfig()


def convert_to_float8_training(
    module: nn.Module,
    *,
    config: Optional[Float8LinearConfig] = None,
    module_filter_fn: Optional[Callable[[nn.Module, str], bool]] = None,
) -> nn.Module:
    """Replace nn.Linear layers with Float8Linear throughout a module.

    Walks the module tree in post-order (children before parents) and swaps
    each nn.Linear that passes the optional filter. The new Float8Linear
    shares the original weight and bias tensors — no copies, no extra memory.

    Args:
        module: Root module to convert.
        config: Float8LinearConfig (only tensorwise supported).
        module_filter_fn: Optional filter(module, fqn) -> bool.

    Returns:
        Module with Float8Linear layers.
    """
    def _convert(mod: nn.Module, prefix: str = ""):
        for name, child in mod.named_children():
            fqn = f"{prefix}.{name}" if prefix else name
            _convert(child, fqn)

            if isinstance(child, nn.Linear) and not isinstance(child, Float8Linear):
                if module_filter_fn is None or module_filter_fn(child, fqn):
                    setattr(mod, name, Float8Linear.from_float(child))

    _convert(module)
    return module


def check_fp8_support() -> tuple[bool, str]:
    """Check if FP8 training is supported on current hardware.

    Returns:
        (supported, reason) tuple
    """
    if not torch.cuda.is_available():
        return False, "CUDA not available"

    device_capability = torch.cuda.get_device_capability()
    major, minor = device_capability

    # Hopper (H100) = SM 90
    # Blackwell (RTX 5090) = SM 100+
    if major >= 9:
        return True, f"GPU supports FP8 (SM {major}{minor})"
    else:
        return False, f"GPU does not support FP8 (SM {major}{minor}, need SM 90+)"


if __name__ == "__main__":
    # Test FP8 support
    supported, reason = check_fp8_support()
    print(f"FP8 Support: {supported} - {reason}")

    # Test FP8 matmul
    if supported and torch.cuda.is_available():
        device = torch.device("cuda")

        # Create test tensors
        B, K, N = 16, 512, 256
        x = torch.randn(B, K, device=device, dtype=COMPUTE_DTYPE)
        w = torch.randn(N, K, device=device, dtype=COMPUTE_DTYPE)

        # Test FP8 matmul
        try:
            x_fp8, x_inv = _to_fp8(x, torch.float8_e4m3fn)
            w_fp8, w_inv = _to_fp8(w, torch.float8_e4m3fn)
            out = torch._scaled_mm(
                x_fp8, w_fp8.t(),
                scale_a=x_inv, scale_b=w_inv,
                out_dtype=COMPUTE_DTYPE,
                use_fast_accum=True,
            )
            print(f"FP8 matmul test passed: input {x.shape} @ weight {w.shape} -> output {out.shape}")
        except Exception as e:
            print(f"FP8 matmul test failed: {e}")
