import os
import torch

import triton
import triton.language as tl

# 让 Triton 在 CPU 后端运行
triton.runtime.driver.set_active_to_cpu()
USE_GPU = False

def get_add_kernel_autotune_config():
    configs = []
    # 这里按“分块”思路：每个 program 处理一段长度为 BLOCK_SIZE 的向量
    for BLOCK_SIZE in [64, 128, 256, 512, 1024]:
        configs.append(triton.Config({"BLOCK_SIZE": BLOCK_SIZE}))
    if os.getenv("ENABLE_AUTOTUNING") == "add_kernel":
        assert len(configs) > 1
        return configs
    # 默认给一个配置
    return [triton.Config({"BLOCK_SIZE": 256})]


@triton.autotune(
    configs=get_add_kernel_autotune_config(),
    key=[],
)
@triton.jit
def add_kernel(
    a_ptr, b_ptr, c_ptr,
    n_elements,
    stride_a, stride_b, stride_c,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    # 本 program 负责的 element 索引范围：pid*BLOCK_SIZE ... pid*BLOCK_SIZE+BLOCK_SIZE-1
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # 支持任意 stride（但要求 contiguous 的话 stride 就是 1）
    a = tl.load(a_ptr + offsets * stride_a, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets * stride_b, mask=mask, other=0.0)
    c = a + b
    tl.store(c_ptr + offsets * stride_c, c, mask=mask)


def add(a: torch.Tensor, b: torch.Tensor):
    assert a.shape == b.shape, "a and b must have the same shape"
    assert a.is_contiguous(), "Tensor a must be contiguous"
    assert b.is_contiguous(), "Tensor b must be contiguous"
    assert a.device == b.device, "a and b must be on the same device"
    assert a.dtype == b.dtype, "a and b must have the same dtype"

    # 拉平成 1D 做元素加法（分块处理）
    a1 = a.reshape(-1)
    b1 = b.reshape(-1)
    n = a1.numel()

    c1 = torch.empty_like(a1)

    grid = lambda META: (triton.cdiv(n, META["BLOCK_SIZE"]),)
    add_kernel[grid](
        a1, b1, c1,
        n,
        a1.stride(0), b1.stride(0), c1.stride(0),
    )
    return c1.reshape(a.shape)


def test_add():
    torch.manual_seed(0)
    # 用一些不整齐的长度来测试 mask 是否正确
    a = torch.randn((179, 167), device="cpu", dtype=torch.float32)
    b = torch.randn((179, 167), device="cpu", dtype=torch.float32)

    triton_output = add(a, b)
    torch_output = a + b

    assert torch.allclose(triton_output, torch_output, atol=0, rtol=0), "❌ Triton and Torch differ"
    print("✅ Triton and Torch match")


# test_add()
