"""NanoFlash: experimental CUSTOM Triton CUDA FlashAttention kernels for SmolVLM.

Implements memory-efficient tiled online-softmax attention, plus recompute-based
custom CUDA backward kernels (no torch SDPA, flash-attn library, or eager attention
fallback in the production path). Triton JIT-compiles the kernels for NVIDIA GPUs.

Contract:
    Q [B,Hq,M,D], K/V [B,Hkv,N,D], dtype float16/bfloat16, D in {32,64,128}.
    GQA Hq/Hkv must be integral. mask optional additive/boolean [B,1|Hq,M,N].
    Zero attention dropout; optional bottom-right aligned causal attention.
This is research software, not a production-optimized FlashAttention replacement.
"""
from __future__ import annotations

import math
import torch
import triton
import triton.language as tl

# Transformers reserves backend names containing "flash" for Hub FlashAttention loaders.
# This name is only a registry key; the custom CUDA/Triton kernels are unchanged.
BACKEND_NAME = "nano_cuda_attn"
CALLS = {"forward": 0, "backward": 0, "vision": 0, "decoder": 0, "dtype_casts": 0}


@triton.jit
def _read_mask(
    mask, b, h, rows, cols,
    MS0: tl.constexpr, MS1: tl.constexpr, MS2: tl.constexpr, MS3: tl.constexpr,
    MASK_H: tl.constexpr, HAS_MASK: tl.constexpr, MASK_BOOL: tl.constexpr,
):
    if HAS_MASK:
        hm = tl.where(MASK_H == 1, 0, h)
        ptrs = mask + b * MS0 + hm * MS1 + rows[:, None] * MS2 + cols[None, :] * MS3
        raw = tl.load(ptrs, mask=(rows[:, None] >= 0) & (cols[None, :] >= 0), other=0)
        if MASK_BOOL:
            return tl.where(raw, 0.0, -float('inf'))
        else:
            return tl.where(raw.to(tl.float32) < -1.0e4, -float('inf'), raw.to(tl.float32))
    else:
        return tl.full((rows.shape[0], cols.shape[0]), 0.0, tl.float32)


@triton.jit
def _fwd(Q, K, V, MASK, O, LSE,
         B: tl.constexpr, HQ: tl.constexpr, HK: tl.constexpr,
         M: tl.constexpr, N: tl.constexpr, D: tl.constexpr, G: tl.constexpr,
         SCALE: tl.constexpr, CAUSAL: tl.constexpr,
         HAS_MASK: tl.constexpr, MASK_BOOL: tl.constexpr, MASK_H: tl.constexpr,
         MS0: tl.constexpr, MS1: tl.constexpr, MS2: tl.constexpr, MS3: tl.constexpr,
         BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
    mi, hi, bi = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    hk = hi // G
    rr = mi * BM + tl.arange(0, BM)
    cc = tl.arange(0, BN)
    dd = tl.arange(0, BD)
    q_ptr = Q + ((bi * HQ + hi) * M + rr[:, None]) * D + dd[None, :]
    q = tl.load(q_ptr, (rr[:, None] < M) & (dd[None, :] < D), other=0)
    maximum = tl.full((BM,), -float('inf'), tl.float32)
    denom = tl.zeros((BM,), tl.float32)
    numerator = tl.zeros((BM, BD), tl.float32)

    for start in range(tl.cdiv(N, BN)):
        nn = start * BN + cc
        kp = K + ((bi * HK + hk) * N + nn[None, :]) * D + dd[:, None]
        vp = V + ((bi * HK + hk) * N + nn[:, None]) * D + dd[None, :]
        k = tl.load(kp, (nn[None, :] < N) & (dd[:, None] < D), other=0)
        v = tl.load(vp, (nn[:, None] < N) & (dd[None, :] < D), other=0)
        scores = tl.dot(q, k).to(tl.float32) * SCALE
        valid = (rr[:, None] < M) & (nn[None, :] < N)
        if CAUSAL:
            valid = valid & (nn[None, :] <= rr[:, None] + (N - M))
        if HAS_MASK:
            # Address outside mask dimensions is clamped before loading.
            sr = tl.minimum(rr, M - 1)
            sc = tl.minimum(nn, N - 1)
            bias = _read_mask(MASK, bi, hi, sr, sc, MS0, MS1, MS2, MS3, MASK_H, HAS_MASK, MASK_BOOL)
            scores = scores + bias
        scores = tl.where(valid, scores, -float('inf'))
        new_max = tl.maximum(maximum, tl.max(scores, axis=1))
        # Avoid NaN for fully masked rows.
        alpha = tl.where(maximum == -float('inf'), 0.0, tl.exp(maximum - new_max))
        p = tl.exp(tl.where(new_max[:, None] == -float('inf'), -float('inf'), scores - new_max[:, None]))
        denom = denom * alpha + tl.sum(p, axis=1)
        numerator = numerator * alpha[:, None] + tl.dot(p.to(q.dtype), v)
        maximum = new_max

    o = numerator / tl.maximum(denom[:, None], 1.0e-20)
    op = O + ((bi * HQ + hi) * M + rr[:, None]) * D + dd[None, :]
    lp = LSE + (bi * HQ + hi) * M + rr
    tl.store(op, o, (rr[:, None] < M) & (dd[None, :] < D))
    lse = tl.where(denom > 0.0, maximum + tl.log(denom), -float('inf'))
    tl.store(lp, lse, rr < M)


@triton.jit
def _dq(Q, K, V, MASK, O, DO, LSE, DQ,
        HQ: tl.constexpr, HK: tl.constexpr, M: tl.constexpr, N: tl.constexpr,
        D: tl.constexpr, G: tl.constexpr, SCALE: tl.constexpr, CAUSAL: tl.constexpr,
        HAS_MASK: tl.constexpr, MASK_BOOL: tl.constexpr, MASK_H: tl.constexpr,
        MS0: tl.constexpr, MS1: tl.constexpr, MS2: tl.constexpr, MS3: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
    mi, hi, bi = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    hk = hi // G
    rr = mi * BM + tl.arange(0, BM)
    cc = tl.arange(0, BN)
    dd = tl.arange(0, BD)
    qp = Q + ((bi * HQ + hi) * M + rr[:, None]) * D + dd[None, :]
    op = O + ((bi * HQ + hi) * M + rr[:, None]) * D + dd[None, :]
    dop = DO + ((bi * HQ + hi) * M + rr[:, None]) * D + dd[None, :]
    q = tl.load(qp, (rr[:, None] < M) & (dd[None, :] < D), other=0)
    o = tl.load(op, (rr[:, None] < M) & (dd[None, :] < D), other=0).to(tl.float32)
    do = tl.load(dop, (rr[:, None] < M) & (dd[None, :] < D), other=0)
    lse = tl.load(LSE + (bi * HQ + hi) * M + rr, rr < M, other=-float('inf'))
    delta = tl.sum(o * do.to(tl.float32), axis=1)
    dq = tl.zeros((BM, BD), dtype=tl.float32)
    for start in range(tl.cdiv(N, BN)):
        nn = start * BN + cc
        kp = K + ((bi * HK + hk) * N + nn[None, :]) * D + dd[:, None]
        vp = V + ((bi * HK + hk) * N + nn[:, None]) * D + dd[None, :]
        k = tl.load(kp, (nn[None, :] < N) & (dd[:, None] < D), other=0)
        v = tl.load(vp, (nn[:, None] < N) & (dd[None, :] < D), other=0)
        scores = tl.dot(q, k).to(tl.float32) * SCALE
        valid = (rr[:, None] < M) & (nn[None, :] < N)
        if CAUSAL:
            valid = valid & (nn[None, :] <= rr[:, None] + (N - M))
        if HAS_MASK:
            bias = _read_mask(MASK, bi, hi, tl.minimum(rr, M - 1), tl.minimum(nn, N - 1),
                              MS0, MS1, MS2, MS3, MASK_H, HAS_MASK, MASK_BOOL)
            scores += bias
        scores = tl.where(valid, scores, -float('inf'))
        p = tl.exp(tl.where(lse[:, None] == -float('inf'), -float('inf'), scores - lse[:, None]))
        dp = tl.dot(do, tl.trans(v)).to(tl.float32)
        ds = p * (dp - delta[:, None]) * SCALE
        dq += tl.dot(ds.to(q.dtype), tl.trans(k)).to(tl.float32)
    dqp = DQ + ((bi * HQ + hi) * M + rr[:, None]) * D + dd[None, :]
    tl.store(dqp, dq, (rr[:, None] < M) & (dd[None, :] < D))


@triton.jit
def _dkdv(Q, K, V, MASK, O, DO, LSE, DK, DV,
          HQ: tl.constexpr, HK: tl.constexpr, M: tl.constexpr, N: tl.constexpr,
          D: tl.constexpr, G: tl.constexpr, SCALE: tl.constexpr, CAUSAL: tl.constexpr,
          HAS_MASK: tl.constexpr, MASK_BOOL: tl.constexpr, MASK_H: tl.constexpr,
          MS0: tl.constexpr, MS1: tl.constexpr, MS2: tl.constexpr, MS3: tl.constexpr,
          BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
    ni, hk, bi = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rr = tl.arange(0, BM)
    nn = ni * BN + tl.arange(0, BN)
    dd = tl.arange(0, BD)
    kp = K + ((bi * HK + hk) * N + nn[None, :]) * D + dd[:, None]
    vp = V + ((bi * HK + hk) * N + nn[:, None]) * D + dd[None, :]
    k = tl.load(kp, (nn[None, :] < N) & (dd[:, None] < D), other=0)
    v = tl.load(vp, (nn[:, None] < N) & (dd[None, :] < D), other=0)
    dk = tl.zeros((BN, BD), tl.float32)
    dv = tl.zeros((BN, BD), tl.float32)
    for g in range(G):
        hi = hk * G + g
        for tile in range(tl.cdiv(M, BM)):
            qq = tile * BM + rr
            qp = Q + ((bi * HQ + hi) * M + qq[:, None]) * D + dd[None, :]
            op = O + ((bi * HQ + hi) * M + qq[:, None]) * D + dd[None, :]
            dop = DO + ((bi * HQ + hi) * M + qq[:, None]) * D + dd[None, :]
            q = tl.load(qp, (qq[:, None] < M) & (dd[None, :] < D), other=0)
            o = tl.load(op, (qq[:, None] < M) & (dd[None, :] < D), other=0).to(tl.float32)
            do = tl.load(dop, (qq[:, None] < M) & (dd[None, :] < D), other=0)
            lse = tl.load(LSE + (bi * HQ + hi) * M + qq, qq < M, other=-float('inf'))
            scores = tl.dot(q, k).to(tl.float32) * SCALE
            valid = (qq[:, None] < M) & (nn[None, :] < N)
            if CAUSAL:
                valid = valid & (nn[None, :] <= qq[:, None] + (N - M))
            if HAS_MASK:
                bias = _read_mask(MASK, bi, hi, tl.minimum(qq, M - 1), tl.minimum(nn, N - 1),
                                  MS0, MS1, MS2, MS3, MASK_H, HAS_MASK, MASK_BOOL)
                scores += bias
            scores = tl.where(valid, scores, -float('inf'))
            p = tl.exp(tl.where(lse[:, None] == -float('inf'), -float('inf'), scores - lse[:, None]))
            dp = tl.dot(do, tl.trans(v)).to(tl.float32)
            delta = tl.sum(o * do.to(tl.float32), axis=1)
            ds = p * (dp - delta[:, None]) * SCALE
            dk += tl.dot(tl.trans(ds.to(q.dtype)), q).to(tl.float32)
            dv += tl.dot(tl.trans(p.to(q.dtype)), do).to(tl.float32)
    dst = ((bi * HK + hk) * N + nn[:, None]) * D + dd[None, :]
    valid_store = (nn[:, None] < N) & (dd[None, :] < D)
    tl.store(DK + dst, dk, valid_store)
    tl.store(DV + dst, dv, valid_store)


class _NanoFlashFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, mask, scaling, causal):
        if not (q.is_cuda and k.is_cuda and v.is_cuda):
            raise RuntimeError("NanoFlash requires CUDA tensors; no silent SDPA fallback")
        if q.dtype not in (torch.float16, torch.bfloat16) or k.dtype != q.dtype or v.dtype != q.dtype:
            raise ValueError("NanoFlash needs matching FP16 or BF16 Q,K,V")
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or k.shape != v.shape:
            raise ValueError("Q/K/V expected [batch, heads, sequence, head_dim]")
        b, hq, m, d = q.shape
        bk, hk, n, kd = k.shape
        if b != bk or d != kd or hq % hk != 0 or d not in (32, 64, 128):
            raise ValueError(f"Unsupported Q={q.shape}, K={k.shape}: require same B/D, Hq multiple of Hkv, D=32/64/128")
        if m < 1 or n < 1:
            raise ValueError("Sequence lengths must be positive")
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        has_mask = mask is not None
        mask_bool = False
        mask_heads = 1
        strides = (0, 0, 0, 0)
        if has_mask:
            if not mask.is_cuda or mask.ndim != 4 or mask.shape[0] != b or mask.shape[2:] != (m, n) or mask.shape[1] not in (1, hq):
                raise ValueError(f"Mask must be CUDA [B,1 or Hq,M,N]; got {mask.shape}")
            mask_bool = mask.dtype == torch.bool
            if not mask_bool and mask.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                raise ValueError("Mask must be boolean or float16/bfloat16/float32 additive")
            mask_heads = mask.shape[1]
            strides = mask.stride()
        bm, bn, bd = 32, 64, d
        kw = dict(HQ=hq, HK=hk, M=m, N=n, D=d, G=hq // hk, SCALE=float(scaling),
                  CAUSAL=bool(causal), HAS_MASK=has_mask, MASK_BOOL=mask_bool,
                  MASK_H=mask_heads, MS0=strides[0], MS1=strides[1], MS2=strides[2], MS3=strides[3],
                  BM=bm, BN=bn, BD=bd)
        out = torch.empty_like(q)
        lse = torch.empty((b, hq, m), dtype=torch.float32, device=q.device)
        _fwd[(triton.cdiv(m, bm), hq, b)](q, k, v, mask if has_mask else q, out, lse, B=b, **kw, num_warps=4)
        ctx.save_for_backward(q, k, v, mask if has_mask else torch.empty(0, device=q.device), out, lse)
        ctx.kw = kw
        ctx.has_mask = has_mask
        CALLS["forward"] += 1
        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v, mask_placeholder, o, lse = ctx.saved_tensors
        mask = mask_placeholder if ctx.has_mask else q
        b, hq, m, d = q.shape
        hk, n = k.shape[1], k.shape[2]
        kw = ctx.kw
        do = do.contiguous()
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        _dq[(triton.cdiv(m, kw["BM"]), hq, b)](q, k, v, mask, o, do, lse, dq, **kw, num_warps=4)
        _dkdv[(triton.cdiv(n, kw["BN"]), hk, b)](q, k, v, mask, o, do, lse, dk, dv, **kw, num_warps=4)
        CALLS["backward"] += 1
        return dq, dk, dv, None, None, None


def flash_attention(q, k, v, attention_mask=None, *, is_causal=False, scaling=None, dropout_p=0.0):
    """Custom CUDA attention, returns [B,Hq,M,D]. No PyTorch SDPA calls."""
    if float(dropout_p) != 0.0:
        raise NotImplementedError("NanoFlash does not implement attention dropout; do not silently omit it")
    if scaling is None:
        scaling = 1.0 / math.sqrt(q.shape[-1])

    # PEFT's prepare_model_for_kbit_training() can upcast non-quantized
    # weights to FP32, especially vision projections. The Triton kernel
    # only supports matching FP16/BF16 Q,K,V. Explicitly cast *outside*
    # our custom autograd.Function so .to(...) and .to(original_dtype)
    # retain PyTorch gradient paths to the original FP32 projections/LoRA.
    # The arithmetic uses lower precision, by design: this is NOT FP32
    # attention and must be benchmarked against a reference.
    original_q_dtype = q.dtype
    supported = (torch.float16, torch.bfloat16)
    if not (q.dtype == k.dtype == v.dtype and q.dtype in supported):
        if not (q.is_floating_point() and k.is_floating_point() and v.is_floating_point()):
            raise TypeError("NanoFlash requires floating-point Q, K and V")
        if not (q.is_cuda and k.is_cuda and v.is_cuda):
            raise RuntimeError("NanoFlash requires CUDA tensors; no SDPA fallback")
        if torch.is_autocast_enabled("cuda"):
            compute_dtype = torch.get_autocast_dtype("cuda")
        else:
            compute_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported()
                             else torch.float16)
        if compute_dtype not in supported:
            raise TypeError(f"Unsupported autocast attention dtype: {compute_dtype}")
        q, k, v = (t.to(dtype=compute_dtype) for t in (q, k, v))
        CALLS["dtype_casts"] += 1

    result = _NanoFlashFunction.apply(
        q, k, v, attention_mask, scaling, is_causal
    )
    # Preserve the attention interface's expected output dtype so the
    # following o_proj linear layer remains compatible without autocast.
    return result.to(dtype=original_q_dtype) if result.dtype != original_q_dtype else result


def _hf_attention_forward(module, query, key, value, attention_mask=None, *, scaling=None,
                          dropout=0.0, is_causal=None, sliding_window=None, **kwargs):
    """Transformers AttentionInterface adapter; output [B,M,H*D]."""
    if sliding_window is not None:
        raise NotImplementedError("NanoFlash has no sliding-window mask support")
    if kwargs.get('output_attentions', False):
        raise NotImplementedError("NanoFlash does not materialize attention weights")
    result = flash_attention(
        query, key, value, attention_mask,
        is_causal=bool(is_causal) if is_causal is not None else bool(getattr(module, 'is_causal', False)),
        scaling=scaling, dropout_p=dropout,
    )
    kind = "vision" if "vision" in type(module).__name__.lower() else "decoder"
    CALLS[kind] += 1
    return result.transpose(1, 2).contiguous(), None


def register_nanoflash():
    """Register BOTH kernel and SDPA-style mask formatting for SmolVLM."""
    from transformers import AttentionInterface, AttentionMaskInterface
    from transformers.masking_utils import sdpa_mask
    AttentionInterface.register(BACKEND_NAME, _hf_attention_forward)
    AttentionMaskInterface.register(BACKEND_NAME, sdpa_mask)
    return BACKEND_NAME


def assert_kernel_ran(min_forward=1, min_backward=0):
    assert CALLS['forward'] >= min_forward, f"Kernel never ran: {CALLS}"
    assert CALLS['backward'] >= min_backward, f"Kernel backward never ran: {CALLS}"
    print("NanoFlash CUDA execution counts:", CALLS)


def numerical_self_test(device='cuda'):
    """Fail-fast parity tests: forward, backward, causal, masks, GQA, KV cache."""
    from torch.nn.functional import scaled_dot_product_attention as reference_only
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required for numerical tests')
    torch.manual_seed(104)
    testcases = [
        # (B,Hq,Hkv,M,N,D,causal,mask_type)
        (1, 4, 4, 128, 128, 64, False, None),
        (1, 4, 4, 128, 128, 64, True, None),
        (2, 4, 4, 65, 77, 64, False, 'float'),
        (1, 4, 4, 33, 49, 64, False, 'bool'),
        (1, 6, 2, 33, 33, 64, True, None),
        (1, 6, 2, 1, 73, 64, True, None),
        (1, 4, 4, 32, 64, 128, False, None),
        (1, 4, 2, 65, 65, 64, False, 'float'),
    ]
    for case_index, (b, hq, hk, m, n, d, causal, mt) in enumerate(testcases):
        dtype = torch.bfloat16 if case_index == len(testcases) - 1 else torch.float16
        q = (torch.randn((b,hq,m,d), device=device, dtype=dtype)*0.35).requires_grad_()
        k = (torch.randn((b,hk,n,d), device=device, dtype=dtype)*0.35).requires_grad_()
        v = (torch.randn((b,hk,n,d), device=device, dtype=dtype)*0.35).requires_grad_()
        mask = None
        if mt:
            visible = torch.ones((b,1,m,n), device=device, dtype=torch.bool)
            visible[..., -9:] = False
            visible[..., 0] = True
            mask = visible if mt == 'bool' else torch.where(visible, 0.0, -float('inf')).to(torch.float32)
        # Explicit reference repeat_interleave is used ONLY in the test, not in NanoFlash.
        kr = k.detach().repeat_interleave(hq//hk, dim=1).requires_grad_()
        vr = v.detach().repeat_interleave(hq//hk, dim=1).requires_grad_()
        qr = q.detach().requires_grad_()
        if causal and m != n:
            # Causal alignment when decoding with cached K/V uses bottom-right mask.
            shifted = torch.arange(n, device=device)[None, :] <= (torch.arange(m, device=device)[:,None] + n-m)
            shifted = shifted[None,None,:,:]
            ref_mask = shifted if mask is None else (shifted & mask if mask.dtype == torch.bool else mask.masked_fill(~shifted, -float('inf')))
            ref_causal = False
        else:
            ref_mask, ref_causal = mask, causal
        # SDPA additive masks must match the query dtype in current PyTorch.
        # Keep boolean masks boolean; preserve additive mask semantics.
        if ref_mask is not None and ref_mask.is_floating_point():
            ref_mask = ref_mask.to(dtype=qr.dtype)
        r = reference_only(qr,kr,vr,attn_mask=ref_mask,is_causal=ref_causal)
        x = flash_attention(q,k,v,mask,is_causal=causal)
        grad = torch.randn_like(x)*0.25
        (drq,drk,drv) = torch.autograd.grad(r,(qr,kr,vr),grad)
        (dxq,dxk,dxv) = torch.autograd.grad(x,(q,k,v),grad)
        drk = drk.view(b,hk,hq//hk,n,d).sum(dim=2)
        drv = drv.view(b,hk,hq//hk,n,d).sum(dim=2)
        metrics = [(a.float()-bb.float()).abs().max().detach().item() for a,bb in [(x,r),(dxq,drq),(dxk,drk),(dxv,drv)]]
        # FP16 dot accumulates using FP32; error increases with repeated low precision products.
        out_tol, grad_tol = (0.05, 0.10) if dtype == torch.bfloat16 else (0.025, 0.04)
        assert metrics[0] <= out_tol and max(metrics[1:]) <= grad_tol, (
            f'NanoFlash parity failure: shape {tuple(q.shape)}, {tuple(k.shape)}, errors={metrics}'
        )
        print(f'PASS dtype={dtype} causal={causal} mask={mt} Q={tuple(q.shape)} K={tuple(k.shape)} '
              f'max_abs_errors (out,dq,dk,dv)={[round(x,5) for x in metrics]}')
    assert_kernel_ran(min_forward=len(testcases), min_backward=len(testcases))
    return True
