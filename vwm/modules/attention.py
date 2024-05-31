import math
from inspect import isfunction
from typing import Any, Optional

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from packaging import version
from torch import nn
from torch.utils.checkpoint import checkpoint

if version.parse(torch.__version__) >= version.parse("2.0.0"):
    SDP_IS_AVAILABLE = True
    from torch.backends.cuda import SDPBackend, sdp_kernel

    BACKEND_MAP = {
        SDPBackend.MATH: {
            "enable_math": True,
            "enable_flash": False,
            "enable_mem_efficient": False
        },
        SDPBackend.FLASH_ATTENTION: {
            "enable_math": False,
            "enable_flash": True,
            "enable_mem_efficient": False
        },
        SDPBackend.EFFICIENT_ATTENTION: {
            "enable_math": False,
            "enable_flash": False,
            "enable_mem_efficient": True
        },
        None: {
            "enable_math": True,
            "enable_flash": True,
            "enable_mem_efficient": True
        }
    }
else:
    from contextlib import nullcontext

    SDP_IS_AVAILABLE = False
    sdp_kernel = nullcontext
    BACKEND_MAP = dict()
    print(
        f"No SDP backend available, likely because you are running in pytorch versions < 2.0. "
        f"In fact, you are using PyTorch {torch.__version__}. You might want to consider upgrading"
    )

try:
    import xformers
    import xformers.ops

    XFORMERS_IS_AVAILABLE = True
except:
    XFORMERS_IS_AVAILABLE = False
    print("No module `xformers`, processing without it")


def exists(val):
    return val is not None


def uniq(arr):
    return {el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    else:
        return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(
            self,
            dim,
            dim_out=None,
            mult=4,
            glu=False,
            dropout=0.0,
            zero_init=False
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = (
            nn.Sequential(
                nn.Linear(dim, inner_dim),
                nn.GELU()
            )
            if not glu
            else GEGLU(dim, inner_dim)
        )

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

        if zero_init:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """

    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(
            qkv, "b (qkv heads c) h w -> qkv b heads c (h w)", heads=self.heads, qkv=3
        )
        k = k.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        out = rearrange(out, "b heads c (h w) -> b (heads c) h w", heads=self.heads, h=h, w=w)
        return self.to_out(out)


class CrossAttention(nn.Module):  # not used, never mind
    def __init__(
            self,
            query_dim,
            context_dim=None,
            heads=8,
            dim_head=64,
            dropout=0.0,
            backend=None,
            zero_init=False,
            **kwargs
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        self.backend = backend

        if zero_init:
            nn.init.zeros_(self.to_out[0].weight)
            nn.init.zeros_(self.to_out[0].bias)

    def forward(
            self,
            x,
            context=None,
            mask=None,
            additional_tokens=None,
            n_times_crossframe_attn_in_self=0,
            **kwargs
    ):
        num_heads = self.heads

        if additional_tokens is not None:
            # get the number of masked tokens at the beginning of the output sequence
            n_tokens_to_mask = additional_tokens.shape[1]
            # add additional token
            x = torch.cat((additional_tokens, x), dim=1)

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        if n_times_crossframe_attn_in_self:
            # reprogramming cross-frame attention as in https://arxiv.org/abs/2303.13439
            assert x.shape[0] % n_times_crossframe_attn_in_self == 0
            n_cp = x.shape[0] // n_times_crossframe_attn_in_self
            k = repeat(
                k[::n_times_crossframe_attn_in_self], "b ... -> (b n) ...", n=n_cp
            )
            v = repeat(
                v[::n_times_crossframe_attn_in_self], "b ... -> (b n) ...", n=n_cp
            )

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=num_heads), (q, k, v))

        with sdp_kernel(**BACKEND_MAP[self.backend]):
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)  # scale is dim_head ** -0.5 per default

        del q, k, v
        out = rearrange(out, "b h n d -> b n (h d)", h=num_heads)

        if additional_tokens is not None:
            # remove additional token
            out = out[:, n_tokens_to_mask:]
        return self.to_out(out)


class MemoryEfficientCrossAttention(nn.Module):  # we are using this implementation
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(
            self,
            query_dim,
            context_dim=None,
            heads=8,
            dim_head=64,
            dropout=0.0,
            zero_init=False,
            causal=False,
            add_lora=False,
            lora_rank=16,
            lora_scale=1.0,
            action_control=False,
            **kwargs
    ):
        super().__init__()
        print(
            f"Setting up {self.__class__.__name__}. "
            f"Query dim is {query_dim}, "
            f"context_dim is {context_dim} and using {heads} heads with a dimension of {dim_head}"
        )
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        self.attention_op: Optional[Any] = None

        if causal:
            self.attn_bias = xformers.ops.LowerTriangularMask()
        else:
            self.attn_bias = None

        if zero_init:
            nn.init.zeros_(self.to_out[0].weight)
            nn.init.zeros_(self.to_out[0].bias)

        self.add_lora = add_lora
        if add_lora:
            self.lora_scale = lora_scale

            self.q_adapter_down = nn.Linear(query_dim, lora_rank, bias=False)
            nn.init.normal_(self.q_adapter_down.weight, std=1 / lora_rank)
            self.q_adapter_up = nn.Linear(lora_rank, inner_dim, bias=False)
            nn.init.zeros_(self.q_adapter_up.weight)

            self.k_adapter_down = nn.Linear(context_dim, lora_rank, bias=False)
            nn.init.normal_(self.k_adapter_down.weight, std=1 / lora_rank)
            self.k_adapter_up = nn.Linear(lora_rank, inner_dim, bias=False)
            nn.init.zeros_(self.k_adapter_up.weight)

            self.v_adapter_down = nn.Linear(context_dim, lora_rank, bias=False)
            nn.init.normal_(self.v_adapter_down.weight, std=1 / lora_rank)
            self.v_adapter_up = nn.Linear(lora_rank, inner_dim, bias=False)
            nn.init.zeros_(self.v_adapter_up.weight)

            self.out_adapter_down = nn.Linear(inner_dim, lora_rank, bias=False)
            nn.init.normal_(self.out_adapter_down.weight, std=1 / lora_rank)
            self.out_adapter_up = nn.Linear(lora_rank, query_dim, bias=False)
            nn.init.zeros_(self.out_adapter_up.weight)

        self.action_control = action_control
        if action_control:
            self.context_dim = context_dim
            self.k_adapter_action_control = nn.Linear(128 * 19, inner_dim, bias=False)
            nn.init.zeros_(self.k_adapter_action_control.weight)
            self.v_adapter_action_control = nn.Linear(128 * 19, inner_dim, bias=False)
            nn.init.zeros_(self.v_adapter_action_control.weight)

    def forward(
            self,
            x,
            context=None,
            mask=None,
            additional_tokens=None,
            n_times_crossframe_attn_in_self=0,
            batchify_xformers=False
    ):
        if additional_tokens is not None:
            # get the number of masked tokens at the beginning of the output sequence
            n_tokens_to_mask = additional_tokens.shape[1]
            # add additional token
            x = torch.cat((additional_tokens, x), dim=1)

        context = default(context, x)
        if self.action_control:
            context, context_ = context[:, :, :self.context_dim], context[:, :, self.context_dim:]
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        if self.add_lora:
            q += self.q_adapter_up(self.q_adapter_down(x)) * self.lora_scale
            k += self.k_adapter_up(self.k_adapter_down(context)) * self.lora_scale
            v += self.v_adapter_up(self.v_adapter_down(context)) * self.lora_scale
        if self.action_control:
            k += self.k_adapter_action_control(context_)
            v += self.v_adapter_action_control(context_)

        if n_times_crossframe_attn_in_self:
            # reprogramming cross-frame attention as in https://arxiv.org/abs/2303.13439
            assert x.shape[0] % n_times_crossframe_attn_in_self == 0
            # n_cp = x.shape[0] // n_times_crossframe_attn_in_self
            k = repeat(
                k[::n_times_crossframe_attn_in_self],
                "b ... -> (b n) ...",
                n=n_times_crossframe_attn_in_self
            )
            v = repeat(
                v[::n_times_crossframe_attn_in_self],
                "b ... -> (b n) ...",
                n=n_times_crossframe_attn_in_self
            )

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v)
        )

        if exists(mask):
            raise NotImplementedError
        else:
            # actually compute the attention, what we cannot get enough of
            if batchify_xformers:
                max_bs = 32768  # >65536 will result in wrong outputs
                n_batches = math.ceil(q.shape[0] / max_bs)
                out = list()
                for i_batch in range(n_batches):
                    batch = slice(i_batch * max_bs, (i_batch + 1) * max_bs)
                    out.append(
                        xformers.ops.memory_efficient_attention(
                            q[batch],
                            k[batch],
                            v[batch],
                            attn_bias=self.attn_bias,
                            op=self.attention_op
                        )
                    )
                out = torch.cat(out, 0)
            else:
                out = xformers.ops.memory_efficient_attention(
                    q,
                    k,
                    v,
                    attn_bias=self.attn_bias,
                    op=self.attention_op
                )

        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        if additional_tokens is not None:
            # remove additional token
            out = out[:, n_tokens_to_mask:]
        if self.add_lora:
            return self.to_out(out) + self.out_adapter_up(self.out_adapter_down(out)) * self.lora_scale
        else:
            return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention  # ampere
    }

    def __init__(
            self,
            dim,
            n_heads,
            d_head,
            dropout=0.0,
            context_dim=None,
            gated_ff=True,
            use_checkpoint=False,
            disable_self_attn=False,
            attn_mode="softmax",
            sdp_backend=None,
            add_lora=False,
            action_control=False
    ):
        super().__init__()
        assert attn_mode in self.ATTENTION_MODES
        if attn_mode != "softmax" and not XFORMERS_IS_AVAILABLE:
            print(
                f"Attention mode `{attn_mode}` is not available. Falling back to native attention. "
                f"This is not a problem in Pytorch >= 2.0. You are running with PyTorch version {torch.__version__}"
            )
            attn_mode = "softmax"
        elif attn_mode == "softmax" and not SDP_IS_AVAILABLE:
            print("We do not support vanilla attention anymore, as it is too expensive")
            if not XFORMERS_IS_AVAILABLE:
                assert (
                    False
                ), "Please install xformers via e.g. `pip install xformers==0.0.16`"
            else:
                print("Falling back to xformers efficient attention")
                attn_mode = "softmax-xformers"
        attn_cls = self.ATTENTION_MODES[attn_mode]
        if version.parse(torch.__version__) >= version.parse("2.0.0"):
            assert sdp_backend is None or isinstance(sdp_backend, SDPBackend)
        else:
            assert sdp_backend is None
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(
            query_dim=dim,
            context_dim=context_dim if self.disable_self_attn else None,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            backend=sdp_backend,
            add_lora=add_lora
        )  # is a self-attn if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            backend=sdp_backend,
            add_lora=add_lora,
            action_control=action_control
        )  # is self-attn if context is None
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.use_checkpoint = use_checkpoint
        if self.use_checkpoint:
            print(f"{self.__class__.__name__} is using checkpointing")

    def forward(self, x, context=None, additional_tokens=None, n_times_crossframe_attn_in_self=0):
        kwargs = {"x": x}

        if context is not None:
            kwargs.update({"context": context})

        if additional_tokens is not None:
            kwargs.update({"additional_tokens": additional_tokens})

        if n_times_crossframe_attn_in_self:
            kwargs.update({"n_times_crossframe_attn_in_self": n_times_crossframe_attn_in_self})

        if self.use_checkpoint:
            # inputs = {"x": x, "context": context}
            # return checkpoint(self._forward, inputs, self.parameters(), self.use_checkpoint)
            return checkpoint(self._forward, x, context)
        else:
            return self._forward(**kwargs)

    def _forward(self, x, context=None, additional_tokens=None, n_times_crossframe_attn_in_self=0):
        # spatial self-attn
        x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None,
                       additional_tokens=additional_tokens,
                       n_times_crossframe_attn_in_self=n_times_crossframe_attn_in_self
                       if not self.disable_self_attn else 0) + x
        # spatial cross-attn
        x = self.attn2(self.norm2(x), context=context, additional_tokens=additional_tokens) + x
        # feedforward
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding) and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image.

    use_linear for more efficiency instead of the 1x1 convs.
    """

    def __init__(
            self,
            in_channels,
            n_heads,
            d_head,
            depth=1,
            dropout=0.0,
            context_dim=None,
            disable_self_attn=False,
            use_linear=False,
            attn_type="softmax",
            use_checkpoint=False,
            sdp_backend=None,
            add_lora=False,
            action_control=False
    ):
        super().__init__()
        print(f"Constructing {self.__class__.__name__} of depth {depth} w/ {in_channels} channels and {n_heads} heads")

        from omegaconf import ListConfig

        if exists(context_dim) and not isinstance(context_dim, (list, ListConfig)):
            context_dim = [context_dim]
        if exists(context_dim) and isinstance(context_dim, list):
            if depth != len(context_dim):
                print(
                    f"WARNING: "
                    f"{self.__class__.__name__}: found context dims {context_dim} of depth {len(context_dim)}, "
                    f"which does not match the specified depth of {depth}. "
                    f"Setting context_dim to {depth * [context_dim[0]]} now"
                )
                # depth does not match context dims
                assert all(
                    map(lambda x: x == context_dim[0], context_dim)
                ), "Need homogenous context_dim to match depth automatically"
                context_dim = depth * [context_dim[0]]
        elif context_dim is None:
            context_dim = [None] * depth
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if use_linear:
            self.proj_in = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim[d],
                    disable_self_attn=disable_self_attn,
                    attn_mode=attn_type,
                    use_checkpoint=use_checkpoint,
                    sdp_backend=sdp_backend,
                    add_lora=add_lora,
                    action_control=action_control
                )
                for d in range(depth)
            ]
        )
        if use_linear:
            self.proj_out = zero_module(
                nn.Linear(inner_dim, in_channels)
            )
        else:
            self.proj_out = zero_module(
                nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
            )
        self.use_linear = use_linear

    def forward(self, x, context=None):
        # NOTE: if no context is given, cross-attn defaults to self-attn
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            if i > 0 and len(context) == 1:
                i = 0  # use same context for each block
            x = block(x, context=context[i])
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in
