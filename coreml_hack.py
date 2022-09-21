## Original code from
## https://gist.github.com/madebyollin/86b9596ffa4ab0fa7674a16ca2aeab3d
## with some modifications to make it work with the latest version of diffusers

import torch
import torch as th
import coremltools as ct
import diffusers

from pathlib import Path

def generate_coreml_model_via_awful_hacks(f, out_name):
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.frontend.torch.torch_op_registry import register_torch_op, _TORCH_OPS_REGISTRY
    import coremltools.converters.mil.frontend.torch.ops as cml_ops
    # def unsliced_attention(self, query, key, value, _sequence_length, _dim):
    #     attn = (torch.einsum("b i d, b j d -> b i j", query, key) * self.scale).softmax(dim=-1)
    #     attn = torch.einsum("b i j, b j d -> b i d", attn, value)
    #     return self.reshape_batch_dim_to_heads(attn)
    # diffusers.models.attention.CrossAttention._attention = unsliced_attention
    orig_einsum = th.einsum
    def fake_einsum(a, b, c):
        if a == 'b i d, b j d -> b i j': return th.bmm(b, c.permute(0, 2, 1))
        if a == 'b i j, b j d -> b i d': return th.bmm(b, c)
        raise ValueError(f"unsupported einsum {a} on {b.shape} {c.shape}")
    th.einsum = fake_einsum
    if "broadcast_to" in _TORCH_OPS_REGISTRY: del _TORCH_OPS_REGISTRY["broadcast_to"]
    @register_torch_op
    def broadcast_to(context, node): return cml_ops.expand(context, node)
    if "gelu" in _TORCH_OPS_REGISTRY: del _TORCH_OPS_REGISTRY["gelu"]
    @register_torch_op
    def gelu(context, node): context.add(mb.gelu(x=context[node.inputs[0]], name=node.name))
    class Undictifier(th.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, *args, **kwargs): return self.m(*args, **kwargs)["sample"]
    print("tracing")
    f_trace = th.jit.trace(Undictifier(f), (th.zeros(2, 4, 64, 64), th.zeros(1), th.zeros(2, 77, 768)), strict=False, check_trace=False)
    print("converting")
    f_coreml_fp16 = ct.convert(f_trace, 
               inputs=[ct.TensorType(shape=(2, 4, 64, 64)), ct.TensorType(shape=(1,)), ct.TensorType(shape=(2, 77, 768))],
               convert_to="mlprogram",  compute_precision=ct.precision.FLOAT16, skip_model_load=True)
    f_coreml_fp16.save(f"{out_name}")
    th.einsum = orig_einsum
    print("the deed is done")

class UNetWrapper:
    def __init__(self, f, out_name="unet.mlpackage"):
        self.in_channels = f.in_channels
        if not Path(out_name).exists():
            print("generating coreml model"); generate_coreml_model_via_awful_hacks(f, out_name); print("saved")
        # not only does ANE take forever to load because it recompiles each time - it then doesn't work!
        # and NSLocalizedDescription = "Error computing NN outputs."; is not helpful... GPU it is
        print("loading saved coreml model"); f_coreml_fp16 = ct.models.MLModel(out_name, compute_units=ct.ComputeUnit.CPU_AND_GPU); print("loaded")
        self.f = f_coreml_fp16
    def __call__(self, sample, timestep, encoder_hidden_states):
        args = {"sample_1": sample.numpy(), "timestep": th.tensor([timestep], dtype=th.int32).numpy(), "input_35": encoder_hidden_states.numpy()}
        for v in self.f.predict(args).values():
            return diffusers.models.unet_2d_condition.UNet2DConditionOutput(sample=th.tensor(v, dtype=th.float32))
 