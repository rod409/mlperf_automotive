import torch

class CustomInverseFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.linalg.inv(x)

    @staticmethod
    def symbolic(g, x):
        # Use the ai.onnx.contrib domain for Python-based custom ops
        return g.op("ai.onnx.contrib::Inverse", x)

class Model(torch.nn.Module):
    def forward(self, x):
        return CustomInverseFunc.apply(x*3) + x

# Export the model
torch.onnx.export(
    Model(), 
    torch.randn(2, 2), 
    "inverse.onnx",
    opset_version=16,
    input_names=['input'],
    output_names=['output'],
    # Explicitly register the contrib domain version
    custom_opsets={"ai.onnx.contrib": 1}
)
