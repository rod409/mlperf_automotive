import numpy as np
import onnxruntime as ort
from onnxruntime_extensions import onnx_op, PyCustomOpDef, get_library_path

# 1. Define the Python implementation
# The domain MUST be "ai.onnx.contrib" for standard Python registration
@onnx_op(op_type="Inverse", domain="ai.onnx.contrib", inputs=[PyCustomOpDef.dt_float])
def python_inverse(x):
    return np.linalg.inv(x).astype(np.float32)

# 2. Setup SessionOptions
so = ort.SessionOptions()
# This library links the @onnx_op functions to the ORT engine
so.register_custom_ops_library(get_library_path())

# 3. Create the session
# This will now find "ai.onnx.contrib::Inverse" via the extensions library
sess = ort.InferenceSession("inverse.onnx", sess_options=so)

# 4. Run Test
test_input = np.random.randn(2, 2).astype(np.float32)
input_name = sess.get_inputs()[0].name
output = sess.run(None, {input_name: test_input})

print("Inference Success!\n", output[0])
