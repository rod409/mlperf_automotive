## Enirvonment setup

Build and run the container

```
cd <path/to/onnx_test>
docker build -t onnx_test .
docker run -v <path/to/onnx_test>:/home/onnx_test -it onnx_test
```

## Inverse custom op tests

Generate a simple onnx file that contains an custom op named inverse. Then run inference with the model.
```
python simple_inverse_model.py
python test_onnx_inverse.py
```
The should output a 2x2 matrix.

## Inference with UniAD model

This fails to run due to custom op not being registered. Contact Rod about getting the uniad onnx file.

```
python test_onnx_model <uniad_onnx_file_path>
```

modify_graph is used to replace the inverse op naming and attributes to match the tests that work. However the results are the same where there is an error with registering the custom op.

```
python modify_graph.py
```
