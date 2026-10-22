import torch
import onnxruntime as ort
import logging
import numpy as np
import torch

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("backend-onnx")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class BackendOnnx():
    def __init__(self, checkpoint, device='cpu'):
        self.checkpoint = checkpoint

    def version(self):
        return torch.__version__

    def name(self):
        return "onnx-SUT"

    def load(self):
        self.ort_sess = ort.InferenceSession(self.checkpoint)
        self.input_img_name = self.ort_sess.get_inputs()[0].name
        return self

    def predict(self, inputs):
        with torch.inference_mode():
            model_input = inputs[0]
            model_input = model_input.reshape(1, *model_input.shape)
            #import pdb; pdb.set_trace()
            outputs = self.ort_sess.run(["dets", "labels"], {"input": model_input})
            boxes_raw = outputs[0][0]   # Shape: (num_queries, 4) in cxcywh normalized format
            logits_raw = outputs[1][0]  # Shape: (num_queries, num_classes) pre-sigmoid
            probs = sigmoid(logits_raw)
            scores = np.max(probs, axis=-1)
            labels = np.argmax(probs, axis=-1)
        #pdb.set_trace()
        return np.column_stack((boxes_raw, scores, labels))
