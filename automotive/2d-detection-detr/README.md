# ABTF 2D DETR based Object Detection Benchmark

This is the reference implementation for the ABTF 2D DETR based object detection benchmark. The reference uses ONNX as a backend.

This model requires a 99.9% latency target and a 99% accuracy constraint of the reference model.

| model | accuracy | dataset | model source | checkpoint source | precision |
| ---- | ---- | ---- | ---- | ---- | ---- |
| RF-DETR | 0.41774314 mAP | ZOD | https://github.com/roboflow/rf-detr | https://github.com/rod409/rf_detr_train | fp32 |

## Download the preprocessed dataset and model checkpoint

TODO: host dataset and checkpoint


## Build and run the Docker container

### CPU only

```
git clone git@github.com:mlcommons/mlperf_automotive.git
cd mlperf_automotive/automotive/2d-detection-detr
docker build -t rf_detr_inference -f dockerfile.cpu .
docker run -it -v ./mlperf_automotive:/mlperf_automotive -v <path to ZOD dataset>:/zod --rm rf_detr_inference
```

## Run the model in performance mode
```
python main.py --dataset-path <path to ZOD dataset>/preprocessed_zod_1280/ --checkpoint <path to checkpoint>/rfdetr-medium.onnx
```

## Run the model in accuracy mode
```
python main.py --dataset-path <path to ZOD dataset>/preprocessed_zod_1280/ --checkpoint <path to checkpoint>/rfdetr-medium.onnx --accuracy
```

## Run the accuracy checker

```
python accuracy.py --mlperf-accuracy-file <path to log outputs>/mlperf_log_accuracy.json --metadata-file <path to ZOD>/preprocessed_zod_1280/mlperf_metadata.json --annotation-file <path to ZOD>/coco_format/valid/_annotations.coco.json
```
