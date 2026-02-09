import torch
#import backend

import onnxruntime as ort
import numpy as np
#from post_process import PostProcess
#from onnxruntime_extensions import onnx_op, PyOp, get_library_path
#from onnxruntime_extensions import PyOrtFunction
import argparse
import os


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


#@onnx_op(op_type="Inverse", domain='ai.onnx.contrib', inputs=[PyOp.dt_float], outputs=[PyOp.dt_float])
#def inverse(x):
#    return np.linalg.inv(x).astype(np.float32)

def _generate_empty_zeros_tracks_trt():
        num_queries = 901
        dim = 512
        query = torch.zeros((901, 512), dtype=torch.float)
        ref_pts = torch.zeros((901, 3), dtype=torch.float)

        # init boxes: xy, wl, z, h, sin, cos, vx, vy, vz
        pred_boxes_init = torch.zeros(
            (len(ref_pts), 10), dtype=torch.float
        )

        output_embedding = torch.zeros(
            (num_queries, dim >> 1)
        )

        obj_idxes = torch.full(
            (len(ref_pts),), -1, dtype=torch.int
        )
        matched_gt_idxes = torch.full(
            (len(ref_pts),), -1, dtype=torch.int
        )
        disappear_time = torch.zeros(
            (len(ref_pts),), dtype=torch.int
        )

        iou = torch.zeros(
            (len(ref_pts),), dtype=torch.float
        )
        scores = torch.zeros(
            (len(ref_pts),), dtype=torch.float
        )
        track_scores = torch.zeros(
            (len(ref_pts),), dtype=torch.float
        )
        # xy, wl, z, h, sin, cos, vx, vy, vz
        pred_boxes = pred_boxes_init

        pred_logits = torch.zeros(
            (len(ref_pts), 10), dtype=torch.float
        )

        mem_bank = torch.zeros(
            (len(ref_pts), 4, dim // 2),
            dtype=torch.float32,
        )
        mem_padding_mask = torch.ones(
            (len(ref_pts), 4), dtype=torch.int
        )
        save_period = torch.zeros(
            (len(ref_pts),), dtype=torch.float32
        )

        return [
            query,
            ref_pts,
            output_embedding,
            obj_idxes,
            matched_gt_idxes,
            disappear_time,
            iou,
            scores,
            track_scores,
            pred_boxes,
            pred_logits,
            mem_bank,
            mem_padding_mask,
            save_period,
        ]

def parse_args():
    parser = argparse.ArgumentParser(
        description='run onnx model')
    parser.add_argument('checkpoint', help='checkpoint file')
    args = parser.parse_args()
    return args

def main():
        args = parse_args()
        #inverse_op = OrtPyFunction.from_customop(op_type='InverseTRT', cpu_impl=inverse)
        session_options = ort.SessionOptions()
        #session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        #session_options.register_custom_ops_library(get_library_path())
        ort_sess = ort.InferenceSession(args.checkpoint, session_options=session_options)
        bevh = 50
        img_h = 256
        img_w = 416
        test_track_instances =[item*0 for item in _generate_empty_zeros_tracks_trt()] 
        input_shapes = dict(
        prev_track_intances0=[-1, 512], 
        prev_track_intances1=[-1, 3],
        prev_track_intances2=[-1, 256],
        prev_track_intances3=[-1],
        prev_track_intances4=[-1],
        prev_track_intances5=[-1],
        prev_track_intances6=[-1],
        prev_track_intances7=[-1],
        prev_track_intances8=[-1],
        prev_track_intances9=[-1, 10],
        prev_track_intances10=[-1, 10],
        prev_track_intances11=[-1, 4, 256],
        prev_track_intances12=[-1, 4],
        prev_track_intances13=[-1],
        prev_timestamp=[1],
        prev_l2g_r_mat=[1, 3, 3],
        prev_l2g_t=[1, 3],
        prev_bev=[bevh**2, 1, 256],
        gt_lane_labels=[1, -1], 
        gt_lane_masks=[1, -1, bevh, bevh],
        gt_segmentation=[1, 7, bevh, bevh],
        img_metas_scene_token=[32],
        timestamp=[1],
        l2g_r_mat=[1, 3, 3], 
        l2g_t=[1, 3], 
        img=[1, 6, 3, img_h, img_w],
        img_metas_can_bus=[18],
        img_metas_lidar2img=[1, 6, 4, 4],
        image_shape=[2],
        command=[1],
        use_prev_bev=[1],
        max_obj_id=[1],
    )
        inputs = {}
        inputs['timestamp'] = np.zeros((1), dtype=np.float32)
        #inputs['image_shape'] = np.array([img_h,img_w]).astype(np.float32)
        inputs['prev_bev'] = np.zeros([bevh**2, 1, 256]).astype(np.float32)
        inputs['max_obj_id'] = np.array([0]).astype(np.int32)
        inputs['prev_track_intances0'] = test_track_instances[0].numpy().astype(np.float32)
        inputs['prev_track_intances1'] = test_track_instances[1].numpy().astype(np.float32)
        inputs['prev_track_intances3'] = test_track_instances[3].numpy().astype(np.int32)
        inputs['prev_track_intances4'] = test_track_instances[4].numpy().astype(np.int32)
        inputs['prev_track_intances5'] = test_track_instances[5].numpy().astype(np.int32)
        inputs['prev_track_intances6'] = test_track_instances[6].numpy().astype(np.float32)
        inputs['prev_track_intances8'] = test_track_instances[8].numpy().astype(np.float32)
        inputs['prev_track_intances9'] = test_track_instances[9].numpy().astype(np.float32)
        inputs['prev_track_intances11'] = test_track_instances[11].numpy().astype(np.float32)
        inputs['prev_track_intances12'] = test_track_instances[12].numpy().astype(np.int32)
        inputs['prev_track_intances13'] = test_track_instances[13].numpy().astype(np.float32)
        inputs['prev_l2g_r_mat'] = np.zeros((1, 3, 3), dtype=np.float32)
        inputs['prev_l2g_t'] = np.zeros((1, 3), dtype=np.float32)
        inputs['prev_timestamp'] = np.zeros([1]).astype(np.float32)
        inputs['use_prev_bev'] = np.array([0]).astype(np.int32)
        inputs['prev_l2g_r_mat'] = np.zeros((1, 3, 3), dtype=np.float32)
        inputs['img'] = np.random.randn(1, 6, 3, img_h, img_w).astype(np.float32)
        inputs['img_metas_can_bus'] = np.random.randn(18).astype(np.float32)
        inputs['img_metas_lidar2img'] = np.random.randn(1, 6, 4, 4).astype(np.float32)
        inputs['command'] = np.random.randn(1).astype(np.float32)
        inputs['l2g_inv'] = np.random.randn(1, 3, 3).astype(np.float32)
        inputs['l2g_r_mat'] = np.random.randn(1, 3, 3).astype(np.float32)
        inputs['l2g_t'] = np.random.randn(1, 3).astype(np.float32)
        #test_func = PyOrtFunction(args.checkpoint)
        #result = test_func(inputs)
        result = ort_sess.run(None, inputs)
        #model_func = PyOrtFunction(args.checkpoint)
        #result = model_func(inputs)
        print(len(result))

if __name__ == '__main__':
    main()