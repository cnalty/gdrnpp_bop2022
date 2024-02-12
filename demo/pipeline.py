# inference with detector, gdrn, and refiner
import os.path as osp
import sys
cur_dir = osp.dirname(osp.abspath(__file__))
PROJ_ROOT = osp.normpath(osp.join(cur_dir, "../"))
sys.path.insert(0, PROJ_ROOT)
import time


from predictor_yolo import YoloPredictor
from predictor_gdrn import GdrnPredictor
import os

import cv2

def full_inference(rgb_img, depth_img=None):
    rgb_img = cv2.resize(rgb_img, (640, 480))
    if depth_img is not None:
        depth_img = cv2.resize(depth_img, (640, 480))

    outputs = yolo_predictor.inference(image=rgb_img)
    data_dict = gdrn_predictor.preprocessing(outputs=outputs, image=rgb_img, depth_img=None)
    out_dict = gdrn_predictor.inference(data_dict)
    poses = gdrn_predictor.postprocessing(data_dict, out_dict)


if __name__ == "__main__":
    yolo_predictor = YoloPredictor(
        exp_name="yolox-x",
        config_file_path=osp.join(PROJ_ROOT,
                                  "configs/yolox/custom/yolox_x_640_augCozyAAEhsv_ranger_30_epochs_custom_test.py"),
        ckpt_file_path="/home/chris/PycharmProjects/gdrnpp_bop2022/output/yolox/custom/yolox_x_640_augCozyAAEhsv_ranger_30_epochs_custom_test/model_0036147.pth",
        fuse=True,
        fp16=False
    )
    gdrn_predictor = GdrnPredictor(
        config_file_path=osp.join(PROJ_ROOT,
                                  "output/lambda_logs/gdrn/custom/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_full_aug_synth_3/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_custom.py"),
        ckpt_file_path=osp.join(PROJ_ROOT,
                                "output/lambda_logs/gdrn/custom/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_full_aug_synth_3/model_final.pth"),
        camera_json_path="/home/chris/datasets/synth_box_test/cam_osu.json",
        path_to_obj_models="/home/chris/datasets/synth_box_2/models",
        # device="cpu"
    )