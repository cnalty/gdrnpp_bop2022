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


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
def get_image_list(rgb_images_path, depth_images_path=None):
    image_names = []

    rgb_file_names = os.listdir(rgb_images_path)
    rgb_file_names.sort()
    for filename in rgb_file_names:
        apath = os.path.join(rgb_images_path, filename)
        ext = os.path.splitext(apath)[1]
        if ext in IMAGE_EXT:
            image_names.append(apath)

    if depth_images_path is not None:
        depth_file_names = os.listdir(depth_images_path)
        depth_file_names.sort()
        for i, filename in enumerate(depth_file_names):
            apath = os.path.join(depth_images_path, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names[i] = (image_names[i], apath)
                # depth_names.append(apath)

    else:
        for i, filename in enumerate(rgb_file_names):
            image_names[i] = (image_names[i], None)

    return image_names

# Identified the problem. Appears to be using the index id as the class number instead of the actual class number.
if __name__ == "__main__":
    image_paths = get_image_list("/home/chris/PycharmProjects/realsense/camera_data/rgb/data_08_02_2024_11_24_28", None)
    yolo_predictor = YoloPredictor(
                       exp_name="yolox-x",
                        config_file_path=osp.join(PROJ_ROOT, "configs/yolox/custom/yolox_x_640_augCozyAAEhsv_ranger_30_epochs_custom_test.py"),
                        ckpt_file_path="/home/chris/PycharmProjects/gdrnpp_bop2022/output/yolox/custom/yolox_x_640_augCozyAAEhsv_ranger_30_epochs_custom_test/model_0036147.pth",
                       fuse=True,
                       fp16=False
                       )
    gdrn_predictor = GdrnPredictor(
        config_file_path=osp.join(PROJ_ROOT,"output/lambda_logs/gdrn/custom/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_full_aug_synth_3/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_custom.py"),
        ckpt_file_path=osp.join(PROJ_ROOT,"output/lambda_logs/gdrn/custom/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_full_aug_synth_3/model_final.pth"),
        camera_json_path="/home/chris/datasets/synth_box_test/cam_osu.json",
        path_to_obj_models="/home/chris/datasets/synth_box_2/models",
        #device="cpu"
    )
    total_time = 0
    iterations = 0
    save_path = "/home/chris/datasets/manual_bbox_output/box_all_instance_45conf_earlymodel_tinynms/{}"
    os.makedirs(osp.dirname(save_path), exist_ok=True)
    for rgb_img, depth_img in image_paths:
        base_name = osp.basename(rgb_img)
        rgb_img = cv2.imread(rgb_img)
        rgb_img = cv2.resize(rgb_img, (640, 480))
        if depth_img is not None:
            depth_img = cv2.imread(depth_img, 0)
            depth_img = cv2.resize(depth_img, (640, 480))
        start_time = time.time()
        outputs = yolo_predictor.inference(image=rgb_img)
        data_dict = gdrn_predictor.preprocessing(outputs=outputs, image=rgb_img, depth_img=None)
        out_dict = gdrn_predictor.inference(data_dict)
        poses = gdrn_predictor.postprocessing(data_dict, out_dict)
        print(poses)
        end_time = time.time()
        total_time += end_time - start_time
        iterations += 1


        if iterations % 100 == 0:
            print("Average time taken so far: {} seconds".format(total_time / float(iterations)))
        gdrn_predictor.gdrn_visualization(batch=data_dict, out_dict=out_dict, image=rgb_img,
                                          save_path=save_path.format(base_name)
                                          )
        #break

    print("Average time taken: {} seconds".format(total_time/float(iterations)))