# inference with detector, gdrn, and refiner
import os.path as osp
import sys
cur_dir = osp.dirname(osp.abspath(__file__))
PROJ_ROOT = osp.normpath(osp.join(cur_dir, "../"))
sys.path.insert(0, PROJ_ROOT)
import time
import numpy as np

from predictor_yolo import YoloPredictor
from predictor_gdrn import GdrnPredictor
import os
from scipy.spatial.transform import Rotation as spR
import cv2


class Pipeline():
    def __init__(self, yolo_config, yolo_ckpt, gdrn_config, gdrn_ckpt, camera_json_path, path_to_obj_models):
        self.yolo_predictor = YoloPredictor(
            exp_name="yolox-x",
            config_file_path=yolo_config,
            ckpt_file_path=yolo_ckpt,
            fuse=True,
            fp16=False
        )
        self.gdrn_predictor = GdrnPredictor(
            config_file_path=gdrn_config,
            ckpt_file_path=gdrn_ckpt,
            camera_json_path=camera_json_path,
            path_to_obj_models=path_to_obj_models,
            # device="cpu"
        )
    def combine_poses(self, poses):
        #unroll the list of poses into a dictionary by class
        combined_poses = {}
        for pose in poses:
            for obj_name in pose.keys():
                if obj_name not in combined_poses:
                    combined_poses[obj_name] = []
                combined_poses[obj_name].append(pose[obj_name])
        # take the median pose for each object
        for obj_name in combined_poses.keys():
            combined_poses[obj_name] = np.median(combined_poses[obj_name], axis=0)
        return combined_poses

    def select_objects(self, poses):
        final_poses = []
        for frame in poses:
            final_poses.append({})
            for obj_name in frame.keys():
                closest_pose = None
                closest_distance = float("inf")
                # obj_name[pose] contains a list of poses for the object in a 4x4 matrix. choose the closest one
                for pose in frame[obj_name]:
                    distance = np.linalg.norm(pose[:3, 3])
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_pose = pose
                final_poses[-1][obj_name] = closest_pose

        return final_poses


    def inference(self, rgb_imgs, depth_img=None):
        poses = []
        for rgb_img in rgb_imgs:
            rgb_img = cv2.resize(rgb_img, (640, 480))
            if depth_img is not None:
                depth_img = cv2.resize(depth_img, (640, 480))

            outputs = self.yolo_predictor.inference(image=rgb_img)
            data_dict = self.gdrn_predictor.preprocessing(outputs=outputs, image=rgb_img, depth_img=None)
            out_dict = self.gdrn_predictor.inference(data_dict)
            poses.append(self.gdrn_predictor.postprocessing(data_dict, out_dict))
        closest_poses = self.select_objects(poses)
        final_poses = self.combine_poses(closest_poses)

        return final_poses

    import numpy as np

    def change_basis(self, original_matrix):
        # Basis change matrix
        basis_change = spR.from_euler('xyz', (-90, 0, -90), degrees=True).as_matrix()
        #basis_change = np.linalg.inv(basis_change)
        # Apply the basis change
        new_matrix = basis_change.T @ original_matrix @ np.linalg.inv(basis_change.T)
        return new_matrix


if __name__ == "__main__":
    from lib.vis_utils.image import vis_image_mask_bbox_cv2, vis_image_bboxes_cv2, vis_image_mask_cv2
    from lib.pysixd import inout, misc
    pipeline = Pipeline(
        yolo_config=osp.join(PROJ_ROOT, "configs/yolox/custom/yolox_x_640_augCozyAAEhsv_ranger_30_epochs_custom_test.py"),
        yolo_ckpt="/home/chris/PycharmProjects/gdrnpp_bop2022/output/yolox/custom/yolox_x_640_augCozyAAEhsv_ranger_30_epochs_custom_test/model_0036147.pth",
        gdrn_config=osp.join(PROJ_ROOT,"output/lambda_logs/gdrn/custom/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_full_aug_synth_3/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_custom.py"),
        gdrn_ckpt=osp.join(PROJ_ROOT,"output/lambda_logs/gdrn/custom/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_full_aug_synth_3/model_final.pth"),
        camera_json_path="/home/chris/datasets/synth_box_test/cam_osu.json",
        path_to_obj_models="/home/chris/datasets/synth_box_2/models"
    )
    image_path = "/home/chris/PycharmProjects/realsense/camera_data/rgb/data_08_02_2024_11_24_28/{:08d}.jpg"
    images = []
    for i in range(104, 115):
        images.append(cv2.imread(image_path.format(i)))

    poses = pipeline.inference(images)
    R = poses['Box'][:3, :3]
    t = poses['Box'][:3, 3]
    eulers = spR.from_matrix(R).as_euler("xyz", degrees=True)

    rotation_label = f"R original: {eulers[0]:.2f}, {eulers[1]:.2f}, {eulers[2]:.2f}"
    print(rotation_label)
    changed_rotation = pipeline.change_basis(R)

    # Should be x forward (out of camera), y left, z up
    eulers = spR.from_matrix(changed_rotation).as_euler("xyz", degrees=True)
    rotation_label = f"R changed: {eulers[0]:.2f}, {eulers[1]:.2f}, {eulers[2]:.2f}"
    print(rotation_label)

    proj_pts_est = misc.project_pts(pipeline.gdrn_predictor.obj_models[1]["pts"], pipeline.gdrn_predictor.cam, R, t)
    mask_pose_est = misc.points2d_to_mask_big(proj_pts_est, 480, 640)
    image_mask_pose_est = vis_image_mask_cv2(images[6], mask_pose_est, color="yellow" if i == 0 else "red")
    #cv2.imshow("blah", image_mask_pose_est)
    #cv2.waitKey(0)
