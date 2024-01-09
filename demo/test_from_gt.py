import torch, cv2
import os.path as osp
import sys
cur_dir = osp.dirname(osp.abspath(__file__))
PROJ_ROOT = osp.normpath(osp.join(cur_dir, "../"))
sys.path.insert(0, PROJ_ROOT)
from lib.pysixd import inout, misc
from lib.vis_utils.image import vis_image_mask_bbox_cv2, vis_image_bboxes_cv2, vis_image_mask_cv2
import json
import numpy as np


def get_models(objs_dir, objs):
    """label based keys."""
    obj_models = {}

    cur_extents = {}
    idx = 0
    for i, obj_name in objs.items():
        model_path = osp.join(objs_dir, f"obj_{i:06d}.ply")
        model = inout.load_ply(model_path, vertex_scale=0.001)
        obj_models[i] = model
        pts = model["pts"]
        xmin, xmax = np.amin(pts[:, 0]), np.amax(pts[:, 0])
        ymin, ymax = np.amin(pts[:, 1]), np.amax(pts[:, 1])
        zmin, zmax = np.amin(pts[:, 2]), np.amax(pts[:, 2])
        size_x = xmax - xmin
        size_y = ymax - ymin
        size_z = zmax - zmin
        cur_extents[idx] = np.array([size_x, size_y, size_z], dtype="float32")
        idx += 1

    return obj_models

def gdrn_visualization(batch, image, obj_models, cam, save_path=None):
    # for crop and resize
    bs = len(batch)

    im_H = 720
    im_W = 1280

    image_mask_pose_est = image

    for i in range(bs):
        R = np.array(batch[i]["cam_R_m2c"]).reshape(3, 3)
        t = np.array(batch[i]["cam_t_m2c"]).reshape(-1, 1)
        print(R)
        print(t)
        print("---------")
        # pose_est = np.hstack([R, t.reshape(3, 1)])
        curr_class = batch[i]["obj_id"]
        # if curr_class not in [3, 17]:
        #    continue
        # print(curr_class)
        proj_pts_est = misc.project_pts(obj_models[curr_class]["pts"], cam, R, t)
        mask_pose_est = misc.points2d_to_mask(proj_pts_est, im_H, im_W)
        image_mask_pose_est = vis_image_mask_cv2(image, mask_pose_est, color="yellow" if i == 0 else "blue")


    if save_path:
        cv2.imwrite(save_path, image_mask_pose_est)
    else:
        cv2.imshow('Main', image)
        if cv2.waitKey(0) == ord('q'):
            exit(0)


def main():
    json_file_path = '/home/chris/PycharmProjects/dataset_construction_utils/test_set/train_isaac/0/scene_gt.json'  # Change to your JSON file path
    with open(json_file_path, 'r') as file:
        frames_data = json.load(file)

    images_dir = '/mnt/sdc1/3d_datasets/synth_box_test_old/train_isaac/0/rgb'

    objs = {1: "Box"}
    obj_models = get_models("/mnt/sdc1/3d_datasets/synth_box_test_old/models", objs)

    camera_json_path = "/mnt/sdc1/3d_datasets/synth_box_test_old/cam_osu_full.json"

    with open(camera_json_path) as f:
        camera_json = json.load(f)
        cam = np.asarray([
            [camera_json['fx'], 0., camera_json['cx']],
            [0., camera_json['fy'], camera_json['cy']],
            [0., 0., 1.]])


    for frame_number, frame_data in frames_data.items():
        image_path = osp.join(images_dir, f"{int(frame_number):04}.jpg")
        image = cv2.imread(image_path)

        gdrn_visualization(frame_data, image, obj_models, cam)

if __name__ == "__main__":
    main()