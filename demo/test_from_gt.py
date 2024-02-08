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
        model = inout.load_ply(model_path, vertex_scale=1)
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
        #print(R)
        print(t)
        print("---------")
        # pose_est = np.hstack([R, t.reshape(3, 1)])
        curr_class = batch[i]["obj_id"]
        # if curr_class not in [3, 17]:
        #    continue
        print(curr_class)
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
    base_dir = "/mnt/sdc1/3d_datasets/synth_box_test"
    scene = "0"
    split = "train_isaac"
    json_file_path = osp.join(base_dir, f"{split}/{scene}/scene_gt.json")
    #json_file_path = '/home/chris/PycharmProjects/dataset_construction_utils/test_set/train_isaac/0/scene_gt.json'  # Change to your JSON file path
    with open(json_file_path, 'r') as file:
        frames_data = json.load(file)

    images_dir = osp.join(base_dir, f"{split}/{scene}/rgb")

    objs = {1: "Box"}
    '''objs = {
        1: "002_master_chef_can",  # [1.3360, -0.5000, 3.5105]
        2: "003_cracker_box",  # [0.5575, 1.7005, 4.8050]
        3: "004_sugar_box",  # [-0.9520, 1.4670, 4.3645]
        4: "005_tomato_soup_can",  # [-0.0240, -1.5270, 8.4035]
        5: "006_mustard_bottle",  # [1.2995, 2.4870, -11.8290]
        6: "007_tuna_fish_can",  # [-0.1565, 0.1150, 4.2625]
        7: "008_pudding_box",  # [1.1645, -4.2015, 3.1190]
        8: "009_gelatin_box",  # [1.4460, -0.5915, 3.6085]
        9: "010_potted_meat_can",  # [2.4195, 0.3075, 8.0715]
        10: "011_banana",  # [-18.6730, 12.1915, -1.4635]
        11: "019_pitcher_base",  # [5.3370, 5.8855, 25.6115]
        12: "021_bleach_cleanser",  # [4.9290, -2.4800, -13.2920]
        13: "024_bowl",  # [-0.2270, 0.7950, -2.9675]
        14: "025_mug",  # [-8.4675, -0.6995, -1.6145]
        15: "035_power_drill",  # [9.0710, 20.9360, -2.1190]
        16: "036_wood_block",  # [1.4265, -2.5305, 17.1890]
        17: "037_scissors",  # [7.0535, -28.1320, 0.0420]
        18: "040_large_marker",  # [0.0460, -2.1040, 0.3500]
        19: "051_large_clamp",  # [10.5180, -1.9640, -0.4745]
        20: "052_extra_large_clamp",  # [-0.3950, -10.4130, 0.1620]
        21: "061_foam_brick",  # [-0.0805, 0.0805, -8.2435]
    }'''
    obj_models = get_models(osp.join(base_dir, "models"), objs)

    camera_json_path = osp.join(base_dir + "_old", f"cam_osu_full.json")

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