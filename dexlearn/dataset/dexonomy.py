import os
from os.path import join as pjoin
from glob import glob
import random

import numpy as np
from torch.utils.data import Dataset

from dexlearn.utils.rot import numpy_quaternion_to_matrix
from dexlearn.utils.util import load_json


class DexonomyDataset(Dataset):
    def __init__(self, config: dict, mode: str, sc_voxel_size: float = None):
        self.config = config
        self.sc_voxel_size = sc_voxel_size
        self.mode = mode

        if self.config.grasp_type_lst is not None:
            self.grasp_type_lst = self.config.grasp_type_lst
        else:
            self.grasp_type_lst = os.listdir(self.config.grasp_path)
        self.grasp_type_num = len(self.grasp_type_lst)
        self.object_pc_folder = pjoin(self.config.object_path, self.config.pc_path)

        if mode == "train" or mode == "eval":
            self.init_train_eval(mode)
        elif mode == "test":
            self.init_test()
        return

    def init_train_eval(self, mode):
        split_name = "test" if mode == "eval" else "train"
        self.obj_id_lst = load_json(
            pjoin(self.config.object_path, self.config.split_path, f"{split_name}.json")
        )

        self.grasp_obj_dict = {}
        self.data_num = 0
        for grasp_type in self.grasp_type_lst:
            self.grasp_obj_dict[grasp_type] = []
            for obj_id in self.obj_id_lst:
                obj_grasp_data = len(
                    glob(
                        pjoin(
                            self.config.grasp_path, grasp_type, f"{obj_id}**", "**.npy"
                        )
                    )
                )
                if obj_grasp_data == 0:
                    continue
                self.data_num += obj_grasp_data
                self.grasp_obj_dict[grasp_type].append(obj_id)
            if len(self.grasp_obj_dict[grasp_type]) == 0:
                self.grasp_obj_dict.pop(grasp_type)
        print(
            f"mode: {mode}, grasp type number: {self.grasp_type_num}, grasp data num: {self.data_num}"
        )
        return

    def init_test(self):
        split_name = self.config.test_split
        self.obj_id_lst = []
        self.test_cfg_lst = []
        self.obj_id_lst = load_json(
            pjoin(self.config.object_path, self.config.split_path, f"{split_name}.json")
        )
        if self.config.mini_test:
            self.obj_id_lst = self.obj_id_lst[:100]
        for o in self.obj_id_lst:
            self.test_cfg_lst.append(
                pjoin(
                    self.config.object_path,
                    "processed_data",
                    o,
                    self.config.test_scene_cfg,
                )
            )
        self.data_num = self.grasp_type_num * len(self.test_cfg_lst)
        print(
            f"Test split: {split_name}, grasp type number: {self.grasp_type_num}, object cfg num: {len(self.test_cfg_lst)}"
        )
        return

    def __len__(self):
        return self.data_num

    def __getitem__(self, id: int):
        ret_dict = {}

        if self.mode == "train" or self.mode == "eval":
            # random select grasp data
            rand_grasp_type = random.choice(self.grasp_type_lst)
            grasp_obj_lst = self.grasp_obj_dict[rand_grasp_type]
            rand_obj_id = random.choice(grasp_obj_lst)
            grasp_npy_lst = glob(
                pjoin(
                    self.config.grasp_path,
                    rand_grasp_type,
                    f"{rand_obj_id}**",
                    "**.npy",
                )
            )
            grasp_path = random.choice(sorted(grasp_npy_lst))
            grasp_data = np.load(grasp_path, allow_pickle=True).item()
            robot_pose = np.stack(
                [
                    grasp_data["pregrasp_qpos"],
                    grasp_data["grasp_qpos"],
                    grasp_data["squeeze_qpos"],
                ],
                axis=0,
            )[
                None
            ]  # 1, 3, 29

            # read point cloud
            pc_path_lst = glob(
                pjoin(
                    self.object_pc_folder,
                    grasp_data["scene_cfg"]["scene_id"],
                    "**/partial_pc**.npy",
                ),
                recursive=True,
            )
            pc_path = random.choice(sorted(pc_path_lst))
            raw_pc = np.load(pc_path, allow_pickle=True)
            idx = np.random.choice(
                raw_pc.shape[0], self.config.num_points, replace=True
            )
            obj_name = grasp_data["scene_cfg"]["task"]["obj_name"]
            object_scale = grasp_data["scene_cfg"]["scene"][obj_name]["scale"]
            pc_scale = float(pc_path.split("scale")[1].split("/")[0]) / 100
            pc = raw_pc[idx] / pc_scale * object_scale

            ret_dict["hand_trans"] = robot_pose[:, :, :3]  # (K, n, 3)
            ret_dict["hand_rot"] = numpy_quaternion_to_matrix(
                robot_pose[:, :, 3:7]
            )  # (K, n, 3, 3)
            ret_dict["hand_joint"] = robot_pose[:, :, 7:]  # (K, n, Q)

        elif self.mode == "test":
            rand_grasp_type = self.grasp_type_lst[id // len(self.test_cfg_lst)]
            cfg_path = self.test_cfg_lst[id % len(self.test_cfg_lst)]
            scene_cfg = np.load(cfg_path, allow_pickle=True).item()

            obj_name = scene_cfg["task"]["obj_name"]
            object_scale = scene_cfg["scene"][obj_name]["scale"]

            # read point cloud
            pc_path_lst = glob(
                pjoin(
                    self.object_pc_folder,
                    scene_cfg["scene_id"],
                    "**/partial_pc**.npy",
                ),
                recursive=True,
            )
            pc_path = random.choice(sorted(pc_path_lst))
            raw_pc = np.load(pc_path, allow_pickle=True)
            idx = np.random.choice(
                raw_pc.shape[0], self.config.num_points, replace=True
            )
            pc_scale = float(pc_path.split("scale")[1].split("/")[0]) / 100
            pc = raw_pc[idx] / pc_scale * object_scale

            ret_dict["save_path"] = pjoin(
                rand_grasp_type,
                scene_cfg["scene_id"],
                f"{os.path.basename(pc_path)}.npy",
            )
            ret_dict["scene_cfg"] = scene_cfg

        ret_dict["point_clouds"] = pc  # (N, 3)
        ret_dict["grasp_type_id"] = int(rand_grasp_type.split("_")[0])
        if self.sc_voxel_size is not None:
            ret_dict["coors"] = pc / self.sc_voxel_size  # (N, 3)
            ret_dict["feats"] = pc  # (N, 3)
        return ret_dict
