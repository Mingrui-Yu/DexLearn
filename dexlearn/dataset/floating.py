import os
from os.path import join as pjoin
import glob
import random

from transforms3d import quaternions as tq
import numpy as np
from torch.utils.data import Dataset

from dexgrasp.utils.rot import numpy_normalize, numpy_quaternion_to_matrix
from dexgrasp.utils.util import load_json


class FloatingGraspDataset(Dataset):
    def __init__(self, config: dict, mode: str):
        self.config = config
        self.mode = mode
        assert mode == "train" or mode == "eval" or mode == "test"
        split_name = mode if mode != "test" else config.data.test_split
        split_name = "test" if mode == "eval" else split_name
        split_path = pjoin(self.config.data.split_path, f"{split_name}.json")
        obj_id_lst = load_json(split_path)
        if (
            mode == "test" and config.data.mini_test
        ):  # TODO: this is just for fast eval!
            obj_id_lst = obj_id_lst[:100]
        if mode == "train":
            # use part of data
            obj_leave_num = len(obj_id_lst) // self.config.data.obj_split_num
            obj_id_lst = obj_id_lst[:obj_leave_num]

        obj_pc_dict = {}
        grasp_data_lst = []
        pc_pose_lst = []
        for obj_id in obj_id_lst:
            grasp_folder = pjoin(self.config.data.grasp_path, obj_id)
            if not os.path.exists(grasp_folder):
                continue
            for grasp_file in os.listdir(grasp_folder):
                grasp_path = pjoin(grasp_folder, grasp_file)
                pc_path = pjoin(
                    self.config.data.pc_path,
                    obj_id,
                    "tabletop_kinect",
                    f"**/**/partial_pc**.npy",
                )
                pc_pose_path = pjoin(
                    self.config.data.object_path, obj_id, "info/tabletop_pose.json"
                )
                pc_data = glob.glob(pc_path)
                grasp_data_lst.append(grasp_path)
                pc_pose_lst.append(pc_pose_path)
                obj_pc_dict[grasp_path] = pc_data
        self.obj_pc_dict = obj_pc_dict
        self.grasp_data_lst = sorted(grasp_data_lst)
        self.pc_pose_lst = sorted(pc_pose_lst)
        assert len(self.grasp_data_lst) == len(self.pc_pose_lst)
        print(f"mode: {mode}, data num: {len(self.grasp_data_lst)}")

    def __len__(self):
        return len(self.grasp_data_lst)

    def __getitem__(self, id: int):
        ret_dict = {}

        # read grasp data
        grasp_path = self.grasp_data_lst[id]
        grasp_data = np.load(grasp_path, allow_pickle=True).item()
        robot_pose = np.stack(
            [
                grasp_data["pregrasp_qpos"],
                grasp_data["grasp_qpos"],
                grasp_data["squeeze_qpos"],
            ],
            axis=1,
        )  # N, 3, 29
        object_pose = grasp_data["obj_pose"]  # 7
        object_scale = grasp_data["obj_scale"]  # 1

        # read point cloud data
        pc_path = random.choice(self.obj_pc_dict[grasp_path])
        raw_pc = np.load(pc_path, allow_pickle=True)
        idx = np.random.choice(
            raw_pc.shape[0], self.config.data.num_points, replace=True
        )
        pc_scale = float(pc_path.split("scale")[1].split("/")[0]) / 100
        pc = raw_pc[idx] / pc_scale * object_scale

        pose_params = load_json(self.pc_pose_lst[id])
        pose_ind = int(pc_path.split("/pose")[-1].split("/")[0])
        pc_pose = np.array(pose_params[pose_ind % len(pose_params)])
        pc_pose[2] *= object_scale

        if self.mode == "train" or self.mode == "eval":
            # use part of data
            grasp_leave_num = (
                1 + robot_pose.shape[0] // self.config.data.grasp_split_num
            )
            robot_pose = robot_pose[:grasp_leave_num]

            if len(robot_pose) == 0:
                return self.__getitem__((id + 1) % len(self))

            all_grasp_num = robot_pose.shape[0]
            indices = np.random.choice(
                all_grasp_num, self.config.data.num_grasps, replace=True
            )
            robot_pose = robot_pose[indices]

            hand_trans = robot_pose[:, :, :3]
            hand_rot = numpy_quaternion_to_matrix(robot_pose[:, :, 3:7])
            obj_trans = object_pose[None, None, :3]
            obj_rot = numpy_quaternion_to_matrix(object_pose[None, None, 3:])
            pc_trans = pc_pose[None, None, :3]
            pc_rot = numpy_quaternion_to_matrix(pc_pose[None, None, 3:])

            # get relative hand pose
            hand_trans = (hand_trans - obj_trans) @ obj_rot.squeeze(0)
            hand_rot = obj_rot.transpose(0, 1, 3, 2) @ hand_rot

            # get hand pose for pc
            hand_trans = (
                hand_trans @ pc_rot.squeeze(0).transpose(0, 2, 1) + pc_trans
            )  # K, n, 3
            hand_rot = pc_rot @ hand_rot  # K, n, 3, 3

            if self.config.data.pose_aug:
                pc, hand_trans, hand_rot, pc_trans, pc_rot = self.pose_augment(
                    pc, hand_trans, hand_rot, pc_trans, pc_rot
                )

            if self.config.data.model_name == "uni3d":
                delta_bias = np.mean(pc, axis=0)
                pc -= delta_bias[None]
                hand_trans -= delta_bias
                pc_trans -= delta_bias

            ret_dict["hand_rot"] = hand_rot  # (K, n, 3, 3)
            ret_dict["hand_trans"] = hand_trans  # (K, n, 3)
            ret_dict["hand_joint"] = robot_pose[:, :, 7:]  # (K, n, Q)

        # save for visualization
        elif self.mode == "test":
            if self.config.data.model_name == "uni3d":
                delta_bias = np.mean(pc, axis=0)
                pc -= delta_bias[None]
                pc_pose[:3] -= delta_bias

            ret_dict["save_path"] = (
                grasp_path.replace(f"{self.config.data.grasp_path}/", "").replace(
                    ".npy", "_cam" + pc_path[:-4].split("partial_pc_")[-1]
                )
                + "_grasp.npy"
            )
            for k in ["obj_path", "obj_scale"]:
                ret_dict[k] = grasp_data[k]
            ret_dict["obj_pose"] = pc_pose

        # np.save(
        #     "debug.npy",
        #     {
        #         "debug_path": pc_path,
        #         "debug_scale": pc_scale,
        #         "pc": pc,
        #         "obj_path": grasp_data["obj_path"],
        #         "obj_scale": grasp_data["obj_scale"],
        #         "obj_trans": pc_trans,
        #         "obj_rot": pc_rot,
        #         "hand_rot": rot,
        #         "hand_trans": trans,
        #         "joint_angle": joint_angles,
        #     },
        # )
        # exit(1)
        ret_dict["point_clouds"] = pc  # (N, 3)
        ret_dict["grasp_type_id"] = 0
        if "MinkUNet" in self.config.algo.model.backbone.name:
            ret_dict["coors"] = (
                pc / self.config.algo.model.backbone.voxel_size
            )  # (N, 3)
            ret_dict["feats"] = pc  # (N, 3)
        return ret_dict

    def pose_augment(self, obj_pc, robot_trans, robot_rot, obj_trans, obj_rot):
        random_trans = np.clip(np.random.randn(1, 3) * 0.02, -0.05, 0.05)
        random_rot = numpy_quaternion_to_matrix(numpy_normalize(np.random.randn(4)))

        aug_obj_pc = obj_pc @ random_rot.T + random_trans
        aug_robot_trans = robot_trans @ random_rot.T + random_trans
        aug_robot_rot = random_rot[None] @ robot_rot
        aug_obj_trans = obj_trans @ random_rot.T + random_trans
        aug_obj_rot = random_rot @ obj_rot

        return aug_obj_pc, aug_robot_trans, aug_robot_rot, aug_obj_trans, aug_obj_rot
