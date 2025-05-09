import os
from os.path import join as pjoin
import glob
import random

from transforms3d import quaternions as tq
import numpy as np
from torch.utils.data import Dataset

from dexgrasp.utils.rot import numpy_normalize, numpy_quaternion_to_matrix
from dexgrasp.utils.util import load_json


class TypeCondGraspDataset(Dataset):
    def __init__(self, config: dict, mode: str):
        self.config = config
        self.mode = mode
        assert mode == "train" or mode == "eval" or mode == "test"
        split_name = mode if mode != "test" else config.data.test_split
        split_name = "test" if mode == "eval" else split_name
        split_path = pjoin(self.config.data.split_path, f"{split_name}.json")
        obj_id_lst = load_json(split_path)

        if mode == "test" and config.data.mini_test:
            obj_id_lst = obj_id_lst[:100]
        if mode == "train":  # use part of data
            obj_leave_num = len(obj_id_lst) // self.config.data.obj_split_num
            obj_id_lst = obj_id_lst[:obj_leave_num]

        # Load object information to list, e.g. pc_path, pose_path, id, scale
        self.obj_pc_dict = {}
        self.obj_pose_dict = {}
        for obj_id in obj_id_lst:
            pc_path_tmp = pjoin(
                self.config.data.pc_path,
                obj_id,
                "tabletop_kinect",
                f"**/**/partial_pc**.npy",
            )
            pc_pose_path = pjoin(
                self.config.data.object_path, obj_id, "info/tabletop_pose.json"
            )
            pc_path_lst = glob.glob(pc_path_tmp)
            if len(pc_path_lst) == 0 or not os.path.exists(pc_pose_path):
                continue
            else:
                self.obj_pc_dict[obj_id] = pc_path_lst
                self.obj_pose_dict[obj_id] = pc_pose_path
        self.obj_id_lst = list(self.obj_pc_dict.keys())
        self.obj_id_num = len(self.obj_id_lst)
        self.obj_scale_lst = [0.05, 0.08, 0.11, 0.14, 0.17, 0.20]  # Only for test
        self.obj_scale_num = len(self.obj_scale_lst)

        if self.config.data.grasp_type_lst is not None:
            self.grasp_type_lst = self.config.data.grasp_type_lst
        else:
            self.grasp_type_lst = os.listdir(self.config.data.grasp_path)
        self.grasp_type_num = len(self.grasp_type_lst)

        # Load grasp data information to list
        if mode != "test":
            self.grasp_obj_dict = {}
            self.grasp_data_num = 0

            for grasp_type in self.grasp_type_lst:
                obj_lst = os.listdir(pjoin(self.config.data.grasp_path, grasp_type))
                valid_obj_lst = list(set(obj_lst).intersection(set(obj_id_lst)))
                if len(valid_obj_lst) > 0:
                    self.grasp_obj_dict[grasp_type] = valid_obj_lst
                else:
                    continue
                for o in valid_obj_lst:
                    self.grasp_data_num += len(
                        glob.glob(
                            pjoin(self.config.data.grasp_path, grasp_type, o, "**.npy")
                        )
                    )
        else:
            self.grasp_data_num = (
                self.grasp_type_num * self.obj_id_num * self.obj_scale_num
            )
        print(
            f"mode: {mode}, grasp type number: {self.grasp_type_num}, grasp data num: {self.grasp_data_num}"
        )

    def __len__(self):
        return self.grasp_data_num

    def __getitem__(self, id: int):
        ret_dict = {}

        if self.mode == "train" or self.mode == "eval":
            # random select grasp data
            rand_grasp_type = random.choice(self.grasp_type_lst)
            grasp_obj_lst = self.grasp_obj_dict[rand_grasp_type]
            rand_obj_id = random.choice(grasp_obj_lst)
            grasp_obj_folder = pjoin(
                self.config.data.grasp_path, rand_grasp_type, rand_obj_id
            )
            grasp_npy = random.choice(os.listdir(grasp_obj_folder))
            grasp_data = np.load(
                pjoin(grasp_obj_folder, grasp_npy), allow_pickle=True
            ).item()
            robot_pose = np.stack(
                [
                    np.concatenate(
                        [grasp_data["pregrasp_pose"], grasp_data["pregrasp_qpos"]]
                    ),
                    np.concatenate([grasp_data["hand_pose"], grasp_data["hand_qpos"]]),
                    np.concatenate(
                        [grasp_data["hand_pose"], grasp_data["squeeze_qpos"]]
                    ),
                ],
                axis=0,
            )[
                None
            ]  # N, 3, 29

            object_pose = grasp_data["obj_pose"]  # 7
            object_scale = grasp_data["obj_scale"]  # 1

            # read point cloud data
            pc_path = random.choice(self.obj_pc_dict[rand_obj_id])
            raw_pc = np.load(pc_path, allow_pickle=True)
            idx = np.random.choice(
                raw_pc.shape[0], self.config.data.num_points, replace=True
            )
            pc_scale = float(pc_path.split("scale")[1].split("/")[0]) / 100
            pc = raw_pc[idx] / pc_scale * object_scale

            pose_params = load_json(self.obj_pose_dict[rand_obj_id])
            pose_ind = int(pc_path.split("/pose")[-1].split("/")[0])
            pc_pose = np.array(pose_params[pose_ind % len(pose_params)])
            pc_pose[2] *= object_scale

            # use part of data
            grasp_leave_num = (
                1 + robot_pose.shape[0] // self.config.data.grasp_split_num
            )
            robot_pose = robot_pose[:grasp_leave_num]

            if len(robot_pose) == 0:
                return self.__getitem__((id + 1) % len(self))

            # randomly select grasps
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

            if "grasp_kp" in grasp_data:
                hand_kp = np.stack(
                    [
                        grasp_data["pregrasp_kp"],
                        grasp_data["grasp_kp"],
                        grasp_data["squeeze_kp"],
                    ],
                    axis=0,
                )[
                    None
                ]  # N 3 28 3
                hand_kp = hand_kp[:grasp_leave_num][indices]
                relative_hand_kp = (hand_kp - hand_trans[:, :, None, :]) @ hand_rot

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

            if "grasp_kp" in grasp_data:
                ret_dict["hand_kp"] = (
                    relative_hand_kp @ hand_rot.transpose(0, 1, 3, 2)
                    + hand_trans[:, :, None, :]
                )
            ret_dict["hand_rot"] = hand_rot  # (K, n, 3, 3)
            ret_dict["hand_trans"] = hand_trans  # (K, n, 3)
            ret_dict["hand_joint"] = robot_pose[:, :, 7:]  # (K, n, Q)

        elif self.mode == "test":
            rand_grasp_type = self.grasp_type_lst[
                id // (self.obj_scale_num * self.obj_id_num)
            ]
            obj_id = list(self.obj_pose_dict.keys())[
                (id % (self.obj_scale_num * self.obj_id_num)) // self.obj_scale_num
            ]
            object_path = pjoin(self.config.data.object_path, obj_id)
            object_pose = np.array([0.0, 0, 0, 1, 0, 0, 0])  # 7
            object_scale = self.obj_scale_lst[id % self.obj_scale_num]  # 1

            # read point cloud data
            pc_path = random.choice(self.obj_pc_dict[obj_id])
            raw_pc = np.load(pc_path, allow_pickle=True)
            idx = np.random.choice(
                raw_pc.shape[0], self.config.data.num_points, replace=True
            )
            pc_scale = float(pc_path.split("scale")[1].split("/")[0]) / 100
            pc = raw_pc[idx] / pc_scale * object_scale

            pose_params = load_json(self.obj_pose_dict[obj_id])
            pose_ind = int(pc_path.split("/pose")[-1].split("/")[0])
            pc_pose = np.array(pose_params[pose_ind % len(pose_params)])
            pc_pose[2] *= object_scale

            if self.config.data.model_name == "uni3d":
                delta_bias = np.mean(pc, axis=0)
                pc -= delta_bias[None]
                pc_pose[:3] -= delta_bias

            ret_dict["save_path"] = pjoin(
                rand_grasp_type,
                obj_id,
                f"scale{str(object_scale*100).zfill(3)}_cam{pc_path[:-4].split('partial_pc_')[-1]}.npy",
            )
            ret_dict["obj_path"] = object_path
            ret_dict["obj_scale"] = object_scale
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
        #         "hand_rot": ret_dict["hand_rot"],
        #         "hand_trans": ret_dict["hand_trans"],
        #         "hand_kp": ret_dict["hand_kp"],
        #         "joint_angle": ret_dict["hand_joint"],
        #     },
        # )
        # exit(1)
        ret_dict["point_clouds"] = pc  # (N, 3)
        ret_dict["grasp_type_id"] = int(rand_grasp_type.split("_")[0])
        if "MinkUNet" in self.config.algo.model.backbone.name:
            ret_dict["coors"] = (
                pc / self.config.algo.model.backbone.voxel_size
            )  # (N, 3)
            ret_dict["feats"] = pc  # (N, 3)
        return ret_dict

    def pose_augment(self, obj_pc, robot_trans, robot_rot, obj_trans, obj_rot):
        raise NotImplementedError
        random_trans = np.clip(np.random.randn(1, 3) * 0.02, -0.05, 0.05)
        random_rot = numpy_quaternion_to_matrix(numpy_normalize(np.random.randn(4)))

        aug_obj_pc = obj_pc @ random_rot.T + random_trans
        aug_robot_trans = robot_trans @ random_rot.T + random_trans
        aug_robot_rot = random_rot[None] @ robot_rot
        aug_obj_trans = obj_trans @ random_rot.T + random_trans
        aug_obj_rot = random_rot @ obj_rot

        return aug_obj_pc, aug_robot_trans, aug_robot_rot, aug_obj_trans, aug_obj_rot
