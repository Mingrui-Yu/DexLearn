import numpy as np

file = "/home/mingrui/mingrui/research/adaptive_grasp/DexGraspBench/output/bodex_tabletop_leap_tac3d/succ_collect/core_bottle_1a7ba1f4c892e2da30711cdbdbc73924/tabletop_ur10e/scale006_pose004_0.npy"

data = np.load(file, allow_pickle=True).item()

a = 1