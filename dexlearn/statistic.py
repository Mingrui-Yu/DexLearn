import sys
import os 
import argparse

import torch
import hydra
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dexgrasp.utils.logger import Logger
from dexgrasp.utils.util import set_seed
from dexgrasp.dataset import create_test_dataloader
from dexgrasp.network.models import get_model


def main_func(config: DictConfig) -> None:
    set_seed(config.seed)
    config.wandb.mode = 'disabled'
    test_loader = create_test_dataloader(config, mode='train')
    
    count = 0
    for data in tqdm(test_loader):
        print(data['grasp_num'].shape)
        count += data['grasp_num'].sum()
    print(count)
    return 

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-e",
        "--exp_name",
        type=str,
        required=True,
        help="experiment folder, e.g. shadow_tabletop_debug",
    )
    args, unknown = parser.parse_known_args()
    
    sys.argv = sys.argv[:1] + unknown + list(OmegaConf.load(f"../../output/experiment/{args.exp_name}/.hydra/overrides.yaml"))
    
    # remove duplicated args. Note: cmd has the priority!
    check_dict = {}
    for argv in sys.argv[1:]:
        arg_key = argv.split('=')[0]
        if arg_key not in check_dict:
            check_dict[arg_key] = True 
        else:
            sys.argv.remove(argv)
            
    hydra.main(config_path="config", config_name="base", version_base=None)(main_func)()
    