from typing import Dict
import json 
import torch 
import random 
import numpy as np

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    

def load_json(file_path):
    if isinstance(file_path, str):
        with open(file_path) as file_p:
            json_params = json.load(file_p)
    else:
        json_params = file_path
    return json_params


def write_json(data: Dict, file_path):
    with open(file_path, "w") as file:
        json.dump(data, file, indent=1)