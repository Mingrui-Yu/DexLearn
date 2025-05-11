# DexGraspLearn

Learning-based grasp synthesis baselines (e.g., regression, cvae, diffusion policy, normalizing flow) for dexterous hands, used in [BODex (ICRA 2025)](https://pku-epic.github.io/BODex/) and [Dexonomy (RSS 2025)](https://pku-epic.github.io/Dexonomy/)


## Installation
```bash
git submodule update --init --recursive --progress

conda create -n dexlearn python=3.10 
conda activate dexlearn

# pytorch
conda install pytorch==2.0.1 pytorch-cuda=11.7 -c pytorch -c nvidia 

# pytorch3d
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch3d/linux-64/pytorch3d-0.7.8-py310_cu118_pyt210.tar.bz2
conda install -y --use-local ./pytorch3d-0.7.8-py310_cu118_pyt210.tar.bz2


# Diffusers 
cd third_party/diffusers
pip install -e .
cd ...

# MinkowskiEngine
cd third_party/MinkowskiEngine
sudo apt install libopenblas-dev
export CUDA_HOME=/usr/local/cuda-11.7
python setup.py install --blas=openblas
cd ...

# nflows
cd third_party/nflows
pip install -e .
cd ...

# dexlearn
pip install -e .
```

## Quick Start
1. Prepare object and grasp pose assets. Download from [hugging face](https://huggingface.co/datasets/JiayiChenPKU/Dexonomy). The folder should be organized as below:
```
DexGraspLearn/assets
|- grasp
|   |_ Dexonomy_GRASP_shadow
|        |_ succgrasp
|_ object
    |- DGN_5k
    |   |- valid_split
    |   |- processed_data
    |   |_ vision_data
    |_ objaverse_5k
        |- valid_split
        |- processed_data
        |_ vision_data
```

2. Running. 
```bash
CUDA_VISIBLE_DEVICES=4 python -m dexlearn.train exp_name=first
```

## License

This work and the dataset are licensed under [CC BY-NC 4.0][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png

## Citation

If you find this work useful for your research, please consider citing:
```
@article{chen2024bodex,
  title={BODex: Scalable and Efficient Robotic Dexterous Grasp Synthesis Using Bilevel Optimization},
  author={Chen, Jiayi and Ke, Yubin and Wang, He},
  journal={arXiv preprint arXiv:2412.16490},
  year={2024}
}
@article{chen2025dexonomy,
        title={Dexonomy: Synthesizing All Dexterous Grasp Types in a Grasp Taxonomy},
        author={Chen, Jiayi and Ke, Yubin and Peng, Lin and Wang, He},
        journal={Robotics: Science and Systems},
        year={2025}
      }
```

## Acknowledgment

Thanks [@haoranliu](https://lhrrhl0419.github.io/) for his codebase and help in normalizing flow.