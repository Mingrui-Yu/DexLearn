# DexGraspLearn

Learning-based grasp synthesis baselines (e.g., regression, cvae, diffusion policy, normalizing flow) for dexterous hands, used in [BODex](https://pku-epic.github.io/BODex/) and [Dexonomy](https://pku-epic.github.io/Dexonomy/)


## Installation
```bash
git submodule update --init --recursive --progress

conda create -n dexlearn python=3.10 
conda activate dexlearn

conda install pytorch==2.5.0 pytorch-cuda=12.1 -c pytorch -c nvidia 

# Diffusers 
cd third_party/diffusers
pip install -e .
cd ...

# MinkowskiEngine
cd third_party/MinkowskiEngine
sudo apt install libopenblas-dev
python setup.py install --blas=openblas
cd ...

# nflows
cd third_party/nflows
pip install -e .
cd ...

pip install -e .
```

## Quick Start

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