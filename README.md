# Learning Convex Decomposition via Feature Fields
**[Project](https://research.nvidia.com/labs/sil/projects/learning-convex-decomp/)**|**[Video](https://www.youtube.com/watch?v=ziSyNg3BACI&t=8s)**|**[Arxiv](https://arxiv.org/abs/2603.09285/)**
 
Yuezhi Yang, Qixing Huang, Mikaela Angelina Uy*, Nicholas Sharp*
 
CVPR 2026 (Oral)

## Overview
<img src="assets/teaser.png" alt="Alt text" width="400" height="400">

This work proposes a new formulation to the long-standing problem of convex decomposition through learning feature fields, enabling the first feed-forward model for open-world convex decomposition. Our method produces high-quality decompositions of 3D shapes into a union of convex bodies, which are essential to accelerate collision detection in physical simulation, amongst many other applications. The key insight is to adopt a feature learning approach and learn a continuous feature field that can later be clustered to yield a good convex decomposition via our self-supervised, purely-geometric objective derived from the classical definition of convexity. Our formulation can be used for single shape optimization, but more importantly, feature prediction unlocks scalable, self-supervised learning on large datasets resulting in the first learned open-world for convex decomposition. Experiments show that our decompositions are higher-quality than alternatives and generalize across open-world objects as well as across representations to meshes, CAD models, and even Gaussian splats.

## Pretrained Model
The link to download our pretrained model is here: [Checkpoint](https://huggingface.co/mikaelaangel/learning-convex-decomp-ckpt/tree/main). Please download model_convexdecomp.ckpt and put it under ./ckpt directory.

## Environment Set-up
We use Python 3.10 with PyTorch 2.4 and CUDA 12.4. The environment and required packages can be installed individually as follows:
```
conda create -n convex_decomposition python=3.10
conda activate convex_decomposition
conda install -y nvidia/label/cuda-12.4.0::cuda 
pip install psutil
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install lightning==2.2 h5py yacs trimesh scikit-image loguru pytz scikit-learn
pip install Werkzeug==2.2.2  pymeshlab  libigl==2.6.1 embreex einops  rtree pyfqmr  matplotlib boto3 numba pymeshlab==2023.12.post3
pip install mesh2sdf tetgen vtk
conda install -y -c conda-forge cmake=3.27 ninja pybind11 compilers
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
```

An environment file is also provided and can be used for installation:
```
conda env create -f environment.yml
conda activate convex_decomposition
```

## Inference
You can either perform convex decomposition on a single file or process the entire directory in batch. You will find the resulting convex components in `./exp_results/<config_name>/<result_name>`.
```
python test.py -c ./configs/<config_name> result_name <result_name> dataset.input_path <data_path> ckpt_path <ckpt_path>
```

For a single mesh file:
```
python test.py -c configs/example.yaml result_name result_data dataset.input_path ./data/camel_mc.ply ckpt_path ./ckpt/model_objaverse.ckpt
```

For a data directory:
```
python test.py -c configs/example.yaml result_name result_data dataset.input_path ./data ckpt_path ./ckpt/model_objaverse.ckpt
```



You can also adjust the termination condition of the algorithm, the decomposition will stop when each components has \<concavity\> score lower than epsilon OR number of convex components is larger than \<num_parts\>
```
python test.py -c ./configs/<config_name> result_name <result_name> dataset.input_path <data_path> ckpt_path <ckpt_path> decomp.eps <epsilon> decomp.max_parts <num_parts>
```
For example:
```
python test.py -c configs/example.yaml result_name exp_data dataset.input_path ./data ckpt_path ./ckpt/model_objaverse.ckpt decomp.eps 0.10 decomp.max_parts 50
```

The algorithm works best if input mesh is manifold and watertight. An optional watertight remesh pre-process is:
```
cd preprocess
python preprocess_input.py --input_dir <input_dir> --output_dir <output_dir>
```

## Training
We train our model in curated subset of Objaverse. You could download these shapes from [Objaverse](https://objaverse.allenai.org/).

Training consumes 256^3 SDF `.h5` files. Generate them with:

```
cd preprocess
python batch_sample_sdf.py --input_dir <input_dir> --output_dir <output_dir>
```



You could start training by runing following commands. You should set `dataset.data_path` to the folder containing your generated `.h5` folder.
```
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node=8 train.py -c configs/example.yaml dataset.data_path <sdf_h5_dir>
```

You should also update the command to reflect the number of GPUs used. We train our model using 8× A100 GPUs.

## Citation
```
@inproceedings{learningconvexdecomp2026,
      title={Learning Convex Decomposition via Feature Fields}, 
      author={Yuezhi Yang and and Qixing Huang and Mikaela Angelina Uy and Nicholas Sharp},
      booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
      year = {2026}
}
```

## References  
We borrow code from the following repositories:  
- [OpenLRM](https://github.com/3DTopia/OpenLRM)  
- [PyTorch 3D UNet](https://github.com/wolny/pytorch-3dunet)  
- [PVCNN](https://github.com/mit-han-lab/pvcnn)  
- [GenSDF](https://github.com/princeton-computational-imaging/gensdf)  

Many thanks to the authors for sharing their code! Review the license terms of this project before use.

## License
This project will download and install additional third-party softwares. Note that these softwares are not distributed by NVIDIA. Review the license terms of these models and projects before use.

This source code is released under the [Apache 2 License](https://www.apache.org/licenses/LICENSE-2.0). Attribution for bundled and adapted third-party code is provided in the root [NOTICE](./NOTICE) file.

See [CONTRIBUTING.md](./CONTRIBUTING.md) for information on how to submit contributions, including the required Developer Certificate of Origin sign-off.
