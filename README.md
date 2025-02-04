<p align="center">
    <img src="assets/disco_logo.png" width="400">
</p>


## DisCO: Portrait Distortion Correction with Perspective-Aware 3D GANs


[![arXiv](https://img.shields.io/badge/arXiv-2302.12253-b31b1b.svg)](https://arxiv.org/abs/2302.12253) ![visitors](https://visitor-badge.laobi.icu/badge?page_id=lightChaserX/DisCO) [![project](https://img.shields.io/badge/project-page-8A2BE2)](https://portrait-disco.github.io/)



[Zhixiang Wang](https://lightchaserx.github.io/)<sup>1,2</sup>, [Yu-Lun Liu](https://yulunalexliu.github.io/)<sup>3</sup>, [Jia-Bin Huang](https://jbhuang0604.github.io/)<sup>4</sup>, [Shin'ichi Satoh](http://research.nii.ac.jp/~satoh/index.html)<sup>2,1</sup>, [Sizhuo Ma](https://sizhuoma.netlify.app/)<sup>5</sup>,   
[Guru Krishnan](https://research.snap.com/team/team-member.html#guru-krishnan)<sup>5</sup>, [Jian Wang](https://jianwang-cmu.github.io/)<sup>5</sup>

<sup>1</sup>University of Tokyo&nbsp;&nbsp;<sup>2</sup>NII&nbsp;&nbsp;<sup>3</sup>NYCU&nbsp;&nbsp;<sup>4</sup>University of Maryland, College Park&nbsp;&nbsp;<sup>5</sup>Snap Inc. 

<img width="667" alt="image" src="https://github.com/lightChaserX/DisCO/assets/11884079/69cc113e-5e9d-48d2-8a3c-d89f4ee65fc3">



## :book:Table Of Contents

- [Update](#update)
- [TODO](#todo)
- [Installation](#installation)
- [Quick Start](#quick_start)

## <a name="update"></a>:new:Update

- **2025.02.04**: ✅ Inversion code release.
- **2024.01.03**: 🚀 Paper is accepted to IJCV.

## <a name="todo"></a>TODO

- [ ] Full system release
- [ ] Killer video release
- [ ] Hugging Face demo release 

## <a name="installation"></a>:gear:Installation

1. Clone EG3D and ensure `Deep3DFaceRecon_pytorch` properly initialized
```shell
git clone https://github.com/NVlabs/eg3d.git
cd eg3d
git submodule update --init --recursive
```

2. Download the checkpoint on FFHQ from <a href="https://catalog.ngc.nvidia.com/orgs/nvidia/teams/research/models/eg3d">NGC Catalog</a>
```shell
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/research/eg3d/1/files?redirect=true&path=ffhq512-128.pkl' -O ffhq512-128.pkl

cd ..
```

2. Clone this repo and install environment
```shell
git clone git@github.com:lightChaserX/DisCO.git
cd DisCO
conda env create -f environment.yml
```



## <a name="quick_start"></a>:flight_departure:Quick Start

1. Modify `example_configs/config.py` accordingly, including the path to input images, pre-trained model weight, etc
2. Process your data
```
python process_in_one_time.py
```
3. Run the following command
```
python run_in_one_time.py
```

## Citation

If you find our work useful, please kindly cite as:
```
@article{wang2024disco,
      title={DisCO: Portrait Distortion Correction with Perspective-Aware 3D GANs},
      author={Wang, Zhixiang and Liu, Yu-Lun and Huang, Jia-Bin and Satoh, Shin'ichi and Ma, Sizhuo and Krishnan, Guru and Wang, Jian},
      journal={International Journal of Computer Vision},
      year={2024}
    }
```


## Acknowledge
This code is built on the following code base <a href='https://github.com/NVlabs/eg3d'>EG3D</a>, <a href='https://github.com/danielroich/PTI'>PTI</a>,  <a href='https://github.com/rotemtzaban/STIT'>STIT</a>, and <a href="https://github.com/vt-vl-lab/3d-photo-inpainting">3d-photo-inpainting</a>. Our functionality also depends on <a href="https://github.com/isl-org/MiDaS">MiDaS</a> for depth estimation, <a href="https://github.com/ZHKKKe/MODNet">MODNet</a> for image matting, <a href="https://github.com/CompVis/stable-diffusion">SD1.5</a> or DALLE2 for background inpainting. 
