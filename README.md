# LIFe-GoM: Generalizable Human Rendering with Learned Iterative Feedback Over Multi-Resolution Gaussians-on-Mesh

ICLR 2025

[Paper](https://arxiv.org/abs/2502.09617) | [Project Page](https://wenj.github.io/LIFe-GoM/)

```bibtex
@inproceedings{wen2025lifegom,
    title={{LIFe-GoM: Generalizable Human Rendering with Learned Iterative Feedback Over Multi-Resolution Gaussians-on-Mesh}},
    author={Jing Wen and Alex Schwing and Shenlong Wang},
    booktitle={ICLR},
    year={2025}
}
```

## Requirements

Our codes are tested in
* CUDA 11.6
* PyTorch 1.13.0
* PyTorch3D 0.7.0

Install the required packages:
```Shell
conda create -n GoMAvatar python=3.8
conda activate GoMAvatar

conda install pytorch==1.13.0 torchvision==0.14.0 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -r requirements.txt

# install pointops 
git clone https://github.com/POSTECH-CVLab/point-transformer.git
cd lib/pointops
python setup.py install
```
If you meet the error `fatal error: THC/THC.h: No such file or directory`, comment out `#include <THC/THC.h>`.

Download the pretrained weights from [this link](https://uofi.box.com/s/bsm1so0fnpstwco1suxchiq9gky3d57o) and put it under this directory.

## Data preparation

Please refer to [DATASET.md](docs/DATASET.md) for data preparation.

## Inference
You can find the checkpoints and rendered results [here](https://uofi.box.com/s/gi8sv875hard3eopjrmly8yxx2r1w7k8).

### Freeview rendering
To render 360 degree freeview images, run:
```Shell
python inference.py --type freeview --cfg $CONFIG_FILE --scene_idx $SCENE_IDX
```
For example, `python inference.py --type freeview --cfg exps/exp_thuman2.0_view3.yaml --scene_idx 0` renders the 360 degree of the first test scene in THuman2.0.

### Novel pose synthesis
Run the following command to synthesize novel poses:
```Shell
python inference.py --type pose --cfg $CONFIG_FILE --scene_idx $SCENE_IDX --pose_path $POSE_FILE --pose_name $NAME
```
An example of `$POSE_FILE` is in `data/pose_example.npz`. You can try `python inference.py --type pose --cfg exps/exp_thuman2.0_view3.yaml --pose_path data/pose_example.npz --pose_name example`.

### Novel view synthesis & evaluation
To replicate the quantitative results in the paper, you can either download the rendered images [here](https://uofi.box.com/s/gi8sv875hard3eopjrmly8yxx2r1w7k8) or render the test set on your own:
```Shell
# 3 input views on THuman2.0
python inference.py --cfg exps/exp_thuman2.0_view3.yaml
# 3 input views on THuman2.0; cross-domain generalization
python inference.py --cfg exps/exp_xhuman_view3_evalonly.yaml
# 5 input views on THuman2.0
python inference.py --cfg exps/exp_thuman2.0_view5.yaml
# 5 input views on AIST++
python inference.py --cfg exps/exp_aistpp_view5.yaml
```
Then, run the evaluation protocol on THuman2.0:
```Shell
# 3 input views
python scripts/eval/compute_metrics_thuman.py --gt_path ./data/thuman2.0/view3_val --ours_path ./log/exp_thuman2.0_view3/eval/view --n_input_frames 3
# 5 input views
python scripts/eval/compute_metrics_thuman.py --gt_path ./data/thuman2.0/view5_val --ours_path ./log/exp_thuman2.0_view5/eval/view --n_input_frames 5
```

Run the following command to evaluate the cross-domain generalization on XHuman:
```Shell
python scripts/eval/compute_metrics_thuman.py --gt_path ./data/xhuman/view3_val --ours_path ./log/exp_xhuman_view3/eval/view --n_input_frames 3
```

On AIST++:
```Shell
python scripts/eval/compute_metrics_aistpp.py --gt_path ./data/aistpp/view5_trainval --ours_path ./log/exp_aistpp_view5/eval/view
```

## Training
```Shell
# 3 input views on THuman2.0
python train.py --cfg exps/exp_thuman2.0_view3.yaml
# 5 input views on THuman2.0
python train.py --cfg exps/exp_thuman2.0_view5.yaml
# 5 input views on AIST++
python train.py --cfg exps/exp_aistpp_view5.yaml
```

## Acknowledgement
This project builds on [HumanNeRF](https://github.com/chungyiweng/humannerf), [GHG](https://github.com/humansensinglab/Generalizable-Human-Gaussians), [ActorsNeRF](https://github.com/JitengMu/ActorsNeRF) and [GoMAvatar](https://github.com/wenj/GoMAvatar). We thank the authors for releasing codes.
