# SG-STSM
This code is currently under submission to The Visual Computer.

## Environment
- Python 3.9
- PyTorch 2.3.1
- causal_conv1d-1.4.0
- mamba_ssm-2.2.2

For installation of the project dependencies, please run:
```
pip install -r requirements.txt
``` 
## Dataset
### Human3.6M
#### Preprocessing
1. Download the fine-tuned Stacked Hourglass detections of [MotionBERT](https://github.com/Walter0807/MotionBERT/blob/main/docs/pose3d.md)'s preprocessed H3.6M data [here](https://1drv.ms/u/s!AvAdh0LSjEOlgU7BuUZcyafu8kzc?e=vobkjZ) and unzip it to 'data/3d'.
2. Slice the motion clips by running the following python code in `data/preprocess` directory:

**For 243 frames**:
```text
python h36m.py  --n-frames 243
```

**For 81 frames**:
```text
python h36m.py --n-frames 81
```

### MPI-INF-3DHP
#### Preprocessing
Please refer to [P-STMO](https://github.com/paTRICK-swk/P-STMO#mpi-inf-3dhp) for dataset setup. After preprocessing, the generated .npz files (`data_train_3dhp.npz` and `data_test_3dhp.npz`) should be located at `data/3d` directory.
#### Visualization
Run it same as the visualization for Human3.6M, but `--dataset` should be set to `mpi`.

## Demo
Our demo is a modified version of the one provided by [MHFormer](https://github.com/Vegetebird/MHFormer) repository. First, you need to download YOLOv3 and HRNet pretrained models [here](https://drive.google.com/drive/folders/1_ENAMOsPM7FXmdYRbkwbFHgzQq_B_NQA?usp=sharing) and put it in the './demo/lib/checkpoint' directory. Next, download our base model checkpoint from [here](https://drive.google.com/file/d/1Iii5EwsFFm9_9lKBUPfN8bV5LmfkNUMP/view) and put it in the './checkpoint' directory. Then, you need to put your in-the-wild videos in the './demo/video' directory.

Run the command below:
```
python demo/vis.py --video sample_video.mp4
```
Sample demo output:

<p align="center"><img src="figure/sample_video.gif" width="60%" alt="" /></p>

## Acknowledgement
Our code refers to the following repositories:

- [MotionBERT](https://github.com/Walter0807/MotionBERT)
- [P-STMO](https://github.com/paTRICK-swk/P-STMO)
- [MHFormer](https://github.com/Vegetebird/MHFormer)
- [MotionAGFormer](https://github.com/TaatiTeam/MotionAGFormer)

We thank the authors for releasing their codes.

```
