# ContrasTR: Contrastive Learning for Multi-Object Tracking with Transformers

Official implementation of the paper [Contrastive Learning for Multi-Object Tracking with Transformers](https://arxiv.org/abs/2311.08043)

TL;DR: We show how object detection models can be turned into multi-object tracking models with almost no overhead. We also introduce a pre-training scheme on detection that improves tracking without needing annotated tracking IDs.

__Published at WACV 2024. 🏆 State-of-the-art on [BDD100K Tracking dataset](https://eval.ai/web/challenges/challenge-page/1836/leaderboard/4312/mMOTA).__

<table>
  <tr style="border: none!important;padding: 0!important;">
    <td style="border: none!important;padding: 3px!important;"><img alt="predictions on video 18" src="./assets/sample_bdd100k_val_018.gif"></td>
    <td style="border: none!important;padding: 3px!important;"><img alt="predictions on video 113" src="./assets/sample_bdd100k_val_113.gif"></td>
    <td style="border: none!important;padding: 3px!important;"><img alt="predictions on video 147" src="./assets/sample_bdd100k_val_147.gif"></td>
  </tr>
  <tr style="border: none!important;padding: 0!important;">
    <td colspan="3" style="border: none!important;padding: 0!important;"><p style="text-align: center">Visualization of predictions on the validation set of BDD100k</p></td>
  </tr>
</table>


### Evaluation

#### MOT17

<table>
    <tr>
        <th>Split</th>
        <th>Backbone</th>
        <th>Epochs</th>
        <th>HOTA</th>
        <th>MOTA</th>
        <th>IDF1</th>
        <th>IDS</th>
        <th>Download</th>
    </tr>
    <tr>
        <td>train/val</td>
        <td>ResNet50</td>
        <td>15</td>
        <td>63.5</td>
        <td>73.6</td>
        <td>76.4</td>
        <td>331</td>
        <td><a href="https://drive.google.com/file/d/1nLz2vOwk6Hjt-tgufMSh9yDfCeAjeGMP/view?usp=sharing">Checkpoint</a></td>
    </tr>
</table>


To evaluate ContrasTR on the MOT17 validation split, download the "train/val" checkpoint and then run the following command:

```bash
python main.py --eval --resume /path/to/mot/checkpoint --dataset_file mot17 --data_path /path/to/mot17 --batch_size 8 --batch_size_val 4 --video_samples 8 --max_prev_frames 20 --objectness_threshold 0.5 --tracking_threshold 0.5
```


#### BDD100K

<table>
    <tr>
        <th>Backbone</th>
        <th>Epochs</th>
        <th>mHOTA</th>
        <th>mMOTA</th>
        <th>mIDF1</th>
        <th>IDS</th>
        <th>Download</th>
    </tr>
    <tr>
        <td>ResNet50</td>
        <td>10</td>
        <td>40.8</td>
        <td>36.7</td>
        <td>49.2</td>
        <td>6695</td>
        <td><a href="https://drive.google.com/file/d/1Bb1tW2azEdg9lgeV4IIuTLkwFa88eufz/view?usp=sharing">Checkpoint</a></td>
    </tr>
    <tr>
        <td>SwinL</td>
        <td>10</td>
        <td>44.4</td>
        <td>41.7</td>
        <td>52.9</td>
        <td>6363</td>
        <td><a href="https://drive.google.com/file/d/1Cp66z3NR7I1mx6tnugQj9ePZXspsy9_p/view?usp=sharing">Checkpoint</a></td>
    </tr>
</table>

To evaluate ContrasTR (with a ResNet50 backbone) on the BDD100K validation split, download the "ResNet50" checkpoint and then run the following command:

```bash
torchrun --nproc_per_node 4 main_mot.py --eval --resume /path/to/contarstr/resnet/checkpoint --dataset_file bdd100k --data_path /path/to/bdd100k/MOT --batch_size 10 --batch_size_val 10 --video_samples 10 --max_prev_frames 9 --objectness_threshold 0.4 --tracking_threshold 0.4
```

To evaluate ContrasTR (with a SwinL backbone) on the BDD100K validation split, download the "SwinL" checkpoint and then run the following command:

```bash
torchrun --nproc_per_node 4 main_mot.py --eval --resume /path/to/contarstr/swinl/checkpoint --dataset_file bdd100k --data_path /path/to/bdd100k/MOT --batch_size 8 --batch_size_val 8 --backbone swin_L_384_22k --video_samples 8 --max_prev_frames 9 --objectness_threshold 0.4 --tracking_threshold 0.4
```

To generate predictions on the BDD100K test set, download the checkpoint and then run the following command:

```bash
torchrun --nproc_per_node 4 main_mot.py --test --resume /path/to/contarstr/resnet/checkpoint --dataset_file bdd100k --data_path /path/to/bdd100k/MOT --batch_size 10 --batch_size_val 10 --video_samples 10 --max_prev_frames 9 --objectness_threshold 0.4 --tracking_threshold 0.4
torchrun --nproc_per_node 4 main_mot.py --test --resume /path/to/contarstr/swinl/checkpoint --dataset_file bdd100k --data_path /path/to/bdd100k/MOT --batch_size 8 --batch_size_val 8 --backbone swin_L_384_22k --video_samples 8 --max_prev_frames 9 --objectness_threshold 0.4 --tracking_threshold 0.4
```

### Training

#### MOT17

To train our detection model on CrowdHuman with 4 GPUs, run the following command:

```bash
torchrun --nproc_per_node 4 main.py --dataset_file crowdh --data_path /path/to/crowdhuman --epochs 50 --lr_drop 1000 --batch_size 2 --contrastive_pretraining --cont_loss_coef 2.0
```

To train ContrasTR on MOT17, run the following command:

```bash
torchrun --nproc_per_node 4 main_mot.py --dataset_file mot17 --data_path /path/to/mot17 --epochs 15 --lr_drop 10 --lr 2e-5 --lr_backbone 2e-6 --batch_size 4 --batch_size_val 1 --video_samples 8 --max_prev_frames 20 --objectness_threshold 0.5 --tracking_threshold 0.5 --cont_loss_coef 2.0 --from_pretrained --resume /path/to/crowdhuman/checkpoint 
```

#### BDD100K

To train our detection model (with a ResNet50 backbone) on BDD100K with 4 GPUs, run the following command:

```bash
torchrun --nproc_per_node 4 main.py --dataset_file bdd100k --data_path /path/to/bdd100k/detection --epochs 36 --lr_drop 1000 --batch_size 6 --contrastive_pretraining --cont_loss_coef 2.0
```

To train ContrasTR (with a ResNet50 backbone) on BDD100K MOT, run the following command:

```bash
torchrun --nproc_per_node 4 main_mot.py --dataset_file bdd100k --data_path /path/to/bdd100k/MOT --epochs 10 --lr_drop 8 --lr 2e-5 --lr_backbone 2e-6 --batch_size 10 --batch_size_val 10 --video_samples 10 --max_prev_frames 9 --objectness_threshold 0.4 --tracking_threshold 0.4 --from_pretrained --resume /path/to/detection/checkpoint
```

To train our detection model (with a SwinL backbone) on BDD100K with 4 GPUs, run the following command:

```bash
torchrun --nproc_per_node 4 main.py --dataset_file bdd100k --data_path /path/to/bdd100k/detection --epochs 36 --lr_drop 1000 --backbone swin_L_384_22k --swin_checkpoint /path/to/swinL/cls/checkpoint --swin_checkpointing --batch_size 3 --contrastive_pretraining --cont_loss_coef 2.0 --bdd_smaller_scales
```

To train ContrasTR (with a SwinL backbone) on BDD100K MOT, run the following command:

```bash
torchrun --nproc_per_node 4 main_mot.py --dataset_file bdd100k --data_path /path/to/bdd100k/MOT --epochs 10 --lr_drop 8 --lr 2e-5 --lr_backbone 2e-6 --backbone swin_L_384_22k --swin_checkpointing --batch_size 8 --batch_size_val 8 --video_samples 8 --max_prev_frames 9 --objectness_threshold 0.4 --tracking_threshold 0.4 --from_pretrained --resume /path/to/detection/checkpoint
```


## Installation

### Requirements

* Linux, CUDA>=12.13, GCC<=12.2
  
* Python>=3.10

We recommend you to use Anaconda to create a virtual environment:

```bash
conda create -n contrastr python=3.10.4 pip
```
Then, activate the environment:

```bash
conda activate contrastr
```
  
* PyTorch>=2.3.0, torchvision>=0.18.0 (following instructions [here](https://pytorch.org/))

```bash
conda install numpy=1.26 pytorch==2.3.0 torchvision==0.18.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```
  
* Other requirements

```bash
conda install pydantic==1.10.12
pip install -r requirements.txt
```

### Compiling CUDA operators
```bash
cd ./models/ops
sh ./make.sh
# unit test (should see all checking is True)
python test.py
```

## Usage

### COCO preparation

Please download the [COCO 2017 dataset](https://cocodataset.org/) and organize it as follows:

```
code_root/
└── data/
    └── coco/
        ├── train2017/
        ├── val2017/
        └── annotations/
        	├── instances_train2017.json
        	└── instances_val2017.json
```

### BDD100K preparation

Please download from [BDD100K](https://bdd-data.berkeley.edu): "100K Images", "Detection 2020 Labels", "MOT 2020 Images"  and "MOT 2020 Labels" and organize it somewhere as follows:

```text
bdd100k
├── images
│   ├── 100k
│   │   ├── test
│   │   ├── train
│   │   └── val
│   └── track
│       ├── test
│       ├── train
│       └── val
└── labels
    ├── box_track_20
    │   ├── train
    │   └── val
    └── det_20
        ├── det_train.json
        └── det_val.json
```

Then organize the detection folder structure as follows:

```text
data/bdd100k_2ddet/
├── test
│   └── images -> /path/to/bdd100k/images/100k/test
├── train
│   ├── annotations
│   │   └── instances_train_half.json
│   └── images -> /path/to/bdd100k/images/100k/train/
└── val
    ├── annotations
    │   └── instances_val_half.json
    └── images -> /path/to/bdd100k/images/100k/val
```

And the MOT folder structure as follows:

```text
data/bdd100k_mot/
├── bdd100k_gt_eval
│   └── val -> /path/to/bdd100k/labels/box_track_20/val
├── test
│   ├── annotations
│   │   └── instances_test.json
│   └── images -> /path/to/bdd100k/images/track/test/
├── train
│   ├── annotations
│   │   ├── images_dimensions.json
│   │   └── instances_train_half.json
│   └── images -> /path/to/bdd100k/images/track/train
└── val
    ├── annotations
    │   ├── images_dimensions.json
    │   └── instances_val_half.json
    └── images -> /path/to/bdd100k/images/track/val
```

Convert the annotations into COCO format with official conversion scripts:

```bash
python -m bdd100k.label.to_coco -m det -i /path/to/labels/det_20/det_train.json -o data/bdd100k_2ddet/train/annotations/instances_train_half.json
python -m bdd100k.label.to_coco -m det -i /path/to/labels/det_20/det_val.json -o data/bdd100k_2ddet/val/annotations/instances_val_half.json
python -m bdd100k.label.to_coco -m box_track -i /path/to/labels/box_track_20/train/ -o data/bdd100k_mot/train/annotations/instances_train_half.json
python -m bdd100k.label.to_coco -m box_track -i /path/to/labels/box_track_20/val/ -o data/bdd100k_mot/val/annotations/instances_val_half.json
```

Create a dummy annotation file in COCO format for the test set (if you want to submit to the BDD100K leaderboard):

```bash
python util/bdd100kmottest_to_coco.py --data_path /path/to/bdd100k/images/track/test/
```

Link the MOT val folder to a local path

```bash
ln -s /path/to/bdd100k/labels/box_track_20/val data/bdd100k_mot/bdd100k_gt_eval/val
```

To visualise predictions or annotations:

```bash
python -m bdd100k.vis.viewer -i data/bdd100k_mot/test/images/ -l checkpoints/experiment_name/submission_files/trackscore_0.5/data/cba301b8-7ef6e928.json
```

### CrowdHuman preparation

Please download [CrowdHuman](https://www.crowdhuman.org) and organise it as follows:

```text
CrowdHuman/
├── annotation_train.odgt
├── annotation_val.odgt
└── Images
```

Then create the following folder structure:

```text
data/crowdhuman/
├── train
│   ├── annotations
│   │   └── instances_train_half.json
│   └── images -> /path/to/CrowdHuman/Images
└── val
    ├── annotations
    │   └── instances_val_half.json
    └── images -> /path/to/CrowdHuman/Images
```

Afterward, execute the following commands:

```bash
python util/crowdhuman2coco.py -d data/crowdhuman/train/images/ -o /path/to/CrowdHuman/annotation_train.odgt -s data/crowdhuman/train/annotations/instances_train_half.json --rm-occ 0

python util/crowdhuman2coco.py -d data/crowdhuman/val/images/ -o /path/to/CrowdHuman/annotation_val.odgt -s data/crowdhuman/val/annotations/instances_val_half.json --rm-occ 0
```

### MOT17 preparation

Please download [MOT17](https://motchallenge.net/data/MOT17/). Then, execute the following command:

```bash
python util/mot17_to_coco.py --data_path path/to/MOT17
```

to obtain the following folder structure:

```text
data/MOT17/
├── mot17_gt_eval
│   ├── seqmaps
│   │   └── MOT17-train.txt
│   ├── train
│   │   ├── MOT17-02-FRCNN
│   │   ├── MOT17-04-FRCNN
│   │   ├── MOT17-05-FRCNN
│   │   ├── MOT17-09-FRCNN
│   │   ├── MOT17-10-FRCNN
│   │   ├── MOT17-11-FRCNN
│   │   └── MOT17-13-FRCNN
│   └── val
│       ├── MOT17-02-FRCNN
│       ├── MOT17-04-FRCNN
│       ├── MOT17-05-FRCNN
│       ├── MOT17-09-FRCNN
│       ├── MOT17-10-FRCNN
│       ├── MOT17-11-FRCNN
│       └── MOT17-13-FRCNN
├── test
│   ├── annotations
│   │   └── instances_test.json
│   └── images -> /path/to/MOT17/test
├── train
│   ├── annotations
│   │   └── instances_train_half.json
│   └── images -> /path/to/MOT17/train
├── train_full
│   ├── annotations
│   │   └── instances_train_full.json
│   └── images -> /path/to/MOT17/train
└── val
    ├── annotations
    │   └── instances_val_half.json
    └── images -> /path/to/MOT17/train
```


We would like to thank the authors of [DETR](https://github.com/facebookresearch/detr), [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR) and [DINO](https://github.com/IDEA-Research/DINO) for their amazing works and codebases.

## Citation

If you make use of this code, please refer to us as:

```bibtex
@inproceedings{ContrasTR_WACV_2024,
  title={Contrastive Learning for Multi-Object Tracking with Transformers},
  author={De Plaen, Pierre-Fran{\c{c}}ois and Marinello, Nicola and Proesmans, Marc and Tuytelaars, Tinne and Van Gool, Luc},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  month={January},
  year={2024}
  pages={6867--6877},
}
```
