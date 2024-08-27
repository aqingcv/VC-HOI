# Leveraging Visual Captions for Enhanced Zero-Shot HOI Detection

## Dataset 
Follow the process of [UPT](https://github.com/fredzzhang/upt).

The downloaded files should be placed as follows. Otherwise, please replace the default path to your custom locations.
```
|- ADA-CM
|   |- hicodet
|   |   |- hico_20160224_det
|   |       |- annotations
|   |       |- images
|   |- vcoco
|   |   |- mscoco2014
|   |       |- train2014
|   |       |-val2014
:   :      
```

## Dependencies
1. Follow the environment setup in [UPT](https://github.com/fredzzhang/upt).

2. Our code is built upon [CLIP](https://github.com/openai/CLIP). Install the local package of CLIP:
```
cd CLIP && python setup.py develop && cd ..
```

3. Download the CLIP weights to `checkpoints/pretrained_clip`.
```
|- ADA-CM
|   |- checkpoints
|   |   |- pretrained_clip
|   |       |- ViT-B-16.pt
|   |       |- ViT-L-14-336px.pt
:   :      
```

4. Download the weights of DETR and put them in `checkpoints/`.


| Dataset | DETR weights |
| --- | --- |
| HICO-DET | [weights](https://drive.google.com/file/d/1BQ-0tbSH7UC6QMIMMgdbNpRw2NcO8yAD/view?usp=sharing)  |
| V-COCO | [weights](https://drive.google.com/file/d/1AIqc2LBkucBAAb_ebK9RjyNS5WmnA4HV/view?usp=sharing) |


```
|- ADA-CM
|   |- checkpoints
|   |   |- detr-r50-hicodet.pth
|   |   |- detr-r50-vcoco.pth
:   :   :
```

## Pre-extracted Features
Download the pre-extracted features from [HERE](https://drive.google.com/file/d/1lUnUQD3XcWyQdwDHMi74oXBcivibGIWN/view?usp=sharing) and the pre-extracted bboxes from [HERE](https://drive.google.com/file/d/1xHGr36idtYSzMYGHKvvxMJyTiaq317Ev/view?usp=sharing). The downloaded files have to be placed as follows.


### HICO-DET
#### Train on HICO-DET:
```
cd ./VC-HOI;
CUDA_VISIBLE_DEVICES=0,1 python main_tip_finetune.py --world-size 2 --pretrained checkpoints/detr-r50-hicodet.pth --output-dir checkpoints/hico_zs_rf_uc --use_insadapter --num_classes 117 --clip_dir_vit checkpoints/pretrained_clip/ViT-B-16.pt --zs --zs_type rare_first --port 1335 

cd ./VC-HOI;
CUDA_VISIBLE_DEVICES=0,1 python main_tip_finetune.py --world-size 2 --pretrained checkpoints/detr-r50-hicodet.pth --output-dir checkpoints/hico --use_insadapter --num_classes 117 --clip_dir_vit checkpoints/pretrained_clip/ViT-B-16.pt --port 1336 

```

#### Test on HICO-DET:
```
CUDA_VISIBLE_DEVICES=1 python main_tip_finetune.py --world-size 1 --pretrained checkpoints/detr-r50-hicodet.pth --output-dir checkpoints/hico --use_insadapter --num_classes 117 --clip_dir_vit checkpoints/pretrained_clip/ViT-B-16.pt --eval --resume ./VC-HOI/checkpoints/hico_zs_rf_uc --zs --zs_type rare_first --port 1336


CUDA_VISIBLE_DEVICES=1 python main_tip_finetune.py --world-size 1 --pretrained checkpoints/detr-r50-hicodet.pth --output-dir checkpoints/hico --use_insadapter --num_classes 117 --clip_dir_vit checkpoints/pretrained_clip/ViT-B-16.pt --eval --resume ./VC-HOI/checkpoints --port 1336
```

### V-COCO
#### Training on V-COCO
```
cd ./VC-HOI;
CUDA_VISIBLE_DEVICES=0,1 python main_tip_finetune.py --world-size 2 --dataset vcoco --data-root ./vcoco --partitions trainval test --pretrained checkpoints/detr-r50-vcoco.pth --output-dir checkpoints/vcoco --use_insadapter --num_classes 24 --clip_dir_vit checkpoints/pretrained_clip/ViT-B-16.pt --port 1337
```

#### Cache detection results for evaluation on V-COCO
```
python main_tip_finetune.py --world-size 1 --dataset vcoco --data-root ./vcoco --partitions trainval test --pretrained checkpoints/detr-r50-vcoco.pth --output-dir checkpoints/vcoco --use_insadapter --num_classes 24 --clip_dir_vit checkpoints/pretrained_clip/ViT-B-16.pt --cache --resume ./checkpoints

python eval_vcoco.py ./checkpoints/vcoco/cache.pkl
```

### Model Zoo


## Acknowledgement
We gratefully thank the authors from [UPT](https://github.com/fredzzhang/upt) for open-sourcing their code.

