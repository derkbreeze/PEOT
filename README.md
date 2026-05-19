# PEOT

![PEOT](figures/teaser.png)

Official implementation of the paper "Learning Probabilistic Embeddings for Unsupervised Action Segmentation"

## Prepare your data
The `data` folder should be aranged in the following way:

```data
data
|--Breakfast
|  `--features
|     `--cereals
|        `--P03_cam01_P03_cereals.txt
|        `...
|     `--coffee
|     `--friedegg
|     `...
|  `--groundTruth
|     `--P03_cam01_P03_cereals
|     `...
|  `--mapping
|     `--mapping.txt
|
|--YTI
|  `...
|
|--FS
|  `...
|--desktop_assembly
|  `...
```

## Training

`python train.py -s -lc 1e-4 d Breakfast`

for `non probabilistic` version of Breakfast dataset

`python train.py -pr -s -lc 1e-4 -d Breakfast`

for `probabilistic` version of Breakfast dataset

## Inference
`python test.py`

and `python test.py -v` if you want to visualize segmentations.

If you have any questions using this code, please open an issue. I'll respond ASAP.

## Citing
If you find this code useful in your research, please consider citing:
```bibtex
@inproceedings{li2022learning,
  title={Learning of Global Objective for Network Flow in Multi-Object Tracking},
  author={Li, Shuai and Kong, Yu and Rezatofighi, Hamid},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={8855--8865},
  year={2022}
}
