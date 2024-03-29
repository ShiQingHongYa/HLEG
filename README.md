## Learnable Hierarchical Label Embedding and Grouping for Visual Intention Understanding

This is the official implementation of the paper: "Learnable Hierarchical Label Embedding and Grouping for Visual Intention Understanding" accepted by IEEE Transactions on Affective Computing, 2023.

## Abstract

Visual intention understanding is to mine the potential and subjective intention behind the images, which includes the user’s hidden emotions and perspectives. Due to the label ambiguity, this paper presents a novel learnable Hierarchical Label Embedding and Grouping (HLEG). It is featured in three aspects: 1) For effectively mining the underlying meaning of images, we build a hierarchical transformer structure to model the hierarchy of labels, formulating a multi-level classification scheme. 2) For the label ambiguity issue, we design a novel learnable label embedding with accumulative grouping integrated into the hierarchical structure, which does not require additional annotation. 3) For multi-level classification, we propose a ”Hard-First” optimization strategy to adaptively adjust the classification optimization at different levels, avoiding over-classification of the coarse labels. HLEG enhances the F1 score (average +1.24\%) and mAP (average +1.48\%) on Intentonomy over prominent baseline models. Comprehensive experiments validate the superiority of our proposed method, achieving state-of-the-art performance under various settings.

![image](https://github.com/ShiQingHongYa/HLEG/blob/main/images/framework.png)

## Results on Intentonomy

![image](https://github.com/ShiQingHongYa/HLEG/blob/main/images/results.png)


## Visualization

![image](https://github.com/ShiQingHongYa/HLEG/blob/main/images/visual_levels.png)

## Quick start

Training and evaluation are as follows:

```sh
# training
python -m torch.distributed.launch --nproc_per_node=2 train.py 
# evaluation
python -m torch.distributed.launch --nproc_per_node=2 eval.py
```

## File Structure

```
├── HLEG
    ├── data
        ├── intentonomy
    ├── data_utils
        ├── get_datasets.py
        ├── get_label_vector.py
        ├── metrics.py
    ├── images
    ├── models
        ├── backbone.py
        ├── transformer.py
    ├── utils
    ├── _init_paths.py
    ├── training.py
    ├── eval.py
    ├── README.md
```
## Reference

If this work is useful to your research, please cite:

```sh
@article{shi2023learnable,
  title={Learnable Hierarchical Label Embedding and Grouping for Visual Intention Understanding},
  author={Shi, QingHongYa and Ye, Mang and Zhang, Ziyi and Du, Bo},
  journal={IEEE Transactions on Affective Computing},
  year={2023},
  publisher={IEEE}
}
```
