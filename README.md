# NYCU Computer Vision 2025 Spring HW2
- StudentID: 313553044
- Name: 江仲恩

## Introduction
In this assignment, we aim to develop a robust digit detection model capable of accurately localizing and classifying digits in each image from the HW2 dataset. To achieve strong generalization performance, we adopt a Faster R-CNN framework with a ResNet-50 backbone, a Region Proposal Network (RPN) as the neck, and classification and localization heads. Additionally, we fine-tune the RPN anchor settings to further improve detection accuracy.

Furthermore, we experiment with different loss functions, such as CIoU Loss and DIoU Loss, to evaluate their impact on model performance. To further analyze the effect on mean Average Precision (mAP), we also fine-tune various anchor size combinations, comparing their contributions to overall accuracy.

Our best configuration achieves a mAP of **0.38** and an accuracy of **0.84**, demonstrating the effectiveness of our model design and optimization strategies.


## How to install

1. Clone the repository
```
git clone https://github.com/CEJiang/NYCU-Computer-Vision-2025-Spring-HW2.git
cd NYCU-Computer-Vision-2025-Spring-HW2
```

2. Create and activate conda environment
```
conda env create -f environment.yml
conda activate cv
```

3. Download the dataset 
- You can download the dataset from the provided [LINK](https://drive.google.com/file/d/13JXJ_hIdcloC63sS-vF3wFQLsUP1sMz5/view?usp=sharing)
- Place it in the following structure
```
NYCU-Computer-Vision-2025-Spring-HW1
├── nycu-hw2-data
│   ├── test
│   ├── train
│   ├── val
│   ├── train.json
│   └── val.json
├── utils
│   ├── __init__.py
│   ├── losses.py
│   ├── memory.py
│   .
│   .
│   .
├── environment.yml
├── main.py
├── train.py
├── test.py
.
.
.
```

4. Run for Train
    1. Train Model 
    ```
    python main.py DATAPATH [--epochs EPOCH] [--batch_size BATCH_SIZE] [--learning_rate LEARNING_RATE] [--decay DECAY] [--eta_min ETA_MIN] [--save SAVE_FOLDER] [--mode train]
    ```
    Example
    ```
    python main.py ./nycu-hw2-data --epochs 15 --batch_size 8 --learning_rate 1e-4 --decay 1e-4 --save saved_models
    ```
    2. Test Model
    ```
    python main.py DATAPATH --mode test
    ```
    Example
    ```
    python main.py ./nycu-hw2-data --mode test
    ```

## Performance snapshot
### Training Parameter Configuration

| Parameter        | Value                                                               |
|------------------|---------------------------------------------------------------------|
| Pretrained Weight| FasterRCNN_ResNet50_FPN_V2                                          |
| Learning Rate    | 0.0001                                                              |
| Batch Size       | 8                                                                   |
| Epochs           | 15                                                                  |
| decay            | 0.0001                                                              |
| Optimizer        | AdamW                                                               |
| Eta_min          | 0.000001                                                            |
| T_max            | 15                                                                  |
| Scheduler        | `CosineAnnealingLR`                                                 |
| Criterion        | `CrossEntropyLoss(Classification)` + `Smooth L1 Loss(Localization)` |

### Training Curve
![Image](https://github.com/CEJiang/NYCU-Computer-Vision-2025-Spring-HW2/blob/main/Image/training_curve.png)
### validate mAP Curve
![Image](https://github.com/CEJiang/NYCU-Computer-Vision-2025-Spring-HW2/blob/main/Image/val_map_curve.png)
### validate AP / AR Curve
![Image](https://github.com/CEJiang/NYCU-Computer-Vision-2025-Spring-HW2/blob/main/Image/ResNet50_Original.png)

### Performance
|                  | mAP                      | Accuracy                 |
|------------------|--------------------------|--------------------------|
| Validation       | 0.4650                   | ******                   |
| Public Test      | 0.3798                   | 0.8360                   |

