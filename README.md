# NYCU Computer Vision 2025 Spring HW2
- StudentID: 313553044
- Name: 江仲恩

## Introduction
In this assignment, we aim to build a robust image classification model for the dataset provided in HW1. To achieve strong generalization performance, we adopt a transfer learning approach using ResNeXt-101 pretrained on ImageNet, and fine-tune its fully connected layers on the target dataset.
To tackle challenges such as visual similarity between species, we introduce several techniques :
- **Strong data augmentations** to enhance model robustness under various lighting conditions, perspectives, and background clutter.
- **Label smoothing and progressive loss switching** to stabilize training during early epochs.
- **Focal Loss** to emphasize hard-to-classify examples and mitigate the effects of class imbalance.
- **Exponential Moving Average (EMA)** of model weights to improve validation stability and final performance.
- **Test-Time Augmentation (TTA)** to further boost prediction accuracy by aggregating results from multiple augmented views during inference.

Our pipeline includes class-balanced loss weighting, cosine learning rate scheduling, and detailed monitoring through training curves and confusion matrices. The final model achieves over 92–95% validation accuracy, with smooth convergence and minimal overfitting.

## How to install

1. Clone the repository
```
git clone https://github.com/CEJiang/NYCU-Computer-Vision-2025-Spring-HW1.git
cd NYCU-Computer-Vision-2025-Spring-HW1
```

2. Create and activate conda environment
```
conda env create -f environment.yml
conda activate cv
```

3. Download the dataset 
- You can download the dataset from the provided [LINK](https://drive.google.com/file/d/1fx4Z6xl5b6r4UFkBrn5l0oPEIagZxQ5u/view)
- Place it in the following structure
```
NYCU-Computer-Vision-2025-Spring-HW1
├── data
│   ├── test
│   ├── train
│   └── val
├── utils
│   ├── __init__.py
│   ├── early_stopping.py
│   ├── losses.py
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
    python main.py ./data --epochs 100 --batch_size 64 --learning_rate 5e-5 --decay 1e-4 --save saved_models
    ```
    2. Test Model
    ```
    python main.py DATAPATH --mode test
    ```

## Performance snapshot
### Training Parameter Configuration

| Parameter        | Value                                               |
|------------------|-----------------------------------------------------|
| Model            | ResNeXt-101                                         |
| Pretrained Weight| IMAGENET1K_V2                                       |
| Learning Rate    | 0.00005                                             |
| Batch Size       | 64                                                  |
| Epochs           | 100                                                 |
| Optimizer        | AdamW                                               |
| Eta_min          | 0.00001                                             |
| T_max            | 50                                                  |
| Scheduler        | `CosineAnnealingLR`                                 |
| label_smoothing  | 0.05                                                |
| Criterion        | `CrossEntropyLoss` -> `SmoothFocal` -> `Focal`      |

### Training Curve
![Image](https://github.com/CEJiang/NYCU-Computer-Vision-2025-Spring-HW1/blob/main/Image/train_curve.png)
### Confusion Matrix
![Image](https://github.com/CEJiang/NYCU-Computer-Vision-2025-Spring-HW1/blob/main/Image/confusion_matrix.png)
### Performance
|                  | Accuracy(%)                                         |
|------------------|-----------------------------------------------------|
| Validation       | 95                                                  |
| Public Test      | 96                                                  |