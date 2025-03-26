# NYCU VRDL spring 2025 HW1

Student ID: 110550138

Name: 鄭博文

## Introduction

A ResNeXt-based image classifier, with training and inference code.

## How to Run

### Install requirements:

  ```bash
  pip install -r requirements.txt
  ```

### Data augmentation:

  ```bash
  python data_augment.py <training_root_directory>
  ```
  The training data root directory is the one that contain subdirectories labeled 0 through 99.

### Configuration:

  Check the comments in `config-example.yml`.

### Training:

  ```bash
  python train.py [--config <config_file_path>]
  ```
  The default value of `config` argument is `./config.yml`.

### Inference:

  ```
  python inference.py --checkpoint <checkpoint_name> [--config <config_file_path>]
  ```
  The checkpoint name should be the stem of `.pth` file in `MODEL_DIR`, with formate `{date}-{time}_epoch_{epoch}`. For example, `20250326-144741_epoch_2` (without .pth).

### Performance:

  The highest accuracy achieved is `0.95` in public testing dataset.