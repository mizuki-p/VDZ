# Integrating the VDZ Module with the M3AE Model

This guide provides step-by-step instructions to integrate the VDZ module with the M3AE model for conducting experiments. Follow these steps to ensure proper setup and execution.

## Setup Instructions

### 1. Prepare the Datasets
Download the required datasets and place them in the `dataset` directory.

### 2. Download Pretrained Models
Download the pretrained **BridgeTower** model from [here](https://huggingface.co/BridgeTower/bridgetower-base-itm-mlm) and save it in the `pretrained_models/base` directory.

### 3. Initialize VDZ Module Parameters
Run the following commands to initialize parameters for the VDZ module:

```bash
cd BridgeTower
cp pretrained_models/base/* pretrained_models/vdz_base/
python init_vdz_model.py
```

### 4. Modify and Run the Training Script
Edit the `stage_lora.sh` script to configure the following parameters:

- `--dataset`: Specify the dataset to use. Options: `[vqa_rad, slake, ovqa]`
- `--saving_path`: Define the path to save the trained model.
- `--pretrained_model_path`: Choose the model path. Options:
  - `pretrained_models/base` (Baseline)
  - `pretrained_models/vdz_base` (VDZ)
- `--include localhost:6,7,8,9`: Specify the GPUs to use for training.

Once configured, execute the script:

```bash
bash stage_lora.sh
```

### 5. Evaluate the Model
Evaluate the trained model on various datasets. Before running the evaluation, ensure that the file paths for model loading and datasets are correctly set. Use the following command:

```bash
python eval_test.py
```
