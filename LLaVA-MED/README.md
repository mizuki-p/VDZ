# Integrating the VDZ Module with the LLaVA-MED Model  

This document provides a step-by-step guide for integrating the VDZ module with the LLaVA-MED v1.5 model and conducting related experiments. The instructions ensure proper setup, fine-tuning, and evaluation of the modified model.  

**Note:** This implementation is based on LLaVA-MED v1.5, which builds upon the open-sourced Mistral model.  

## Overview  

The integration of the VDZ module into LLaVA-MED involves the following key aspects:  

1. **Framework and Implementation**  
   - LLaVA-MED is built upon the LLaVA framework and utilizes the Mistral model, both of which are implemented using the Hugging Face Transformers library. Familiarity with the Hugging Face ecosystem is recommended for further experimentation.  

2. **Fine-tuning Strategy**  
   - Parameter-efficient fine-tuning (PEFT) is employed to adapt LLaVA-MED, specifically using LoRA (Low-Rank Adaptation) from the Hugging Face PEFT library.  
   - To streamline the process, we first create initialized but untrained LoRA parameter files in the `output/vdz_base` directory.  
   - Both the LoRA layers and the proposed VDZ module are set as trainable.  

3. **Integration with LLaVA-MED**  
   - The `llava` directory is a modified copy of the official LLaVA-MED repository. Several files have been adjusted to incorporate the VDZ module.  
   - Directly check the modified files may decrease sanity, so if just want to disable the VDZ module and conduct experiments using only LoRA-based fine-tuning, one can modify the function `if_add_vdz_module` in `LLaVA-MED/llava/model/mistral/vdz.py` to always return `False`. (Note: This may require regenerating the LoRA parameter files as described in Step 4.)  

## Installation and Setup  

### 1. Install Dependencies  
Install all required libraries from `requirements.txt`.  

### 2. Prepare Datasets  
Download the required datasets and place them in the `dataset` directory.  

### 3. Obtain the Pretrained LLaVA-MED v1.5 Model  
Download the pretrained [LLaVA-MED v1.5](https://huggingface.co/microsoft/llava-med-v1.5-mistral-7b) model and place it in the `downloaded` directory.  

### 4. Initialize LoRA Model  
Run the following command to create the initial LoRA parameter files:  
```bash
python make_peft_model.py
```  

### 5. Fine-tuning  
Execute the fine-tuning process by running the following script:  
```bash
bash run_vdz.sh
```  
**Key parameters in `run_vdz.sh`:**  
```bash
--include localhost:num1,num2...  # Specify GPUs to use  
--dataset xxx                     # Choose dataset: [rad, slake, ovqa]  
--saving_path xxx                 # Directory for saving trained models  
--run_name xxx                    # Weights & Biases (wandb) run name  
--output_dir xxx                   # Directory for storing temporary parameters (intermediate backups)  
```  

### 6. Evaluation  
To evaluate the fine-tuned model, run the following command:  
```bash
python eval_test.py --model-path <saving_path used in run_vdz.sh> --dataset <rad/slake/ovqa> --output-path <directory for generated results>
```  
After evaluation, use the corresponding `check_ans_xxx.py` script to compute performance metrics. The `eval_test.py` script generates an output file, which should be referenced in the `output_file` variable within the `check_ans_xxx.py` script to view the results.  
