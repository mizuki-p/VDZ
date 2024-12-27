#### Integrating the VDZ Module with the M3AE Model

---

## Steps to Apply the VDZ Module

1. **Clone the M3AE repository**  
   Use the following command to clone the original M3AE repository:  
   ```bash
   git clone https://github.com/zhjohnchan/M3AE
   ```

2. **Merge files from this directory into the M3AE repository**  
   Copy the files provided in this directory to their respective locations in the cloned M3AE repository. Overwrite any existing files if prompted.

3. **Reproduce experiments following the M3AE documentation**  
   Refer to the M3AE documentation for instructions on running experiments. Our modifications include support for the OVQA dataset and the integration of the VDZ module.

---

## File Descriptions

The directory structure and files provided in this package are described below.  

```
M3AE/
 |- prepro/                     # Data preprocessing
 |--- prepro_finetuning_data.py # Preprocesses the OVQA dataset
 |- m3ae/
 |--- datamodules/              # Dataset classes for OVQA
 |------ __init__.py
 |------ vqa_ovqa_datamodule.py
 |--- datasets/
 |------ __init__.py
 |--- modules/
 |------ language_encoders/
 |--------- bert_model.py       # Adds the VDZ module (search for "VDZ" to locate the implementation)
 |------ m3ae_module.py         # Attaches the VDZ module to each transformer block
 |--- config.py                 # Updates to hyper-parameters
```

### Key Modifications:
- **Preprocessing (prepro_finetuning_data.py):**  
  Added support for the OVQA dataset, including data loading and formatting.

- **Dataset Classes (vqa_ovqa_datamodule.py):**  
  Implemented a custom datamodule for handling OVQA-specific data.

- **Model Integration (bert_model.py & m3ae_module.py):**  
  Integrated the VDZ module into the model's architecture. Look for comments containing "VDZ" to locate the specific changes.

- **Configuration (config.py):**  
  Updated hyper-parameters to optimize training with the VDZ module.

---

Follow these steps to integrate and test the VDZ module with the M3AE model. For additional support, consult the M3AE repository documentation or reach out to us directly.
