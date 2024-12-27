# Visual De-duplication and Zoom Model for Medical Visual Question Answering (VDZ)

---

## Introduction

Medical Visual Question Answering (Med-VQA) represents a crucial advancement in facilitating computer-assisted clinical diagnosis. Although recent advancements in large-scale vision-language models and specialized smaller multi-modal models have demonstrated considerable progress in Med-VQA tasks, their performance remains constrained due to the presence of massive redundant visual information. Our investigation reveals pervasive redundancy within vision embeddings, coupled with the existence of divergent vision embeddings in contrast to these redundant patterns. Further empirical analysis demonstrates that these divergent vision embeddings contribute disproportionately to model performance relative to redundant embeddings, yet receive comparatively less attention within current architecture. To address this imbalance, we introduce a novel plug-and-play approach: the V̲isual D̲e-duplication and Z̲oom (VDZ) model. This approach identifies and eliminates redundant embeddings while optimizing the model's attention allocation towards high-impact divergent embeddings, thereby enhancing model performance. We validate the effectiveness of our method across four representative models, including two large-scale vision-language models (with over 2 billion parameters) and two compact multi-modal models (with fewer than 500 million parameters), utilizing three distinct Med-VQA datasets. The experimental results establish the efficacy of our proposed method. Furthermore, we conduct extensive ablation studies to investigate the differential impact of each VDZ component and its implementation strategy on overall performance.

---

## Release Notes

- **[December 2, 2024]**: The VDZ module code is now available!  
  Complete training and evaluation code will be released in two months.

---

## Repository Structure

```plaintext
.
 |- M3AE/
 |- LLaVA-MED/
 |- BridgeTower/
 |- Pali-Gemma/
 |- VDZ.py
 |- README.md
 |- LICENSE
```

- **VDZ.py**: Contains the core functionality of the VDZ module. Use this file to quickly test our method.  
- **Model Directories (e.g., M3AE, LLaVA-MED)**: Include modified files for reproducing results. Detailed usage instructions can be found in the README files within each directory.

---

## Model Downloads

Fine-tuned model parameters will be provided via Baidu YunPan. 

| Model          | Dataset | URL  |
|----------------|---------|------|
| M3AE           | VQA-RAD | TBD  |
| M3AE           | SLAKE   | TBD  |
| M3AE           | OVQA    | TBD  |
| LLaVA-MED      | VQA-RAD | TBD  |
| LLaVA-MED      | SLAKE   | TBD  |
| LLaVA-MED      | OVQA    | TBD  |
| BridgeTower    | VQA-RAD | TBD  |
| BridgeTower    | SLAKE   | TBD  |
| BridgeTower    | OVQA    | TBD  |
| Pali-Gemma     | VQA-RAD | TBD  |
| Pali-Gemma     | SLAKE   | TBD  |
| Pali-Gemma     | OVQA    | TBD  |

---

## Datasets

We provide dataset download instructions to support further research in Med-VQA.

- **VQA-RAD:**  
  Access the dataset [here](https://osf.io/89kps/). Note that this version is not pre-split into training and testing sets. We used a preprocessed version available [here](https://github.com/aioz-ai/MICCAI19-MedVQA?tab=readme-ov-file#Preprocessing).

- **SLAKE:**  
  The dataset is available [here](https://www.med-vqa.com/slake/). Only the English subset was used for training and testing in our experiments.

- **OVQA:**  
  Download the dataset [here](http://47.94.174.82/verify). Credentials are required for access.

---

## Citation

If you find our work useful, please cite us:

```

```
