# Visual De-duplication and Zoom Model for Medical Visual Question Answering

---

## Introduction

> Medical Visual Question Answering (Med-VQA) represents a crucial advancement in facilitating computer-assisted clinical diagnosis. Although recent advancements in large-scale vision-language models and specialized smaller multi-modal models have demonstrated considerable progress in Med-VQA tasks, their performance remains constrained due to the presence of massive redundant visual information. Our investigation reveals pervasive redundancy within vision embeddings, coupled with the existence of divergent vision embeddings in contrast to these redundant patterns. Further empirical analysis demonstrates that these divergent vision embeddings contribute disproportionately to model performance relative to redundant embeddings, yet receive comparatively less attention within current architecture. To address this imbalance, we introduce a novel plug-and-play approach: the $\underline{V}$isual $\underline{D}$e-duplication and $\underline{Z}$oom (VDZ) model. This approach identifies and eliminates redundant embeddings while optimizing the model's attention allocation towards high-impact divergent embeddings, thereby enhancing model performance. We validate the effectiveness of our method across four representative models, including two large-scale vision-language models (with over 2 billion parameters) and two compact multi-modal models (with fewer than 500 million parameters), utilizing three distinct Med-VQA datasets. The experimental results establish the efficacy of our proposed method. Furthermore, we conduct extensive ablation studies to investigate the differential impact of each VDZ component and its implementation strategy on overall performance.

---
## Release 
- **[December 2, 2024]** We have released the code for the VDZ module. The complete training and evaluation code will be available in two weeks.

---
## Repository Structure
```

-
 |- M3AE
 |- LLaVA-MED
 |- BridgeTower
 |- Pali-Gemma
 |-
 |- VDZ.py
 |- README.md
 |- LICENSE
```

Since the VDZ method functions like a plugin, we have extracted its main functionality into the VDZ.py file. For those who wish to quickly test our proposed method, please refer to the VDZ.py file.

If you would like to reproduce the results for each model, navigate to the corresponding directory and read the provided README file. (Only the modified files from the original work are included in each directory.)

---

## Model Download

We will provide the fine-tuned model parameters via Baidu YunPan.


| Model          | Dataset | URL |
| -------------- | ------- | --- |
| M3AE           | VQA-RAD |     |
| M3AE           | SLAKE   |     |
| M3AE           | OVQA    |     |
|                |         |     |
| LLaVA-MED      | VQA-RAD |     |
| LLaVA-MED      | SLAKE   |     |
| LLaVA-MED      | OVQA    |     |
|                |         |     |
| BridgeTower    | VQA-RAD |     |
| BridgeTower    | SLAKE   |     |
| BridgeTower    | OVQA    |     |
|                |         |     |
| PaliGemma      | VQA-RAD |     |
| PaliGemma      | SLAKE   |     |
| PaliGemma      | OVQA    |     |

---
## Dataset

We provide the dataset download instructions here for those interested in advancing research in this area.

- **VQA-RAD:**  
  The dataset is available at [this URL](https://osf.io/89kps/). However, this version is not pre-split into training and testing sets. Our experiments were conducted using [this version](https://github.com/aioz-ai/MICCAI19-MedVQA?tab=readme-ov-file#Preprocessing).

- **SLAKE:**  
  The dataset can be accessed [here](https://www.med-vqa.com/slake/). Please note that we used only the English subset for training and testing.

- **OVQA:**  
  The dataset is available [here](http://47.94.174.82/verify). Please note that downloading the OVQA dataset requires credentials.
---

## Citation
> 
> 