# LLMs Are Few-Shot In-Context Low-Resource Language Learners

This is the official repository ["LLMs Are Few-Shot In-Context Low-Resource Language Learners"](https://arxiv.org/pdf/2403.16512) paper.

### Code Structure
- [`dataset/prepare_dataset.ipynb`](https://github.com/SamuelCahyawijaya/in-context-alignment/blob/main/dataset/prepare_dataset.ipynb) => This code is for preparing all the dataasets in the experiment, including aligning AmericasNLI data to the Spanish subset in XNLI
- [`analysis/final_metrics.ipynb`](https://github.com/SamuelCahyawijaya/in-context-alignment/blob/main/analysis/final_metrics.ipynb) and [analysis/final_metrics_plotly.ipynb](https://github.com/SamuelCahyawijaya/in-context-alignment/blob/main/analysis/final_metrics_plotly.ipynb) => These notebooks are for data visualization used in our paper
- [`run_script_final/`](https://github.com/SamuelCahyawijaya/in-context-alignment/tree/main/run_script_final) => This folder provides all the shell commands for running the experiments
- [`src/`](https://github.com/SamuelCahyawijaya/in-context-alignment/tree/main/src) => This folder contains codes related to prompting, indexing, retrieval, and other utility functions used in our experiments
- [`*_sample_preview.py` and `main_*.py`](https://github.com/SamuelCahyawijaya/in-context-alignment/tree/main) => These codes are for checking the samples and running the main experiments, respecitively.

### Citing Information
If you are inspired by our work and/or use the code in this repo, please cite:
```
@misc{cahyawijaya2024llms,
      title={LLMs Are Few-Shot In-Context Low-Resource Language Learners}, 
      author={Samuel Cahyawijaya and Holy Lovenia and Pascale Fung},
      year={2024},
      eprint={2403.16512},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
