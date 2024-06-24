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
@inproceedings{cahyawijaya-etal-2024-llms,
    title = "{LLM}s Are Few-Shot In-Context Low-Resource Language Learners",
    author = "Cahyawijaya, Samuel  and
      Lovenia, Holy  and
      Fung, Pascale",
    editor = "Duh, Kevin  and
      Gomez, Helena  and
      Bethard, Steven",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.naacl-long.24",
    pages = "405--433",
    abstract = "In-context learning (ICL) empowers large language models (LLMs) to perform diverse tasks in underrepresented languages using only short in-context information, offering a crucial avenue for narrowing the gap between high-resource and low-resource languages.Nonetheless, there is only a handful of works explored ICL for low-resource languages with most of them focusing on relatively high-resource languages, such as French and Spanish. In this work, we extensively study ICL and its cross-lingual variation (X-ICL) on 25 low-resource and 7 relatively higher-resource languages.Our study not only assesses the effectiveness of ICL with LLMs in low-resource languages but also identifies the shortcomings of in-context label alignment, and introduces a more effective alternative: query alignment. Moreover, we provide valuable insights into various facets of ICL for low-resource languages.Our study concludes the significance of few-shot in-context information on enhancing the low-resource understanding quality of LLMs through semantically relevant information by closing the language gap in the target language and aligning the semantics between the targeted low-resource and the high-resource language that the model is proficient in. Our work highlights the importance of advancing ICL research, particularly for low-resource languages.",
}
```
