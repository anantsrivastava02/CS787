# 🔍 Text Fluoroscopy

[![Paper](https://img.shields.io/badge/Paper-EMNLP2024-blue)](https://aclanthology.org/2024.emnlp-main.885.pdf)
[![GitHub Stars](https://img.shields.io/github/stars/Fish-and-Sheep/Text-Fluoroscopy?style=social)](https://github.com/Fish-and-Sheep/Text-Fluoroscopy)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/downloads/)

**Text Fluoroscopy**: we propose Text Fluoroscopy, a black-box method for LLM-generated text detection through intrinsic features. Our method captures the text’s intrinsic features by identifying the layer with the largest distribution difference from the last and first layers when projected to the vocabulary space.
![Text Fluoroscopy Architecture](./assets/framework.png)


## 📋 Overview

Most LLM detection methods focus on either semantic features (from the last layer) or simple linguistic features (from early layers). **Text Fluoroscopy** takes a different approach by:

1. Finding the model layer with the largest distribution difference from both first and last layers
2. Extracting intrinsic text features that reveal the actual "fingerprints" of AI generation
3. Achieving superior generalization and robustness across different domains and against paraphrase attacks

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- PyTorch 1.10+
- CUDA-compatible GPU

### Installation

```bash
git clone https://github.com/Fish-and-Sheep/Text-Fluoroscopy.git
cd Text-Fluoroscopy
pip install -r requirements.txt
```

### Step 1: Download the LLM Model

```bash
huggingface-cli download --resume-download Alibaba-NLP/gte-Qwen1.5-7B-instruct \
  --local-dir ../huggingface_model/gte-Qwen1.5-7B-instruct \
  --cache-dir ../huggingface_model/gte-Qwen1.5-7B-instruct \
  --local-dir-use-symlinks False
```

### Step 2: Extract Features and Calculate KL Divergence

```bash
python gte-qwen/save_KL_with_first_and_last_layer.py

python gte-qwen/save_embedding.py
```

### Step 3: Train Classifier and Test

```bash
python embedding_classify/classify_with_max_KL_layer.py
```

## ⚡ Performance Optimization

Although our dynamic layer-selection method is effective, the computational overhead of examining each layer introduces time delays. To address this limitation, we provide an alternative approach:

| Methods | ChatGPT | GPT-4 | Claude3 |
|------------------|---------|-------|---------|
| Detection with the Last Layer | 0.07s | 0.09s | 0.08s |
| Text Fluoroscopy | 0.52s | 0.51s | 0.49s |
| Detection with the 30-th layer | 0.08s | 0.08s | 0.07s |

As shown above:
- We identified that using the fixed 30th layer offers an excellent balance between accuracy and speed
- This optimization reduces processing time by ~6.5× with less than 0.7% accuracy reduction
- The 30th layer approach maintains the core benefits of our method while being nearly as fast as last-layer-only methods

![Text Fluoroscopy Architecture](./assets/results.png)



## 📚 Citation

If you find our work useful, please consider citing our paper:

```bibtex
@inproceedings{yu2024textfluoroscopy,
    title={Text Fluoroscopy: Detecting LLM-Generated Text through Intrinsic Features},
    author={Yu, Xiao and Chen, Kejiang and Yang, Qi and Zhang, Weiming and Yu, Nenghai},
    booktitle={Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
    year={2024},
    url={https://aclanthology.org/2024.emnlp-main.885.pdf}
}
```

## 📬 Contact

For questions or feedback, please open an issue or contact us at [Xiao Yu](mailto:yuxiao1217@mail.ustc.edu.cn).