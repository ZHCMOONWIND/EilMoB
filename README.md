------

# EilMoB: Emotion-aware Incongruity Learning and Modality Bridging Network for Multi-modal Sarcasm Detection

This repository contains the official implementation of the paper:

> **[EilMoB: Emotion-aware Incongruity Learning and Modality Bridging Network for Multi-modal Sarcasm Detection](https://dl.acm.org/doi/abs/10.1145/3731715.3733321)** 
> *ICMR*, 2025.

## 📦 Installation

Clone this repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

## 📁 Directory Structure

```plaintext
.
├── code/               # Source code for model architecture, training, and evaluation
├── imgs/               # Image data used in multi-modal sarcasm detection
│   ├── train/          # Training images
│   ├── val/            # Validation images
│   └── test/           # Test images
├── llm/                # Emotion-aware information extracted via LLM
│   ├── train.txt       # EAI for training data
│   ├── val.txt         # EAI for validation data
│   └── test.txt        # EAI features for test data
├── text_data/          # MMSD2.0 textual data
├── README.md           # This documentation file
└── requirements.txt    # Python dependencies
```

## 📊 Dataset Description

This project is based on the **MMSD** and **MMSD2.0** datasets, containing multi-modal sarcastic posts with aligned image and text data.

- The `text_data_clean/` directory holds the processed text portions of the MMSD2.0 dataset.
- The `imgs/` folder contains corresponding visual modality data, categorized into:
  - `train/`, `val/`, and `test/` based on the dataset split.
- The `llm/` folder holds emotion-aware textual features extracted using a Multi-modal Large Language Model (MLLM), specifically **Qwen2-VL-7B-Instruct**.  
  -Each line corresponds to an **Emotion-aware Incongruity (EAI)** description for a sample in the dataset split (`train.txt`, `val.txt`, `test.txt`).

EAI information is generated through a **three-aspect prompt template** released in the paper, which can be directly reused to regenerate EAI data.  
The prompt guides the MLLM to extract:
1. **Image Description** — captures visual entities, embedded text, and relational context.
2. **Emotion Analysis** — infers the dominant emotional tone conveyed by the post.
3. **Alignment Check** — identifies incongruities between the image and textual emotion.

Since EAI is purely textual, it can be seamlessly integrated with the text modality to reduce modality gaps.  
Users can reproduce these files by applying the public prompt to any MLLM supporting visual-text input (e.g., Qwen2-VL or similar).


## 🚀 Usage Instructions

### Step 1: Data Preparation

Ensure the following before training:

- `imgs/train/`, `imgs/val/`, `imgs/test/` are filled with image files (e.g., `.jpg`, `.png`) for each split.
- `llm/train.txt`, `llm/val.txt`, `llm/test.txt` are aligned with their respective text/image entries.
- `text_data_clean/` is present with structured cleaned text input from MMSD2.0.

### Step 2: Training the Model

To train the sarcasm detection model, run:

```bash
python main.py
```

This script will automatically reference the corresponding visual and emotion-aware inputs from the `imgs/` and `llm/` folders.

## 📌 Notes

- Make sure all images and LLM files are aligned in order across splits.
- If you customize datasets or introduce new ones, ensure consistency in folder structure.

## 📖 Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{zhao2025eilmob,
  title={EilMoB: Emotion-aware Incongruity Learning and Modality Bridging Network for Multi-modal Sarcasm Detection},
  author={Zhao, Haochen and Xu, Yongxiu and Lin, Xinkui and Lu, Jiarui and Xu, Hongbo and Wang, Yubin},
  booktitle={Proceedings of the 2025 International Conference on Multimedia Retrieval},
  pages={1868--1876},
  year={2025}
}
```

------



