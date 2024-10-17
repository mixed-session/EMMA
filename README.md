# Egocentric Memory Enhanced Mixed-Session Conversation Agent (EMMA)

This repository provides the implementation for the paper **["Mixed-Session Conversation with Egocentric Memory"](https://arxiv.org/abs/2410.02503)** by Jihyoung Jang, Taeyoung Kim, and Hyounghun Kim, presented at EMNLP 2024 Findings.

- **[Paper](https://arxiv.org/abs/2410.02503)**
- **[Project Page](https://mixed-session.github.io/)**

## Dataset and Model

The MiSC dataset is publicly available on Hugging Face. The adapter for EMMA's generation module is available on Hugging Face, and the weights for the retriever module can be downloaded from Google Drive via the link below:

- **[MiSC Dataset](https://huggingface.co/datasets/jihyoung/MiSC)** 
- **[Adapter for Generation Module](https://huggingface.co/jihyoung/EMMA)** 
- **[Retriever Module Weights](https://drive.google.com/file/d/1fu8tpraorSGQc4abFtptHBxmGzjkKM73/view?usp=sharing)** 

## Usage

To set up and run the code, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/mixed-session/EMMA.git
    ```

2. Create and activate the conda environment:

    ```bash
    conda env create -f environment.yml
    conda activate emma
    ```

3. Download the retriever module weights before running the code.

4. Run the EMMA model:

    ```bash
    python emma.py
    ```

## Citation

```bibtex
@article{jang2024mixed,
  title={Mixed-Session Conversation with Egocentric Memory},
  author={Jang, Jihyoung and Kim, Taeyoung and Kim, Hyounghun},
  journal={arXiv preprint arXiv:2410.02503},
  year={2024}
}
```
