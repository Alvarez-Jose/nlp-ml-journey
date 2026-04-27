<!--
  STARTER README for github.com/Alvarez-Jose/nlp-ml-journey
  This repo currently has no description; this draft frames it as a learning log.
  Personalize the bracketed sections, then drop into the repo as README.md.
-->

# nlp-ml-journey

> A curated learning log of NLP / ML notebooks built during my M.S. coursework at UC Santa Cruz. Each notebook is self-contained and reproduces a concept end-to-end — from data prep through evaluation — with notes on what worked, what didn't, and what surprised me.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface&logoColor=black)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange?logo=jupyter&logoColor=white)

## Why this exists

Coursework notebooks tend to rot. This repo is the opposite — a small, well-organized set of notebooks I keep coming back to as references when I'm starting a new model or pipeline.

## Notebooks

<!-- Replace the rows below with the actual notebooks in your repo. -->

| Notebook | Topic | Key takeaways |
|---|---|---|
| `01_text_classification_baseline.ipynb` | TF-IDF + logistic regression on [REPLACE: dataset] | Strong baseline; rank-based thresholding for multilabel |
| `02_finetuning_transformers.ipynb` | Fine-tuning BERT / DeBERTa for sequence classification | Layer-wise learning rate decay, gradient accumulation |
| `03_retrieval_basics.ipynb` | BM25 vs. dense retrieval | When sparse beats dense, and why hybrid wins |
| `04_[REPLACE]` | [REPLACE] | [REPLACE] |

## Setup

```bash
git clone https://github.com/Alvarez-Jose/nlp-ml-journey
cd nlp-ml-journey
pip install -r requirements.txt
jupyter lab
```

## What's intentionally not here

- Production code (live in their own project repos)
- Coursework solutions for currently-running classes (academic integrity)
- Anything proprietary from research or industry work

## Related

- [REPLACE: link to your DeBERTa multilabel pipeline repo when it's up]
- [REPLACE: link to your RAG repo when it's up]

---

**Author:** Antonio Alvarez Maciel · M.S. NLP, UC Santa Cruz · [LinkedIn](https://linkedin.com/in/jose-alvarez-maciel)
