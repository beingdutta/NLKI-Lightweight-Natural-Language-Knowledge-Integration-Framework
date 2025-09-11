# 🌟 NLKI: Lightweight Natural Language Knowledge Integration Framework

[![Paper](https://img.shields.io/badge/Paper-EMNLP%20Findings%202025-blue)](https://arxiv.org/abs/2508.19724)  
[![Website](https://img.shields.io/badge/Project-Website-green)](https://beingdutta.github.io/NLKI-Project-Page-EMNLP-2025-Findings/)  

Official **codebase** for the EMNLP Findings 2025 paper:  
**“NLKI: A Lightweight Natural Language Knowledge Integration Framework for Improving Small VLMs in Commonsense VQA Tasks”**  

---

## 🖼️ Overview
**NLKI** is a modular framework for **commonsense visual–question answering (VQA)** and **visual entailment**, designed to improve **small vision–language models (≤250M parameters)** such as **ViLT, VisualBERT, and FLAVA**.  

The framework combines:
1. 📖 **Knowledge Retrieval** → Dense retrieval with a fine-tuned **ColBERTv2**  
2. 💡 **LLM Explanations** → Natural language rationales crafted by **Llama-3**, conditioned on enriched object prompts  
3. 🧩 **Knowledge Integration** → Injecting retrieved facts + explanations into sVLMs  
4. 🛡️ **Noise-Robust Training** → Leveraging **Symmetric Cross Entropy (SCE)** and **Generalized Cross Entropy (GCE)**  

---

## 📊 Key Results
- 🚀 **+7% accuracy** on CRIC, AOKVQA, and e-SNLI-VE  
- 🧠 **Reduced hallucination** via retriever–explainer synergy  
- 🛡️ **Noise-aware training** adds **+2.5% (CRIC)** and **+5.5% (AOKVQA)**  
- ⚖️ **Parameter efficiency** → 250M-parameter **FLAVA** matches/exceeds **2–4B models** (Qwen-2 VL-2B, SmolVLM-2.5B)  

---

## 📂 Repository Structure
```text
├── Captioning/          # Visual captioning modules
├── FLOPS-Calculation/   # FLOPs and efficiency computation
├── Generation/          # LLM-based explanation generation
├── Object-Detection/    # Object tagging and detection scripts
├── Retrieval/           # ColBERTv2 retriever training and inference
├── Train-Test/          # Training and evaluation scripts (CE, SCE, GCE losses)
├── requirements.txt     # Dependencies
└── README.md

```
## ⚙️ Installation
```bash
git clone https://github.com/beingdutta/NLKI-Lightweight-Natural-Language-Knowledge-Integration-Framework.git
cd NLKI-Lightweight-Natural-Language-Knowledge-Integration-Framework
pip install -r requirements.txt


```
## 🚀 Usage

#### Train with standard CE loss
python Train-Test/train.py --dataset CRIC --loss CE

#### Train with noise-robust losses (SCE / GCE)
python Train-Test/train.py --dataset AOKVQA --loss SCE


## ✨ Citation
```bibtex
@misc{dutta2025nlkilightweightnaturallanguage,
  title        = {NLKI: A Lightweight Natural Language Knowledge Integration Framework for Improving Small VLMs in Commonsense VQA Tasks},
  author       = {Aritra Dutta and Swapnanil Mukherjee and Deepanway Ghosal and Somak Aditya},
  year         = {2025},
  eprint       = {2508.19724},
  archivePrefix= {arXiv},
  primaryClass = {cs.CL},
  url          = {https://arxiv.org/abs/2508.19724}
}

```
## ⚖️ License & Data Disclaimer

This repository is released under the MIT License.

The model weights and code are provided for research purposes.

All copyrights for the training data remain with their original owners; no claim of ownership is made.