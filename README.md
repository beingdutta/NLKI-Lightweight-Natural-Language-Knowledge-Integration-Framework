# 🌟 NLKI: Lightweight Natural Language Knowledge Integration Framework

[![Paper](https://img.shields.io/badge/Paper-EMNLP%20Findings%202025-blue)](https://arxiv.org/abs/XXXX.XXXXX)  
[![Website](https://img.shields.io/badge/Project-Website-green)](https://beingdutta.github.io/NLKI-Project-Page-EMNLP-2025-Findings/)  

Official codebase for the EMNLP Findings 2025 paper:  
**“NLKI: A Lightweight Natural Language Knowledge Integration Framework for Improving Small VLMs in Commonsense VQA Tasks”**  

---

## 🖼️ Project Overview

**NLKI** is a modular framework for **commonsense visual–question answering (VQA)** and **visual entailment**, designed to enhance **small vision–language models (≤250M parameters)** such as **ViLT, VisualBERT, and FLAVA**.  

The framework consists of:  
1. 📖 **Knowledge Retrieval** → Dense retrieval with a fine-tuned **ColBERTv2**  
2. 💡 **LLM-based Explanations** → Natural language rationales crafted by **Llama-3** with enriched object prompts  
3. 🧩 **Integration into VLMs** → Injecting both retrieved knowledge and explanations into small VLMs  
4. 🛡️ **Noise-Robust Training** → Using **Symmetric Cross Entropy (SCE)** and **Generalized Cross Entropy (GCE)** for noisy benchmarks  

---

## 📊 Key Highlights
- 🚀 **Improves small VLMs by up to 7%** on CRIC, AOKVQA, and e-SNLI-VE  
- 🧠 **Reduces hallucination** by leveraging retriever + explainer synergy  
- 🛡️ **Noise-aware training** adds another **2.5% (CRIC)** and **5.5% (AOKVQA)**  
- ⚖️ **Parameter-efficient reasoning** → 250M FLAVA matches or exceeds **2–4B parameter models** like Qwen-2 VL-2B and SmolVLM-2.5B  

---

## 📂 Repository Structure

```bash
├── Captioning/          # Visual captioning modules  
├── FLOPS-Calculation/   # FLOPs and efficiency computation  
├── Generation/          # LLM-based explanation generation  
├── Object-Detection/    # Object tagging and detection scripts  
├── Retrieval/           # ColBERTv2 retriever training and inference  
├── Train-Test/          # Training and evaluation scripts (with CE, SCE, GCE losses)  
├── requirements.txt     # Dependencies  
└── README.md

## ⚡ Installation

```bash
git clone https://github.com/beingdutta/NLKI-Lightweight-Natural-Language-Knowledge-Integration-Framework.git
cd NLKI-Lightweight-Natural-Language-Knowledge-Integration-Framework

# Create environment
conda create -n nlki python=3.10
conda activate nlki

# Install dependencies
pip install -r requirements.txt
---