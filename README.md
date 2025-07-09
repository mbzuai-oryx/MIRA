## MIRA (Multimodal Intelligent Retrieval and Augmentation framework)
Official Code Base for MIRA: A Novel Framework for Fusing Modalities in Medical RAG

Multimodal Large Language Models (MLLMs) have significantly advanced AI-assisted medical diagnosis but often generate factually inconsistent responses that deviate from established medical knowledge. Retrieval-Augmented Generation (RAG) enhances factual accuracy by integrating external sources, but it presents two key challenges. First, insufficient retrieval can miss critical information, whereas excessive retrieval can introduce irrelevant or misleading content, disrupting model output. Second, even when the model initially provides correct answers, over-reliance on retrieved data can lead to factual errors. To address these issues, we introduce Multimodal Intelligent Retrieval and Augmentation (MIRA) framework designed to optimize factual accuracy in MLLM. MIRA consists of two key components: (1) a calibrated Rethinking and Rearrangement module that dynamically adjusts the number of retrieved contexts to manage factual risk, and (2) A medical RAG framework integrating image embeddings and a medical knowledge base with a query-rewrite module for efficient multimodal reasoning. This enables the model to effectively integrate both its inherent knowledge and external references. Our evaluation of publicly available medical VQA and report generation benchmarks demonstrates that MIRA substantially enhances factual accuracy and overall performance, achieving new state-of-the-art results. Code, model, and processed data will be publicly released after acceptance.

[Website](https://tommyix.github.io/MIRA/)

---
## Acknowledgement

The Codebase of MIRA is built on [LLaVA-Next](https://github.com/LLaVA-VL/LLaVA-NeXT), many thanks to them. If you are citing MIRA, please also cite their paper.

---

Now working: writing README for instruction on training / Reproduction of MIRA.