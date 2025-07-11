<p align="center">
    <img src="https://i.imgur.com/waxVImv.png" alt="Oryx Video-ChatGPT">
</p>
<!-- centred logo -->
<div align="center" style="margin:24px 0;">
  <img src="images/logo.png" width="85%" />
</div>

<!-- bottom full-width GIF -->
<p align="center">
    <img src="https://i.imgur.com/waxVImv.png" alt="Oryx Video-ChatGPT">
</p>
<h1 align="left" style="margin:24px 0;">
  MIRA: ANovel Framework for Fusing Modalities in Medical RAG
</h1>

<div align="center">

[![arXiv:2505.24876](https://img.shields.io/badge/arXiv-2505.24876-b31b1b)](https://arxiv.org/abs/2505.24876)
[![Hugging Face](https://img.shields.io/badge/Dataset-HuggingFace-orange?logo=huggingface)](https://huggingface.co/datasets/Tajamul21/Agent-X)
[![Download](https://img.shields.io/badge/Dataset-Download-blue?logo=cloud)](https://github.com/Tajamul21/Agent-X-Benchmark/releases/download/v0.1.0/agent-X_dataset.zip)
[![Leaderboard](https://img.shields.io/badge/View-Leaderboard-green)](#-leaderboard-may-2025)

</div>



#### Authors: [Tajamul Ashraf](https://www.tajamulashraf.com)\*, and [Rao Muhammad Anwer](https://mbzuai.ac.ae/study/faculty/rao-muhammad-anwer/),  


\* Equally contribution, **Correspondence:** [Tajamul Ashraf](https://www.tajamulashraf.com)
<div align="left" style="margin:24px 0;">
  <img src="https://user-images.githubusercontent.com/74038190/212284115-f47cd8ff-2ffb-4b04-b5bf-4d1c14c0247f.gif"
       width="100%" />
</div>



## Updates



[2025-06-02]: **Agent-X paper published on [![arXiv](https://img.shields.io/badge/https://arxiv.org/abs/2507.07902)](https://arxiv.org/abs/2507.07902)**

[2025-05-29]: **Released evaluation & deployment code for MIRA** 
 
[2025-05-22]:  **Published the MIRA dataset on Hugging Face** 

## Introduction

>Multimodal Large Language Models (MLLMs) have significantly advanced AI-assisted medical diagnosis, but they often generate factually inconsistent responses that deviate from established medical knowledge. Retrieval-Augmented Generation (RAG) enhances factual accuracy by integrating external sources, but it presents two key challenges. First, insufficient retrieval can miss critical information, whereas excessive retrieval can introduce irrelevant or misleading content, disrupting model output. Second, even when the model initially provides correct answers, over-reliance on retrieved data can lead to factual errors. 


## What is MIRA?

We introduce the Multimodal Intelligent Retrieval and Augmentation (MIRA) framework, designed to optimize factual accuracy in MLLM. MIRA consists of two key components: (1) a calibrated Rethinking and Rearrangement module that dynamically adjusts the number of retrieved contexts to manage factual risk, and (2) A medical RAG framework integrating image embeddings and a medical knowledge base with a query-rewrite module for efficient multimodal reasoning. This enables the model to integrate both its inherent knowledge and external references effectively. Our evaluation of publicly available medical VQA and report generation benchmarks demonstrates that MIRA substantially enhances factual accuracy and overall performance, achieving new state-of-the-art results.




<div align="center">
 <img src="images/data_statistics.png" width="800"/>
</div>


## Our Pipeline
We design the Agent-X benchmark using a semi-automated pipeline that ensures each task is solvable with a defined tool subset and requires deep reasoning over realistic, multimodal scenarios. The pipeline begins with an LMM (Large Multimodal Model) generating candidate queries based on visual input and an available toolset. These queries are then refined by human annotators for clarity and realism. Next, the refined queries are passed back to the LMM to produce step-by-step reasoning traces, including tool calls, intermediate outputs, and final answers. Each trace is manually reviewed for logical consistency and correctness.
<div align="center">
 <img src="images/pipeline.png" width="800"/>
</div>

## 🏆 Leaderboard, May 2025

### Evaluation Protocol

We evaluate models on the Agent-X benchmark across **three distinct modes**:

1. **Step-by-Step**: Assesses the agent’s ability to execute individual reasoning steps, focusing on how well it follows structured tool-use sequences grounded in visual inputs.

2. **Deep Reasoning**: Evaluates the coherence and logical consistency of the full reasoning trace. This mode emphasizes the agent’s capacity to integrate visual and textual context to produce semantically meaningful and factually accurate explanations.

3. **Outcome**: Measures the agent’s overall task-solving performance by verifying the correctness of the final answer and appropriate tool usage.

We report results using **GPT-4** and **Qwen-15B** as evaluation judges. For each metric, the **best-performing value is shown in bold and underlined**, while the **second-best is italicized**.


### With GPT-4o as a judge
| **Model** | Ground<sub>s</sub> | Tool<sub>p</sub> | Tool<sub>acc</sub> | Fact<sub>acc</sub> | Context<sub>s</sub> | Fact<sub>p</sub> | Sem<sub>acc</sub> | Goal<sub>acc</sub> | Goal<sub>acc</sub><sup>*</sup> | Tool<sub>acc</sub><sup>s</sup> |
|----------|--------------------|------------------|--------------------|--------------------|---------------------|------------------|-------------------|-----------------------|------------------------------|---------------------------|
| *Open-source* |||||||||||
| Phi-4-VL-Instruct|0.13|0.21|0.24|0.61|0.19|0.47|0.40|0.11|0.26|0.42|
| InternVL-2.5-8B|0.45|0.31|0.47|0.68|0.47|0.52|0.60|0.28|0.55|0.58|
| Gemma-3-4B|0.26|0.30|0.78|0.61|*0.54*|0.38|0.54|0.27|*0.67*|0.60|
| InternVL-3-8B|0.46|0.34|0.54|0.68|0.45|*0.70*|0.40|0.20|0.59|0.62|
| VideoLLaMA-3-7B|0.45|0.28|0.46|0.65|0.46|0.62|0.54|0.28|0.54|0.54|
| Qwen-2.5-VL-7B|*0.54*|*0.43*|0.63|*0.75*|<ins><strong>0.57</strong></ins>|0.56|0.67|0.36|0.65|*0.67*|
| Pixtral-12B|0.12|0.20|0.63|0.45|0.19|0.26|0.34|0.07|0.55|0.54|
| LLaMA-3.2-11B-Vision|0.03|0.15|0.14|0.70|0.08|*0.70*|0.24|0.07|0.26|0.42|
| Kimi-VL-A3B-Thinking|0.26|0.19|0.5|0.62|0.42|0.52|0.65|0.29|0.29|0.48|
| mPLUG-Owl3-7B-240728|0.10|0.14|0.30|0.49|0.25|0.32|0.37|0.11|0.26|0.50|
| *Closed-source* |  |  |  |  |  |  |  |  |  |  |
| Gemini-1.5-Pro  | 0.43 | 0.23 | *0.84* | 0.62 | 0.45 | 0.53 | 0.62 | 0.04 | 0.56 | 0.48 |
| Gemini-2.5-Pro  | 0.40 | 0.36 | 0.81 | 0.72 | 0.48 | 0.64 | *0.73* | *0.40* | 0.56 | 0.62 |
| GPT-4o          | <ins><strong>0.60</strong></ins> | <ins><strong>0.47</strong></ins> | 0.72 | <ins><strong>0.81</strong></ins> | <ins><strong>0.57</strong></ins> | <ins><strong>0.79</strong></ins> | 0.59 | 0.37 | <ins><strong>0.70</strong></ins> | <ins><strong>0.68</strong></ins> |
| OpenAI o4-mini  | 0.42 | 0.32 | <ins><strong>0.89</strong></ins> | 0.71 | 0.51 | 0.60 | <ins><strong>0.80</strong></ins> | <ins><strong>0.45</strong></ins> | *0.67* | 0.63 |

### With Qwen-15B  as a judge
| **Model** | Ground<sub>s</sub> | Tool<sub>p</sub> | Tool<sub>acc</sub> | Fact<sub>acc</sub> | Context<sub>s</sub> | Fact<sub>p</sub> | Sem<sub>acc</sub> | Goal<sub>acc</sub> | Goal<sub>acc</sub><sup>*</sup> | Tool<sub>acc</sub><sup>s</sup> |
|----------|--------------------|------------------|--------------------|--------------------|---------------------|------------------|-------------------|-----------------------|------------------------------|---------------------------|
| *Open-source* |||||||||||
| Phi-4-VL-Instruct        | 0.27 | 0.11 | 0.32 | 0.54 | 0.39 | 0.59 | 0.46 | 0.16 | 0.35 | 0.39 |
| InternVL2.5-8B          | 0.38 | 0.16 | 0.49 | 0.63 | 0.51 | 0.61 | 0.55 | 0.29 | 0.53 | 0.53 |
| Gemma-3-4B              | 0.50 | 0.24 | 0.67 | 0.74 | 0.66 | 0.59 | 0.74 | 0.30 | 0.68 | 0.68 |
| InternVL3-8B            | 0.41 | 0.16 | 0.51 | 0.71 | 0.61 | 0.60 | 0.69 | 0.23 | 0.51 | 0.62 |
| VideoLLaMA3-7B          | 0.39 | 0.15 | 0.40 | 0.68 | 0.56 | 0.60 | 0.68 | 0.27 | 0.53 | 0.56 |
| Qwen2.5-VL-7B            | 0.51 | 0.27 | 0.63 | 0.77 | 0.66 | 0.64 | 0.77 | 0.37 | 0.62 | 0.67 |
| Pixtral-12B              | 0.30 | 0.17 | 0.68 | 0.59 | 0.50 | 0.42 | 0.58 | 0.10 | 0.68 | 0.58 |
| LLaMA-3.2-11B-Vision    | 0.16 | 0.06 | 0.12 | 0.49 | 0.17 | 0.74 | 0.20 | 0.10 | 0.11 | 0.15 |
| Kimi-VL-A3B-Thinking    | 0.47 | 0.20 | 0.59 | 0.79 | *0.64* | 0.68 | *0.74* | 0.35 | 0.60 | 0.62 |
| mPLUG-Owl3-7B-240728    | 0.30 | 0.11 | 0.31 | 0.59 | 0.48 | 0.48 | 0.56 | 0.16 | 0.45 | 0.48 |
| *Closed-source* |||||||||||
| Gemini-1.5-Pro       | *0.57* | *0.36* | 0.80 | 0.82 | 0.73 | 0.76 | 0.63 | 0.05 | <ins><strong>0.77</strong></ins> | *0.71* |
| Gemini-2.5-Pro       | <ins><strong>0.63</ins></strong> | <ins><strong>0.40</ins></strong> | *0.84* | *0.86* | *0.76* | <ins><strong>0.80</strong></ins> | *0.83* | *0.50* | *0.74* | <ins><strong>0.72</ins></strong> |
| GPT-4o              | 0.46 | 0.27 | 0.63 | 0.72 | 0.59 | 0.75 | 0.69 | 0.44 | 0.48 | 0.56 |
| OpenAI-o4-mini       | <ins><strong>0.63</strong></ins> | 0.35 | <ins><strong>0.86</ins></strong> | <ins><strong>0.89</ins></strong> | <ins><strong>0.78</strong></ins> | *0.79* | <ins><strong>0.88</strong></ins> | <ins><strong>0.53</strong></ins> | 0.64 | 0.69 |




## 📂 Submodules

### Generation Pipeline  
See [`generation/README.md`](generation/README.md) for details on:
- Frame extraction from video clips  
- Query generation using GPT-4o  
- Step-by-step reasoning trace generation  
> 📁 Path: `generation/README.md`

---

### Analysis & Evaluation  
See [`analysis/README.md`](analysis/README.md) for:
- Error analysis notebook  
- Model comparison plots  
- Tool usage breakdown and visualizations  
> 📁 Path: `analysis/README.md`

---

### Evaluation Scripts  
See [`eval/`](eval) for:
- Scripted evaluation of model inference results  
- Accuracy metrics, binary matching scores, and goal success analysis  
- Useful for benchmarking your model outputs against Agent-X GT  
> 📁 Path: `eval/`


## 📝 Citation
If you use Agent-Xin your research, please cite the following paper:
```
@misc{ashraf2025agentxevaluatingdeepmultimodal,
      title={Agent-X: Evaluating Deep Multimodal Reasoning in Vision-Centric Agentic Tasks}, 
      author={Tajamul Ashraf and Amal Saqib and Hanan Ghani and Muhra AlMahri and Yuhao Li and Noor Ahsan and Umair Nawaz and Jean Lahoud and Hisham Cholakkal and Mubarak Shah and Philip Torr and Fahad Shahbaz Khan and Rao Muhammad Anwer and Salman Khan},
      year={2025},
      eprint={2505.24876},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.24876}, 
}

```



















## MIRA (Multimodal Intelligent Retrieval and Augmentation framework)
Official Code Base for MIRA: A Novel Framework for Fusing Modalities in Medical RAG

Multimodal Large Language Models (MLLMs) have significantly advanced AI-assisted medical diagnosis but often generate factually inconsistent responses that deviate from established medical knowledge. Retrieval-Augmented Generation (RAG) enhances factual accuracy by integrating external sources, but it presents two key challenges. First, insufficient retrieval can miss critical information, whereas excessive retrieval can introduce irrelevant or misleading content, disrupting model output. Second, even when the model initially provides correct answers, over-reliance on retrieved data can lead to factual errors. To address these issues, we introduce Multimodal Intelligent Retrieval and Augmentation (MIRA) framework designed to optimize factual accuracy in MLLM. MIRA consists of two key components: (1) a calibrated Rethinking and Rearrangement module that dynamically adjusts the number of retrieved contexts to manage factual risk, and (2) A medical RAG framework integrating image embeddings and a medical knowledge base with a query-rewrite module for efficient multimodal reasoning. This enables the model to effectively integrate both its inherent knowledge and external references. Our evaluation of publicly available medical VQA and report generation benchmarks demonstrates that MIRA substantially enhances factual accuracy and overall performance, achieving new state-of-the-art results. Code, model, and processed data will be publicly released after acceptance.

[Website](https://tommyix.github.io/MIRA/)

---
## Acknowledgement

The Codebase of MIRA is built on [LLaVA-Next](https://github.com/LLaVA-VL/LLaVA-NeXT), many thanks to them. If you are citing MIRA, please also cite their paper.

---

Now working: writing README for instruction on training / Reproduction of MIRA.
