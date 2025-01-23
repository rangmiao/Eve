

<h2 align="center"> <a href="https://arxiv.org/pdf/2501.04322"> <img src="assets/Eve-remove-background.png"  width="30"  >  Eve: Efficient Multimodal Vision Language Models with Elastic Visual Experts</a></h2>
<a href='https://arxiv.org/pdf/2501.04322'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>


## Abstract
Multimodal vision language models (VLMs) have made significant progress with the support of continuously increasing model sizes and data volumes. Running VLMs on edge devices has become a challenge for their widespread application. There are several efficient VLM efforts, but they often sacrifice linguistic capabilities to enhance multimodal abilities, or require extensive training. To address this quandary, we introduce the innovative framework of Efficient Vision Language Models with Elastic Visual Experts (Eve). By strategically incorporating adaptable visual expertise at multiple stages of training, Eve strikes a balance between preserving linguistic abilities and augmenting multimodal capabilities. This balanced approach results in a versatile model with only 1.7B parameters that delivers significant improvements in both multimodal and linguistic tasks. Notably, in configurations below 3B parameters, Eve distinctly outperforms in language benchmarks and achieves state-of-the-art results 68.87% in VLM Benchmarks. Additionally, its multimodal accuracy outstrips that of the larger 7B LLaVA-1.5 model.
<p align="center"><img src="assets/VLM_language.PNG" width="80%" height="80%"></p>

### 😮 Highlights

Eve shows excellent performance in VLM task and Language task.

- ##### 🔥 High performance in VLM and Language task, but with fewer parameters
  
  with only **1.8B  parameters** that delivers significant improvements in multimodal capabilities and maintains the integrity of existing language abilities.
- ##### Proprietary Visual and Language experts, each focusing on their specific tasks to enhance multiple capabilities of the model.

## Methods
Our proposed model, Eve, incorporates a sophisticated three-stage framework, strategically integrating elastic vision experts at each stage. A key focus of our approach is the preservation of linguistic capabilities throughout the training process. Notably, the linguistic proficiency of the model remains unaffected by the variations in pre-training data used for the visual encoder during the first two stages of training. This stability in linguistic performance is a significant accomplishment, as it ensures that the model’s ability to process and comprehend language is not compromised by changes in the visual encoder’s pretraining.
<p align="center"><img src="assets/method.PNG"></p>
## ⚙️Requirements and Installation
We recommend the requirements as follows.

Python == 3.10
Pytorch == 2.0.1
CUDA Version >= 11.7
Transformers == 4.37.0
Tokenizers==0.15.1
Install required packages:
```shell
git clone https://github.com/rangmiao/Eve
cd Eve
conda create -n eve python=3.10 -y
conda activate eve
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install -e ".[train]"
```
## 🗝️Training
The training  instruction can be find in scripts/va/zhizi, including three stage script. 
## 👍 Acknowledgement

* [MoELLaVA](https://github.com/PKU-YuanGroup/MoE-LLaVA) The codebase we built upon and it is an efficient large language and vision assistant.
* [LLaVA](https://github.com/haotian-liu/LLaVA) This is an important work in the field of VLM.
## ✏️ Citation
If you find our paper useful in your research, please consider giving a star :star: and citation :pencil:.

```BibTeX
@article{Rang2025eve,
  title={Eve: Efficient Multimodal Vision Language Models with Elastic Visual Experts},
  author={Rang M, Bi Z, Liu C, Tang, Y., Han, K., & Wang, Y.et al},
  journal={arXiv preprint rXiv:2501.04322, 2025},
  year={2025}
}
```
