## Objective
Fine-tune and evaluate VisionMamba (Vim) on the Flame wildfire dataset. Propose, implement and compare Hybrid models with existing SOTA models that don't make use of self-attention, thereby, avoiding it's computational costs. The idea is to verify or refute claims that Mamba (VisionMamba), in this case and State Space Models in general, are able to capture long-range dependencies without the need for self-attention.

## Flame wildfire dataset
https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs

## Branches
Right now, we're working on three branches
- `main` - This branch contains the original code from the VisionMamba repository. We have used this branch to fine-tune and evaluate the VisionMamba model on the Flame wildfire dataset.
- `control` - This branch contains code for our controlled experiments with other models.
- `hybrid` - This branch contains code for our VisionMambaTiny+EfficientNetB0 hybrid model.

## Entension of original model and code
1. Added support for single GPU training and fine-tuning bypassing torch.distributed module
2. Addressed bugs related to conv1d versioning in mamba module's setup.py
3. Added support for splitting train data into train and validation sets
4. Added support for binary classifier
5. Control experiment conditions
6. Added test.py for separate evaluation on a previously unseen test set
7. Bash scripts for training, fine-tuning and testing separate variants for the VisionMamba model
8. Code refactoring for better readability
9. Documentation on environment setup

## Setup
For detailed setup instructions, please go through our Notion page.

## Queries
Please direct all VisionMamba related queries to Sayeedur Rahman (sayeedur.rahman@g.bracu.ac.bd) or open an issue on Github.


## Model Weights

| Model | #param. | Top-1 Acc. | Top-5 Acc. | Hugginface Repo |
|:------------------------------------------------------------------:|:-------------:|:----------:|:----------:|:----------:|
| [Vim-tiny](https://huggingface.co/hustvl/Vim-tiny-midclstok)    |       7M       |   76.1   | 93.0 | https://huggingface.co/hustvl/Vim-tiny-midclstok |
| [Vim-tiny<sup>+</sup>](https://huggingface.co/hustvl/Vim-tiny-midclstok)    |       7M       |   78.3   | 94.2 | https://huggingface.co/hustvl/Vim-tiny-midclstok |
| [Vim-small](https://huggingface.co/hustvl/Vim-small-midclstok)    |       26M       |   80.5   | 95.1 | https://huggingface.co/hustvl/Vim-small-midclstok |
| [Vim-small<sup>+</sup>](https://huggingface.co/hustvl/Vim-small-midclstok)    |       26M       |   81.6   | 95.4 | https://huggingface.co/hustvl/Vim-small-midclstok |
| [Vim-base](https://huggingface.co/hustvl/Vim-base-midclstok)    |       98M       |   81.9   | 95.8 | https://huggingface.co/hustvl/Vim-base-midclstok |

**Notes:**
- <sup>+</sup> means that we finetune at finer granularity with short schedule.
## Evaluation on Provided Weights
To evaluate `Vim-Ti` on ImageNet-1K, run:
```bash
python main.py --eval --resume /path/to/ckpt --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --data-path /path/to/imagenet
```
## Acknowledgement :heart: from VisionMamba authors
This project is based on Mamba ([paper](https://arxiv.org/abs/2312.00752), [code](https://github.com/state-spaces/mamba)), Causal-Conv1d ([code](https://github.com/Dao-AILab/causal-conv1d)), DeiT ([paper](https://arxiv.org/abs/2012.12877), [code](https://github.com/facebookresearch/deit)). Thanks for their wonderful works.

## Citation
If you find Vim is useful in your research or applications, please consider giving us a star 🌟 and citing it by the following BibTeX entry.

```bibtex
@inproceedings{vim,
  title={Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model},
  author={Zhu, Lianghui and Liao, Bencheng and Zhang, Qian and Wang, Xinlong and Liu, Wenyu and Wang, Xinggang},
  booktitle={Forty-first International Conference on Machine Learning}
}
```
## Original Repository
https://github.com/hustvl/Vim
