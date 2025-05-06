# RobustBind

**RobustBind** is a fine-tuning and evaluation framework for training adversarially robust multi-modal models built on top of [UniBind](https://github.com/qc-ly/UniBind). It supports robust fine-tuning using PGD/APGD and evaluation across 6 modalities and 12 datasets.

> 💡 This project builds on the UniBind CVPR 2024 codebase, adapted for adversarial robustness research.

---

## 📦 Dataset & Checkpoint Download

All pretrained weights, LoRA weights, and center embeddings are available here:

📂 [Google Drive – Datasets & Checkpoints](https://drive.google.com/open?id=19_Pgqt4aU1JYAx8TgLq99jEA4QPyNPyG&usp=drive_fs)

---

## ⚙️ Setup

```bash
git clone https://github.com/TensorNeural/RobustBind
cd RobustBind

conda create -n robustbind python=3.9 -y
conda activate robustbind

# Install dependencies
conda install pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia
pip install -r requirements.txt
```

---

## 🚀 Training

```bash
bash train.sh
```

- Trains with PGD using `train_robust_unibind.py`
- Supports all 6 modalities and 12 datasets
- Saves outputs to `output/{modality}/{dataset}`

---

## 🧪 Evaluation

```bash
bash eval_unibind.sh
```

- Evaluates clean and robust accuracy via `eval_unibind.py`
- Supports LoRA weights like `eps2_lora_weights.pt`, `eps4_lora_weights.pt`
- Prints accuracy at multiple epsilons

---

## Modalities

- image
- audio
- video
- event
- thermal
- point

## Datasets

- ImageNet-1K
- Places365
- ESC-50
- Urban-Sound-8K
- LLVIP
- RGB-T
- ModalNet40
- ShapeNet
- MSR-VTT
- UCF-101
- N-Caltech-101
- N-ImageNet-1K

---

## Acknowledgements

This repo is built on [UniBind (CVPR 2024)](https://github.com/qc-ly/UniBind).

We thank:
- [ImageBind](https://github.com/facebookresearch/ImageBind)
- [PointBind](https://github.com/ZiyuGuo99/Point-Bind_Point-LLM)
- [LLaMA-Adapter](https://github.com/OpenGVLab/LLaMA-Adapter)
- [E-CLIP](https://github.com/jiazhou-garland/E-CLIP)

---

## 📖 Citation

```bibtex
@article{lyu2024unibind,
  title={UniBind: LLM-Augmented Unified and Balanced Representation Space to Bind Them All},
  author={Lyu, Yuanhuiyi and Zheng, Xu and Zhou, Jiazhou and Wang, Lin},
  journal={arXiv preprint arXiv:2403.12532},
  year={2024}
}
```

---

## 📬 Contact

```
Yang Liu, magicoder20@gmail.com
Zheng Xu
```
