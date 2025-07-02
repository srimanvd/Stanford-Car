# Stanford Car Classification with Adversarial Robustness

This project tackles car classification using the [Stanford Cars Dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html), incorporating transfer learning, adversarial training, and data augmentation to build accurate and robust models.

## 🚗 Overview

We fine-tuned three pre-trained models on the dataset:

- **ResNet50**
- **EfficientNetV2-S**
- **EfficientNetV2-XL**

The project was made more challenging due to **adversarial perturbations** in the test set. To address this, we used **FGSM (Fast Gradient Sign Method)** during training to build adversarially robust models.

## 🔍 Key Features

- **Transfer Learning** with state-of-the-art CNN architectures
- **Adversarial Training** using FGSM for robustness
- **Data Augmentation** through random cropping and rotation
- **Model Optimization** with PyTorch pruning and inference speedups

## 🏆 Results

- ✅ **91% classification accuracy** with EfficientNetV2
- 🛡️ **15% improvement in robustness** against adversarial examples
- ⚡ **3× faster training** using NVIDIA GH200 GPU with hyperparameter sweeps
- 🚀 **5× faster inference** via model pruning and PyTorch-level optimizations

## 🛠️ Tech Stack

- **Framework:** PyTorch
- **Models:** ResNet50, EfficientNetV2-S, EfficientNetV2-XL
- **Tools:** Jupyter Notebooks, TorchVision, NVIDIA GH200

## 📁 Project Structure

```plaintext
.
├── resnet50_car_classification.ipynb
├── efficientnetv2s_car_classification.ipynb
├── efficientnetv2xl_car_classification.ipynb
└── README.md
