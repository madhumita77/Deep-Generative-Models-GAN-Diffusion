# Deep Generative Models: GAN and Diffusion Model for Image Synthesis

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Status](https://img.shields.io/badge/status-completed-success.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![GAN](https://img.shields.io/badge/GAN-Vanilla%20%26%20Conditional-blue)
![Diffusion](https://img.shields.io/badge/Diffusion-DDPM-purple)

This repository contains implementations of state-of-the-art generative models developed as part of the ADIPCV (Advanced Deep Learning for Image Processing and Computer Vision) course. The project demonstrates proficiency in implementing complex neural network architectures from scratch using PyTorch.

## 🎯 Project Overview

This assignment focuses on implementing and comparing different generative modeling approaches:

1. **Generative Adversarial Networks (GANs)** - Both vanilla and conditional variants
2. **Denoising Diffusion Probabilistic Models (DDPM)** - Modern diffusion-based generation

## 📁 Repository Structure

```

├── GAN_cGAN.ipynb                 \# Part 1: GAN and cGAN Implementation
├── diffusion_model.ipynb           \# Part 2: DDPM Implementation
├── requirements.txt                  # Python dependencies
└── README.md                       \# This file

```

## 🚀 Implemented Models

### Part 1: Generative Adversarial Networks (GANs)

#### 🎲 Vanilla GAN
- **Architecture**: Fully connected layers with LeakyReLU activations
- **Generator**: 
  - Input: Random noise vector (latent dimension: 100)
  - Output: 28×28 grayscale images with Tanh activation
- **Discriminator**: 
  - Input: 28×28 images
  - Output: Binary classification (real/fake)
- **Training**: 50 epochs on MNIST dataset

#### 🎯 Conditional GAN (cGAN)
- **Enhanced Architecture**: Label conditioning using embedding layers
- **Generator**: Concatenates noise vector with one-hot encoded labels
- **Discriminator**: Processes both image and label information
- **Capability**: Generate specific digits on demand

**Key Features:**
- Flexible architecture design for easy extension to cGAN
- Comprehensive training loop with adversarial loss
- Real-time visualization of generated samples every 10 epochs
- Model persistence for inference

### Part 2: Denoising Diffusion Probabilistic Models (DDPM)

#### 🌊 Diffusion Process Implementation
- **Forward Process**: Gradual noise addition over T=1000 timesteps
- **Noise Schedule**: Linear schedule from 1e-4 to 0.02
- **Reverse Process**: Learned denoising using U-Net architecture

#### 🏗️ U-Net Architecture
- **Encoder**: 3 convolutional layers (32→64→128 channels)
- **Decoder**: 3 transposed convolution layers for upsampling
- **Time Embedding**: Sinusoidal positional encoding for timestep conditioning
- **Skip Connections**: Feature preservation across encoder-decoder

**Key Features:**
- Complete DDPM pipeline from scratch
- Time-conditioned denoising network
- Iterative sampling procedure
- Progressive generation visualization

## 📊 Training Configuration

### Hyperparameters
```


# Common Settings

BATCH_SIZE = 128
LEARNING_RATE = 0.0002 (GAN) / 0.001 (DDPM)
OPTIMIZER = Adam
DEVICE = CUDA (if available)

# GAN Specific

LATENT_DIM = 100
EPOCHS = 50
BETA1 = 0.5, BETA2 = 0.999

# DDPM Specific

TIMESTEPS = 1000
EPOCHS = 10

```

### Dataset
- **MNIST**: Handwritten digits (0-9)
- **Preprocessing**: Normalization to [-1, 1] range
- **Size**: 28×28 grayscale images

## 🎨 Results & Visualizations

### GAN Outputs
- Progressive improvement in image quality over epochs
- Clear digit generation after 50 epochs of training
- Conditional generation of specific digit classes

### DDPM Outputs  
- High-quality digit generation through iterative denoising
- Smooth transition from noise to coherent images
- Superior image fidelity compared to GAN approach

## 🛠️ Technical Implementation Details

### Loss Functions
- **GAN**: Binary Cross-Entropy Loss
- **DDPM**: Mean Squared Error (MSE) for noise prediction

### Training Strategies
- **GAN**: Alternating discriminator and generator training
- **DDPM**: Single-step noise prediction training

### Inference
- **GAN**: Single forward pass through generator
- **DDPM**: Multi-step iterative denoising (1000 steps)

## 📈 Performance Metrics

The models demonstrate:
- **Convergence**: Stable training without mode collapse
- **Quality**: Visually coherent digit generation
- **Diversity**: Generation of all digit classes
- **Conditioning**: Successful label-based generation (cGAN)

## 🔧 Usage

### Running GAN Training
```


# Load the GAN notebook

jupyter notebook Ass_3_GAN.ipynb

# Execute all cells for complete training pipeline

```

### Running DDPM Training  
```


# Load the DDPM notebook

jupyter notebook diffusion_model.ipynb

# Execute all cells for diffusion model training

```

## 📋 Requirements

```

torch>=1.9.0
torchvision>=0.10.0
matplotlib>=3.3.0
numpy>=1.21.0
jupyter>=1.0.0

```

## 🎓 Course Context

This project was developed as part of the **Advanced Deep Learning for Image Processing and Computer Vision (ADIPCV)** course, demonstrating:

- **Deep Learning Mastery**: Implementation of complex architectures from research papers
- **PyTorch Proficiency**: Custom model development without high-level abstractions
- **Generative Modeling**: Understanding of different generative paradigms
- **Research Implementation**: Translation of theoretical concepts to working code

## 🔍 Key Learning Outcomes

1. **GAN Architecture Design**: Understanding adversarial training dynamics
2. **Conditional Generation**: Label-based controllable synthesis  
3. **Diffusion Models**: Modern probabilistic generative modeling
4. **Model Comparison**: Empirical evaluation of different approaches
5. **Implementation Skills**: From-scratch development using PyTorch

## 📚 References

- Goodfellow, I. et al. "Generative Adversarial Networks" (2014)
- Mirza, M. & Osindero, S. "Conditional Generative Adversarial Nets" (2014)  
- Ho, J. et al. "Denoising Diffusion Probabilistic Models" (2020)

## 📄 License

This project is developed for educational purposes as part of academic coursework Advanced Deep Learning for Image Processing and Computer Vision (ADIPCV).

---

## 📧 Contact

Madhumita Gayen - madhumitagayen07@gmail.com

Project Link: [https://github.com/madhumita77/multisite_product_webscrapping](https://github.com/madhumita77/multisite_product_webscrapping)

```

This README provides a comprehensive overview of your project, highlighting both the technical implementation details and the academic context. It's structured to showcase your skills to potential employers while providing clear documentation for anyone wanting to understand or use your code.
<span style="display:none">[^2][^2]</span>

<div style="text-align: center">⁂</div>

[^1]: diffusion_model.ipynb

[^2]: GAN_cGAN.ipynb

