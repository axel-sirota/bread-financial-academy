#!/usr/bin/env python3
"""
Complete Week 1 PyTorch Basics Notebook Builder
Builds the entire notebook programmatically to avoid JSON escaping hell
"""
import json

def create_markdown_cell(content):
    """Create a markdown cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": content.split('\n')
    }

def create_code_cell(content):
    """Create a code cell."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": content.split('\n')
    }

# All cells in order
cells = []

# Cell 0: Header
cells.append(create_markdown_cell("""# Week 1: PyTorch Basics - Building Neural Networks

**Bread Financial - AI for Data Scientists Academy**

---

## Learning Objectives

By the end of this session, you will be able to:

- Work with PyTorch tensors and perform basic tensor operations
- Understand automatic differentiation with autograd
- Build feedforward neural networks using `nn.Module`
- Implement a complete training loop with proper best practices
- Train a neural network to classify handwritten digits (MNIST dataset)
- Evaluate model performance and visualize predictions

## Prerequisites

Before starting this notebook, you should have:

- ✅ Python programming fundamentals
- ✅ Basic NumPy and Pandas knowledge
- ✅ Watched pre-class videos on: neural networks, backpropagation, activation functions

## Session Format

- **2-hour hands-on session**
- Instructor will demo key concepts (live coding)
- You will complete labs independently
- Solutions shared after class

---

## 🚀 Important: GPU Setup for Google Colab

If you're running this notebook on Google Colab, **enable GPU acceleration** for faster training:

1. Click on **Runtime** in the top menu
2. Select **Change runtime type**
3. Under **Hardware accelerator**, select **T4 GPU**
4. Click **Save**

The notebook will work fine on CPU, but GPU makes training much faster!

---"""))

# Cell 1: Section 0 Header
cells.append(create_markdown_cell("""## Section 0: Environment Setup

Let's start by setting up our environment and understanding what we're building today."""))

# Cell 2: Environment Setup Code
cells.append(create_code_cell("""# Import all necessary libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

# Check PyTorch version
print(f"PyTorch version: {torch.__version__}")

# Device configuration - automatically use GPU if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if device.type == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Set random seed for reproducibility
# This ensures everyone gets the same results
torch.manual_seed(42)

print("\\n✅ Environment setup complete!")"""))

# Cell 3: Real-World Context
cells.append(create_markdown_cell("""## What Are We Building Today?

### Real-World Context: Automated Check Processing

Imagine you're a data scientist at a bank. Thousands of checks arrive daily, and clerks manually type in the check amounts. This is slow, expensive, and error-prone.

**Your mission**: Build an AI system that automatically reads handwritten digits on checks.

Today, we'll start with the MNIST dataset - 70,000 images of handwritten digits (0-9). This is a classic dataset that simulates the digit recognition problem banks face.

Let's look at what we're working with:"""))

# Cell 4: MNIST Preview
cells.append(create_code_cell("""# Load a sample of MNIST data to visualize
# Don't worry about the details yet - we'll explain everything later!
sample_data = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

# Display 10 sample digits
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    image, label = sample_data[i]
    ax.imshow(image.squeeze(), cmap='gray')
    ax.set_title(f'Label: {label}', fontsize=14)
    ax.axis('off')

plt.suptitle('MNIST Handwritten Digits - Examples', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

print(f"\\nDataset size: {len(sample_data):,} training images")
print(f"Image dimensions: 28x28 pixels (grayscale)")
print(f"Number of classes: 10 (digits 0-9)")
print(f"\\n🎯 Goal: Build a neural network that achieves ~95% accuracy!")"""))

print(f"Created {len(cells)} cells so far (Section 0 complete)")
print("Building Topic 1...")

# TOPIC 1: PyTorch Tensors & Operations

# Cell 5: Topic 1 Theory
cells.append(create_markdown_cell("""---

# Topic 1: PyTorch Tensors & Operations

## Why PyTorch?

You might be wondering: "We already know NumPy, why learn PyTorch?"

**PyTorch offers three critical advantages:**

1. **GPU Acceleration**: PyTorch tensors can run on GPUs, making computations 10-100x faster
2. **Automatic Differentiation**: PyTorch automatically computes gradients (derivatives) for us - essential for training neural networks
3. **Deep Learning Ecosystem**: Built-in layers, optimizers, and tools specifically designed for neural networks

## What is a Tensor?

A **tensor** is a multi-dimensional array - just like NumPy arrays, but optimized for deep learning:

- **Scalar** (0D tensor): A single number → `5`
- **Vector** (1D tensor): Array of numbers → `[1, 2, 3, 4]`
- **Matrix** (2D tensor): Table of numbers → `[[1, 2], [3, 4]]`
- **3D+ tensors**: Images (height × width × channels), videos, etc.

### Example: Representing an MNIST Image

Each MNIST digit is a **2D tensor** of shape `(28, 28)` containing pixel intensities:

```python
# Conceptual example (actual values)
digit_image = torch.tensor([
    [0.0, 0.0, 0.5, 0.8, 0.8, 0.5, ...],  # Row 1 (28 pixels)
    [0.0, 0.2, 0.9, 1.0, 1.0, 0.9, ...],  # Row 2
    ...  # 28 rows total
])  # Shape: (28, 28)
```

---

## Demo: Tensor Basics

The instructor will demonstrate tensor creation and operations. **Pay attention to**:
- How to create tensors
- Basic operations (add, multiply, reshape)
- Moving tensors between CPU and GPU"""))

# Continuing with more cells... [Rest of notebook creation code]
#  I'll add remaining cells in batches to keep under token limits

# Write notebook
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open('week_01_pytorch_basics_NEW.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print(f"\\nCreated NEW notebook with {len(cells)} cells")
print(f"File size: {len(json.dumps(notebook)) / 1024:.1f} KB")
print("\\nNext: Continue adding remaining topics (2-5)")
