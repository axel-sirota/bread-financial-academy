# Week 1: PyTorch Basics - Notebook Build Guide

This guide contains all the cell content for Week 1. **Copy each cell into the Jupyter notebook in order.**

---

## Section 0: Environment Setup & Motivation

### Cell 1: Markdown - Section Header

```markdown
## Section 0: Environment Setup

Let's start by setting up our environment and understanding what we're building today.
```

### Cell 2: Code - Environment Setup

```python
# Import all necessary libraries
import torch as pt
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

# Check PyTorch version
print(f"PyTorch version: {pt.__version__}")

# Device configuration - automatically use GPU if available, otherwise CPU
device = pt.device('cuda' if pt.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if device.type == 'cuda':
    print(f"GPU Name: {pt.cuda.get_device_name(0)}")
    print(f"GPU Memory: {pt.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Set random seed for reproducibility
# This ensures everyone gets the same results
pt.manual_seed(42)

print("\n✅ Environment setup complete!")
```

### Cell 3: Markdown - What We're Building

```markdown
## What Are We Building Today?

### Real-World Context: Automated Check Processing

Imagine you're a data scientist at a bank. Thousands of checks arrive daily, and clerks manually type in the check amounts. This is slow, expensive, and error-prone.

**Your mission**: Build an AI system that automatically reads handwritten digits on checks.

Today, we'll start with the MNIST dataset - 70,000 images of handwritten digits (0-9). This is a classic dataset that simulates the digit recognition problem banks face.

Let's look at what we're working with:
```

### Cell 4: Code - MNIST Preview

```python
# Load a sample of MNIST data to visualize
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

print(f"\nDataset size: {len(sample_data):,} training images")
print(f"Image dimensions: 28x28 pixels (grayscale)")
print(f"Number of classes: 10 (digits 0-9)")
print(f"\n🎯 Goal: Build a neural network that achieves ~95% accuracy!")
```

---

## Topic 1: PyTorch Tensors & Operations

### Cell 5: Markdown - Theory

```markdown
---

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
digit_image = pt.tensor([
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
- Moving tensors between CPU and GPU
```

### Cell 6: Code - Demo Creating Tensors

```python
# Demo: Creating Tensors

# From Python lists
tensor_1d = pt.tensor([1, 2, 3, 4, 5])
print("1D Tensor (vector):")
print(tensor_1d)
print(f"Shape: {tensor_1d.shape}\n")

# 2D tensor (matrix)
tensor_2d = pt.tensor([[1, 2, 3], [4, 5, 6]])
print("2D Tensor (matrix):")
print(tensor_2d)
print(f"Shape: {tensor_2d.shape}\n")

# Special tensors
zeros = pt.zeros(3, 4)  # 3x4 matrix of zeros
ones = pt.ones(2, 3)    # 2x3 matrix of ones
random = pt.randn(2, 2) # 2x2 matrix with random values (normal distribution)

print("Special tensors:")
print(f"Zeros:\n{zeros}\n")
print(f"Ones:\n{ones}\n")
print(f"Random:\n{random}\n")
```

### Cell 7: Code - Demo Tensor Operations

```python
# Demo: Tensor Operations

x = pt.tensor([1.0, 2.0, 3.0])
y = pt.tensor([4.0, 5.0, 6.0])

# Element-wise operations (performed on corresponding elements)
print("Element-wise operations:")
print(f"x + y = {x + y}")  # [1+4, 2+5, 3+6] = [5, 7, 9]
print(f"x * y = {x * y}")  # [1*4, 2*5, 3*6] = [4, 10, 18]
print(f"x ** 2 = {x ** 2}\n")  # [1^2, 2^2, 3^2] = [1, 4, 9]

# Matrix operations
A = pt.tensor([[1, 2], [3, 4]])
B = pt.tensor([[5, 6], [7, 8]])

# Matrix multiplication (dot product)
C = pt.mm(A, B)  # or A @ B
print("Matrix multiplication A @ B:")
print(C)
```

### Cell 8: Code - Demo Reshaping

```python
# Demo: Reshaping Tensors

# This is CRITICAL for neural networks!
# We often need to flatten images or change tensor dimensions

# Create a tensor representing an image (batch_size=1, channels=1, height=28, width=28)
image = pt.randn(1, 1, 28, 28)
print(f"Original image shape: {image.shape}")

# Flatten the image to a vector (needed for feedforward networks)
# 28 * 28 = 784 pixels
flattened = image.view(1, 784)  # or image.reshape(1, 784)
print(f"Flattened image shape: {flattened.shape}")

# Alternative: use -1 to infer dimension automatically
flattened_auto = image.view(image.size(0), -1)  # -1 means "figure out this dimension"
print(f"Auto-flattened shape: {flattened_auto.shape}")

print("\n💡 Key insight: .view() and .reshape() let us change tensor shape without copying data")
```

### Cell 9: Code - Demo Device Management

```python
# Demo: Device Management (CPU vs GPU)

# Create tensor on CPU (default)
cpu_tensor = pt.tensor([1, 2, 3])
print(f"CPU tensor device: {cpu_tensor.device}")

# Move tensor to GPU if available
if device.type == 'cuda':
    gpu_tensor = cpu_tensor.to(device)  # Copy to GPU
    print(f"GPU tensor device: {gpu_tensor.device}")

    # For operations to work, tensors must be on the SAME device
    # This would ERROR: cpu_tensor + gpu_tensor

    # Move back to CPU
    back_to_cpu = gpu_tensor.cpu()
    print(f"Back to CPU: {back_to_cpu.device}")
else:
    print("GPU not available, staying on CPU")

print("\n💡 Best practice: Use .to(device) to automatically handle CPU/GPU")
```

### Cell 10: Markdown - Lab 1 Instructions

```markdown
---

## Lab 1: Tensor Operations Practice

Now it's your turn! Complete the following exercises to practice working with tensors.

### Exercise 1.1: Create Tensors

Create the following tensors:
1. A 1D tensor with values `[10, 20, 30, 40, 50]`
2. A 3x3 matrix of zeros
3. A 2x4 matrix of random values (use `pt.randn()`)
4. A tensor representing a batch of 5 MNIST images (shape should be `(5, 1, 28, 28)`)

**Hints:**
- Use `pt.tensor()` for specific values
- Use `pt.zeros()`, `pt.ones()`, `pt.randn()` for special tensors
- Check shapes with `.shape`
```

### Cell 11: Code - Lab 1.1 Starter

```python
# Your code here

# 1. Create 1D tensor
tensor_1 = None  # YOUR CODE

# 2. Create 3x3 zeros
tensor_2 = None  # YOUR CODE

# 3. Create 2x4 random
tensor_3 = None  # YOUR CODE

# 4. Create batch of MNIST-shaped tensors
tensor_4 = None  # YOUR CODE

# Print shapes to verify
print(f"tensor_1 shape: {tensor_1.shape if tensor_1 is not None else 'Not created'}")
print(f"tensor_2 shape: {tensor_2.shape if tensor_2 is not None else 'Not created'}")
print(f"tensor_3 shape: {tensor_3.shape if tensor_3 is not None else 'Not created'}")
print(f"tensor_4 shape: {tensor_4.shape if tensor_4 is not None else 'Not created'}")
```

### Cell 12: Markdown - Lab 1.2 Instructions

```markdown
### Exercise 1.2: Tensor Operations

Given two tensors `a` and `b`, perform the following operations:

1. Element-wise addition
2. Element-wise multiplication
3. Compute the mean of tensor `a`
4. Find the maximum value in tensor `b`
```

### Cell 13: Code - Lab 1.2 Starter

```python
# Given tensors
a = pt.tensor([2.0, 4.0, 6.0, 8.0])
b = pt.tensor([1.0, 3.0, 5.0, 7.0])

# Your code here

# 1. Element-wise addition
addition = None  # YOUR CODE

# 2. Element-wise multiplication
multiplication = None  # YOUR CODE

# 3. Mean of a
mean_a = None  # YOUR CODE

# 4. Max of b
max_b = None  # YOUR CODE

print(f"a + b = {addition}")
print(f"a * b = {multiplication}")
print(f"Mean of a = {mean_a}")
print(f"Max of b = {max_b}")
```

### Cell 14: Markdown - Lab 1.3 Instructions

```markdown
### Exercise 1.3: Reshaping for Neural Networks

**Scenario**: You have a batch of 10 MNIST images (shape `(10, 1, 28, 28)`). To feed them into a feedforward neural network, you need to flatten each image into a vector of 784 pixels.

**Task**: Flatten the batch so the shape becomes `(10, 784)`.

**Hint**: Use `.view()` or `.reshape()` with size `(batch_size, -1)`
```

### Cell 15: Code - Lab 1.3 Starter

```python
# Create a batch of 10 random MNIST-like images
batch_images = pt.randn(10, 1, 28, 28)
print(f"Original shape: {batch_images.shape}")

# Your code here: Flatten the images
flattened_batch = None  # YOUR CODE

print(f"Flattened shape: {flattened_batch.shape if flattened_batch is not None else 'Not flattened'}")
print(f"Expected shape: (10, 784)")

# Verify dimensions
if flattened_batch is not None and flattened_batch.shape == (10, 784):
    print("\n✅ Correct! You successfully flattened the batch.")
else:
    print("\n❌ Shape doesn't match. Try again!")
```

---

## Topic 2: Autograd & Computational Graphs

### Cell 16: Markdown - Theory

```markdown
---

# Topic 2: Autograd & Computational Graphs

## Why Do We Need Automatic Differentiation?

Training a neural network requires computing **gradients** (derivatives) to know how to adjust weights:

1. **Forward pass**: Input → Network → Prediction → Loss
2. **Backward pass**: Compute gradients of loss with respect to all weights
3. **Weight update**: Adjust weights in the direction that reduces loss

Manually computing gradients for complex networks is extremely tedious and error-prone. **Autograd** does this automatically!

## How Autograd Works

PyTorch builds a **computational graph** tracking all operations:

```
x (requires_grad=True) → multiply by W → add b → loss
                            ↓                ↓      ↓
                        gradients computed automatically
```

When you call `.backward()`, PyTorch:
1. Traverses the graph backwards (chain rule)
2. Computes gradients for all tensors with `requires_grad=True`
3. Stores gradients in the `.grad` attribute

---

## Demo: Autograd in Action

The instructor will demonstrate:
- How to enable gradient tracking
- Computing gradients with `.backward()`
- **Why we need `zero_grad()`** (critical for training!)
```

### Cell 17: Code - Demo Simple Gradients

```python
# Demo: Simple Gradient Computation

# Create a tensor and tell PyTorch to track operations on it
x = pt.tensor([2.0], requires_grad=True)
print(f"x = {x}")
print(f"requires_grad: {x.requires_grad}\n")

# Perform some operations
# Let's compute y = 3x^2 + 2x + 1
y = 3 * x**2 + 2 * x + 1
print(f"y = 3x² + 2x + 1 = {y}")
print(f"y requires_grad: {y.requires_grad}\n")

# Compute gradients (dy/dx)
# Mathematically: dy/dx = 6x + 2 = 6(2) + 2 = 14
y.backward()  # This computes gradients!

print(f"Gradient dy/dx = {x.grad}")
print(f"Expected: 6x + 2 = 6(2) + 2 = 14")
print("\n✅ Autograd computed the derivative automatically!")
```

### Cell 18: Code - Demo zero_grad() Critical

```python
# Demo: Why zero_grad() is CRITICAL

print("🚨 COMMON MISTAKE: Forgetting to zero gradients\n")

# Create tensor
x = pt.tensor([3.0], requires_grad=True)

# First computation
y1 = x ** 2  # y1 = 9, dy1/dx = 2x = 6
y1.backward()
print(f"First computation: x² = {y1.item():.1f}, gradient = {x.grad.item():.1f}")

# Second computation WITHOUT zeroing gradients
y2 = x ** 3  # y2 = 27, dy2/dx = 3x² = 27
y2.backward()  # ⚠️ This ADDS to existing gradient!
print(f"Second computation (no zero): x³ = {y2.item():.1f}, gradient = {x.grad.item():.1f}")
print(f"❌ Wrong! Gradient accumulated: 6 + 27 = 33\n")

# Correct way: Zero gradients before each new computation
x = pt.tensor([3.0], requires_grad=True)
y1 = x ** 2
y1.backward()
print(f"First: gradient = {x.grad.item():.1f}")

x.grad.zero_()  # 🔑 Zero the gradients!
y2 = x ** 3
y2.backward()
print(f"Second (with zero): gradient = {x.grad.item():.1f}")
print(f"✅ Correct! Gradient = 27\n")

print("💡 KEY TAKEAWAY: Always call zero_grad() before computing new gradients!")
print("   In training loops: optimizer.zero_grad() does this for all model parameters.")
```

### Cell 19: Markdown - Lab 2 Instructions

```markdown
---

## Lab 2: Autograd Practice

### Exercise 2.1: Compute Gradients

Given the function `f(x) = x³ - 2x² + 5x - 1`:

1. Create a tensor `x = 4.0` with gradient tracking enabled
2. Compute `f(x)`
3. Compute the gradient `df/dx`
4. Verify your answer (derivative: `f'(x) = 3x² - 4x + 5`, so `f'(4) = 3(16) - 4(4) + 5 = 48 - 16 + 5 = 37`)

**Hints:**
- Use `requires_grad=True`
- Call `.backward()` on the result
- Access gradient with `.grad`
```

### Cell 20: Code - Lab 2.1 Starter

```python
# Your code here

# 1. Create x with gradient tracking
x = None  # YOUR CODE

# 2. Compute f(x) = x³ - 2x² + 5x - 1
f = None  # YOUR CODE

# 3. Compute gradient
# YOUR CODE

# 4. Print results
if x is not None and f is not None:
    print(f"f(4) = {f.item():.1f}")
    if x.grad is not None:
        print(f"f'(4) = {x.grad.item():.1f}")
        print(f"Expected: 37.0")
        if abs(x.grad.item() - 37.0) < 0.01:
            print("\n✅ Correct gradient!")
        else:
            print("\n❌ Gradient doesn't match. Check your computation.")
```

### Cell 21: Markdown - Lab 2.2 Instructions

```markdown
### Exercise 2.2: Understanding Gradient Accumulation

**Task**: Demonstrate the gradient accumulation problem and fix it.

1. Create `x = 2.0` with gradient tracking
2. Compute `y1 = 5x²` and get the gradient (should be 20)
3. WITHOUT zeroing, compute `y2 = 3x` and get the gradient
4. Observe the accumulated gradient
5. Now zero the gradient and recompute `y2` to get the correct gradient (should be 3)
```

### Cell 22: Code - Lab 2.2 Starter

```python
# Your code here

# Step 1: Create x
x = None  # YOUR CODE

# Step 2: Compute y1 and gradient
# YOUR CODE

print(f"After y1 = 5x²: gradient = ???")  # Replace ??? with actual gradient

# Step 3: Compute y2 WITHOUT zeroing
# YOUR CODE

print(f"After y2 = 3x (no zero): gradient = ???")  # Replace ??? with actual gradient
print("This should be WRONG (accumulated)\n")

# Step 4: Zero gradients
# YOUR CODE

# Step 5: Recompute y2
# YOUR CODE

print(f"After y2 = 3x (with zero): gradient = ???")  # Replace ??? with actual gradient
print("This should be CORRECT (3.0)")
```

---

**Continue in Part 2 of this guide for Topic 3: Neural Networks & MNIST Lab...**
