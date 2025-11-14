#!/usr/bin/env python3
"""
Script to build the Week 1 PyTorch notebook incrementally.
This ensures we don't create files that are too large.
"""
import json

def add_cells_to_notebook(notebook_path, new_cells):
    """Add cells to an existing notebook."""
    with open(notebook_path) as f:
        nb = json.load(f)

    nb['cells'].extend(new_cells)

    with open(notebook_path, 'w') as f:
        json.dump(nb, f, indent=1)

    print(f"Added {len(new_cells)} cells. Total cells now: {len(nb['cells'])}")
    return len(nb['cells'])

# Topic 1 Demo and Lab cells
topic1_demo_lab_cells = [
    # Demo: Creating Tensors
    {
        "cell_type": "code",
        "execution_count": null,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Demo: Creating Tensors\n",
            "\n",
            "# From Python lists\n",
            "tensor_1d = torch.tensor([1, 2, 3, 4, 5])\n",
            "print(\"1D Tensor (vector):\")\n",
            "print(tensor_1d)\n",
            "print(f\"Shape: {tensor_1d.shape}\\n\")\n",
            "\n",
            "# 2D tensor (matrix)\n",
            "tensor_2d = torch.tensor([[1, 2, 3], [4, 5, 6]])\n",
            "print(\"2D Tensor (matrix):\")\n",
            "print(tensor_2d)\n",
            "print(f\"Shape: {tensor_2d.shape}\\n\")\n",
            "\n",
            "# Special tensors\n",
            "zeros = torch.zeros(3, 4)  # 3x4 matrix of zeros\n",
            "ones = torch.ones(2, 3)    # 2x3 matrix of ones\n",
            "random = torch.randn(2, 2) # 2x2 matrix with random values (normal distribution)\n",
            "\n",
            "print(\"Special tensors:\")\n",
            "print(f\"Zeros:\\n{zeros}\\n\")\n",
            "print(f\"Ones:\\n{ones}\\n\")\n",
            "print(f\"Random:\\n{random}\\n\")"
        ]
    },
    # Demo: Tensor Operations
    {
        "cell_type": "code",
        "execution_count": null,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Demo: Tensor Operations\n",
            "\n",
            "x = torch.tensor([1.0, 2.0, 3.0])\n",
            "y = torch.tensor([4.0, 5.0, 6.0])\n",
            "\n",
            "# Element-wise operations (performed on corresponding elements)\n",
            "print(\"Element-wise operations:\")\n",
            "print(f\"x + y = {x + y}\")  # [1+4, 2+5, 3+6] = [5, 7, 9]\n",
            "print(f\"x * y = {x * y}\")  # [1*4, 2*5, 3*6] = [4, 10, 18]\n",
            "print(f\"x ** 2 = {x ** 2}\\n\")  # [1^2, 2^2, 3^2] = [1, 4, 9]\n",
            "\n",
            "# Matrix operations\n",
            "A = torch.tensor([[1, 2], [3, 4]])\n",
            "B = torch.tensor([[5, 6], [7, 8]])\n",
            "\n",
            "# Matrix multiplication (dot product)\n",
            "C = torch.mm(A, B)  # or A @ B\n",
            "print(\"Matrix multiplication A @ B:\")\n",
            "print(C)"
        ]
    },
    # Demo: Reshaping
    {
        "cell_type": "code",
        "execution_count": null,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Demo: Reshaping Tensors\n",
            "\n",
            "# This is CRITICAL for neural networks!\n",
            "# We often need to flatten images or change tensor dimensions\n",
            "\n",
            "# Create a tensor representing an image (batch_size=1, channels=1, height=28, width=28)\n",
            "image = torch.randn(1, 1, 28, 28)\n",
            "print(f\"Original image shape: {image.shape}\")\n",
            "\n",
            "# Flatten the image to a vector (needed for feedforward networks)\n",
            "# 28 * 28 = 784 pixels\n",
            "flattened = image.view(1, 784)  # or image.reshape(1, 784)\n",
            "print(f\"Flattened image shape: {flattened.shape}\")\n",
            "\n",
            "# Alternative: use -1 to infer dimension automatically\n",
            "flattened_auto = image.view(image.size(0), -1)  # -1 means \"figure out this dimension\"\n",
            "print(f\"Auto-flattened shape: {flattened_auto.shape}\")\n",
            "\n",
            "print(\"\\n💡 Key insight: .view() and .reshape() let us change tensor shape without copying data\")"
        ]
    },
    # Demo: Device Management
    {
        "cell_type": "code",
        "execution_count": null,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Demo: Device Management (CPU vs GPU)\n",
            "\n",
            "# Create tensor on CPU (default)\n",
            "cpu_tensor = torch.tensor([1, 2, 3])\n",
            "print(f\"CPU tensor device: {cpu_tensor.device}\")\n",
            "\n",
            "# Move tensor to GPU if available\n",
            "if device.type == 'cuda':\n",
            "    gpu_tensor = cpu_tensor.to(device)  # Copy to GPU\n",
            "    print(f\"GPU tensor device: {gpu_tensor.device}\")\n",
            "    \n",
            "    # For operations to work, tensors must be on the SAME device\n",
            "    # This would ERROR: cpu_tensor + gpu_tensor\n",
            "    \n",
            "    # Move back to CPU\n",
            "    back_to_cpu = gpu_tensor.cpu()\n",
            "    print(f\"Back to CPU: {back_to_cpu.device}\")\n",
            "else:\n",
            "    print(\"GPU not available, staying on CPU\")\n",
            "\n",
            "print(\"\\n💡 Best practice: Use .to(device) to automatically handle CPU/GPU\")"
        ]
    },
    # Lab 1 Instructions
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "\n",
            "## Lab 1: Tensor Operations Practice\n",
            "\n",
            "Now it's your turn! Complete the following exercises to practice working with tensors.\n",
            "\n",
            "### Exercise 1.1: Create Tensors\n",
            "\n",
            "Create the following tensors:\n",
            "1. A 1D tensor with values `[10, 20, 30, 40, 50]`\n",
            "2. A 3x3 matrix of zeros\n",
            "3. A 2x4 matrix of random values (use `torch.randn()`)\n",
            "4. A tensor representing a batch of 5 MNIST images (shape should be `(5, 1, 28, 28)`)\n",
            "\n",
            "**Hints:**\n",
            "- Use `torch.tensor()` for specific values\n",
            "- Use `torch.zeros()`, `torch.ones()`, `torch.randn()` for special tensors\n",
            "- Check shapes with `.shape`"
        ]
    },
    # Lab 1.1 Starter
    {
        "cell_type": "code",
        "execution_count": null,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Your code here\n",
            "\n",
            "# 1. Create 1D tensor\n",
            "tensor_1 = None  # YOUR CODE\n",
            "\n",
            "# 2. Create 3x3 zeros\n",
            "tensor_2 = None  # YOUR CODE\n",
            "\n",
            "# 3. Create 2x4 random\n",
            "tensor_3 = None  # YOUR CODE\n",
            "\n",
            "# 4. Create batch of MNIST-shaped tensors\n",
            "tensor_4 = None  # YOUR CODE\n",
            "\n",
            "# Print shapes to verify\n",
            "print(f\"tensor_1 shape: {tensor_1.shape if tensor_1 is not None else 'Not created'}\")\n",
            "print(f\"tensor_2 shape: {tensor_2.shape if tensor_2 is not None else 'Not created'}\")\n",
            "print(f\"tensor_3 shape: {tensor_3.shape if tensor_3 is not None else 'Not created'}\")\n",
            "print(f\"tensor_4 shape: {tensor_4.shape if tensor_4 is not None else 'Not created'}\")"
        ]
    },
    # Lab 1.2 Instructions
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Exercise 1.2: Tensor Operations\n",
            "\n",
            "Given two tensors `a` and `b`, perform the following operations:\n",
            "\n",
            "1. Element-wise addition\n",
            "2. Element-wise multiplication\n",
            "3. Compute the mean of tensor `a`\n",
            "4. Find the maximum value in tensor `b`"
        ]
    },
    # Lab 1.2 Starter
    {
        "cell_type": "code",
        "execution_count": null,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Given tensors\n",
            "a = torch.tensor([2.0, 4.0, 6.0, 8.0])\n",
            "b = torch.tensor([1.0, 3.0, 5.0, 7.0])\n",
            "\n",
            "# Your code here\n",
            "\n",
            "# 1. Element-wise addition\n",
            "addition = None  # YOUR CODE\n",
            "\n",
            "# 2. Element-wise multiplication\n",
            "multiplication = None  # YOUR CODE\n",
            "\n",
            "# 3. Mean of a\n",
            "mean_a = None  # YOUR CODE\n",
            "\n",
            "# 4. Max of b\n",
            "max_b = None  # YOUR CODE\n",
            "\n",
            "print(f\"a + b = {addition}\")\n",
            "print(f\"a * b = {multiplication}\")\n",
            "print(f\"Mean of a = {mean_a}\")\n",
            "print(f\"Max of b = {max_b}\")"
        ]
    },
    # Lab 1.3 Instructions
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Exercise 1.3: Reshaping for Neural Networks\n",
            "\n",
            "**Scenario**: You have a batch of 10 MNIST images (shape `(10, 1, 28, 28)`). To feed them into a feedforward neural network, you need to flatten each image into a vector of 784 pixels.\n",
            "\n",
            "**Task**: Flatten the batch so the shape becomes `(10, 784)`.\n",
            "\n",
            "**Hint**: Use `.view()` or `.reshape()` with size `(batch_size, -1)`"
        ]
    },
    # Lab 1.3 Starter
    {
        "cell_type": "code",
        "execution_count": null,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Create a batch of 10 random MNIST-like images\n",
            "batch_images = torch.randn(10, 1, 28, 28)\n",
            "print(f\"Original shape: {batch_images.shape}\")\n",
            "\n",
            "# Your code here: Flatten the images\n",
            "flattened_batch = None  # YOUR CODE\n",
            "\n",
            "print(f\"Flattened shape: {flattened_batch.shape if flattened_batch is not None else 'Not flattened'}\")\n",
            "print(f\"Expected shape: (10, 784)\")\n",
            "\n",
            "# Verify dimensions\n",
            "if flattened_batch is not None and flattened_batch.shape == (10, 784):\n",
            "    print(\"\\n✅ Correct! You successfully flattened the batch.\")\n",
            "else:\n",
            "    print(\"\\n❌ Shape doesn't match. Try again!\")"
        ]
    }
]

# Add Topic 1 demo and lab cells
notebook_path = '/Users/axelsirota/repos/bread-financial-academy/exercises/week_01_pytorch_basics/week_01_pytorch_basics.ipynb'
add_cells_to_notebook(notebook_path, topic1_demo_lab_cells)
