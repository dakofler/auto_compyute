# AutoCompyute

AutoCompyute: Lightweight Deep Learning with Pure Python

AutoCompyute is a lightweight and efficient deep learning library that provides automatic differentiation using only NumPy as the backend for computation. CuPy can be used as a drop-in replacement for NumPy. Designed for simplicity and performance, it enables users to build and train deep learning models with minimal dependencies while leveraging GPU acceleration. The package supports:

- Flexible tensor operations with gradient tracing.
- Customizable neural network layers and loss functions.
- Optimized computation for both CPU and GPU.
- A focus on clarity, making it ideal for research and education.

Whether you're exploring the fundamentals of autograd or developing deep learning models, this library offers a pure Python solution with a streamlined API.

## How It Works

At its core, it features a `Tensor` object for storing data and gradients, and `Op` objects for defining differentiable operations.

### Tensor Object

The `Tensor` object is the fundamental data structure in this autograd engine. It holds numerical data as a NumPy array and tracks gradients for backpropagation.

**Most Important Attributes:**
- `data`: A NumPy array containing the numerical values.
- `grad`: A NumPy array holding the computed gradients (initialized as `None`).
- `ctx`: Stores a reference to the operation (`Op`) that created this tensor.
- `parents`: References the parent tensors involved in its computation.

### Op Object

The `Op` object represents a differentiable operation applied to tensors. Each operation implements both a forward and backward pass.

### How the Engine Works

1. **Computation Graph Construction**: When an operation (e.g., `Tensor.add`) is called, an `Op` instance is created. It stores input tensors and performs the forward computation, returning a new output tensor. The output tensor maintains references to the `Op` and parent tensors, forming a computational graph.

2. **Backpropagation**: Calling `backward()` on the final tensor initiates gradient computation. The gradients propagate in reverse through the computational graph by calling `backward()` on each `Op`, which distributes gradients to parent tensors.

3. **Gradient Storage**: As the gradients are propagated, they are stored in the `grad` attribute of each `Tensor`, enabling parameter updates for optimization.
