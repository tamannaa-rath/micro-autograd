# my-micrograd

I built this project to understand how neural networks actually work under the hood. Most of the time we just use `import torch`, but I wanted to see if I could write the math myself from scratch.

This is a tiny **autograd engine**. It can track a bunch of math operations and then use the chain rule to calculate the gradients (slopes) automatically.

## What's inside

* **`engine.py`**: This is the heart of the project. It has a `Value` class that stores a number and remembers which operations created it.
* **`nn.py`**: This uses the engine to build neurons and layers. You can use it to create a Multi-Layer Perceptron (MLP) just like in PyTorch.
* **`train.py`**: A small script that shows the network "learning" by decreasing the loss over time.



## How it works

The main idea is that every time you do something like `a * b + c`, the engine remembers how those numbers are connected. When you call `.backward()`, it walks through those connections in reverse order and calculates how much every single number affected the final result.



I even tested it against the real PyTorch library, and the gradients matched exactly.

## How to run it

Make sure you have Python installed. Clone the repo and run the training script:

```bash
python train.py
