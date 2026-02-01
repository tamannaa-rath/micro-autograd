from nn import MLP

# The dataset: 4 examples with 3 features each
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
# Desired targets for each example
ys = [1.0, -1.0, -1.0, 1.0]

# The training loop (20 iterations)
for k in range(20):
    
    # 1. Forward pass
    # Get predictions for all inputs in xs
    ypred = [n(x) for x in xs]
    # Calculate Mean Squared Error (MSE) loss
    loss = sum(((yout - ygt)**2 for ygt, yout in zip(ys, ypred)), Value(0.0))
    
    # 2. Backward pass
    # Zero out gradients before starting backprop
    for p in n.parameters():
        p.grad = 0.0
    loss.backward()
    
    # 3. Update (Stochastic Gradient Descent)
    # Move parameters in the opposite direction of the gradient
    for p in n.parameters():
        p.data += -0.05 * p.grad
        
    print(k, loss.data)