def gradient_descent(x, y, w=0.5, alpha=0.00001, ε=1e-8, max_iterations=10000, print_progress=1000):
    iterations = 0
    dw = float('inf')
    while abs(dw) > ε and iterations <= max_iterations:
        g = gradient(x, y, w)  # calculate current gradient
        dw = alpha * float(g)  # ensure g is converted to a float
        w -= dw
        iterations += 1
        if iterations % print_progress == 0:
            print(f"Iteration {iterations}. w = {w:.4f}, dw = {dw:.4f}")
    print("Terminated!")
    print(f"Iteration {iterations - 1}. w = {w:.4f}.")
    return w

def gradient(x, y, w):
    # Ensure this function returns a single numeric value
    return sum((w * xi - yi) * xi for xi, yi in zip(x, y)) / len(x)

import numpy as np
x = np.array(x, dtype=float)
y = np.array(y, dtype=float)

gradient_descent(x, y, w=0.5, alpha=0.00001)
gradient_descent(x, y, w=0.5, alpha=0.00005, print_progress=2)
plot_gradient_descent(x, y, w=0.5, alpha=0.00005)
plot_gradient_descent(x, y, w=0.5, alpha=0.00018, max_iterations=4)