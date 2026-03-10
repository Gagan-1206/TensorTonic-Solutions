def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Minimize f(x) = a*x^2 + b*x + c using gradient descent.
    """

    x = float(x0)

    for _ in range(steps):
        # gradient of f(x) = ax^2 + bx + c
        grad = 2 * a * x + b

        # gradient descent update
        x = x - lr * grad

    return float(x)