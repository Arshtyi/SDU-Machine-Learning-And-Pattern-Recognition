import numpy as np
import matplotlib.pyplot as plt

X = np.loadtxt("res/ex2x.dat")
y = np.loadtxt("res/ex2y.dat")
m = y.size
X = np.concatenate([np.ones((m, 1)), X], axis=1)
X_raw = X.copy()
mu = np.mean(X[:, 1:], axis=0)
sigma = np.std(X[:, 1:], axis=0)
X[:, 1:] = (X[:, 1:] - mu) / sigma


def compute_cost(X, y, theta):
    m = y.size
    J = (1 / (2 * m)) * np.sum((X @ theta - y) ** 2)
    return J


def gradient_descent(X, y, theta, alpha, num_iters):
    m = y.size
    J_history = []
    for i in range(num_iters):
        theta = theta - (alpha / m) * (X.T @ (X @ theta - y))
        J_history.append(compute_cost(X, y, theta))
    return theta, J_history


alphas = [0.01, 0.03, 0.1, 0.3, 1]
num_iters = 50
theta_init = np.zeros(X.shape[1])
plt.figure()
for alpha in alphas:
    theta = theta_init.copy()
    _, J_hist = gradient_descent(X, y, theta, alpha, num_iters)
    plt.plot(range(num_iters), J_hist, label=f"alpha={alpha}")
plt.xlabel("Number of iterations")
plt.ylabel("Cost J")
plt.legend()
plt.title("Cost function vs. Iterations for different learning rates")
plt.show()

alpha = 0.1
num_iters = 400
theta = np.zeros(X.shape[1])
theta, J_history = gradient_descent(X, y, theta, alpha, num_iters)
x_test = np.array([1, 1650, 3])
x_test[1:] = (x_test[1:] - mu) / sigma
price = x_test @ theta
print(f"预测房价（梯度下降）：{price:.2f}")
print(f"最终 theta：{theta}")

X_ne = X_raw
theta_ne = np.linalg.pinv(X_ne.T @ X_ne) @ X_ne.T @ y
x_test_ne = np.array([1, 1650, 3])
price_ne = x_test_ne @ theta_ne
print(f"预测房价（正规方程）：{price_ne:.2f}")
print(f"正规方程 theta：{theta_ne}")
