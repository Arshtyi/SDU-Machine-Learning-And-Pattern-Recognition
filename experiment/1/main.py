import numpy as np
import matplotlib.pyplot as plt

x = np.loadtxt("res/ex1x.dat")
y = np.loadtxt("res/ex1y.dat")

plt.scatter(x, y, marker="o")
plt.xlabel("Age in years")
plt.ylabel("Height in meters")
plt.title("Training data")
plt.show()

m = len(y)
x = np.c_[np.ones(m), x]

theta = np.zeros(x.shape[1])
learning_rate = 0.07
max_iter_times = 1500
loss_set = []

for _ in range(max_iter_times):
    pre_y = np.dot(x, theta)
    loss = pre_y - y
    gradients = np.dot(x.T, loss) / m
    theta -= learning_rate * gradients
    cost = np.sum(loss**2) / (2 * m)
    loss_set.append(cost)

print("Iteration times:", max_iter_times)
print(f"Trained weights: theta_0={theta[0]:.4f}, theta_1={theta[1]:.4f}")

fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121)
ax1.scatter(x[:, 1], y, marker="o", label="Training data")
ax1.plot(x[:, 1], np.dot(x, theta), color="red", label="Linear regression")
ax1.set_xlabel("Age in years")
ax1.set_ylabel("Height in meters")
ax1.set_title("Linear Regression Fit")
ax1.legend()
ax2 = fig.add_subplot(122)
ax2.plot(range(max_iter_times), loss_set)
ax2.set_title("Loss vs. Iterations")
ax2.set_xlabel("Step")
ax2.set_ylabel("Loss")
plt.show()

x_to_pre = np.array([[1, 3.5], [1, 7]])
predictions = np.dot(x_to_pre, theta)
print(f"Prediction for age 3.5: {predictions[0]:.4f} meters")
print(f"Prediction for age 7.0: {predictions[1]:.4f} meters")

theta0_vals = np.linspace(-3, 3, 100)
theta1_vals = np.linspace(-1, 1, 100)
loss_j = np.zeros((len(theta0_vals), len(theta1_vals)))

for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        pred = theta0_vals[i] * x[:, 0] + theta1_vals[j] * x[:, 1]
        loss_j[i, j] = np.sum((pred - y) ** 2) / (2 * m)

loss_j = loss_j.T
T0, T1 = np.meshgrid(theta0_vals, theta1_vals)

fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(121, projection="3d")
ax1.plot_surface(T0, T1, loss_j, cmap="viridis", alpha=0.9)
ax1.set_xlabel(r"$\theta_0$")
ax1.set_ylabel(r"$\theta_1$")
ax1.set_zlabel("Loss")
ax1.set_title("Surface plot")
ax2 = fig.add_subplot(122)
contour = ax2.contour(T0, T1, loss_j, np.logspace(-2, 3, 20))
ax2.plot(theta[0], theta[1], "rx", markersize=10, linewidth=2)
ax2.set_xlabel(r"$\theta_0$")
ax2.set_ylabel(r"$\theta_1$")
ax2.set_title("Contour plot")
plt.show()
