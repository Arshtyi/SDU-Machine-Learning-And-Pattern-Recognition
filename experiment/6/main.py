from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


DATA_DIR = Path(__file__).resolve().parent / "res"


def load_linear_data() -> tuple[np.ndarray, np.ndarray]:
    x = np.loadtxt(DATA_DIR / "ex5Linx.dat")
    y = np.loadtxt(DATA_DIR / "ex5Liny.dat")
    return x.reshape(-1), y.reshape(-1)


def load_logistic_data() -> tuple[np.ndarray, np.ndarray]:
    x = np.loadtxt(DATA_DIR / "ex5Logx.dat", delimiter=",")
    y = np.loadtxt(DATA_DIR / "ex5Logy.dat")
    return x, y.reshape(-1)


def poly_features_1d(x: np.ndarray, degree: int = 5) -> np.ndarray:
    return np.column_stack([x**p for p in range(degree + 1)])


def regularized_linear_normal_eq(
    x_design: np.ndarray, y: np.ndarray, lambda_: float
) -> np.ndarray:
    n_features = x_design.shape[1]
    reg = np.eye(n_features)
    reg[0, 0] = 0.0
    a = x_design.T @ x_design + lambda_ * reg
    b = x_design.T @ y
    return np.linalg.solve(a, b)


def map_feature(
    u: np.ndarray | float, v: np.ndarray | float, degree: int = 6
) -> np.ndarray:
    u_arr = np.asarray(u)
    v_arr = np.asarray(v)
    out = [np.ones_like(u_arr, dtype=float)]
    for i in range(1, degree + 1):
        for j in range(i + 1):
            out.append((u_arr ** (i - j)) * (v_arr**j))
    return np.column_stack(out)


def sigmoid(z: np.ndarray) -> np.ndarray:
    z_clip = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z_clip))


def logistic_cost_grad_hess(
    theta: np.ndarray, x_design: np.ndarray, y: np.ndarray, lambda_: float
) -> tuple[float, np.ndarray, np.ndarray]:
    m = x_design.shape[0]
    h = sigmoid(x_design @ theta)
    h_safe = np.clip(h, 1e-12, 1.0 - 1e-12)

    cost = -np.sum(y * np.log(h_safe) + (1 - y) * np.log(1 - h_safe)) / m + (
        lambda_ / (2 * m)
    ) * np.sum(theta[1:] ** 2)

    grad = (x_design.T @ (h - y)) / m
    grad[1:] += (lambda_ / m) * theta[1:]

    w = h * (1.0 - h)
    weighted_x = x_design * w[:, None]
    hess = (x_design.T @ weighted_x) / m
    reg = np.eye(x_design.shape[1])
    reg[0, 0] = 0.0
    hess += (lambda_ / m) * reg

    return cost, grad, hess


def newton_method_regularized_logistic(
    x_design: np.ndarray,
    y: np.ndarray,
    lambda_: float,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> tuple[np.ndarray, list[float]]:
    theta = np.zeros(x_design.shape[1], dtype=float)
    costs: list[float] = []

    for _ in range(max_iter):
        cost, grad, hess = logistic_cost_grad_hess(theta, x_design, y, lambda_)
        costs.append(cost)
        step = np.linalg.solve(hess, grad)
        theta_next = theta - step

        if np.linalg.norm(theta_next - theta, ord=2) < tol:
            theta = theta_next
            final_cost, _, _ = logistic_cost_grad_hess(theta, x_design, y, lambda_)
            costs.append(final_cost)
            break

        theta = theta_next

    return theta, costs


def plot_linear_data(x: np.ndarray, y: np.ndarray) -> None:
    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, color="tab:blue", edgecolor="black")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Linear Regression Data")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()


def plot_linear_fit(
    x: np.ndarray, y: np.ndarray, theta: np.ndarray, lambda_: float, degree: int = 5
) -> None:
    x_plot = np.linspace(x.min() - 0.1, x.max() + 0.1, 400)
    y_plot = poly_features_1d(x_plot, degree=degree) @ theta

    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, color="tab:blue", edgecolor="black", label="Data")
    plt.plot(x_plot, y_plot, color="tab:red", linewidth=2, label="Polynomial fit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Regularized Linear Regression (lambda={lambda_})")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()


def plot_logistic_data(x: np.ndarray, y: np.ndarray) -> None:
    pos = y == 1
    neg = y == 0

    plt.figure(figsize=(6, 5))
    plt.scatter(x[pos, 0], x[pos, 1], marker="+", s=80, label="y=1")
    plt.scatter(x[neg, 0], x[neg, 1], marker="o", edgecolor="black", label="y=0")
    plt.xlabel("u")
    plt.ylabel("v")
    plt.title("Logistic Regression Data")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()


def plot_decision_boundary(
    x: np.ndarray, y: np.ndarray, theta: np.ndarray, lambda_: float
) -> None:
    pos = y == 1
    neg = y == 0

    u = np.linspace(-1.0, 1.5, 200)
    v = np.linspace(-1.0, 1.5, 200)
    uu, vv = np.meshgrid(u, v)
    z = map_feature(uu.ravel(), vv.ravel()) @ theta
    z = z.reshape(uu.shape)

    plt.figure(figsize=(6, 5))
    plt.scatter(x[pos, 0], x[pos, 1], marker="+", s=80, label="y=1")
    plt.scatter(x[neg, 0], x[neg, 1], marker="o", edgecolor="black", label="y=0")
    plt.contour(u, v, z, levels=[0.0], linewidths=2, colors="tab:red")
    plt.xlabel("u")
    plt.ylabel("v")
    plt.title(f"Regularized Logistic Regression (lambda={lambda_})")
    plt.xlim(-1.0, 1.5)
    plt.ylim(-1.0, 1.5)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()


def run_regularized_linear_regression() -> None:
    x, y = load_linear_data()
    x_design = poly_features_1d(x, degree=5)
    lambdas = [0.0, 1.0, 10.0]

    plot_linear_data(x, y)

    print("=== Regularized Linear Regression ===")
    for lambda_ in lambdas:
        theta = regularized_linear_normal_eq(x_design, y, lambda_)
        print(f"lambda={lambda_}")
        print("theta=", np.array2string(theta, precision=8, suppress_small=False))
        print(f"||theta||_2={np.linalg.norm(theta):.8f}")
        print()

        plot_linear_fit(x, y, theta, lambda_, degree=5)


def run_regularized_logistic_regression() -> None:
    x, y = load_logistic_data()
    x_design = map_feature(x[:, 0], x[:, 1], degree=6)
    lambdas = [0.0, 1.0, 10.0]

    plot_logistic_data(x, y)

    print("=== Regularized Logistic Regression (Newton Method) ===")
    for lambda_ in lambdas:
        theta, costs = newton_method_regularized_logistic(x_design, y, lambda_)
        print(f"lambda={lambda_}")
        print("theta=", np.array2string(theta, precision=8, suppress_small=False))
        print(f"||theta||_2={np.linalg.norm(theta):.8f}")
        if costs:
            print(
                f"J(theta)_start={costs[0]:.10f}, J(theta)_end={costs[-1]:.10f}, iterations={len(costs) - 1}"
            )
        print()

        plot_decision_boundary(x, y, theta, lambda_)


def main() -> None:
    run_regularized_linear_regression()
    run_regularized_logistic_regression()


if __name__ == "__main__":
    main()
