#import "dependency.typ": *
#import "template.typ": *

#set page(height: auto)
#set par(justify: true)
#show: report.with(
    institute: "计算机科学与技术",
    course: "机器学习",
    student-id: "202400130242",
    student-name: "彭靖轩",
    class: "24智能",
    date: datetime.today(),
    lab-title: "Experiment6: Regularization",
    exp-time: "2",
)
#show figure.where(kind: "image"): it => {
    set image(width: 67%)
    it
}
#show: zebraw.with(
    lang: false,
)

#exp-block([
    = 实验目的
    - 理解正则化的概念和作用
    - 掌握正则化线性回归和正则化逻辑回归的实现方法
])
#exp-block([
    = 硬件环境
    - CPU: 9600x
])
#exp-block([
    = 软件环境
    - Python 3.10
])
#exp-block()[
    = 实验步骤与内容
    == 准备数据
    分别读入提前准备好的数据
    ```python
    DATA_DIR = Path(__file__).resolve().parent / "res"
    def load_linear_data() -> tuple[np.ndarray, np.ndarray]:
        x = np.loadtxt(DATA_DIR / "ex5Linx.dat")
        y = np.loadtxt(DATA_DIR / "ex5Liny.dat")
        return x.reshape(-1), y.reshape(-1)
    def load_logistic_data() -> tuple[np.ndarray, np.ndarray]:
        x = np.loadtxt(DATA_DIR / "ex5Logx.dat", delimiter=",")
        y = np.loadtxt(DATA_DIR / "ex5Logy.dat")
        return x, y.reshape(-1)
    ```
    == 实现正则化线性回归
    首先绘制数据点的散点图@F1 ,然后对于不同的正则化参数lambda,使用正规方程求解线性回归的参数theta,并输出theta的值和范数,最后绘制拟合曲线.@F2 , @F3 , @F4
    ```python
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
    ```
    #figure(image("../output/1.png"))<F1>
    #figure(image("../output/2.png"))<F2>
    #figure(image("../output/3.png"))<F3>
    #figure(image("../output/4.png"))<F4>
    == 实现正则化逻辑回归
    首先绘制数据点的散点图@F5 ,然后对于不同的正则化参数lambda,使用牛顿法求解逻辑回归的参数theta,并输出theta的值和范数,以及迭代过程中损失函数的变化情况,最后绘制决策边界.@F6 , @F7 , @F8
    ```python
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
    ```
    #figure(image("../output/5.png"))<F5>
    #figure(image("../output/6.png"))<F6>
    #figure(image("../output/7.png"))<F7>
    #figure(image("../output/8.png"))<F8>
    == 结果
    上述方案的结果为
    ```txt
    === Regularized Linear Regression ===
    lambda=0.0
    theta= [ 0.47252877  0.68135289 -1.38012842 -5.97768747  2.44173268  4.73711433]
    ||theta||_2=8.16868030

    lambda=1.0
    theta= [ 0.3975953  -0.42066637  0.12959211 -0.3974739   0.17525553 -0.33938772]
    ||theta||_2=0.80976562

    lambda=10.0
    theta= [ 0.52047074 -0.18250706  0.06064258 -0.14817721  0.07433006 -0.12795737]
    ||theta||_2=0.59306886

    === Regularized Logistic Regression (Newton Method) ===
    lambda=0.0
    theta= [   26.78199709    20.08445532    68.34254602  -319.82003964
      -151.49181011  -107.10926645  -133.2614877   -479.48631125
      -399.36464784  -365.86768314  1312.02163901  1350.45252692
      1351.54298638   557.87586485   272.73460664   353.32601844
      1084.05165274  1373.39644719  1661.46150131  1136.52583771
       619.76919023 -1505.00771121 -2291.86922051 -3343.7935502
     -3116.8826241  -2646.25078599 -1299.73858143  -527.78656277]
    ||theta||_2=7172.69462044
    J(theta)_start=0.6931471806, J(theta)_end=0.1998374976, iterations=16

    lambda=1.0
    theta= [ 1.31927447  0.71087555  1.18826648 -1.98580543 -0.92898104 -1.51897319
      0.12441196 -0.3673571  -0.4123502  -0.16569679 -1.47520467 -0.05464385
     -0.6437332  -0.28088938 -1.20629848 -0.27455762 -0.20613518 -0.0663025
     -0.27695517 -0.3139599  -0.4448237  -1.07849593  0.02702622 -0.30566142
      0.01408538 -0.33417087 -0.14871289 -0.91862637]
    ||theta||_2=4.24000928
    J(theta)_start=0.6931471806, J(theta)_end=0.5246326405, iterations=6

    lambda=10.0
    theta= [ 0.34739681  0.00932067  0.16347324 -0.4422708  -0.11256354 -0.28970135
     -0.06766223 -0.05880069 -0.06835876 -0.107587   -0.33766442 -0.01341784
     -0.11923118 -0.02831858 -0.28998164 -0.117519   -0.0374963  -0.02380563
     -0.04911808 -0.04294791 -0.18744184 -0.25646482 -0.00317254 -0.05910708
     -0.00062841 -0.06437231 -0.01232483 -0.27312304]
    ||theta||_2=0.93841846
    J(theta)_start=0.6931471806, J(theta)_end=0.6475839593, iterations=4
    ```
]
#exp-block()[
    = 结论分析与体会
    - 正则化可以有效地防止模型过拟合，提高模型的泛化能力。
    - 通过调整正则化参数lambda，可以控制模型的复杂度，从而找到最佳的拟合效果。
    - 在本实验中，随着lambda的增加，线性回归模型的参数theta的范数逐渐减小，拟合曲线变得更加平滑；而逻辑回归模型的参数theta的范数也逐渐减小，决策边界变得更加简单。
]

#exp-block()[
    = 源代码
    ```python
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
    ```
]
