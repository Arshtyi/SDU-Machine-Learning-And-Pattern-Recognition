#import "dependency.typ": *
#import "template.typ": *

#show: zebraw

#set page(height: auto)
#set par(justify: true)

#show: report.with(
    institute: "计算机科学与技术",
    course: "机器学习",
    student-id: "202400130242",
    student-name: "彭靖轩",
    class: "24智能",
    date: datetime.today(),
    lab-title: "Experiment 2: Multivariate Linear Regression",
    exp-time: "2",
)

#show figure.where(kind: "image"): it => {
    set image(width: 67%)
    it
}


#exp-block([
    = 实验目的
    - 理解多变量线性回归的原理和实现方法
    - 掌握梯度下降算法在多变量线性回归中的应用
    - 学习正规方程求解多变量线性回归参数的方法
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
    == 数据预处理
    首先加载并处理数据集
    ```python
    X = np.loadtxt("res/ex2x.dat")
    y = np.loadtxt("res/ex2y.dat")
    m = y.size
    X = np.concatenate([np.ones((m, 1)), X], axis=1)
    X_raw = X.copy()
    mu = np.mean(X[:, 1:], axis=0)
    sigma = np.std(X[:, 1:], axis=0)
    X[:, 1:] = (X[:, 1:] - mu) / sigma
    ```
    == 梯度下降算法
    定义代价函数和梯度下降函数，并测试不同学习率的收敛情况
    ```python
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
    ```
    == 训练
    对不同的学习率进行训练并绘制代价函数曲线,结果如@F1
    ```python
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
    ```
    #figure(image("../output/1.png"))<F1>
    == 利用梯度下降算法进行预测
    选择合适的学习率（如 0.1），训练到收敛，并预测房价
    ```python
    alpha = 0.1
    num_iters = 400
    theta = np.zeros(X.shape[1])
    theta, J_history = gradient_descent(X, y, theta, alpha, num_iters)
    x_test = np.array([1, 1650, 3])
    x_test[1:] = (x_test[1:] - mu) / sigma
    price = x_test @ theta
    print(f"预测房价（梯度下降）：{price:.2f}")
    print(f"最终 theta：{theta}")
    ```
    == 正规方程求解
    不需要特征缩放，但要加截距项，直接求解参数并预测房价
    ```python
    X_ne = X_raw
    theta_ne = np.linalg.pinv(X_ne.T @ X_ne) @ X_ne.T @ y
    x_test_ne = np.array([1, 1650, 3])
    price_ne = x_test_ne @ theta_ne
    print(f"预测房价（正规方程）：{price_ne:.2f}")
    print(f"正规方程 theta：{theta_ne}")
    ```
    == 结果比较
    两种方法得到的预测结果为
    ```txt
    预测房价（梯度下降）：340412.66
    最终 theta：[340412.65957447 109447.79558639  -6578.3539709 ]
    预测房价（正规方程）：293081.46
    正规方程 theta：[89597.90954361   139.21067402 -8738.01911255]
    ```
]
#exp-block()[
    = 结论分析与体会
    - 梯度下降算法在多变量线性回归中能够有效地找到参数，但需要选择合适的学习率和迭代次数
    - 正规方程求解参数不需要迭代，但在特征数量较多时计算成本较高
]
