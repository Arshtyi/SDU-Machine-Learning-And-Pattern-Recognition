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
    lab-title: "Experiment4: Logistic Regression and Newton's Method",
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
    - 理解逻辑回归的原理和算法
    - 掌握牛顿法在逻辑回归中的应用
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
    读入并简单处理数据,绘制对应的散点图@F1
    ```python
    data_dir = Path("res")
    x_raw = np.loadtxt(data_dir / "ex4x.dat")
    y = np.loadtxt(data_dir / "ex4y.dat")
    m = x_raw.shape[0]
    x = np.c_[np.ones(m), x_raw]

    pos = y == 1
    neg = y == 0
    plt.figure(figsize=(6, 5))
    plt.scatter(x_raw[pos, 0], x_raw[pos, 1], marker="+", s=80, label="Admitted")
    plt.scatter(x_raw[neg, 0], x_raw[neg, 1], marker="o", s=35, label="Not admitted")
    plt.xlabel("Exam 1")
    plt.ylabel("Exam 2")
    plt.title("Training Data")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show(block=True)
    plt.close()
    ```
    #figure(image("../output/1.png"))<F1>
    == 训练模型
    定义sigmoid函数和代价函数,使用牛顿法迭代更新参数theta,记录每次迭代的代价值,绘制代价函数的收敛曲线@F2
    ```python
    def sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))
    def cost(theta: np.ndarray, x_mat: np.ndarray, y_vec: np.ndarray) -> float:
        h = sigmoid(x_mat @ theta)
        eps = 1e-12
        return float(-np.mean(y_vec * np.log(h + eps) + (1 - y_vec) * np.log(1 - h + eps)))
    theta = np.zeros(x.shape[1], dtype=float)
    max_iter = 30
    tol = 1e-8
    cost_history: list[float] = []

    for _ in range(max_iter):
        h = sigmoid(x @ theta)
        grad = (x.T @ (h - y)) / m
        w = h * (1 - h)
        hessian = (x.T @ (x * w[:, None])) / m
        step = np.linalg.solve(hessian, grad)
        theta = theta - step
        cost_history.append(cost(theta, x, y))
        if np.linalg.norm(step, ord=2) < tol:
            break
    iters_used = len(cost_history)
    ```
    #figure(image("../output/2.png"))<F2>
    == 结果分析
    根据训练得到的参数theta,绘制决策边界@F3
    ```python
    x1_line = np.linspace(x_raw[:, 0].min() - 2, x_raw[:, 0].max() + 2, 200)
    if abs(theta[2]) < 1e-12:
        x2_line = np.full_like(x1_line, np.nan)
    else:
        x2_line = -(theta[0] + theta[1] * x1_line) / theta[2]
    plt.figure(figsize=(6, 5))
    plt.scatter(x_raw[pos, 0], x_raw[pos, 1], marker="+", s=80, label="Admitted")
    plt.scatter(x_raw[neg, 0], x_raw[neg, 1], marker="o", s=35, label="Not admitted")
    plt.plot(x1_line, x2_line, "r-", linewidth=2, label="Decision boundary")
    plt.xlabel("Exam 1")
    plt.ylabel("Exam 2")
    plt.title("Logistic Regression with Newton Method")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show(block=True)
    plt.close()
    ```
    #figure(image("../output/3.png"))<F3>
    == 预测分析
    计算当Exam1=20, Exam2=80时被录取和不被录取的概率
    ```python
    exam = np.array([1.0, 20.0, 80.0])
    p_admit = float(sigmoid(exam @ theta))
    p_not_admit = 1.0 - p_admit
    print("theta =", np.array2string(theta, precision=6, suppress_small=True))
    print("iterations =", iters_used)
    print("P(not admitted | Exam1=20, Exam2=80) =", f"{p_not_admit:.6f}")
    ```
    ```txt
    theta = [-16.378743   0.148341   0.158908]
    iterations = 7
    P(not admitted | Exam1=20, Exam2=80) = 0.668022
    ```
]#exp-block()[
    = 结论分析与体会
    - 通过本次实验,我深入理解了逻辑回归的原理和算法,以及牛顿法在优化中的应用。
    - 实验中,我成功实现了逻辑回归模型的训练过程,并通过绘制代价函数的收敛曲线和决策边界,直观地展示了模型的训练效果。
    - 通过预测分析,我计算了特定输入条件下被录取和不被录取的概率,进一步验证了模型的实用性和准确性。
]
