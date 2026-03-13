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
    lab-title: "Experiment 1: Linear Regression",
    exp-time: "2",
)

#show figure.where(kind: "image"): it => {
    set image(width: 67%)
    it
}


#exp-block([
    = 实验目的
    - 掌握单变量线性回归的基本原理
    - 熟悉梯度下降算法
    - 理解并可视化代价函数
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
    首先从准备好的路径加载数据集,并将其分为输入特征和目标变量
    ```python
    x = np.loadtxt("res/ex1x.dat")
    y = np.loadtxt("res/ex1y.dat")
    ```
    绘制原数据的散点图如@F1
    ```python
    plt.scatter(x, y, marker="o")
    plt.xlabel("Age in years")
    plt.ylabel("Height in meters")
    plt.title("Training data")
    plt.show()
    ```
    #figure(image("../output/1.png"))<F1>
    == 梯度下降算法
    接下来实现梯度下降算法来拟合线性回归模型
    ```python
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
    ```
    得到的拟合结果如@F2
    // #subpar.grid(
    //     columns: (1fr, 1fr),
    //     figure(image("../output/2.png")), <F2-S1>,
    //     figure(image("../output/3.png")), <F2-S2>,
    //     label: <F2>,
    //     // caption: [@F2-S1: 线性回归拟合结果 @F2-S2: 代价函数曲线],
    // )
    #figure(image("../output/2.png"))<F2>
    == 预测未知数据
    使用训练好的模型对未知数据进行预测
    ```python
    x_to_pre = np.array([[1, 3.5], [1, 7]])
    predictions = np.dot(x_to_pre, theta)
    ```
    == 代价函数的可视化
    最后可视化代价函数的曲面图和等高线图来观察优化过程(@F3)
    ```python
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
    ```
    #figure(image("../output/3.png"))<F3>
]
#exp-block()[
    = 结论分析与体会
    - 单变量线性回归模型能够较好地拟合一些简单的数据集,但对于复杂的数据可能表现不佳
    - 梯度下降算法是优化线性回归模型的有效方法,但需要选择合适的学习率和迭代次数
    - 代价函数的可视化有助于理解优化过程和模型的收敛情况
]
