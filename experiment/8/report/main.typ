#import "@preview/codly:1.3.0": *
#import "@preview/cetz:0.4.2"
#import "@preview/numbly:0.1.0": numbly
#import "@preview/zebraw:0.6.1": *

// Define some variables for the report
#let institute = "计算机科学与技术"
#let course = "机器学习"
#let student-id = "202400130242"
#let student-name = "彭靖轩"
#let date = datetime.today().display("[year].[month].[day]")
#let lab-title = "实验8-监督学习"
#let class = "24智能"
#let font = (
    main: "IBM Plex Serif",
    mono: "Fira Code",
    cjk: "Noto Serif CJK SC",
    math: "New Computer Modern Math",
)

// Color palette
#let palette = (
    link: rgb("#1D4F91"),
    ref: rgb("#6A3FB5"),
)

// Set up styles
#set document(title: lab-title, author: student-name)
#set text(
    font: (font.main, font.cjk),
    size: 11pt,
    lang: "en",
    region: "us",
)
#set smartquote(quotes: "\"\"")
#set page(
    paper: "a4",
    margin: (x: 25pt, y: 25pt),
    footer: context {
        set align(center)
        set text(9pt)
        counter(page).display("1 / 1", both: true)
    },
    header: counter(footnote).update(0),
)
#set heading(
    numbering: numbly(
        "{1:1}",
        "{2:1}.",
        "({3:1})",
    ),
)
#show heading.where(level: 1): it => block(above: .6em, it.body)
#show heading: it => if (it.level == 1) {
    show h.where(amount: .3em): none
    text(size: 15pt, it)
} else {
    text(size: 13pt, it)
}
#show heading.where(level: 1): set heading(supplement: [实验])
#set par(justify: true, first-line-indent: (amount: 2em, all: true))
#show raw: set text(font: ((name: font.mono, covers: "latin-in-cjk"), font.cjk))
#show link: it => text(fill: palette.link, style: "italic", underline(evade: false, it))
#show ref: set text(fill: palette.ref)
#let cite = cite.with(style: "ieee")
#set footnote(numbering: "[1]")
#set list(indent: 6pt, marker: sym.bullet.tri)
#set enum(indent: 6pt, numbering: numbly(n => emph(strong(numbering("1.", n)))))

#{
    show heading: it => align(center, text(size: 18pt, tracking: 0.1em, weight: "bold", it))
    heading(
        numbering: none,
        level: 1,
        bookmarked: false,
        outlined: false,
    )[#institute 学院 #underline(offset: 4pt, extent: 6pt, [#course]) 课程实验报告]
    set text(size: 12pt)
    set table.cell(align: left + horizon, inset: 6pt)
    table(
        columns: (1fr,) * 3,
        [学号: #student-id], [姓名: #student-name], [班级: #class],
        [实验题目: #lab-title], [日期: #date], [实验课时: 2],
    )
}

#show: zebraw.with(
    numbering-separator: true,
    radius: 10pt,
    lang: false,
)
= 实验目的
- 理解监督学习的基本流程, 掌握从数据读取, 特征处理, 训练集测试集划分到模型训练与评估的完整步骤.
- 学习使用回归模型完成 APP 评分预测任务, 对比线性回归, KNN 回归和 SVR 的预测效果.
- 学习使用分类模型完成糖尿病预测任务, 掌握逻辑回归分类, 交叉验证, 网格搜索和模型评价方法.
- 熟悉 MSE, MAE, R2, accuracy, precision, recall, F1, ROC-AUC 等评价指标的含义和使用场景.
- 掌握 PCA 降维, 相关性分析, 数据分布可视化等常用数据分析方法.
= 硬件环境
- CPU: 96x
= 软件环境
- Python: ^=3.11
= 实验步骤与内容
== 环境准备
配置相关环境:
```python
from pathlib import Path
from collections import Counter
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 160)
sns.set_theme(style="whitegrid", font="SimHei")
plt.rcParams["axes.unicode_minus"] = False
BASE_DIR = Path.cwd()
if not (BASE_DIR / "res").exists() and (BASE_DIR / "experiment" / "8" / "res").exists():
    BASE_DIR = BASE_DIR / "experiment" / "8"
RES_DIR = BASE_DIR / "res"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42
print("数据目录:", RES_DIR.resolve())
print("输出目录:", OUTPUT_DIR.resolve())
```
== 评分预测
=== 数据准备
读取所需数据:
```python
app_data = pd.read_csv(RES_DIR / "AppDataV2.csv", index_col=0)
app_after_var = pd.read_csv(RES_DIR / "data_after_var", index_col=0)
app_after_pca = pd.read_csv(RES_DIR / "after_pca.csv", index_col=0)
app_after_filter = pd.read_csv(RES_DIR / "df_after_filter.csv", index_col=0)
def split_app_xy(df, target="Rating"):
    X_part = df.drop(columns=[target])
    y_part = df[target]
    return X_part, y_part
X, Y = split_app_xy(app_data)
X_var, Y_var = split_app_xy(app_after_var)
X_pca, Y_pca = split_app_xy(app_after_pca)
X_filter, Y_filter = split_app_xy(app_after_filter)
app_shapes = pd.DataFrame(
    {
        "样本数": [len(app_data), len(app_after_var), len(app_after_pca), len(app_after_filter)],
        "特征数": [X.shape[1], X_var.shape[1], X_pca.shape[1], X_filter.shape[1]],
    },
    index=["原始特征", "方差选择后", "PCA 后", "过滤式选择后"],
)
app_shapes.to_csv(OUTPUT_DIR / "app_dataset_shapes.csv")
display(app_shapes)
display(app_data.head())
app_data.info()
```
#figure(image("../output/1.png"))<F1>
=== 划分训练集与测试集
使用上述数据划分得到训练集和测试集:
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=10
)
print("训练集:", X_train.shape, y_train.shape)
print("测试集:", X_test.shape, y_test.shape)
```
```
训练集: (8192, 40) (8192,)
测试集: (2048, 40) (2048,)
```
=== 线性回归
进行线性回归:
```python
linreg = LinearRegression()
linreg.fit(X_train, y_train)
linreg_pred_train = linreg.predict(X_train)
linreg_pred_test = linreg.predict(X_test)
linreg_mse_train = mean_squared_error(y_train, linreg_pred_train)
linreg_mse_test = mean_squared_error(y_test, linreg_pred_test)
print("训练集 MSE:", linreg_mse_train)
print("测试集 MSE:", linreg_mse_test)
```
```
训练集 MSE: 0.2300317037732973
测试集 MSE: 0.22310522452148027
```
=== KNN回归
作KNN回归:
```python
knn_model = KNeighborsRegressor(n_neighbors=50)
knn_model.fit(X_train, y_train)
knn_pred_train = knn_model.predict(X_train)
knn_pred_test = knn_model.predict(X_test)
knn_mse_train = mean_squared_error(y_train, knn_pred_train)
knn_mse_test = mean_squared_error(y_test, knn_pred_test)
print("训练集 MSE:", knn_mse_train)
print("测试集 MSE:", knn_mse_test)
```
```
训练集 MSE: 0.20456525146484375
测试集 MSE: 0.206983958984375
```
=== SVR回归+对比
作SVR回归并进行效果对比:
```python
svr = SVR(kernel="rbf", C=1.0, gamma="scale")
svr.fit(X_train, y_train)
svr_pred_train = svr.predict(X_train)
svr_pred_test = svr.predict(X_test)
svr_mse_train = mean_squared_error(y_train, svr_pred_train)
svr_mse_test = mean_squared_error(y_test, svr_pred_test)
model_mse = pd.DataFrame(
    data=[
        [linreg_mse_train, knn_mse_train, svr_mse_train],
        [linreg_mse_test, knn_mse_test, svr_mse_test],
    ],
    columns=["LinearRegression", "KNN", "SVR"],
    index=["training set", "test set"],
)
model_metrics = pd.DataFrame(
    {
        "model": ["LinearRegression", "KNN", "SVR"],
        "train_mse": [linreg_mse_train, knn_mse_train, svr_mse_train],
        "test_mse": [linreg_mse_test, knn_mse_test, svr_mse_test],
        "test_mae": [
            mean_absolute_error(y_test, linreg_pred_test),
            mean_absolute_error(y_test, knn_pred_test),
            mean_absolute_error(y_test, svr_pred_test),
        ],
        "test_r2": [
            r2_score(y_test, linreg_pred_test),
            r2_score(y_test, knn_pred_test),
            r2_score(y_test, svr_pred_test),
        ],
    }
).sort_values("test_mse")
model_mse.to_csv(OUTPUT_DIR / "app_regression_mse.csv")
model_metrics.to_csv(OUTPUT_DIR / "app_regression_metrics.csv", index=False)
display(model_mse)
display(model_metrics)
```
#figure(image("../output/2.png"))<F2>
```python
ax = model_mse.plot(kind="bar", figsize=(9, 5), rot=0)
ax.set_title("APP 评分预测模型 MSE 对比")
ax.set_ylabel("MSE")
ax.legend(title="模型")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "app_regression_mse_compare.png", dpi=160)
plt.show()
```
#figure(image("../output/3.png"))<F3>
=== KNN 交叉验证选择 K 值
选择最佳的K值:
```python
k_range = range(15, 100)
k_scores = []
for k in k_range:
    knn = KNeighborsRegressor(n_neighbors=k)
    scores = cross_val_score(
        knn,
        X_train,
        y_train,
        cv=10,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    k_scores.append(-scores.mean())
knn_cv_results = pd.DataFrame({"n_neighbors": list(k_range), "cv_mse": k_scores})
best_k = int(knn_cv_results.loc[knn_cv_results["cv_mse"].idxmin(), "n_neighbors"])
knn_cv_results.to_csv(OUTPUT_DIR / "app_knn_cv_results.csv", index=False)
print("交叉验证 MSE 最小的 K 值:", best_k)
display(knn_cv_results.head())
```
#figure(image("../output/4.png"))<F4>
并绘制K值的影响曲线:
```python
plt.figure(figsize=(10, 5))
plt.plot(knn_cv_results["n_neighbors"], knn_cv_results["cv_mse"], marker="o", markersize=3)
plt.axvline(best_k, color="tomato", linestyle="--", label=f"best k={best_k}")
plt.xlabel("Value of K for KNN")
plt.ylabel("Cross-Validation MSE")
plt.title("不同 K 值对 KNN 回归模型的影响")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "app_knn_cv_mse.png", dpi=160)
plt.show()
```
#figure(image("../output/5.png"))<F5>
=== 决策树分类器网格搜索
进行网格搜索参数:
```python
y_train_int = y_train.round().astype(int)
y_test_int = y_test.round().astype(int)
dtree = DecisionTreeClassifier(random_state=RANDOM_STATE)
params = [
    {
        "criterion": ["gini"],
        "max_depth": [30, 50, 60, 100],
        "min_samples_leaf": [2, 3, 5, 10],
        "min_impurity_decrease": [0.1, 0.2, 0.5],
    },
    {"criterion": ["gini", "entropy"]},
    {"max_depth": [30, 60, 100], "min_impurity_decrease": [0.1, 0.2, 0.5]},
]
best_tree_model = GridSearchCV(
    dtree,
    param_grid=params,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
)
best_tree_model.fit(X_train, y_train_int)
tree_search_results = pd.DataFrame(best_tree_model.cv_results_).sort_values("rank_test_score")
tree_search_results.to_csv(OUTPUT_DIR / "app_decision_tree_grid_search.csv", index=False)
print("最优分类器:", best_tree_model.best_params_)
print("最优分数:", best_tree_model.best_score_)
display(tree_search_results[["rank_test_score", "mean_test_score", "std_test_score", "params"]].head())
```
#figure(image("../output/6.png"))<F6>
=== SVR 随机搜索
进行随机搜索:
```python
params_svr = {
    "kernel": ["rbf"],
    "C": np.logspace(-3, 2, 6),
    "gamma": [0.001, 0.01, 0.1, 1, 2, 4, 6, 8],
}
best_svr_model = RandomizedSearchCV(
    SVR(),
    param_distributions=params_svr,
    n_iter=12,
    cv=3,
    scoring="neg_mean_squared_error",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
best_svr_model.fit(X, Y)
svr_search_results = pd.DataFrame(best_svr_model.cv_results_).sort_values("rank_test_score")
svr_search_results.to_csv(OUTPUT_DIR / "app_svr_random_search.csv", index=False)
print("最优回归器:", best_svr_model.best_params_)
print("最优分数:", best_svr_model.best_score_)
display(svr_search_results[["rank_test_score", "mean_test_score", "std_test_score", "params"]].head())
```
#figure(image("../output/7.png"))<F7>
=== 总结
通过线性回归、KNN 回归和 SVR 的 MSE 对比,可以直接观察不同监督回归算法在 APP 评分预测任务上的拟合效果;随后使用交叉验证、网格搜索和随机搜索完成超参数选择流程.
== 糖尿病预测
=== 数据导入
导入数据:
```python
diabetes_path = RES_DIR / "pima-indians-diabetes.data"
df = pd.read_csv(diabetes_path)
print("数据尺寸:", df.shape)
display(df.head())
display(df.describe().T)
df.info()
```
#figure(image("../output/8.png"))<F8>
进行初步考察:
```python
target_counts = df["Outcome"].value_counts().sort_index()
target_ratio = df["Outcome"].value_counts(normalize=True).sort_index().rename("ratio")
target_summary = pd.concat([target_counts.rename("count"), target_ratio], axis=1)
target_summary.to_csv(OUTPUT_DIR / "diabetes_target_distribution.csv")
display(target_summary)
print("目标变量分布:", Counter(df["Outcome"]))
ax = target_counts.plot(kind="bar", rot=0, figsize=(5, 4), color=["#4C78A8", "#F58518"])
ax.set_title("糖尿病标签分布")
ax.set_xlabel("Outcome")
ax.set_ylabel("样本数")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "diabetes_target_distribution.png", dpi=160)
plt.show()
```
#figure(image("../output/9.png"))<F9>
=== 特征相关性可视化
对特征相关性进行可视化:
```python
corr_matrix = df.corr(method="spearman")
corr_matrix.to_csv(OUTPUT_DIR / "diabetes_spearman_corr.csv")
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.5)
plt.title("糖尿病数据 Spearman 相关系数热力图")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "diabetes_corr_heatmap.png", dpi=160)
plt.show()
```
#figure(image("../output/10.png"))<F10>
=== 特征分布可视化
对特征分布进行可视化:
```python
feature_cols = [col for col in df.columns if col != "Outcome"]
fig, axes = plt.subplots(3, 3, figsize=(14, 11))
axes = axes.ravel()
for ax, col in zip(axes, feature_cols):
    sns.histplot(data=df, x=col, hue="Outcome", kde=True, element="step", stat="density", common_norm=False, ax=ax)
    ax.set_title(col)
for ax in axes[len(feature_cols):]:
    ax.axis("off")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "diabetes_feature_distributions.png", dpi=160)
plt.show()
```
#figure(image("../output/11.png"))<F11>
=== PCA 降维分析
对输入特征进行降维选择PCA:
```python
X_diabetes = df[feature_cols]
y_diabetes = df["Outcome"]
scaler = StandardScaler()
X_std = pd.DataFrame(
    scaler.fit_transform(X_diabetes),
    columns=feature_cols,
    index=df.index,
)
pca = PCA(n_components=0.99, random_state=RANDOM_STATE)
X_pca = pd.DataFrame(
    pca.fit_transform(X_std),
    columns=[f"PC{i + 1}" for i in range(pca.n_components_)],
    index=df.index,
)
pca_summary = pd.DataFrame(
    {
        "component": X_pca.columns,
        "variance_ratio": pca.explained_variance_ratio_,
        "cumulative_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
    }
)
pca_summary.to_csv(OUTPUT_DIR / "diabetes_pca_summary.csv", index=False)
X_pca.join(y_diabetes).to_csv(OUTPUT_DIR / "diabetes_after_pca.csv")
print("PCA 后数据形状:", X_pca.shape)
display(pca_summary)
```
#figure(image("../output/12.png"))<F12>
统计解释质量:
```python
plt.figure(figsize=(7, 4))
plt.plot(pca_summary["component"], pca_summary["cumulative_variance_ratio"], marker="o")
plt.axhline(0.99, color="tomato", linestyle="--", label="99%")
plt.ylim(0, 1.05)
plt.xlabel("主成分")
plt.ylabel("累计解释方差比")
plt.title("PCA 累计解释方差")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "diabetes_pca_cumulative_variance.png", dpi=160)
plt.show()
```
#figure(image("../output/13.png"))<F13>
=== 分层划分训练集与测试集
```python
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
    X_std,
    y_diabetes,
    test_size=0.1,
    random_state=RANDOM_STATE,
    stratify=y_diabetes,
)
print("训练集标签分布:", Counter(y_train_d))
print("测试集标签分布:", Counter(y_test_d))
print("训练集:", X_train_d.shape, "测试集:", X_test_d.shape)
```
```
训练集标签分布: Counter({0: 450, 1: 241})
测试集标签分布: Counter({0: 50, 1: 27})
训练集: (691, 8) 测试集: (77, 8)
```
=== 逻辑回归建模与 5 折交叉验证
```python
logreg = LogisticRegression(solver="liblinear", max_iter=500, random_state=RANDOM_STATE)
cv_f1_scores = cross_val_score(logreg, X_train_d, y_train_d, cv=5, scoring="f1", n_jobs=-1)
logreg.fit(X_train_d, y_train_d)
y_pred_d = logreg.predict(X_test_d)
y_prob_d = logreg.predict_proba(X_test_d)[:, 1]
base_metrics = pd.DataFrame(
    {
        "metric": ["cv_f1_mean", "cv_f1_std", "accuracy", "precision", "recall", "f1", "roc_auc"],
        "value": [
            cv_f1_scores.mean(),
            cv_f1_scores.std(),
            accuracy_score(y_test_d, y_pred_d),
            precision_score(y_test_d, y_pred_d),
            recall_score(y_test_d, y_pred_d),
            f1_score(y_test_d, y_pred_d),
            roc_auc_score(y_test_d, y_prob_d),
        ],
    }
)
base_metrics.to_csv(OUTPUT_DIR / "diabetes_logreg_base_metrics.csv", index=False)
print("5 折交叉验证 F1:", cv_f1_scores)
display(base_metrics)
print(classification_report(y_test_d, y_pred_d, digits=4))
```
#figure(image("../output/14.png"))<F14>
```python
base_cm = confusion_matrix(y_test_d, y_pred_d)
plt.figure(figsize=(5, 4))
sns.heatmap(base_cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("基础逻辑回归混淆矩阵")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "diabetes_logreg_base_confusion_matrix.png", dpi=160)
plt.show()
```
#figure(image("../output/15.png"))<F15>
=== 逻辑回归网格搜索
进行逻辑回归网格搜索:
```python
c_range = [0.001, 0.01, 0.1, 1.0]
solvers = ["liblinear", "lbfgs", "newton-cg", "sag"]
max_iters = [80, 100, 150, 200, 300]
tuned_parameters = dict(solver=solvers, C=c_range, max_iter=max_iters)
logreg_grid = GridSearchCV(
    LogisticRegression(random_state=RANDOM_STATE),
    param_grid=tuned_parameters,
    cv=5,
    scoring="f1",
    n_jobs=-1,
)
logreg_grid.fit(X_train_d, y_train_d)
grid_results = pd.DataFrame(logreg_grid.cv_results_).sort_values("rank_test_score")
grid_results.to_csv(OUTPUT_DIR / "diabetes_logreg_grid_search.csv", index=False)
print("最优参数:", logreg_grid.best_params_)
print("最优 F1:", logreg_grid.best_score_)
display(grid_results[["rank_test_score", "mean_test_score", "std_test_score", "params"]].head())
```
#figure(image("../output/16.png"))<F16>
=== 最终预测
```python
best_logreg = logreg_grid.best_estimator_
y_pred_best = best_logreg.predict(X_test_d)
y_prob_best = best_logreg.predict_proba(X_test_d)[:, 1]
final_metrics = pd.DataFrame(
    {
        "metric": ["accuracy", "precision", "recall", "f1", "roc_auc"],
        "value": [
            accuracy_score(y_test_d, y_pred_best),
            precision_score(y_test_d, y_pred_best),
            recall_score(y_test_d, y_pred_best),
            f1_score(y_test_d, y_pred_best),
            roc_auc_score(y_test_d, y_prob_best),
        ],
    }
)
predictions = pd.DataFrame(
    {
        "y_true": y_test_d,
        "y_pred": y_pred_best,
        "positive_probability": y_prob_best,
    },
    index=y_test_d.index,
).sort_index()
final_metrics.to_csv(OUTPUT_DIR / "diabetes_logreg_final_metrics.csv", index=False)
predictions.to_csv(OUTPUT_DIR / "diabetes_logreg_predictions.csv")
print(classification_report(y_test_d, y_pred_best, digits=4))
display(final_metrics)
display(predictions.head())
```
#figure(image("../output/17.png"))<F17>
```python
final_cm = confusion_matrix(y_test_d, y_pred_best)
plt.figure(figsize=(5, 4))
sns.heatmap(final_cm, annot=True, fmt="d", cmap="Greens", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("网格搜索后逻辑回归混淆矩阵")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "diabetes_logreg_final_confusion_matrix.png", dpi=160)
plt.show()
```
#figure(image("../output/18.png"))<F18>
=== 总结
本部分完成了数据读取、标签均衡性检查、相关性和分布可视化、PCA 降维分析、分层抽样、逻辑回归 5 折交叉验证、网格搜索和最终预测结果保存.
= 实验总结
- 本实验围绕监督学习中的回归任务和分类任务展开. 在 APP 评分预测部分, 通过线性回归, KNN 回归和 SVR 建立评分预测模型, 并使用 MSE, MAE 和 R2 等指标比较模型效果. 实验结果表明, 不同模型对同一数据集的拟合能力存在差异, 通过交叉验证选择 KNN 的 K 值, 以及使用网格搜索和随机搜索优化模型参数, 可以进一步提升模型的稳定性和泛化能力.
- 在糖尿病预测部分, 首先对数据进行了标签分布统计, 特征相关性分析和特征分布可视化, 随后使用标准化和 PCA 分析数据结构. 在建模阶段, 使用逻辑回归完成二分类预测, 并通过 5 折交叉验证和网格搜索选择较优参数. 最终结合混淆矩阵, accuracy, precision, recall, F1 和 ROC-AUC 等指标对模型性能进行了综合评价.
