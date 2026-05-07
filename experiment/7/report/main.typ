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
#let lab-title = "实验7-特征工程"
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
- 熟悉并掌握最常用的特征选择与降维方法
= 硬件环境
- CPU: 96x
= 软件环境
- Python: ^=3.11
= 实验步骤与内容
== 数据准备
读取原始数据及其基本信息:
```python
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    f_classif,
    chi2,
    mutual_info_classif,
    RFE,
    SelectFromModel,
)
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler, MinMaxScaler
warnings.filterwarnings("ignore", category=UserWarning)
pd.set_option("display.max_columns", 120)
DATA_PATH = Path("res/AppDataV2.csv")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
raw_df = pd.read_csv(DATA_PATH, index_col=0)
raw_data = raw_df.drop(columns=["Rating"])
labels = raw_df["Rating"]
print("原始数据形状:", raw_df.shape)
display(raw_df.head())
display(raw_df.dtypes.to_frame("dtype").T)
```
#figure(image("../output/1.png"))<F1>
== 特征选择
=== Filter
==== 无量纲化
进行无量纲化:
```python
scaler = StandardScaler()
data = pd.DataFrame(scaler.fit_transform(raw_data), columns=raw_data.columns, index=raw_data.index)
print("特征矩阵:", raw_data.shape, "标签:", labels.shape)
display(data.head())
```
#figure(image("../output/2.png"))<F2>
=== 方差选择法
接着使用方差选择法:
```python
var_selector = VarianceThreshold(threshold=0.01)
data_after_var_arr = var_selector.fit_transform(raw_data)
var_selected_columns = raw_data.columns[var_selector.get_support()].tolist()
data_after_var = pd.DataFrame(data_after_var_arr, columns=var_selected_columns, index=raw_data.index)
data_after_var_with_label = pd.concat([data_after_var, labels], axis=1)
data_after_var_with_label.to_csv(OUTPUT_DIR / "data_after_var.csv")
print("方差选择后形状:", data_after_var.shape)
print("删除的低方差特征:", raw_data.columns[~var_selector.get_support()].tolist())
display(data_after_var.head())
```
#figure(image("../output/3.png"))<F3>
==== Pearson 相关系数
对于数值数据我们进行 Pearson 相关系数计算,结果还可以进一步取Top k:
```python
numerical_cols = ["Reviews", "Size", "Installs", "Price"]
pearson_rows = []
for col in numerical_cols:
    corr, p_value = pearsonr(raw_data[col], labels)
    pearson_rows.append({"feature": col, "pearson_r": corr, "abs_r": abs(corr), "p_value": p_value})
pearson_summary = pd.DataFrame(pearson_rows).sort_values("abs_r", ascending=False).reset_index(drop=True)
pearson_summary.to_csv(OUTPUT_DIR / "pearson_summary.csv", index=False)
pearson_top_cols = pearson_summary.head(3)["feature"].tolist()
data_numerical = raw_data[pearson_top_cols].copy()
data_numerical.to_csv(OUTPUT_DIR / "filter_pearson_top3.csv")
print("Pearson Top3:", pearson_top_cols)
display(pearson_summary)
display(data_numerical.head())
```
#figure(image("../output/4.png"))<F4>
==== ANOVA
对于分类数据,我们使用 ANOVA 进行选择:
```python
categorical_cols = [
    col for col in raw_data.columns
    if col in ["Type", "Content Rating", "Genres"] or col.startswith("Category_")
]
anova_k = min(30, len(categorical_cols))
anova_selector = SelectKBest(score_func=f_classif, k=anova_k)
data_categorical_arr = anova_selector.fit_transform(raw_data[categorical_cols], labels)
anova_selected_columns = pd.Index(categorical_cols)[anova_selector.get_support()].tolist()
data_categorical = pd.DataFrame(data_categorical_arr, columns=anova_selected_columns, index=raw_data.index)
anova_summary = pd.DataFrame({
    "feature": categorical_cols,
    "f_score": anova_selector.scores_,
    "p_value": anova_selector.pvalues_,
}).sort_values("f_score", ascending=False).reset_index(drop=True)
anova_summary.to_csv(OUTPUT_DIR / "anova_summary.csv", index=False)
data_categorical.to_csv(OUTPUT_DIR / "filter_anova_top30.csv")
print("ANOVA 选择特征数:", len(anova_selected_columns))
display(anova_summary.head(15))
display(data_categorical.head())
```
#figure(image("../output/5.png"))<F5>
==== 合并结果
最后我们将数值和分类特征的选择结果进行合并,得到最终的特征子集:
```python
df_after_filter = pd.concat([data_numerical, data_categorical], axis=1)
df_after_filter_with_label = pd.concat([df_after_filter, labels], axis=1)
df_after_filter_with_label.to_csv(OUTPUT_DIR / "df_after_filter.csv")
print("Filter 后形状:", df_after_filter.shape)
display(df_after_filter.head())
```
#figure(image("../output/6.png"))<F6>
=== Wrapper
使用RFE+Lasso进行特征选择:
```python
wrapper_selection = RFE(
    estimator=Lasso(alpha=0.001, max_iter=50000, random_state=42),
    n_features_to_select=30,
    step=1,
)
wrapper_selection.fit(data, labels)
wrapper_selected_columns = raw_data.columns[wrapper_selection.support_].tolist()
df_after_wrapper = raw_data[wrapper_selected_columns].copy()
df_after_wrapper_with_label = pd.concat([df_after_wrapper, labels], axis=1)
df_after_wrapper_with_label.to_csv(OUTPUT_DIR / "df_after_wrapper.csv")
wrapper_ranking = pd.DataFrame({
    "feature": raw_data.columns,
    "selected": wrapper_selection.support_,
    "ranking": wrapper_selection.ranking_,
}).sort_values(["ranking", "feature"]).reset_index(drop=True)
wrapper_ranking.to_csv(OUTPUT_DIR / "wrapper_ranking.csv", index=False)
print("Wrapper 后形状:", df_after_wrapper.shape)
display(wrapper_ranking.head(40))
display(df_after_wrapper.head())
```
#figure(image("../output/7.png"))<F7>
=== Embedded
下面使用基于L1的特征选择:
```python
lasso = Lasso(alpha=0.001, max_iter=50000, random_state=42)
lasso.fit(data, labels)
embedded_model = SelectFromModel(lasso, prefit=True, threshold="mean")
embedded_support = embedded_model.get_support()
embedded_selected_columns = raw_data.columns[embedded_support].tolist()
df_after_embedded = raw_data[embedded_selected_columns].copy()
df_after_embedded_with_label = pd.concat([df_after_embedded, labels], axis=1)
df_after_embedded_with_label.to_csv(OUTPUT_DIR / "df_after_embedded.csv")
embedded_importance = pd.DataFrame({
    "feature": raw_data.columns,
    "coef": lasso.coef_,
    "abs_coef": np.abs(lasso.coef_),
    "selected": embedded_support,
}).sort_values("abs_coef", ascending=False).reset_index(drop=True)
embedded_importance.to_csv(OUTPUT_DIR / "embedded_lasso_coef.csv", index=False)
print("Embedded 后形状:", df_after_embedded.shape)
display(embedded_importance.head(20))
display(df_after_embedded.head())
```
#figure(image("../output/8.png"))<F8>
== 降维
=== PCA
使用PCA进行降维:
```python
pca = PCA(n_components=0.85, random_state=42)
data_after_pca = pd.DataFrame(
    pca.fit_transform(data),
    columns=[f"PC{i + 1}" for i in range(pca.n_components_)],
    index=raw_data.index,
)
data_after_pca_withlabels = pd.concat([data_after_pca, labels], axis=1)
data_after_pca_withlabels.to_csv(OUTPUT_DIR / "after_pca.csv")
explained_summary = pd.DataFrame({
    "component": data_after_pca.columns,
    "explained_variance_ratio": pca.explained_variance_ratio_,
    "cumulative_ratio": np.cumsum(pca.explained_variance_ratio_),
})
explained_summary.to_csv(OUTPUT_DIR / "pca_explained_variance.csv", index=False)
print(f"PCA 后维度: {raw_data.shape[1]} -> {pca.n_components_}")
print(f"累计解释方差: {pca.explained_variance_ratio_.sum():.4f}")
display(explained_summary.head(15))
display(data_after_pca.head())
```
#figure(image("../output/9.png"))<F9>
=== PCA 可视化
对PCA结果进行可视化:
```python
plot_df = data_after_pca_withlabels.copy()
plot_df["RatingLevel"] = plot_df["Rating"].round().astype(int).clip(1, 5)
plt.figure(figsize=(8, 6))
for rating_level in sorted(plot_df["RatingLevel"].unique()):
    subset = plot_df[plot_df["RatingLevel"] == rating_level]
    plt.scatter(subset["PC1"], subset["PC2"], s=12, alpha=0.55, label=str(rating_level))
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Result by Rounded Rating")
plt.legend(title="Rating")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pca_scatter.png", dpi=160)
plt.show()
```
#figure(image("../output/10.png"))<F10>
== 思考题
=== 数据准备
准备iris数据集:
```python
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="target")
iris_df = pd.concat([X, y], axis=1)
print("Iris 数据形状:", iris_df.shape)
display(iris_df.sample(5, random_state=42))
```
#figure(image("../output/11.png"))<F11>
=== 方差选择法
使用方差选择法:
```python
iris_var_selector = VarianceThreshold(threshold=0.6)
iris_after_var = pd.DataFrame(
    iris_var_selector.fit_transform(X),
    columns=X.columns[iris_var_selector.get_support()],
)
iris_var_summary = pd.DataFrame({
    "feature": X.columns,
    "variance": X.var(ddof=0).values,
    "selected": iris_var_selector.get_support(),
}).sort_values("variance", ascending=False).reset_index(drop=True)
iris_after_var.to_csv(OUTPUT_DIR / "iris_after_variance.csv", index=False)
iris_var_summary.to_csv(OUTPUT_DIR / "iris_variance_summary.csv", index=False)
display(iris_var_summary)
display(iris_after_var.head())
```
#figure(image("../output/12.png"))<F12>
=== 相关系数法
使用 Pearson 相关系数法:
```python
iris_pearson_rows = []
for col in X.columns:
    corr, p_value = pearsonr(X[col], y)
    iris_pearson_rows.append({"feature": col, "pearson_r": corr, "abs_r": abs(corr), "p_value": p_value})
iris_pearson_summary = pd.DataFrame(iris_pearson_rows).sort_values("abs_r", ascending=False).reset_index(drop=True)
iris_pearson_top2 = iris_pearson_summary.head(2)["feature"].tolist()
iris_after_pearson = X[iris_pearson_top2].copy()
iris_pearson_summary.to_csv(OUTPUT_DIR / "iris_pearson_summary.csv", index=False)
iris_after_pearson.to_csv(OUTPUT_DIR / "iris_after_pearson_top2.csv", index=False)
print("Pearson Top2:", iris_pearson_top2)
display(iris_pearson_summary)
display(iris_after_pearson.head())
```
#figure(image("../output/13.png"))<F13>
=== 卡方检验法
使用卡方检验法:
```python
iris_chi2_selector = SelectKBest(score_func=chi2, k=2)
iris_after_chi2 = pd.DataFrame(
    iris_chi2_selector.fit_transform(X, y),
    columns=X.columns[iris_chi2_selector.get_support()],
)
iris_chi2_summary = pd.DataFrame({
    "feature": X.columns,
    "chi2_score": iris_chi2_selector.scores_,
    "p_value": iris_chi2_selector.pvalues_,
    "selected": iris_chi2_selector.get_support(),
}).sort_values("chi2_score", ascending=False).reset_index(drop=True)
iris_after_chi2.to_csv(OUTPUT_DIR / "iris_after_chi2_top2.csv", index=False)
iris_chi2_summary.to_csv(OUTPUT_DIR / "iris_chi2_summary.csv", index=False)
display(iris_chi2_summary)
display(iris_after_chi2.head())
```
#figure(image("../output/14.png"))<F14>
=== 互信息法
使用互信息法:
```python
mi_scores = mutual_info_classif(X, y, random_state=42)
iris_mi_summary = pd.DataFrame({
    "feature": X.columns,
    "mutual_info": mi_scores,
}).sort_values("mutual_info", ascending=False).reset_index(drop=True)
iris_mi_top2 = iris_mi_summary.head(2)["feature"].tolist()
iris_after_mi = X[iris_mi_top2].copy()
iris_mi_summary.to_csv(OUTPUT_DIR / "iris_mutual_info_summary.csv", index=False)
iris_after_mi.to_csv(OUTPUT_DIR / "iris_after_mutual_info_top2.csv", index=False)
print("互信息 Top2:", iris_mi_top2)
display(iris_mi_summary)
display(iris_after_mi.head())
```
#figure(image("../output/15.png"))<F15>
= 总结
本次实验我们对一个应用数据集进行了特征选择和降维的完整流程,包括Filter、Wrapper和Embedded三大类方法的应用,以及PCA降维和可视化。通过对数值和分类特征分别进行选择,我们得到了一个更精简的特征子集,并通过PCA将其降维。

在思考题部分,我们使用了Iris数据集对方差选择法、相关系数法、卡方检验法和互信息法进行了实践,比较了不同方法的选择结果。总体来说,特征选择和降维是机器学习中非常重要的步骤,合理的特征处理可以显著提升模型性能和泛化能力。
