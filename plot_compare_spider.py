import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 8
})

# =========================================================
# 1. 指标名称
# =========================================================
metrics = [
    "AUC",
    "Accuracy",
    "Sensitivity",
    "Specificity",
    "F1"
]

# =========================================================
# 2. BUS 数据集上的 Baseline 结果
#
# 请将下面的示例方法和数值替换为表1中的真实数据。
# 每个方法必须按照以下顺序填写：
# [AUC, Accuracy, Sensitivity, Specificity, F1]
# =========================================================
baseline_data = {
    "FixMatch":      [84.08, 77.43, 70.31, 84.80, 76.02],
    "AdaMatch":      [84.15, 77.17, 70.31, 84.28, 75.81],
    "FlexMatch":     [84.02, 78.16, 74.36, 82.09, 77.60],
    "SoftMatch":     [83.84, 77.73, 72.46, 83.20, 76.80],
    "FreeMatch":     [83.94, 78.24, 74.15, 82.48, 77.61],
    "ReFixMatch":    [83.81, 77.66, 72.00, 83.53, 76.64],
    "SimMatch":      [84.31, 78.35, 74.68, 82.14, 77.82],

    # 下面继续添加表1中的其他方法
    # "SRC-MT": [AUC, Accuracy, Sensitivity, Specificity, F1],
    # "RankMatch": [AUC, Accuracy, Sensitivity, Specificity, F1],
    # "Zeng et al.":         [AUC, Accuracy, Sensitivity, Specificity, F1],
    # "SimMatchV2":         [AUC, Accuracy, Sensitivity, Specificity, F1],
    "InfluMatch":  [84.70, 78.47, 73.40, 83.73, 77.62],
}

# =========================================================
# 3. 数据检查
# =========================================================
for method, values in baseline_data.items():

    if len(values) != len(metrics):
        raise ValueError(
            f"{method} 的数据必须包含 {len(metrics)} 个指标："
            f"{metrics}"
        )

    if np.any(np.asarray(values, dtype=float) <= 0):
        raise ValueError(
            f"{method} 包含非正数，请检查数据是否填写正确。"
        )

# =========================================================
# 4. 雷达图角度
# =========================================================
num_metrics = len(metrics)

base_angles = np.linspace(
    0,
    2 * np.pi,
    num_metrics,
    endpoint=False
)

angles = np.concatenate([
    base_angles,
    [base_angles[0]]
])

# =========================================================
# 5. 创建图形
# =========================================================
fig, ax = plt.subplots(
    figsize=(10.5, 9),
    subplot_kw={"polar": True}
)

# 雷达图方向
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# 指标标签
ax.set_xticks(base_angles)
ax.set_xticklabels(
    metrics,
    fontsize=11,
    fontweight="bold"
)

ax.tick_params(
    axis="x",
    pad=12
)

# =========================================================
# 6. 统一纵坐标范围
#
# 所有方法使用相同范围，保证比较公平。
# 如果表1的指标范围接近 60–90，可以保留此设置。
# =========================================================
ax.set_ylim(60, 90)
ax.set_yticks([60, 70, 80, 90])
ax.set_yticklabels(
    ["60", "70", "80", "90"],
    fontsize=9,
    color="dimgray"
)

# =========================================================
# 7. 自动生成颜色
# =========================================================
num_methods = len(baseline_data)

colors = plt.cm.tab20(
    np.linspace(0, 1, max(num_methods, 2))
)

# =========================================================
# 8. 绘制各个方法
# =========================================================
for color, (method, values) in zip(
    colors,
    baseline_data.items()
):

    values = np.asarray(values, dtype=float)

    values_closed = np.concatenate([
        values,
        [values[0]]
    ])

    ax.plot(
        angles,
        values_closed,
        color=color,
        linewidth=1.5,
        marker="o",
        markersize=3.5,
        alpha=0.90,
        label=method,
        zorder=3
    )

# =========================================================
# 9. 网格和外圈
# =========================================================
ax.grid(
    color="gray",
    linestyle="--",
    linewidth=0.6,
    alpha=0.55
)

ax.spines["polar"].set_color("gray")
ax.spines["polar"].set_linewidth(0.8)

# =========================================================
# 10. 图题
# =========================================================
ax.set_title(
    "BUS dataset: Baseline comparison",
    fontsize=14,
    fontweight="bold",
    pad=28
)

# =========================================================
# 11. 图例
#
# 方法较多时，图例放在图外，避免遮挡雷达图。
# =========================================================
handles, labels = ax.get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="center left",
    bbox_to_anchor=(0.88, 0.50),
    frameon=False,
    fontsize=8.5,
    ncol=1,
    handlelength=2.0,
    labelspacing=0.8
)

# =========================================================
# 12. 布局
# =========================================================
plt.subplots_adjust(
    left=0.06,
    right=0.78,
    top=0.87,
    bottom=0.08
)

# =========================================================
# 13. 保存
# =========================================================
plt.savefig(
    "BUS_baseline_radar_chart.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.15
)

plt.close(fig)