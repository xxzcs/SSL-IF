import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 10,
    "legend.fontsize": 10
})

# =========================
# 指标名称
# =========================
metrics = [
    "AUC",
    "Acc",
    "Sen",
    "Spe",
    "F1"
]

# =========================
# 实验数据
# =========================
data = {
    "AdaMatch": {
        "baseline": [84.15, 77.17, 70.31, 84.28, 75.81],
        "irc":      [84.70, 78.40, 73.53, 83.45, 77.60]
    },
    "FlexMatch": {
        "baseline": [84.02, 78.16, 74.36, 82.09, 77.60],
        "irc":      [84.65, 78.70, 75.30, 82.23, 78.25]
    },
    "SoftMatch": {
        "baseline": [83.84, 77.73, 72.46, 83.20, 76.80],
        "irc":      [84.55, 78.58, 74.47, 82.84, 77.96]
    },
    "FreeMatch": {
        "baseline": [83.94, 78.24, 74.15, 82.48, 77.61],
        "irc":      [84.66, 78.70, 75.01, 82.53, 78.18]
    },
    "ReFixMatch": {
        "baseline": [83.81, 77.66, 72.00, 83.53, 76.64],
        "irc":      [84.38, 78.28, 73.99, 82.73, 77.61]
    },
    "SimMatch": {
        "baseline": [84.31, 78.35, 74.68, 82.14, 77.82],
        "irc":      [84.63, 79.04, 77.13, 81.03, 78.92]
    }
}

# =========================
# 指标颜色
# =========================
metric_colors = {
    "AUC": "#4C78A8",
    "Acc": "#59A14F",
    "Sen": "#F28E2B",
    "Spe": "#E15759",
    "F1":  "#9467BD"
}

# =========================
# 计算性能增量
# =========================
methods = []
delta_values = []

for method, values in data.items():

    baseline = np.asarray(values["baseline"], dtype=float)
    irc = np.asarray(values["irc"], dtype=float)

    if len(baseline) != len(metrics):
        raise ValueError(
            f"{method} 的 baseline 数据必须包含 {len(metrics)} 个数值"
        )

    if len(irc) != len(metrics):
        raise ValueError(
            f"{method} 的 irc 数据必须包含 {len(metrics)} 个数值"
        )

    if np.all(irc == 0):
        print(f"警告：{method} 的 IRC 数据仍为占位值，已跳过。")
        continue

    methods.append(method)
    delta_values.append(irc - baseline)

delta_values = np.asarray(delta_values, dtype=float)

# =========================
# 创建图形
# =========================
fig, ax = plt.subplots(figsize=(15, 7.5))

# 方法中心位置
x = np.arange(len(methods)) * 0.98

# 增大柱间距离，避免柱子互相覆盖
bar_width = 0.145
offset_step = 0.165
offsets = (np.arange(len(metrics)) - 2) * offset_step

all_bars = []

# =========================
# 绘制分组柱状图
# =========================
for metric_index, metric in enumerate(metrics):

    values = delta_values[:, metric_index]

    bars = ax.bar(
        x + offsets[metric_index],
        values,
        width=bar_width,
        color=metric_colors[metric],
        edgecolor="white",
        linewidth=0.6,
        label=metric,
        zorder=3
    )

    all_bars.append(bars)

# =========================
# 添加增量标签
# =========================
for method_index in range(len(methods)):

    for metric_index in range(len(metrics)):

        value = delta_values[method_index, metric_index]
        bar = all_bars[metric_index][method_index]

        x_position = bar.get_x() + bar.get_width() / 2

        if value >= 0:
            y_position = value + 0.075
            vertical_alignment = "bottom"
        else:
            y_position = value - 0.075
            vertical_alignment = "top"

        ax.text(
            x_position,
            y_position,
            f"{value:.2f}",
            ha="center",
            va=vertical_alignment,
            rotation=0,
            fontsize=8.8,
            color="black",
            zorder=6,
            clip_on=False
        )

# =========================
# 零刻度参考线
# =========================
ax.axhline(
    y=0,
    color="black",
    linewidth=1.0,
    zorder=4
)

# =========================
# 坐标轴设置
# =========================
ax.set_xticks(x)
ax.set_xticklabels(
    methods,
    fontsize=11
)

ax.set_xlabel(
    "SSL frameworks",
    fontsize=12,
    labelpad=8
)

ax.set_ylabel(
    "Performance change (percentage points)",
    fontsize=12,
    labelpad=8
)

# =========================
# 纵坐标范围
# =========================
min_delta = np.min(delta_values)
max_delta = np.max(delta_values)

lower = min(-1.20, min_delta - 0.60)
upper = max(1.35, max_delta + 0.70)

ax.set_ylim(lower, upper)

# =========================
# 网格线
# =========================
ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.6,
    alpha=0.55,
    zorder=0
)

# =========================
# 坐标轴边框
# =========================
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(0.8)
ax.spines["bottom"].set_linewidth(0.8)

# =========================
# 图题与图例
# =========================
fig.suptitle(
    "Performance changes after incorporating IRC",
    fontsize=14,
    fontweight="bold",
    y=0.98
)

handles, labels = ax.get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.925),
    ncol=5,
    frameon=False,
    fontsize=10.5,
    handlelength=1.2,
    columnspacing=1.3,
    handletextpad=0.45
)

# =========================
# 布局
# =========================
plt.subplots_adjust(
    left=0.09,
    right=0.98,
    top=0.82,
    bottom=0.16
)

# =========================
# 保存 PDF
# =========================
plt.savefig(
    "generalization_delta_bar_chart.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.15
)

plt.close(fig)