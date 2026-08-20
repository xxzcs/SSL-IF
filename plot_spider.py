import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 10,
    "legend.fontsize": 10
})

# 图中使用缩写，避免标签压线
metrics = [
    "AUC",
    "Acc",
    "Sen",
    "Spe",
    "F1"
]

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

        # 请替换为 SimMatch + IRC 的真实均值
        "irc":      [0, 0, 0, 0, 0]
    }
}

# =========================
# 雷达图角度
# =========================
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

# =========================
# 创建 2 行 3 列图
# =========================
fig, axes = plt.subplots(
    2,
    3,
    figsize=(13, 9),
    subplot_kw={"polar": True}
)

axes = axes.flatten()

for ax, (method, values) in zip(axes, data.items()):

    baseline = np.asarray(
        values["baseline"],
        dtype=float
    )

    irc = np.asarray(
        values["irc"],
        dtype=float
    )

    # 检查数据长度
    if len(baseline) != num_metrics:
        raise ValueError(
            f"{method} 的 baseline 数据必须包含 {num_metrics} 个数值"
        )

    if len(irc) != num_metrics:
        raise ValueError(
            f"{method} 的 irc 数据必须包含 {num_metrics} 个数值"
        )

    baseline_plot = np.concatenate([
        baseline,
        [baseline[0]]
    ])

    irc_plot = np.concatenate([
        irc,
        [irc[0]]
    ])

    # 判断 IRC 是否已经替换为真实数据
    irc_available = not np.all(irc == 0)

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

    # 增大指标标签与图形之间的距离
    ax.tick_params(
        axis="x",
        pad=12
    )

    # 所有子图使用相同坐标范围
    ax.set_ylim(60, 90)
    ax.set_yticks([60, 70, 80, 90])

    ax.set_yticklabels(
        ["60", "70", "80", "90"],
        fontsize=9,
        color="dimgray"
    )

    # =========================
    # Baseline
    # =========================
    ax.plot(
        angles,
        baseline_plot,
        color="#1f77b4",
        linewidth=2.0,
        linestyle="--",
        marker="o",
        markersize=3.5,
        label="Baseline",
        zorder=3
    )

    ax.fill(
        angles,
        baseline_plot,
        color="#1f77b4",
        alpha=0.10,
        zorder=1
    )

    # =========================
    # + IRC
    # =========================
    if irc_available:

        ax.plot(
            angles,
            irc_plot,
            color="#d62728",
            linewidth=2.0,
            linestyle="-",
            marker="o",
            markersize=3.5,
            label="+ IRC",
            zorder=3
        )

        ax.fill(
            angles,
            irc_plot,
            color="#d62728",
            alpha=0.12,
            zorder=1
        )

    # 网格线
    ax.grid(
        color="gray",
        linestyle="--",
        linewidth=0.6,
        alpha=0.55,
        zorder=0
    )

    # 外圈
    ax.spines["polar"].set_color("gray")
    ax.spines["polar"].set_linewidth(0.8)

    # 方法名称放在子图下方
    ax.text(
        0.5,
        -0.22,
        method,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold",
        clip_on=False
    )

# =========================
# 统一图例
# =========================
handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.015),
    ncol=2,
    frameon=False,
    fontsize=12
)

# =========================
# 调整布局
# =========================
plt.subplots_adjust(
    left=0.06,
    right=0.94,
    top=0.91,
    bottom=0.20,
    wspace=0.42,
    hspace=0.72
)

# =========================
# 保存为 PDF
# =========================
plt.savefig(
    "generalization_radar_charts.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.15
)

plt.close(fig)