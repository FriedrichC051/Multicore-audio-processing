#!/usr/bin/env python3
"""Read the three result tables and write the presentation figures. No new timings."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = Path(__file__).resolve().parent / "figures"
DEADLINE_US = 21333.33


def table_rows(text, header_prefix):
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(header_prefix))
    rows = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip().strip("|").split("|")])
    return rows


def phase2_by_axis(text, axis):
    rows = []
    for cells in table_rows(text, "| 轴 |"):
        if cells[0] == axis:
            rows.append(cells)
    return rows


def main():
    p1 = (RESULTS / "phase1_results.md").read_text(encoding="utf-8")
    p2 = (RESULTS / "phase2_results.md").read_text(encoding="utf-8")
    p3 = (RESULTS / "phase3_results.md").read_text(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)

    cuda_rows = table_rows(p1, "| kernel |")
    fp32 = [r for r in cuda_rows if r[0].startswith("f32_")]
    fixed = [r for r in cuda_rows if r[0] in ("q15_i64", "i8_dp4a")]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    names = [r[0].replace("f32_", "") for r in fp32]
    p50 = [float(r[2]) for r in fp32]
    snr = [r[5] for r in fp32]
    bars = axes[0].bar(range(len(names)), p50, color="#3d6b99")
    axes[0].set_xticks(range(len(names)), names, rotation=30, ha="right")
    axes[0].set_ylabel("device p50 (us)")
    axes[0].set_title("FP32 kernels")
    for bar, value, snr_db in zip(bars, p50, snr):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 0.6, f"{value:.1f}\n{snr_db} dB",
                     ha="center", va="bottom", fontsize=7)
    axes[0].set_ylim(0, max(p50) * 1.35)

    names2 = [r[0] for r in fixed]
    p50_2 = [float(r[2]) for r in fixed]
    snr2 = [r[5] for r in fixed]
    bars2 = axes[1].bar(range(len(names2)), p50_2, color="#b36b2c")
    axes[1].set_xticks(range(len(names2)), names2)
    axes[1].set_ylabel("device p50 (us)")
    axes[1].set_title("Q15 and int8 DP4A")
    for bar, value, snr_db in zip(bars2, p50_2, snr2):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 0.8, f"{value:.1f} us\nSNR {snr_db} dB",
                     ha="center", va="bottom", fontsize=8)
    axes[1].set_ylim(0, max(p50_2) * 1.45)
    fig.tight_layout()
    fig.savefig(OUT / "ceiling.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.0))
    series = [
        ("sources", "sources", 4, "source count"),
        ("buf", "block length", 2, "BUF"),
        ("fir-time", "FIR order, time domain", 3, "FIR"),
    ]
    for ax, (axis, title, x_col, xlabel) in zip(axes, series):
        rows = phase2_by_axis(p2, axis)
        if axis == "buf":
            rows = [r for r in rows if r[1] == "time"]
        xs = [int(r[x_col]) for r in rows]
        cuda = [float(r[8]) for r in rows]
        avx = [float(r[12]) for r in rows]
        xpos = range(len(xs))
        ax.plot(xpos, cuda, "o-", color="#3d6b99", label="CUDA e2e p50")
        ax.plot(xpos, avx, "s--", color="#b36b2c", label="AVX2 p50")
        ax.axhline(DEADLINE_US, color="#666666", lw=0.8, label="21.33 ms deadline")
        ax.set_yscale("log")
        ax.set_xticks(list(xpos), [str(x) for x in xs])
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.set_ylabel("p50 (us)")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "range.png", dpi=150)
    plt.close(fig)

    default_section, eight_section = p3.split("## 8 声源补点", 1)
    path_rows = table_rows(default_section, "| 通路 |")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    labels = ["pageable", "async reg.", "mapped", "resident"]
    kernel = [float(r[5]) for r in path_rows]
    copy = [float(r[6]) for r in path_rows]
    e2e = [float(r[2]) for r in path_rows]
    xpos = range(len(labels))
    axes[0].bar(xpos, kernel, color="#3d6b99", label="kernel p50")
    axes[0].bar(xpos, copy, bottom=kernel, color="#d0d7e2", label="copy p50")
    axes[0].scatter(xpos, e2e, color="#1a1a1a", zorder=3, label="e2e p50")
    axes[0].set_xticks(list(xpos), labels)
    axes[0].set_ylabel("us")
    axes[0].set_title("Default point, 1 source")
    axes[0].legend(fontsize=7)

    buf_default = [r for r in phase2_by_axis(p2, "buf") if r[2] == "1024" and r[1] == "time"][0]
    azimuth = phase2_by_axis(p2, "azimuth")[0]
    az_labels = ["coeffs fixed", "new HRIR / block"]
    az_vals = [float(buf_default[8]), float(azimuth[8])]
    axes[1].bar(az_labels, az_vals, color=["#3d6b99", "#b36b2c"])
    for i, value in enumerate(az_vals):
        axes[1].text(i, value + 3, f"{value:.2f}", ha="center", fontsize=8)
    axes[1].set_ylabel("e2e p50 (us)")
    axes[1].set_title("Coefficient update")
    axes[1].set_ylim(0, max(az_vals) * 1.25)
    fig.tight_layout()
    fig.savefig(OUT / "paths.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
