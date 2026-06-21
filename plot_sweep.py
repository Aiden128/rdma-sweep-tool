#!/usr/bin/env python3
"""Generate static SVG chart from sweep results.  Zero dependencies."""

import json, math, sys
from pathlib import Path


# ── colour palette ──────────────────────────────────────────────────────────

COLORS = [
    "#2563eb", "#dc2626", "#16a34a", "#ca8a04",
    "#9333ea", "#0891b2", "#be123c", "#d1d5db",
]
BG = "#f8fafc"
CARD_BG = "#ffffff"
AXIS = "#94a3b8"
TEXT = "#1e293b"
LABEL = "#64748b"


# ── helpers ─────────────────────────────────────────────────────────────────


def _fmt(v: float, d: int = 0) -> str:
    if v >= 1000:
        return f"{v:.0f}"
    return f"{v:.{d}f}"


def _svg_roundrect(x: float, y: float, w: float, h: float, r: float = 6) -> str:
    return (
        f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='{r}' "
        f"fill='{CARD_BG}' stroke='#e2e8f0' stroke-width='1'/>"
    )


def _svg_text(
    x: float, y: float, text: str, size: int = 13,
    anchor: str = "start", weight: str = "normal", fill: str = TEXT,
) -> str:
    return (
        f"<text x='{x}' y='{y}' font-family='system-ui,sans-serif' "
        f"font-size='{size}' font-weight='{weight}' fill='{fill}' "
        f"text-anchor='{anchor}'>{text}</text>"
    )


# ── chart: line ─────────────────────────────────────────────────────────────


def _line_chart(
    x_vals: list[float],
    y_vals: list[float],
    color: str,
    title: str,
    ylabel: str,
    x: float,
    y: float,
    w: float,
    h: float,
) -> list[str]:
    pad_l, pad_r, pad_b, pad_t = 50, 20, 40, 40
    cx = x + pad_l
    cy = y + pad_t
    cw = w - pad_l - pad_r
    ch = h - pad_t - pad_b

    ymn, ymx = 0, max(y_vals) * 1.1 or 1
    xmn, xmx = min(x_vals), max(x_vals)

    def px(vx: float) -> float:
        if xmx == xmn:
            return cx + cw / 2
        return cx + (math.log2(vx) - math.log2(xmn)) / (math.log2(xmx) - math.log2(xmn)) * cw

    def py(vy: float) -> float:
        return cy + ch - (vy - ymn) / (ymx - ymn) * ch

    lines: list[str] = []
    lines.append(_svg_text(x + w / 2, y + 16, title, 14, "middle", "bold"))
    lines.append(_svg_text(x + pad_l / 2, y + pad_t + ch / 2, ylabel, 11, "middle", "normal", LABEL))

    # grid
    for i in range(5):
        gy = cy + ch * i / 4
        lines.append(
            f"<line x1='{cx}' y1='{gy}' x2='{cx + cw}' y2='{gy}' "
            f"stroke='#e2e8f0' stroke-width='1'/>"
        )
        lines.append(
            _svg_text(cx - 6, gy + 4, _fmt(ymn + (ymx - ymn) * (1 - i / 4), 0), 10, "end", "normal", LABEL)
        )

    # axis line
    lines.append(
        f"<line x1='{cx}' y1='{cy + ch}' x2='{cx + cw}' y2='{cy + ch}' "
        f"stroke='{AXIS}' stroke-width='1'/>"
    )

    # x labels
    for vx in x_vals:
        lx = px(vx)
        lines.append(
            f"<line x1='{lx}' y1='{cy + ch}' x2='{lx}' y2='{cy + ch + 4}' "
            f"stroke='{AXIS}' stroke-width='1'/>"
        )
        lines.append(
            _svg_text(lx, cy + ch + 18, str(int(vx)), 10, "middle", "normal", LABEL)
        )

    # data line
    pts = " ".join(f"{px(vx)},{py(vy)}" for vx, vy in zip(x_vals, y_vals))
    lines.append(
        f"<polyline points='{pts}' fill='none' stroke='{color}' "
        f"stroke-width='2.5' stroke-linejoin='round'/>"
    )

    # dots + labels
    for vx, vy in zip(x_vals, y_vals):
        dx, dy = px(vx), py(vy)
        lines.append(f"<circle cx='{dx}' cy='{dy}' r='4' fill='{color}'/>")
        lines.append(
            _svg_text(dx, dy - 10, _fmt(vy, 0), 10, "middle", "normal", TEXT)
        )

    return lines


# ── chart: stacked bar ──────────────────────────────────────────────────────


def _stacked_bar(
    x_labels: list[str],
    series: list[tuple[str, list[float]]],
    title: str,
    x: float,
    y: float,
    w: float,
    h: float,
) -> list[str]:
    pad_l, pad_r, pad_b, pad_t = 50, 160, 40, 40
    cx = x + pad_l
    cy = y + pad_t
    cw = w - pad_l - pad_r
    ch = h - pad_t - pad_b

    n = len(x_labels)
    bw = min(cw / n * 0.7, 50)
    gap = (cw - bw * n) / (n + 1)

    lines: list[str] = []
    lines.append(_svg_text(x + w / 2, y + 16, title, 14, "middle", "bold"))
    lines.append(_svg_text(x + pad_l / 2, y + pad_t + ch / 2, "Self %", 11, "middle", "normal", LABEL))

    # grid
    for i in range(5):
        gy = cy + ch * i / 4
        lines.append(
            f"<line x1='{cx}' y1='{gy}' x2='{cx + cw}' y2='{gy}' "
            f"stroke='#e2e8f0' stroke-width='1'/>"
        )
        lines.append(
            _svg_text(cx - 6, gy + 4, _fmt(100 - 100 * i / 4, 0), 10, "end", "normal", LABEL)
        )

    # axis
    lines.append(
        f"<line x1='{cx}' y1='{cy + ch}' x2='{cx + cw}' y2='{cy + ch}' "
        f"stroke='{AXIS}' stroke-width='1'/>"
    )

    n_series = len(series)

    for si in range(n):
        bx = cx + gap + si * (bw + gap)
        # x label
        lines.append(
            _svg_text(bx + bw / 2, cy + ch + 18, x_labels[si], 10, "middle", "normal", LABEL)
        )
        bottom = 0.0
        for ci, (_, vals) in enumerate(series):
            v = vals[si]
            if v <= 0:
                continue
            bh = v / 100 * ch
            color = COLORS[ci % len(COLORS)]
            lines.append(
                f"<rect x='{bx}' y='{cy + ch - bottom - bh}' "
                f"width='{bw}' height='{bh}' fill='{color}'/>"
            )
            bottom += v

    # legend
    lx = cx + cw + 12
    ly = cy + 4
    for ci, (name, _) in enumerate(series):
        color = COLORS[ci % len(COLORS)]
        display = name if len(name) < 32 else name[:29] + "..."
        lines.append(
            f"<rect x='{lx}' y='{ly}' width='10' height='10' fill='{color}' rx='2'/>"
        )
        lines.append(_svg_text(lx + 16, ly + 10, display, 10, "start", "normal", TEXT))
        ly += 18

    return lines


# ── table ────────────────────────────────────────────────────────────────────


def _svg_table(
    headers: list[str],
    rows: list[list[str]],
    title: str,
    x: float,
    y: float,
    w: float,
) -> list[str]:
    lines: list[str] = []
    lines.append(_svg_text(x + w / 2, y + 16, title, 14, "middle", "bold"))
    ncols = len(headers)
    col_w = w / ncols
    ty = y + 30
    lw = 0.5

    def _cell(tx: float, ty: float, txt: str, bold: bool = False, bg: str = "") -> str:
        attr = f"fill='{bg}'" if bg else ""
        res = f"<rect x='{tx}' y='{ty}' width='{col_w}' height='24' {attr}/>"
        res += _svg_text(
            tx + col_w / 2, ty + 16, txt, 11, "middle",
            "bold" if bold else "normal", TEXT,
        )
        return res

    # header
    for ci, hdr in enumerate(headers):
        lines.append(_cell(x + ci * col_w, ty, hdr, True, "#f1f5f9"))
        lines.append(
            f"<line x1='{x + ci * col_w}' y1='{ty}' x2='{x + ci * col_w}' "
            f"y2='{ty + 24 * (len(rows) + 1)}' stroke='#e2e8f0' stroke-width='{lw}'/>"
        )
    lines.append(
        f"<line x1='{x + ncols * col_w}' y1='{ty}' x2='{x + ncols * col_w}' "
        f"y2='{ty + 24 * (len(rows) + 1)}' stroke='#e2e8f0' stroke-width='{lw}'/>"
    )

    # rows
    for ri, row in enumerate(rows):
        ry = ty + 24 * (ri + 1)
        bg = "#f8fafc" if ri % 2 == 1 else ""
        for ci, val in enumerate(row):
            lines.append(_cell(x + ci * col_w, ry, val, ci == 0, bg))

    return lines


# ── main ────────────────────────────────────────────────────────────────────


def main(result_dir: str) -> None:
    out = Path(result_dir)
    summary = json.loads((out / "summary.json").read_text())

    qps = [e["params"]["qp"] for e in summary]
    bw = [e["BW_average"] for e in summary]
    rates = [e["MsgRate"] for e in summary]

    # Perf data
    perf_data = []
    for i in range(len(summary)):
        p = out / f"{i+1:04d}" / "result.json"
        if p.exists():
            perf_data.append(json.loads(p.read_text())["_process"]["server_perf"])
        else:
            perf_data.append({})

    all_syms = set()
    for pd in perf_data:
        for s, v in sorted(pd.items(), key=lambda x: -x[1])[:5]:
            if v > 0:
                all_syms.add(s)
    sym_total = {s: sum(pd.get(s, 0) for pd in perf_data) for s in all_syms}
    top_syms = sorted(sym_total, key=lambda s: -sym_total[s])[:7]

    series: list[tuple[str, list[float]]] = []
    for sym in top_syms:
        series.append((sym, [pd.get(sym, 0) for pd in perf_data]))

    # CPU table
    cores = sorted(
        [k for k in summary[0]["cpu_per_core"] if k.startswith("cpu") and k != "cpu"],
        key=lambda c: int(c.replace("cpu", "")),
    )
    headers = ["QP"] + cores
    cpu_rows = [
        [str(q)] + [f'{summary[i]["cpu_per_core"].get(c, 0):.1f}' for c in cores]
        for i, q in enumerate(qps)
    ]

    # SVG dimensions
    W, H = 1100, 900
    M = 16
    half_w = (W - 3 * M) / 2
    card_h1 = 220
    card_h2 = 280

    elements: list[str] = []
    elements.append(
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' "
        f"viewBox='0 0 {W} {H}' style='background:{BG}'>"
    )

    # title
    elements.append(_svg_text(W / 2, 26, "RDMA Write BW Sweep", 18, "middle", "bold"))
    elements.append(
        _svg_text(
            W / 2, 42,
            "SoftRoCE (rxe0) · 64K msg · ib_write_bw · server perf record -g",
            12, "middle", "normal", LABEL,
        )
    )

    y_off = 56

    # Row 1: BW + MsgRate
    for col, (title, ylabel, yvals, color) in enumerate([
        ("Bandwidth (MB/s)", "MB/s", bw, "#2563eb"),
        ("Message Rate (Mmsg/s)", "Mmsg/s", [r * 1000 for r in rates], "#16a34a"),
    ]):
        cx = M + col * (half_w + M)
        elements.append(_svg_roundrect(cx, y_off, half_w, card_h1))
        elements += _line_chart(
            [float(q) for q in qps], yvals, color, title, ylabel,
            cx + 8, y_off + 4, half_w - 16, card_h1 - 8,
        )

    # Row 2: stacked bar
    y_off += card_h1 + M
    elements.append(_svg_roundrect(M, y_off, W - 2 * M, card_h2))
    elements += _stacked_bar(
        [str(q) for q in qps], series, "Top CPU Consumers (self %)",
        M + 8, y_off + 4, W - 2 * M - 16, card_h2 - 8,
    )

    # Row 3: CPU table
    y_off += card_h2 + M
    table_h = 30 + 24 * (len(cpu_rows) + 1)
    elements.append(_svg_roundrect(M, y_off, W - 2 * M, table_h))
    elements += _svg_table(
        headers, cpu_rows, "Per-Core CPU Utilization (%)",
        M + 8, y_off + 4, W - 2 * M - 16,
    )

    elements.append("</svg>")

    svg = "\n".join(elements)
    svg_path = out / "chart.svg"
    svg_path.write_text(svg)
    print(f"chart -> {svg_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
