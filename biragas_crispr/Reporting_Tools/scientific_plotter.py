"""
ScientificPlotter — Publication-Quality Plots for BiRAGAS CRISPR Complete
===========================================================================
Generates all scientific figures using PyMuPDF (no matplotlib dependency).
Supports: bar charts, heatmaps, waterfall plots, volcano plots, network
diagrams, radar charts, stacked bars, confidence interval plots.

All 150+ metrics across all engines can be visualized.
"""

import fitz
import math
import os
from typing import Any, Dict, List, Optional, Tuple

# Colors
NAVY = (0.10, 0.14, 0.49)
BLUE = (0.08, 0.40, 0.75)
LIGHT_BLUE = (0.56, 0.79, 0.96)
GREEN = (0.18, 0.49, 0.20)
LIGHT_GREEN = (0.50, 0.78, 0.50)
RED = (0.83, 0.18, 0.18)
LIGHT_RED = (0.94, 0.60, 0.60)
PURPLE = (0.42, 0.10, 0.60)
LIGHT_PURPLE = (0.73, 0.56, 0.87)
ORANGE = (0.96, 0.49, 0.00)
TEAL = (0.00, 0.59, 0.65)
GRAY = (0.37, 0.39, 0.41)
LGRAY = (0.93, 0.94, 0.95)
WHITE = (1, 1, 1)
BLACK = (0, 0, 0)

PALETTE = [BLUE, RED, GREEN, PURPLE, ORANGE, TEAL, LIGHT_BLUE, LIGHT_RED, LIGHT_GREEN, LIGHT_PURPLE]


class ScientificPlotter:
    """Publication-quality scientific figure generator using PyMuPDF."""

    def __init__(self, width: int = 792, height: int = 612):
        self.W = width
        self.H = height
        self.doc = None
        self.page = None
        self._page_count = 0

    def new_document(self) -> fitz.Document:
        self.doc = fitz.open()
        self._page_count = 0
        return self.doc

    def new_page(self, title: str = "", landscape: bool = True) -> fitz.Page:
        w = self.W if landscape else 612
        h = self.H if landscape else 792
        self.page = self.doc.new_page(width=w, height=h)
        self._page_count += 1
        # Header
        self.page.draw_rect(fitz.Rect(0, 0, w, 32), color=NAVY, fill=NAVY)
        self.page.insert_text(fitz.Point(20, 22), title or "BiRAGAS CRISPR Complete v3.0",
                               fontname="hebo", fontsize=10, color=WHITE)
        self.page.insert_text(fitz.Point(w - 150, 22), "Ayass Bioscience LLC",
                               fontname="helv", fontsize=8, color=(0.7, 0.75, 0.9))
        # Footer
        self.page.insert_text(fitz.Point(20, h - 12), f"Page {self._page_count}",
                               fontname="helv", fontsize=7, color=GRAY)
        return self.page

    def save(self, path: str):
        if self.doc:
            self.doc.save(path)
            self.doc.close()

    # ══════════════════════════════════════════════════════════════════════════
    # BAR CHART — Horizontal bars with labels
    # ══════════════════════════════════════════════════════════════════════════

    def bar_chart(self, x: float, y: float, w: float, h: float,
                  labels: List[str], values: List[float],
                  title: str = "", colors: Optional[List] = None,
                  max_val: float = 0, show_values: bool = True,
                  value_format: str = "{:.3f}"):
        """Draw horizontal bar chart."""
        p = self.page
        if not max_val:
            max_val = max(abs(v) for v in values) * 1.1 if values else 1.0
        n = len(labels)
        if n == 0:
            return
        bar_h = min(18, (h - 20) / n)
        gap = 3

        # Title
        if title:
            p.insert_text(fitz.Point(x, y + 12), title, fontname="hebo", fontsize=10, color=NAVY)
            y += 18

        for i, (label, val) in enumerate(zip(labels, values)):
            by = y + i * (bar_h + gap)
            # Label
            p.insert_text(fitz.Point(x, by + bar_h - 3), label[:18],
                           fontname="helv", fontsize=7, color=BLACK)
            # Bar background
            bx = x + 120
            bar_w = w - 130
            p.draw_rect(fitz.Rect(bx, by, bx + bar_w, by + bar_h),
                         color=LGRAY, fill=LGRAY)
            # Bar fill
            fill_w = abs(val) / max_val * bar_w
            color = colors[i % len(colors)] if colors else PALETTE[i % len(PALETTE)]
            if val >= 0:
                p.draw_rect(fitz.Rect(bx, by, bx + fill_w, by + bar_h),
                             color=color, fill=color)
            else:
                p.draw_rect(fitz.Rect(bx + bar_w - fill_w, by, bx + bar_w, by + bar_h),
                             color=color, fill=color)
            # Value label
            if show_values:
                p.insert_text(fitz.Point(bx + fill_w + 5, by + bar_h - 3),
                               value_format.format(val),
                               fontname="helv", fontsize=7, color=GRAY)

    # ══════════════════════════════════════════════════════════════════════════
    # WATERFALL PLOT — For knockout ensemble methods
    # ══════════════════════════════════════════════════════════════════════════

    def waterfall_plot(self, x: float, y: float, w: float, h: float,
                       labels: List[str], values: List[float],
                       title: str = "", colors: Optional[List] = None):
        """Draw waterfall/cascade plot showing cumulative contributions."""
        p = self.page
        n = len(labels)
        if n == 0:
            return

        if title:
            p.insert_text(fitz.Point(x, y + 12), title, fontname="hebo", fontsize=10, color=NAVY)
            y += 20

        max_cumulative = sum(abs(v) for v in values) * 1.1
        bar_w = (w - 20) / n
        cumulative = 0

        for i, (label, val) in enumerate(zip(labels, values)):
            bx = x + 10 + i * bar_w
            prev_cum = cumulative
            cumulative += val

            # Normalize to chart height
            y_base = y + h - 20 - (prev_cum / max_cumulative * (h - 40))
            y_top = y + h - 20 - (cumulative / max_cumulative * (h - 40))

            color = colors[i % len(colors)] if colors else PALETTE[i % len(PALETTE)]
            p.draw_rect(fitz.Rect(bx + 2, min(y_base, y_top), bx + bar_w - 2, max(y_base, y_top)),
                         color=color, fill=color)
            # Label
            p.insert_text(fitz.Point(bx + 2, y + h - 5), label[:8],
                           fontname="helv", fontsize=6, color=GRAY)
            # Value
            p.insert_text(fitz.Point(bx + 2, min(y_base, y_top) - 3),
                           f"{val:.3f}", fontname="helv", fontsize=6, color=BLACK)

    # ══════════════════════════════════════════════════════════════════════════
    # RADAR CHART — For multi-dimensional scoring
    # ══════════════════════════════════════════════════════════════════════════

    def radar_chart(self, cx: float, cy: float, radius: float,
                    labels: List[str], values: List[float],
                    title: str = "", color=BLUE, max_val: float = 1.0):
        """Draw radar/spider chart for multi-dimensional scores."""
        p = self.page
        n = len(labels)
        if n < 3:
            return

        if title:
            p.insert_text(fitz.Point(cx - radius, cy - radius - 15), title,
                           fontname="hebo", fontsize=10, color=NAVY)

        # Draw axes
        angles = [2 * math.pi * i / n - math.pi / 2 for i in range(n)]
        for i, angle in enumerate(angles):
            ex = cx + radius * math.cos(angle)
            ey = cy + radius * math.sin(angle)
            p.draw_line(fitz.Point(cx, cy), fitz.Point(ex, ey), color=LGRAY, width=0.5)
            # Label
            lx = cx + (radius + 15) * math.cos(angle) - 20
            ly = cy + (radius + 15) * math.sin(angle) + 3
            p.insert_text(fitz.Point(lx, ly), labels[i][:12],
                           fontname="helv", fontsize=6, color=GRAY)

        # Draw concentric circles
        for r_frac in [0.25, 0.5, 0.75, 1.0]:
            r = radius * r_frac
            # Approximate circle with polygon
            pts = [fitz.Point(cx + r * math.cos(a), cy + r * math.sin(a))
                   for a in [2 * math.pi * j / 36 for j in range(37)]]
            for j in range(len(pts) - 1):
                p.draw_line(pts[j], pts[j + 1], color=LGRAY, width=0.3)

        # Draw data polygon
        data_pts = []
        for i, val in enumerate(values):
            r = radius * min(1.0, val / max_val)
            data_pts.append(fitz.Point(cx + r * math.cos(angles[i]),
                                        cy + r * math.sin(angles[i])))
        # Fill polygon
        for i in range(len(data_pts)):
            j = (i + 1) % len(data_pts)
            p.draw_line(data_pts[i], data_pts[j], color=color, width=2)
        # Draw dots
        for pt in data_pts:
            p.draw_circle(pt, 3, color=color, fill=color)

    # ══════════════════════════════════════════════════════════════════════════
    # HEATMAP — For combination synergy matrix
    # ══════════════════════════════════════════════════════════════════════════

    def heatmap(self, x: float, y: float, w: float, h: float,
                row_labels: List[str], col_labels: List[str],
                matrix: List[List[float]], title: str = "",
                vmin: float = -1.0, vmax: float = 1.0):
        """Draw heatmap with color scale."""
        p = self.page
        nr = len(row_labels)
        nc = len(col_labels)
        if nr == 0 or nc == 0:
            return

        if title:
            p.insert_text(fitz.Point(x, y + 12), title, fontname="hebo", fontsize=10, color=NAVY)
            y += 18

        cell_w = min(40, (w - 80) / nc)
        cell_h = min(25, (h - 40) / nr)
        ox = x + 80  # offset for row labels
        oy = y + 20  # offset for col labels

        # Column labels
        for j, cl in enumerate(col_labels):
            p.insert_text(fitz.Point(ox + j * cell_w + 2, oy - 3), cl[:8],
                           fontname="helv", fontsize=6, color=GRAY)

        # Cells
        for i in range(nr):
            # Row label
            p.insert_text(fitz.Point(x, oy + i * cell_h + cell_h - 4),
                           row_labels[i][:12], fontname="helv", fontsize=7, color=BLACK)
            for j in range(nc):
                val = matrix[i][j] if i < len(matrix) and j < len(matrix[i]) else 0
                # Color: blue (negative) → white (0) → red (positive)
                t = (val - vmin) / (vmax - vmin) if vmax != vmin else 0.5
                t = max(0, min(1, t))
                if t < 0.5:
                    r = 0.5 + t
                    g = 0.5 + t
                    b = 1.0
                else:
                    r = 1.0
                    g = 1.0 - (t - 0.5) * 2
                    b = 1.0 - (t - 0.5) * 2

                cx = ox + j * cell_w
                cy = oy + i * cell_h
                p.draw_rect(fitz.Rect(cx, cy, cx + cell_w - 1, cy + cell_h - 1),
                             color=(0.8, 0.8, 0.8), fill=(r, g, b))
                # Value text
                p.insert_text(fitz.Point(cx + 2, cy + cell_h - 5),
                               f"{val:.2f}", fontname="helv", fontsize=5, color=BLACK)

    # ══════════════════════════════════════════════════════════════════════════
    # CONFIDENCE INTERVAL PLOT — For knockout methods
    # ══════════════════════════════════════════════════════════════════════════

    def ci_plot(self, x: float, y: float, w: float, h: float,
                labels: List[str], means: List[float],
                ci_lows: List[float], ci_highs: List[float],
                title: str = ""):
        """Draw confidence interval forest plot."""
        p = self.page
        n = len(labels)
        if n == 0:
            return

        if title:
            p.insert_text(fitz.Point(x, y + 12), title, fontname="hebo", fontsize=10, color=NAVY)
            y += 20

        all_vals = ci_lows + ci_highs + means
        vmin = min(all_vals) - 0.1
        vmax = max(all_vals) + 0.1
        plot_x = x + 100
        plot_w = w - 120

        row_h = min(20, (h - 20) / n)

        for i, (label, mean, lo, hi) in enumerate(zip(labels, means, ci_lows, ci_highs)):
            ry = y + i * row_h

            # Label
            p.insert_text(fitz.Point(x, ry + row_h - 4), label[:14],
                           fontname="helv", fontsize=7, color=BLACK)

            # Scale to plot
            def scale(v):
                return plot_x + (v - vmin) / (vmax - vmin) * plot_w

            # CI line
            p.draw_line(fitz.Point(scale(lo), ry + row_h / 2),
                         fitz.Point(scale(hi), ry + row_h / 2),
                         color=BLUE, width=1.5)
            # CI caps
            for v in [lo, hi]:
                p.draw_line(fitz.Point(scale(v), ry + 3),
                             fitz.Point(scale(v), ry + row_h - 3),
                             color=BLUE, width=1)
            # Mean dot
            p.draw_circle(fitz.Point(scale(mean), ry + row_h / 2), 3, color=RED, fill=RED)

            # Value
            p.insert_text(fitz.Point(scale(hi) + 5, ry + row_h - 4),
                           f"{mean:.3f} [{lo:.3f}, {hi:.3f}]",
                           fontname="helv", fontsize=6, color=GRAY)

        # Zero line
        zero_x = plot_x + (0 - vmin) / (vmax - vmin) * plot_w
        if plot_x <= zero_x <= plot_x + plot_w:
            p.draw_line(fitz.Point(zero_x, y), fitz.Point(zero_x, y + n * row_h),
                         color=GRAY, width=0.5, dashes="[3]")

    # ══════════════════════════════════════════════════════════════════════════
    # STACKED BAR — For ACE stream contributions
    # ══════════════════════════════════════════════════════════════════════════

    def stacked_bar(self, x: float, y: float, w: float, h: float,
                    labels: List[str], stacks: List[List[float]],
                    stack_labels: List[str], title: str = ""):
        """Draw stacked horizontal bar chart."""
        p = self.page
        n = len(labels)
        if n == 0:
            return

        if title:
            p.insert_text(fitz.Point(x, y + 12), title, fontname="hebo", fontsize=10, color=NAVY)
            y += 20

        max_total = max(sum(abs(v) for v in stack) for stack in stacks) * 1.1
        bar_h = min(18, (h - 30) / n)
        bx = x + 100
        bar_w = w - 120

        for i, (label, stack) in enumerate(zip(labels, stacks)):
            by = y + i * (bar_h + 3)
            p.insert_text(fitz.Point(x, by + bar_h - 3), label[:14],
                           fontname="helv", fontsize=7, color=BLACK)
            cum_x = bx
            for j, val in enumerate(stack):
                seg_w = abs(val) / max_total * bar_w
                color = PALETTE[j % len(PALETTE)]
                p.draw_rect(fitz.Rect(cum_x, by, cum_x + seg_w, by + bar_h),
                             color=color, fill=color)
                cum_x += seg_w

        # Legend
        ly = y + n * (bar_h + 3) + 5
        for j, sl in enumerate(stack_labels[:8]):
            lx = x + j * 90
            p.draw_rect(fitz.Rect(lx, ly, lx + 8, ly + 8), color=PALETTE[j], fill=PALETTE[j])
            p.insert_text(fitz.Point(lx + 12, ly + 7), sl[:12],
                           fontname="helv", fontsize=6, color=GRAY)

    # ══════════════════════════════════════════════════════════════════════════
    # TABLE — Professional formatted table
    # ══════════════════════════════════════════════════════════════════════════

    def table(self, x: float, y: float, headers: List[str], rows: List[List],
              col_widths: Optional[List[float]] = None, title: str = "",
              fontsize: float = 7):
        """Draw a professional table."""
        p = self.page
        nc = len(headers)
        if not col_widths:
            col_widths = [120] * nc

        if title:
            p.insert_text(fitz.Point(x, y + 12), title, fontname="hebo", fontsize=10, color=NAVY)
            y += 18

        rh = fontsize + 8

        # Header
        cx = x
        for i, h in enumerate(headers):
            p.draw_rect(fitz.Rect(cx, y, cx + col_widths[i], y + rh), color=NAVY, fill=NAVY)
            p.insert_text(fitz.Point(cx + 3, y + fontsize + 1), str(h)[:25],
                           fontname="hebo", fontsize=fontsize, color=WHITE)
            cx += col_widths[i]
        y += rh

        for ri, row in enumerate(rows):
            cx = x
            bg = LGRAY if ri % 2 == 0 else WHITE
            for ci, cell in enumerate(row):
                if ci < nc:
                    p.draw_rect(fitz.Rect(cx, y, cx + col_widths[ci], y + rh),
                                 color=(0.85, 0.86, 0.87), fill=bg)
                    p.insert_text(fitz.Point(cx + 3, y + fontsize + 1),
                                   str(cell)[:30], fontname="helv", fontsize=fontsize, color=BLACK)
                    cx += col_widths[ci]
            y += rh

        return y + 5

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE PROGRESS — 7-phase pipeline visualization
    # ══════════════════════════════════════════════════════════════════════════

    def phase_progress(self, x: float, y: float, w: float,
                       phase_results: Dict, title: str = ""):
        """Draw 7-phase pipeline progress with pass/fail indicators."""
        p = self.page
        if title:
            p.insert_text(fitz.Point(x, y + 12), title, fontname="hebo", fontsize=10, color=NAVY)
            y += 20

        phases = list(phase_results.keys())
        phase_names = ["P1: Screening→DAG", "P2: Network Scoring", "P3: QA & Validation",
                       "P4: Mechanisms", "P5: Pharmaceutical", "P6: Stratification", "P7: Reporting"]

        box_w = (w - 20) / 7
        for i, phase_key in enumerate(phases[:7]):
            data = phase_results[phase_key]
            n_run = data.get('modules_run', 0)
            n_fail = data.get('modules_failed', 0)
            color = GREEN if n_fail == 0 else RED

            bx = x + i * box_w
            # Box
            p.draw_rect(fitz.Rect(bx + 2, y, bx + box_w - 2, y + 50),
                         color=color, fill=(*color, 0.15) if n_fail == 0 else (*color, 0.1))
            p.draw_rect(fitz.Rect(bx + 2, y, bx + box_w - 2, y + 5),
                         color=color, fill=color)
            # Phase name
            name = phase_names[i] if i < len(phase_names) else phase_key
            p.insert_text(fitz.Point(bx + 5, y + 18), name[:16],
                           fontname="hebo", fontsize=7, color=NAVY)
            # Status
            status = f"{n_run}/4 PASS" if n_fail == 0 else f"{n_run-n_fail}/{n_run} FAIL"
            p.insert_text(fitz.Point(bx + 5, y + 35), status,
                           fontname="helv", fontsize=8, color=color)
            # Arrow
            if i < 6:
                p.insert_text(fitz.Point(bx + box_w - 8, y + 22), "→",
                               fontname="helv", fontsize=12, color=GRAY)

        return y + 60
