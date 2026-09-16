# Copyright 2026 IQM
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Plotting helpers for the qubit selector demo notebooks.

The helpers share one light palette and one plot style so that every figure in a talk reads as part of the
same system. Call :func:`apply_demo_style` once at the top of a notebook, then use the ``plot_*`` functions.

Everything here works off a qiskit ``IQMBackend`` and plain dicts of numbers, so ``matplotlib``,
``rustworkx`` and ``iqm-client[qiskit]`` are the only things it needs.
"""

from collections.abc import Mapping, Sequence
from math import dist
import re
from statistics import median
from typing import Protocol

from iqm.qiskit_iqm.iqm_backend import IQMBackendBase
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from rustworkx import spring_layout


class LayoutScoreLike(Protocol):
    """What :func:`plot_score_breakdown` needs of a score, so that this module stays qiskit-only.

    A ``LayoutScore`` from ``iqm.qubit_selector.core`` satisfies it, but nothing here imports the core.
    """

    layout: list[str]
    """Physical component names of the layout."""

    total: float
    """Total error budget in nats."""

    error: float
    """Error the budget corresponds to."""

    two_qubit: float
    """Two-qubit gate contribution, in nats."""

    single_qubit: float
    """Single-qubit gate contribution, in nats."""

    readout: float
    """Readout contribution, in nats."""


SURFACE = "#fcfcfb"
"""Background of the figure and the plot area."""

INK_PRIMARY = "#0b0b0b"
"""Ink for titles and value labels."""

INK_SECONDARY = "#52514e"
"""Ink for subtitles and axis labels."""

INK_MUTED = "#898781"
"""Ink for tick labels and annotations that must stay recessive."""

GRID = "#e1e0d9"
"""Hairline gridlines."""

AXIS = "#c3c2b7"
"""Baseline, axis line, and the fill of de-emphasized bars."""

SERIES_BLUE = "#2a78d6"
"""Categorical slot 1, used for the primary series and for anything selected."""

SERIES_ORANGE = "#eb6834"
"""Categorical slot 2, used for the series compared against the primary one."""

SERIES_AQUA = "#1baf7a"
"""Categorical slot 3, used for a third series."""

_TWO_QUBIT_KEYS = ("CZ", "CLIFFORD")
"""Calibration keys whose entries are qubit pairs rather than single qubits."""

_COHERENCE_KEYS = ("t1", "t2")
"""Calibration keys reported as times rather than as fidelities."""

_TITLES = {
    "CZ": "CZ gate error per qubit pair",
    "CLIFFORD": "Clifford gate error per qubit pair",
    "1Q": "Single-qubit gate error per qubit",
    "readout": "Readout error per qubit",
    "readout_qndness": "Readout QNDness error per qubit",
    "t1": "T1 coherence time per qubit",
    "t2": "T2 coherence time per qubit",
}
"""Human readable chart titles for the calibration metric keys."""

_MAX_TICK_LABELS = 48
"""Above this many categories the tick labels are thinned out so they stay readable."""

_LARGE_CHIP = 24
"""Above this many qubits the node labels of the QPU plot are shrunk."""

_LAYOUT_SEED = 7
"""Seed of the force directed fallback layout, so that an unknown chip is drawn the same way every time."""

# Coordinates of the two crystal chips we demo on, in qiskit index order. Taken from
# ``iqm.benchmarks.utils_plots.GraphPositions``, which keeps the same maps for the benchmark plots.
# fmt: off
_GARNET_POSITIONS = [
    (5, 7), (6, 6), (3, 7), (4, 6), (5, 5), (6, 4), (7, 3), (2, 6), (3, 5), (4, 4),
    (5, 3), (6, 2), (1, 5), (2, 4), (3, 3), (4, 2), (5, 1), (1, 3), (2, 2), (3, 1)
]
"""Node coordinates of the 20-qubit crystal (Garnet)."""

_EMERALD_POSITIONS = [
    (10, 10), (11, 9), (7, 11), (8, 10), (9, 9), (10, 8), (11, 7), (5, 11), (6, 10), (7, 9),
    (8, 8), (9, 7), (10, 6), (11, 5), (3, 11), (4, 10), (5, 9), (6, 8), (7, 7), (8, 6),
    (9, 5), (10, 4), (2, 10), (3, 9), (4, 8), (5, 7), (6, 6), (7, 5), (8, 4), (9, 3),
    (10, 2), (2, 8), (3, 7), (4, 6), (5, 5), (6, 4), (7, 3), (8, 2), (9, 1), (1, 7),
    (2, 6), (3, 5), (4, 4), (5, 3), (6, 2), (7, 1), (1, 5), (2, 4), (3, 3), (4, 2),
    (5, 1), (1, 3), (2, 2), (3, 1)
]
"""Node coordinates of the 54-qubit crystal (Emerald)."""
# fmt: on

_CRYSTAL_POSITIONS = {20: _GARNET_POSITIONS, 54: _EMERALD_POSITIONS}
"""Predefined node coordinates, keyed by the number of qubits of the chip."""


def apply_demo_style() -> None:
    """Apply the shared plot style to matplotlib.

    Sets the surface colors, the recessive grid and axes, and the type sizes used by all helpers in this
    module. Call it once per notebook, before the first plot.
    """
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "figure.dpi": 120,
            "savefig.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.labelcolor": INK_SECONDARY,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.size": 10,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2,
            "text.color": INK_PRIMARY,
            "xtick.color": INK_MUTED,
            "xtick.labelsize": 8,
            "ytick.color": INK_MUTED,
            "ytick.labelsize": 9,
        }
    )


def _set_headers(ax: Axes, title: str, subtitle: str) -> None:
    """Draw a left aligned bold title with a muted subtitle underneath it."""
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=INK_PRIMARY, pad=24)
    ax.text(
        0.0,
        1.02,
        subtitle,
        transform=ax.transAxes,
        fontsize=10,
        color=INK_SECONDARY,
        va="bottom",
        ha="left",
    )


def _pretty_label(name: str) -> str:
    """Turn a two-qubit metrics key such as ``"['QB1', 'QB2']"`` into ``"QB1-QB2"``, leaving others alone."""
    components = re.findall(r"[A-Za-z]+\d+", name)
    return "-".join(components) if name.startswith("[") else name


def _thin_tick_labels(ax: Axes, labels: Sequence[str]) -> None:
    """Show every label if they fit, otherwise show every n-th one."""
    step = 1 + len(labels) // _MAX_TICK_LABELS
    ax.set_xticks(range(0, len(labels), step))
    ax.set_xticklabels(labels[::step], rotation=90)


def plot_calibration_data(key: str, data: dict[str, float]) -> Axes:
    """Plot one calibration metric across the chip.

    Errors are plotted as ``1 - fidelity`` and coherence metrics as times. A dashed median line makes the
    spread across the device visible at a glance, and the worst entry is labelled directly.

    Args:
        key: Calibration metric key, for example ``CalibrationType.CZ.value``.
        data: Mapping from qubit or qubit pair name to the calibrated value.

    Returns:
        The axes the metric was drawn on.

    """
    is_coherence = key in _COHERENCE_KEYS
    entries = list(data.items())
    if key in _TWO_QUBIT_KEYS:
        # Each pair is reported in both directions, so keep only one of the two.
        entries = entries[::2]

    labels = [_pretty_label(name) for name, _ in entries]
    values = [value if is_coherence else 1 - value for _, value in entries]

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.bar(range(len(values)), values, color=SERIES_BLUE, width=0.8)

    mid = median(values)
    ax.axhline(mid, color=INK_MUTED, linestyle="--", linewidth=1)

    if is_coherence:
        subtitle = f"Higher is better. Dashed line: median {mid:.1f} us"
    else:
        subtitle = f"Lower is better. Dashed line: median {mid:.2%}"
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.set_xlim(-1, len(values))
    ax.set_ylim(0, max(values) * 1.08)
    ax.set_xlabel("Qubit pairs" if key in _TWO_QUBIT_KEYS else "Qubits")
    ax.set_ylabel("Coherence time (us)" if is_coherence else "Error")
    _thin_tick_labels(ax, labels)
    _set_headers(ax, _TITLES.get(key, f"{key} metrics"), subtitle)
    fig.tight_layout()
    return ax


def plot_cost_spread(costs: Sequence[float], num_qubits: int, metric: str = "cost") -> Axes:
    """Plot the cost of every candidate layout, with the selected one highlighted.

    Args:
        costs: Layout costs sorted best first, as returned by ``CostEvaluator.get_top_layouts`` or
            ``select_layout``.
        num_qubits: Number of qubits in the circuit the layouts were generated for.
        metric: What the values are called, used in the axis label and the subtitle.

    Returns:
        The axes the costs were drawn on.

    """
    percentages = [c * 100 for c in costs]
    colors = [SERIES_BLUE] + [AXIS] * (len(percentages) - 1)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(range(len(percentages)), percentages, color=colors, width=0.8)
    ax.set_xlabel(f"Candidate layout, sorted by {metric}")
    ax.set_ylabel(metric.capitalize())
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=1))
    ax.set_xlim(-1, len(percentages))
    ax.legend(
        handles=[
            Patch(facecolor=SERIES_BLUE, label="selected layout"),
            Patch(facecolor=AXIS, label="other candidates"),
        ],
        loc="upper left",
    )
    _set_headers(
        ax,
        f"Layout quality across the top {len(percentages)} candidates",
        f"{num_qubits}-qubit circuit. Lower {metric} is better: "
        f"{percentages[0]:.2f}% for the selected layout, {percentages[-1]:.2f}% for the worst one shown",
    )
    fig.tight_layout()
    return ax


def plot_score_breakdown(scores: Sequence[LayoutScoreLike], top_n: int = 10) -> Axes:
    """Plot where the error budget of each candidate layout goes.

    ``score_layouts`` returns the budget decomposed into two-qubit, single-qubit and readout terms, all in
    nats and additive, so they stack into the total the ranking is based on.

    Args:
        scores: Layout scores sorted best first, as returned by ``score_layouts``.
        top_n: How many of the best candidates to show.

    Returns:
        The axes the breakdown was drawn on.

    """
    shown = list(scores[:top_n])
    ranks = range(len(shown))
    terms = [
        ("two-qubit gates", [s.two_qubit for s in shown], SERIES_BLUE),
        ("single-qubit gates", [s.single_qubit for s in shown], SERIES_ORANGE),
        ("readout", [s.readout for s in shown], SERIES_AQUA),
    ]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    bottom = [0.0] * len(shown)
    for name, values, color in terms:
        # A surface-colored edge keeps a visible gap between the stacked segments.
        ax.bar(ranks, values, bottom=bottom, color=color, width=0.7, label=name, edgecolor=SURFACE, linewidth=1)
        bottom = [base + value for base, value in zip(bottom, values, strict=True)]

    ax.set_xticks(list(ranks))
    ax.set_xticklabels([str(rank + 1) for rank in ranks])
    ax.set_xlabel("Candidate layout, sorted by total budget")
    ax.set_ylabel("Error budget (nats)")
    ax.legend(loc="upper left")
    best = shown[0]
    _set_headers(
        ax,
        f"Where the error budget goes, top {len(shown)} layouts",
        f"Best layout {', '.join(best.layout)}: {best.total:.3f} nats in total, {best.error:.2%} error",
    )
    fig.tight_layout()
    return ax


def _chip(backend: IQMBackendBase) -> tuple[list[str], list[tuple[int, int]]]:
    """Qubit names in qiskit index order, and the coupler list with each coupling listed once."""
    names = [backend.index_to_qubit_name(i) for i in range(backend.num_qubits)]
    couplers = {(min(a, b), max(a, b)) for a, b in backend.coupling_map.get_edges()}
    return names, sorted(couplers)


def _qubit_positions(backend: IQMBackendBase, names: Sequence[str]) -> dict[int, tuple[float, float]]:
    """Node coordinates of the chip, predefined for the known crystals and force directed for the rest."""
    known = _CRYSTAL_POSITIONS.get(len(names))
    # The predefined tables assume the standard QB1..QBn order; anything else gets a generated layout
    # rather than a wrong picture.
    if known is not None and list(names) == [f"QB{i + 1}" for i in range(len(names))]:
        return dict(enumerate(known))
    graph = backend.coupling_map.graph.to_undirected(multigraph=False)
    placed = spring_layout(graph, scale=3, num_iter=500, seed=_LAYOUT_SEED)
    return {int(node): (float(xy[0]), float(xy[1])) for node, xy in placed.items()}


def _layout_indices(names: Sequence[str], layout: Sequence[int | str] | None) -> list[int]:
    """Normalize a layout to node indices, accepting either qiskit indices or component names."""
    if layout is None:
        return []
    return [names.index(qubit) if isinstance(qubit, str) else int(qubit) for qubit in layout]


def plot_qpu_layout(  # noqa: PLR0913
    backend: IQMBackendBase,
    layout: Sequence[int | str] | None = None,
    title: str | None = None,
    color: str = SERIES_BLUE,
    label: str = "selected qubits",
    ax: Axes | None = None,
) -> Axes:
    """Draw the qubits and couplers of the chip and highlight a layout on it.

    The highlighted qubits and the couplers between them are drawn in the highlight color, everything else
    stays recessive, so a layout can be read off the chip at a glance. Nodes are labelled with their qubit
    name, and the layout is spelled out in the subtitle.

    Args:
        backend: IQM backend whose chip is drawn.
        layout: Qubits to highlight, either as qiskit indices or as component names, for example the best
            layout of ``get_top_layouts``. Nothing is highlighted if it is omitted.
        title: Chart title. Defaults to the number of qubits of the chip.
        color: Highlight color of the layout.
        label: Legend label of the highlighted qubits.
        ax: Axes to draw on. A new figure is created if it is omitted.

    Returns:
        The axes the chip was drawn on.

    """
    names, couplers = _chip(backend)
    positions = _qubit_positions(backend, names)
    selected = _layout_indices(names, layout)
    chosen = set(selected)

    xs = [x for x, _ in positions.values()]
    ys = [y for _, y in positions.values()]
    span_x = max(max(xs) - min(xs), 1.0)
    span_y = max(max(ys) - min(ys), 1.0)
    spacing = min((dist(positions[a], positions[b]) for a, b in couplers), default=1.0)
    radius = 0.3 * spacing

    own_figure = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(8, min(9.0, max(4.5, 8.0 * span_y / span_x))))
    for a, b in couplers:
        highlighted = a in chosen and b in chosen
        ax.plot(
            [positions[a][0], positions[b][0]],
            [positions[a][1], positions[b][1]],
            color=color if highlighted else AXIS,
            linewidth=3.0 if highlighted else 1.2,
            solid_capstyle="round",
            zorder=2 if highlighted else 1,
        )

    label_size = 9 if len(positions) <= _LARGE_CHIP else 6.5
    for index, (x, y) in positions.items():
        highlighted = index in chosen
        ax.add_patch(
            Circle(
                (x, y),
                radius,
                facecolor=color if highlighted else SURFACE,
                edgecolor=color if highlighted else AXIS,
                linewidth=1.2,
                zorder=3,
            )
        )
        ax.text(
            x,
            y,
            names[index],
            ha="center",
            va="center",
            fontsize=label_size,
            fontweight="bold" if highlighted else "normal",
            color=SURFACE if highlighted else INK_MUTED,
            zorder=4,
        )

    if chosen:
        ax.legend(
            handles=[_node_handle(color, color, label), _node_handle(SURFACE, AXIS, "other qubits")],
            loc="upper right",
        )
    # Echo the layout exactly as it was passed in, whether that was qiskit indices or component names.
    given = ", ".join(str(qubit) for qubit in layout) if layout is not None else ""
    subtitle = f"Layout: {given}" if selected else f"{len(positions)} qubits, {len(couplers)} couplers"

    ax.set_xlim(min(xs) - 2 * radius, max(xs) + 2 * radius)
    ax.set_ylim(min(ys) - 2 * radius, max(ys) + 2 * radius)
    ax.set_aspect("equal")
    ax.axis("off")
    _set_headers(ax, title or f"{len(positions)}-qubit QPU", subtitle)
    if own_figure:
        ax.figure.tight_layout()
    return ax


def plot_layout_comparison(
    backend: IQMBackendBase, naive_layout: Sequence[int | str], selected_layout: Sequence[int | str]
) -> tuple[Axes, Axes]:
    """Draw the chip twice side by side, with the naive and the selected qubits highlighted.

    The colors match :func:`plot_fidelity_comparison`, so the two figures can be read together.

    Args:
        backend: IQM backend whose chip is drawn.
        naive_layout: Qubits the default transpilation ended up using, as indices or component names.
        selected_layout: Qubits chosen by the qubit selector, as indices or component names.

    Returns:
        The axes of the naive layout and of the selected layout.

    """
    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 7.5))
    plot_qpu_layout(backend, naive_layout, title="Naive layout", color=SERIES_ORANGE, label="qubits used", ax=left)
    plot_qpu_layout(
        backend, selected_layout, title="Qubit selector layout", color=SERIES_BLUE, label="qubits used", ax=right
    )
    # The panels keep an equal aspect, so leave room for the headers instead of letting them run off the top.
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return left, right


def _node_handle(facecolor: str, edgecolor: str, label: str) -> Line2D:
    """Legend handle shaped like a qubit node."""
    return Line2D(
        [],
        [],
        marker="o",
        linestyle="",
        markersize=9,
        markerfacecolor=facecolor,
        markeredgecolor=edgecolor,
        color=edgecolor,
        label=label,
    )


def plot_fidelity_comparison(fidelity_naive: float, fidelity_selected: float, num_qubits: int) -> Axes:
    """Compare the measured fidelity of the naive layout and the selected layout.

    Args:
        fidelity_naive: Fidelity obtained on the layout chosen by the default transpilation.
        fidelity_selected: Fidelity obtained on the layout chosen by the qubit selector.
        num_qubits: Number of qubits in the circuit that was executed.

    Returns:
        The axes the comparison was drawn on.

    """
    labels = ["Naive layout", "Qubit selector layout"]
    values = [fidelity_naive, fidelity_selected]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(labels, values, color=[SERIES_ORANGE, SERIES_BLUE], width=0.5)
    for index, value in enumerate(values):
        ax.annotate(
            f"{value:.3f}",
            xy=(index, value),
            xytext=(0, 6),
            textcoords="offset points",
            color=INK_PRIMARY,
            fontsize=12,
            fontweight="bold",
            ha="center",
        )

    improvement = (fidelity_selected / fidelity_naive - 1) * 100 if fidelity_naive else float("nan")
    ax.set_ylim(0, min(1.0, max(values) * 1.25))
    ax.set_ylabel("Hellinger fidelity")
    ax.tick_params(axis="x", labelsize=10, colors=INK_SECONDARY)
    _set_headers(
        ax,
        f"{num_qubits}-qubit GHZ state on hardware",
        f"Selecting the layout improves the fidelity by {improvement:.1f}%",
    )
    fig.tight_layout()
    return ax


def plot_fidelity_groups(
    groups: Mapping[str, Mapping[str, float]],
    threshold: float | None = 0.5,
    title: str = "GHZ fidelity by configuration",
    subtitle: str | None = None,
    ylabel: str = "GHZ state fidelity",
) -> Axes:
    """Plot a few configurations measured the same way, each with the same set of numbers.

    Where :func:`plot_fidelity_levels` draws a ladder of runs that each add something to the previous one,
    this draws runs that are *alternatives* to one another: one group of bars per configuration, one bar per
    series inside it, so the comparison to make is between groups rather than along them.

    Args:
        groups: Configuration name to a mapping of series name to fidelity. Every group must carry the same
            series, drawn in the order the first group gives them.
        threshold: Fidelity to mark with a reference line, or ``None`` for no line. The GHZ default of 0.5
            is the sufficient condition for genuine multipartite entanglement.
        title: Chart title.
        subtitle: Chart subtitle. Defaults to naming the first group as the reference.
        ylabel: Label of the value axis.

    Returns:
        The axes the comparison was drawn on.

    Raises:
        ValueError: If ``groups`` is empty or the groups do not carry the same series.

    """
    if not groups:
        raise ValueError("plot_fidelity_groups needs at least one group.")
    names = list(groups)
    series = list(groups[names[0]])
    for name in names:
        if list(groups[name]) != series:
            raise ValueError(f"Group {name!r} carries {list(groups[name])}, expected the same series as {series}.")

    colors = [SERIES_ORANGE, SERIES_BLUE, SERIES_AQUA]
    width = 0.8 / len(series)
    positions = range(len(names))

    fig, ax = plt.subplots(figsize=(3.2 * len(names) + 2.0, 5.0))
    for index, label in enumerate(series):
        # Centre the group on its tick: the offset runs from -0.4 to +0.4 as index runs over the series.
        offset = (index - (len(series) - 1) / 2) * width
        values = [groups[name][label] for name in names]
        ax.bar(
            [position + offset for position in positions],
            values,
            width=width * 0.9,
            color=colors[index % len(colors)],
            label=label,
        )
        for position, value in zip(positions, values, strict=True):
            ax.annotate(
                f"{value:.3f}",
                xy=(position + offset, value),
                xytext=(0, 6),
                textcoords="offset points",
                color=INK_PRIMARY,
                fontsize=11,
                fontweight="bold",
                ha="center",
            )

    # The gain over the reference group goes into the tick label rather than above the bars, where it would
    # run into the threshold line as soon as a group gets close to it.
    reference = groups[names[0]]
    tick_labels = [names[0]]
    for name in names[1:]:
        gains = [(groups[name][label] / reference[label] - 1) * 100 for label in series if reference[label]]
        gain = f"{sum(gains) / len(gains):+.1f}% on average" if gains else "no reference"
        tick_labels.append(f"{name}\n{gain}")

    if threshold is not None:
        ax.axhline(threshold, color=INK_MUTED, linestyle="--", linewidth=1)

    highest = max(max(group.values()) for group in groups.values())
    ax.set_xticks(list(positions))
    ax.set_xticklabels(tick_labels)
    ax.set_ylim(0, min(1.0, max(highest, threshold or 0) * 1.35))
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelsize=10, colors=INK_SECONDARY)
    ax.legend(loc="upper left")
    _set_headers(ax, title, subtitle if subtitle is not None else f"Every group compared against {names[0].lower()}")
    fig.tight_layout()
    return ax


def plot_fidelity_levels(
    fidelities: Mapping[str, float],
    threshold: float | None = 0.5,
    title: str = "GHZ fidelity by error reduction level",
    subtitle: str | None = None,
) -> Axes:
    """Plot a ladder of runs where each step adds one more error reduction technique.

    The first entry is the baseline and is drawn in the comparison color; every later one is drawn in the
    highlight color and labelled with its improvement over that baseline.

    Args:
        fidelities: Mapping from level name to measured fidelity, drawn in the order given.
        threshold: Fidelity to mark with a reference line, or ``None`` for no line. The GHZ default of 0.5
            is the sufficient condition for genuine multipartite entanglement.
        title: Chart title.
        subtitle: Chart subtitle. Defaults to naming the baseline.

    Returns:
        The axes the ladder was drawn on.

    """
    names = list(fidelities)
    values = [fidelities[name] for name in names]
    base_value = values[0]

    fig, ax = plt.subplots(figsize=(2.6 * len(names) + 2.0, 5.0))
    ax.bar(names, values, color=[SERIES_ORANGE] + [SERIES_BLUE] * (len(names) - 1), width=0.55)
    for index, value in enumerate(values):
        ax.annotate(
            f"{value:.3f}",
            xy=(index, value),
            xytext=(0, 6),
            textcoords="offset points",
            color=INK_PRIMARY,
            fontsize=12,
            fontweight="bold",
            ha="center",
        )
        if index and base_value:
            ax.annotate(
                f"{(value / base_value - 1) * 100:+.1f}%",
                xy=(index, value),
                xytext=(0, 24),
                textcoords="offset points",
                color=INK_SECONDARY,
                fontsize=10,
                ha="center",
            )

    handles = [Patch(facecolor=SERIES_ORANGE, label=names[0]), Patch(facecolor=SERIES_BLUE, label="with mitigation")]
    if threshold is not None:
        ax.axhline(threshold, color=INK_MUTED, linestyle="--", linewidth=1)
        handles.append(Line2D([], [], color=INK_MUTED, linestyle="--", linewidth=1, label=f"F = {threshold}"))

    ax.set_ylim(0, min(1.0, max(*values, threshold or 0) * 1.35))
    ax.set_ylabel("GHZ state fidelity")
    ax.tick_params(axis="x", labelsize=10, colors=INK_SECONDARY)
    ax.legend(handles=handles, loc="upper left")
    _set_headers(ax, title, subtitle if subtitle is not None else f"Each step adds one technique to {names[0].lower()}")
    fig.tight_layout()
    return ax
