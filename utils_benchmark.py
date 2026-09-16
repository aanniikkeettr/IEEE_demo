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

"""Chip plots for the GHZ benchmark notebook, in the style of the other demo figures.

This is a simplified take on ``iqm.benchmarks.utils_plots.plot_layout_fidelity_graph``: the CZ error is
carried by the **edge width** alone, and every qubit is drawn the same size, so the picture answers one
question only - which couplers does the GHZ tree use, and how good are they.

Geometry, palette and headers come from :mod:`utils`, so a chip drawn here and a chip drawn there are the
same picture. The GHZ tree itself is computed with the benchmark's own ``get_edges`` and ``get_cx_map``, so
it is the tree the benchmark will actually run.
"""

from collections.abc import Mapping, Sequence
from math import log
from statistics import median

from iqm.benchmarks.utils import extract_fidelities_unified
from iqm.qiskit_iqm.iqm_backend import IQMBackendBase
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
from utils import (
    AXIS,
    INK_MUTED,
    SERIES_BLUE,
    SERIES_ORANGE,
    SURFACE,
    _chip,
    _qubit_positions,
    _set_headers,
)

try:  # The development sources in GHZ_optimal/, activated by ghz_optimal_patch.patch().
    from iqm.benchmarks.entanglement.ghz_optimal import TreeSearchOptions, select_ghz_tree

    _HAS_TREE_SEARCH = True
except ImportError:  # The released iqm-benchmarks, where the tree search still lives in ghz.py.
    from iqm.benchmarks.entanglement.ghz import get_cx_map, get_edges

    TreeSearchOptions = None  # type: ignore[assignment,misc]
    _HAS_TREE_SEARCH = False


_MIN_EDGE_WIDTH = 0.7
"""Line width of the best coupler on the chip."""

_MAX_EDGE_WIDTH = 5.0
"""Line width of the worst coupler on the chip."""

_TREE_EDGE_SCALE = 1.6
"""Line width multiplier for a coupler the GHZ tree uses. A multiplier rather than a constant, so that a
bad CZ inside the tree still looks bad."""

_TREE_EDGE_FLOOR = 0.8
"""Line width added to a coupler the GHZ tree uses, so that even the best one is clearly visible."""

_LARGE_CHIP = 24
"""Above this many qubits the node labels are shrunk."""


def cz_fidelities(backend: IQMBackendBase) -> dict[tuple[int, int], float]:
    """Fetch the CZ fidelity of every calibrated coupler, keyed by a sorted pair of qiskit indices.

    Args:
        backend: IQM backend to read the calibration of. Crystal devices only, since a Star device
            calibrates CZ gates against a resonator rather than against another qubit.

    Returns:
        Mapping from ``(lower index, higher index)`` to the CZ fidelity of that coupler.

    """
    qubit_mapping, metrics = extract_fidelities_unified(backend)
    # The metrics are keyed by an enumeration of the calibrated qubits; qubit "QBn" is qiskit index n - 1.
    to_qiskit = {index: number - 1 for number, index in qubit_mapping.items()}
    return {
        (min(to_qiskit[a], to_qiskit[b]), max(to_qiskit[a], to_qiskit[b])): fidelity
        for (a, b), fidelity in metrics["cz_gate_fidelity"].items()
    }


def _clamped(fidelities: dict[tuple[int, int], float]) -> dict[tuple[int, int], float]:
    """Replace fidelities of 1.0 or above with the median, since a zero-weight edge breaks the tree search."""
    valid = [fidelity for fidelity in fidelities.values() if fidelity < 1.0]
    fallback = median(valid) if valid else 0.99
    return {pair: (fidelity if fidelity < 1.0 else fallback) for pair, fidelity in fidelities.items()}


def ghz_tree_edges(
    backend: IQMBackendBase,
    qubit_layout: Sequence[int],
    tree_options: "TreeSearchOptions | None" = None,
    rank: int = 0,
) -> list[tuple[int, int]]:
    """Return the couplers the ``"tree"`` GHZ routine would use on this layout, in chronological order.

    This calls the benchmark's own spanning-tree code, so the answer is the circuit
    ``state_generation_routine="tree"`` will actually build, not an approximation of it. Which code that is
    depends on the environment: with the development sources of ``GHZ_optimal/`` in place it is the ranked
    search of ``ghz_optimal``, which is what ``tree_options`` configures, and otherwise the minimum spanning
    tree of the released ``ghz`` module.

    Args:
        backend: IQM backend the layout lives on.
        qubit_layout: Qiskit indices the GHZ state is prepared on.
        tree_options: What the tree search minimizes, matching the ``tree_*`` fields of the configuration
            that will be run. ``None`` uses the search defaults.
        rank: Which of the ranked candidate trees to return, 0 being the best-scoring one.

    Returns:
        The CZ pairs of the state generation circuit, earliest first.

    Raises:
        RuntimeError: If ``tree_options`` or a non-zero ``rank`` is given but the ranked tree search is not
            available, since silently ignoring them would draw a tree that is not the one being run.

    """
    if _HAS_TREE_SEARCH:
        _qubit_mapping, metrics = extract_fidelities_unified(backend)
        candidate = select_ghz_tree(list(qubit_layout), backend, metrics, len(qubit_layout), tree_options, rank)
        return list(candidate.pairs)

    if tree_options is not None or rank:
        raise RuntimeError(
            "tree_options and rank need the GHZ_optimal sources; call ghz_optimal_patch.patch() before "
            "importing this module."
        )
    fidelities = _clamped(cz_fidelities(backend))
    graph = get_edges(
        list(backend.coupling_map.get_edges()),
        list(qubit_layout),
        [list(pair) for pair in fidelities],
        list(fidelities.values()),
    )
    return [(control, target) for control, target in get_cx_map(list(qubit_layout), graph)]


def _edge_widths(
    couplers: Sequence[tuple[int, int]], fidelities: dict[tuple[int, int], float]
) -> dict[tuple[int, int], float]:
    """Line width per coupler, scaled linearly in the CZ error so that a thin edge is a good edge."""
    weights = {pair: -log(fidelities[pair]) for pair in couplers if pair in fidelities}
    worst = max(weights.values(), default=1.0) or 1.0
    span = _MAX_EDGE_WIDTH - _MIN_EDGE_WIDTH
    # Couplers without calibration data are drawn at the minimum width rather than dropped.
    return {pair: _MIN_EDGE_WIDTH + span * weights.get(pair, 0.0) / worst for pair in couplers}


def plot_cz_graph(  # noqa: PLR0913
    backend: IQMBackendBase,
    qubit_layout: Sequence[int] | None = None,
    tree_edges: Sequence[tuple[int, int]] | None = None,
    title: str | None = None,
    color: str = SERIES_BLUE,
    ax: Axes | None = None,
) -> Axes:
    """Draw the chip with the CZ error carried by the edge width, and highlight a layout and its GHZ tree.

    Every qubit is drawn the same size; only the edges vary. **A thinner edge is a better CZ**, since the
    width is proportional to ``-log(fidelity)``. The couplers in ``tree_edges`` are drawn on top in the
    highlight color, so it is immediately visible whether the tree found thin edges or had to take thick
    ones.

    Args:
        backend: IQM backend whose chip is drawn.
        qubit_layout: Qiskit indices to highlight. Nothing is highlighted if it is omitted.
        tree_edges: Couplers to draw as the GHZ path, for example from :func:`ghz_tree_edges`.
        title: Chart title. Defaults to the number of qubits of the chip.
        color: Highlight color of the layout and its tree.
        ax: Axes to draw on. A new figure is created if it is omitted.

    Returns:
        The axes the chip was drawn on.

    """
    names, couplers = _chip(backend)
    positions = _qubit_positions(backend, names)
    fidelities = _clamped(cz_fidelities(backend))
    widths = _edge_widths(couplers, fidelities)
    chosen = set(qubit_layout or [])
    tree = {(min(a, b), max(a, b)) for a, b in (tree_edges or [])}

    xs = [x for x, _ in positions.values()]
    ys = [y for _, y in positions.values()]
    own_figure = ax is None
    if ax is None:
        span_x = max(max(xs) - min(xs), 1.0)
        span_y = max(max(ys) - min(ys), 1.0)
        _, ax = plt.subplots(figsize=(8, min(9.0, max(4.5, 8.0 * span_y / span_x))))

    for pair in couplers:
        a, b = pair
        in_tree = pair in tree
        ax.plot(
            [positions[a][0], positions[b][0]],
            [positions[a][1], positions[b][1]],
            color=color if in_tree else AXIS,
            linewidth=widths[pair] * _TREE_EDGE_SCALE + _TREE_EDGE_FLOOR if in_tree else widths[pair],
            solid_capstyle="round",
            zorder=2 if in_tree else 1,
        )

    radius = 0.26 * min((_distance(positions[a], positions[b]) for a, b in couplers), default=1.0)
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

    handles = [
        Line2D([], [], color=AXIS, linewidth=2.0, label="coupler, width = CZ error"),
        Line2D([], [], color=color, linewidth=4.0, label="CZ used by the GHZ tree"),
    ]
    ax.legend(handles=handles, loc="upper right")

    if tree:
        tree_errors = [1 - fidelities[pair] for pair in tree if pair in fidelities]
        mean_error = sum(tree_errors) / len(tree_errors) if tree_errors else float("nan")
        subtitle = f"Thinner is better. {len(tree)} CZ gates in the tree, mean error {mean_error:.2%}"
    else:
        subtitle = "Thinner is better"
    ax.set_xlim(min(xs) - 2 * radius, max(xs) + 2 * radius)
    ax.set_ylim(min(ys) - 2 * radius, max(ys) + 2 * radius)
    ax.set_aspect("equal")
    ax.axis("off")
    _set_headers(ax, title or f"{len(positions)}-qubit QPU", subtitle)
    if own_figure:
        ax.figure.tight_layout()
    return ax


def _distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    """Euclidean distance between two node positions."""
    return ((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2) ** 0.5


def plot_tree_comparison(
    backend: IQMBackendBase,
    naive_layout: Sequence[int],
    selected_layout: Sequence[int],
    titles: tuple[str, str] = ("Naive layout", "Qubit selector layout"),
) -> tuple[Axes, Axes]:
    """Draw the GHZ tree on two layouts side by side, to compare which couplers each one gets to use.

    The colors match the other demo figures: the naive layout in the comparison color, the selected one in
    the highlight color.

    Args:
        backend: IQM backend whose chip is drawn.
        naive_layout: Qiskit indices of the layout on the left.
        selected_layout: Qiskit indices of the layout on the right.
        titles: Titles of the left and right panel.

    Returns:
        The axes of the left and the right panel.

    """
    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 7.5))
    for layout, title, color, ax in (
        (naive_layout, titles[0], SERIES_ORANGE, left),
        (selected_layout, titles[1], SERIES_BLUE, right),
    ):
        plot_cz_graph(backend, layout, ghz_tree_edges(backend, layout), title=title, color=color, ax=ax)
    # The panels keep an equal aspect, so leave room for the headers instead of letting them run off the top.
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return left, right


def plot_tree_variants(
    backend: IQMBackendBase,
    qubit_layout: Sequence[int],
    variants: Mapping[str, Sequence[tuple[int, int]]],
) -> tuple[Axes, ...]:
    """Draw several GHZ trees over the *same* layout side by side, one panel each.

    The companion of :func:`plot_tree_comparison`: there the layout changes and the tree follows, here the
    layout is fixed and only what the tree search was asked to minimize changes, so the panels differ by the
    couplers chosen and nothing else.

    Args:
        backend: IQM backend whose chip is drawn.
        qubit_layout: Qiskit indices the GHZ state is prepared on, the same in every panel.
        variants: Panel title to the CZ pairs of that tree, drawn left to right in the order given.

    Returns:
        One axes per variant, in the same order.

    """
    colors = (SERIES_ORANGE, SERIES_BLUE, INK_MUTED)
    fig, axes = plt.subplots(1, len(variants), figsize=(7.0 * len(variants), 7.5), squeeze=False)
    for ax, (title, tree_edges), color in zip(axes[0], variants.items(), colors, strict=False):
        plot_cz_graph(backend, qubit_layout, tree_edges, title=title, color=color, ax=ax)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return tuple(axes[0])
