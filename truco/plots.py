"""Gráficos de estrategias y de cruces entre políticas."""

from collections import defaultdict

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from truco.games.envido import Action

# Un color fijo por acción, el mismo en todos los gráficos.
ACTION_COLORS = {
    Action.PASO: "#2a78d6",
    Action.ENVIDO: "#eb6834",
    Action.REAL_ENVIDO: "#6250d6",
    Action.QUIERO: "#008300",
    Action.NO_QUIERO: "#e34948",
}
ACTION_LABELS = {
    Action.PASO: "Paso",
    Action.ENVIDO: "Envido",
    Action.REAL_ENVIDO: "Real envido",
    Action.QUIERO: "Quiero",
    Action.NO_QUIERO: "No quiero",
}
_LETTER_LABELS = {"P": "paso", "E": "envido", "R": "real envido", "Q": "quiero", "N": "no quiero"}

_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_AXIS = "#c3c2b7"


def describe_history(historia: str) -> str:
    """Ej.: "PE" -> "mano: paso → pie: envido". Las acciones alternan empezando por el mano."""
    if not historia:
        return "inicio"
    roles = ("mano", "pie")
    return " → ".join(f"{roles[i % 2]}: {_LETTER_LABELS[c]}" for i, c in enumerate(historia))


def _tantos_positions(tantos: list[int]) -> list[float]:
    # Los tantos 8 a 19 no existen: se dejan fuera y se separan los dos grupos con un hueco.
    return [t if t <= 7 else t - 11 for t in tantos]


def envido_strategy_figure(rows: list[dict], es_mano: int, historia: str) -> Figure:
    """Barras apiladas: para cada valor de tantos, la probabilidad de cada acción (suman 1)."""
    rows = sorted(rows, key=lambda r: r["tantos"])
    tantos = [r["tantos"] for r in rows]
    x = _tantos_positions(tantos)
    actions = [a for a in Action if any(r[a.name] > 0 for r in rows)]

    fig = Figure(figsize=(10, 4.2), facecolor=_SURFACE)
    ax = fig.add_subplot()
    ax.set_facecolor(_SURFACE)

    bottom = [0.0] * len(rows)
    for action in actions:
        heights = [r[action.name] for r in rows]
        ax.bar(
            x,
            heights,
            bottom=bottom,
            width=0.8,
            color=ACTION_COLORS[action],
            edgecolor=_SURFACE,
            linewidth=1,
            label=ACTION_LABELS[action],
            zorder=2,
        )
        bottom = [b + h for b, h in zip(bottom, heights, strict=True)]

    ax.set_xticks(x, [str(t) for t in tantos])
    ax.set_xlim(min(x) - 0.6, max(x) + 0.6)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Tantos", color=_INK)
    ax.set_ylabel("Probabilidad", color=_INK)
    ax.tick_params(colors=_MUTED, length=0)
    ax.grid(axis="y", color=_GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(_AXIS)

    rol = "Mano" if es_mano else "Pie"
    ax.set_title(f"{rol} · {describe_history(historia)}", loc="left", color=_INK, pad=28)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.0),
        ncol=len(actions),
        frameon=False,
        labelcolor=_INK,
    )
    fig.tight_layout()
    return fig


_ROLE_COLORS = {"mano": "#2a78d6", "pie": "#eb6834"}


def envido_value_figure(by_tantos: dict[str, dict[int, float]], total: dict[str, float]) -> Figure:
    """Barras agrupadas: puntos esperados por mano según los tantos, siendo mano y siendo pie."""
    tantos = sorted(by_tantos["mano"])
    x = _tantos_positions(tantos)
    width = 0.4

    fig = Figure(figsize=(10, 4.2), facecolor=_SURFACE)
    ax = fig.add_subplot()
    ax.set_facecolor(_SURFACE)
    for offset, role in ((-width / 2, "mano"), (width / 2, "pie")):
        ax.bar(
            [p + offset for p in x],
            [by_tantos[role][t] for t in tantos],
            width=width,
            color=_ROLE_COLORS[role],
            edgecolor=_SURFACE,
            linewidth=1,
            label=f"{role.capitalize()} (promedio {total[role]:+.3f})",
            zorder=2,
        )

    ax.axhline(0, color=_AXIS, linewidth=1, zorder=1)
    ax.set_xticks(x, [str(t) for t in tantos])
    ax.set_xlim(min(x) - 0.6, max(x) + 0.6)
    ax.set_xlabel("Tantos", color=_INK)
    ax.set_ylabel("Puntos esperados por mano", color=_INK)
    ax.tick_params(colors=_MUTED, length=0)
    ax.grid(axis="y", color=_GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)

    ax.set_title("Valor esperado del envido según los tantos", loc="left", color=_INK, pad=28)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, labelcolor=_INK)
    fig.tight_layout()
    return fig


def envido_strategy_figures(rows: list[dict], jugador: int = 0) -> dict[str, Figure]:
    """Un gráfico por (es_mano, historia) para `jugador`. Claves: "mano_inicio", "pie_PE", ..."""
    groups = defaultdict(list)
    for r in rows:
        if r["jugador"] == jugador:
            groups[(r["es_mano"], r["historia"])].append(r)

    figures = {}
    for (es_mano, historia), group in sorted(groups.items(), key=lambda kv: (-kv[0][0], kv[0][1])):
        name = f"{'mano' if es_mano else 'pie'}_{historia or 'inicio'}"
        figures[name] = envido_strategy_figure(group, es_mano, historia)
    return figures


# Diverging: blue (column wins) <-> gray (even) <-> red (row wins).
_DIVERGING = LinearSegmentedColormap.from_list(
    "row_wins", ["#104281", "#3987e5", "#f0efec", "#e34948", "#8f1f1f"]
)
_SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4


def crossplay_heatmap_figure(means: np.ndarray, labels: list[str], num_checkpoints: int) -> Figure:
    """Cross-play matrix: the first `num_checkpoints` rows are checkpoints, the rest fixed
    opponents (one of them "aleatorio")."""
    # The scale leaves out the random opponent and the initial (untrained, almost random) network:
    # their lopsided results would wash out the rest.
    weak = [0, labels.index("aleatorio")]
    others = np.delete(np.delete(np.abs(means), weak, axis=0), weak, axis=1)
    limit = max(float(others.max()), 0.05)

    fig = Figure(figsize=(9, 8), facecolor=_SURFACE)
    ax = fig.add_subplot()
    image = ax.imshow(means, cmap=_DIVERGING, vmin=-limit, vmax=limit)
    edge = num_checkpoints - 0.5  # checkpoints | fixed opponents
    ax.axhline(edge, color=_INK, linewidth=1.5)
    ax.axvline(edge, color=_INK, linewidth=1.5)
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7, color=_MUTED)
    ax.set_yticks(range(len(labels)), labels, fontsize=7, color=_MUTED)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xlabel("Column (opponent)", color=_INK)
    ax.set_ylabel("Row", color=_INK)
    ax.set_title(
        "Points per hand of the row against the column (red: the row wins)",
        loc="left",
        color=_INK,
    )
    colorbar = fig.colorbar(image, ax=ax, shrink=0.8, extend="both")
    colorbar.ax.tick_params(colors=_MUTED, length=0)
    colorbar.outline.set_visible(False)
    colorbar.set_label(f"Points per hand (clipped at ±{limit:.2f})", color=_INK)
    fig.tight_layout()
    return fig


def crossplay_strength_figure(updates: list[int], series: dict[str, np.ndarray]) -> Figure:
    """One line per series: points per hand of each checkpoint against something."""
    fig = Figure(figsize=(10, 4.5), facecolor=_SURFACE)
    ax = fig.add_subplot()
    ax.set_facecolor(_SURFACE)
    ax.axhline(0, color=_AXIS, linewidth=1, zorder=1)
    for color, (name, values) in zip(_SERIES, series.items(), strict=False):
        ax.plot(updates, values, color=color, linewidth=2, marker="o", markersize=4, label=name)
    ax.set_xlabel("PPO update of the checkpoint", color=_INK)
    ax.set_ylabel("Points per hand", color=_INK)
    ax.tick_params(colors=_MUTED, length=0)
    ax.grid(axis="y", color=_GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_title("Each checkpoint against...", loc="left", color=_INK, pad=28)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.0),
        ncol=len(series),
        frameon=False,
        labelcolor=_INK,
    )
    fig.tight_layout()
    return fig
