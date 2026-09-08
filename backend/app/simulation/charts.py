"""A small, deterministic SVG chart writer.  No plotting library, on purpose.

Day 14 needs five or six line charts and one column chart, written to files
that get committed, embedded in Markdown and diffed in review.  Three
requirements decided the approach, and they point away from matplotlib:

**Reproducibility.**  The brief is that running the experiment twice produces
the same plots.  A PNG stamps creation metadata, renders through whatever
FreeType and Agg versions happen to be installed, and is a binary blob in
``git diff``.  The text emitted here is a pure function of the numbers: same
input, byte-identical file, and a review can *read* the change.

**CI installs ``./backend[dev]`` and nothing else.**  Adding matplotlib for six
charts would put ~50 MB of transitive dependency into an environment that
currently has none of it, to draw axes and polylines.

**The charts are simple.**  Lines, a shaded interquartile band, stacked columns,
a legend.  No projections, no statistical fitting, no interactivity.  The
honest amount of code for that is this file, and everything in it is testable
arithmetic rather than a call into a renderer that cannot be asserted on.

What this is not
----------------

Not a plotting library.  There is no autoscaling heuristic beyond nice ticks,
no layout engine, and no support for a chart form Day 14 does not draw.  If a
later phase needs a different form, the right move is to add it here
deliberately or to take the dependency then - not to bend these two functions.

Design rules, and where they come from
--------------------------------------

The visual specification is the project's data-visualisation guidance, applied
rather than reinvented:

* **Colour by identity, in a fixed slot order** (:data:`SERIES_SLOTS`).  A
  policy keeps its hue whichever chart it appears in, and a chart that drops a
  series never repaints the survivors.  The first three slots are the ones
  validated for colour-vision deficiency against *every* pair rather than only
  adjacent ones - which is why the three policies use exactly those three.
* **Text never wears the data colour.**  Titles, ticks, labels and legend text
  are ink tokens; identity comes from the coloured line-key beside the label.
  A legend is always drawn for two or more series, so identity is never
  colour-alone.
* **Thin marks, recessive chrome.**  2px lines, hairline solid gridlines one
  step off the surface, an interquartile band as a 10%-opacity wash.
* **Direct labels are selective, and never stacked.**  End labels are drawn
  only when the series separate at the right edge; when two would land within
  :data:`_MIN_LABEL_GAP` pixels the panel drops them all and lets the legend
  carry identity, because nudging a label off its own line reads as noise.
* **Dark mode is selected, not inverted.**  Both palettes are declared as CSS
  custom properties and swapped by ``prefers-color-scheme``; the dark values
  are the guidance's own dark steps, not a programmatic lightening.

Every accompanying figure also ships its numbers as a Markdown table in the
report, which is the table view that a colour-only reading would otherwise
gate.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# palette
# ---------------------------------------------------------------------------

#: ``slot -> (light, dark)``.  Slots 1-3 are validated across every pair in both
#: modes and are what a three-series chart must use; 4-8 clear the adjacent-pair
#: gate and are for stacked columns, where only neighbours touch.
SERIES_SLOTS: dict[int, tuple[str, str]] = {
    1: ("#2a78d6", "#3987e5"),  # blue
    2: ("#eb6834", "#d95926"),  # orange
    3: ("#1baf7a", "#199e70"),  # aqua
    4: ("#eda100", "#c98500"),  # yellow
    5: ("#e87ba4", "#d55181"),  # magenta
    6: ("#008300", "#008300"),  # green
    7: ("#4a3aa7", "#9085e9"),  # violet
    8: ("#e34948", "#e66767"),  # red
}

#: Slot 0 is not a series.  It is the reference channel - ground truth, a
#: threshold line - drawn in muted ink and dashed, so it can never be mistaken
#: for one of the things being compared.
REFERENCE_SLOT = 0

_INK: tuple[str, str] = ("#0b0b0b", "#ffffff")
_INK_2: tuple[str, str] = ("#52514e", "#c3c2b7")
_MUTED: tuple[str, str] = ("#898781", "#898781")
_GRID: tuple[str, str] = ("#e1e0d9", "#2c2c2a")
_AXIS: tuple[str, str] = ("#c3c2b7", "#383835")
_SURFACE: tuple[str, str] = ("#fcfcfb", "#1a1a19")

_FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'

# ---------------------------------------------------------------------------
# layout constants
# ---------------------------------------------------------------------------

_PAD = 20
_TITLE_Y = 28
#: Baseline-to-baseline spacing for the stacked header lines.
_HEADER_LINE_H = 20.0
#: Air between the last header line and the top of the first panel.
_HEADER_GAP = 16.0
_PANEL_TITLE_H = 22
_Y_GUTTER = 46
_X_AXIS_H = 26
_PANEL_GAP_X = 26
_PANEL_GAP_Y = 40
_LEGEND_ROW_H = 22
_LEGEND_KEY_W = 18
_AXIS_LABEL_H = 20
#: Two end labels closer than this collide; the panel then drops all of them.
_MIN_LABEL_GAP = 13.0


def _f(value: float) -> str:
    """A coordinate, at fixed precision so the output is byte-stable.

    ``-0.00`` is normalised to ``0.00``: the sign of a rounded-away negative is
    a property of the floating-point unit, not of the data, and it would make
    two identical runs differ in a diff.
    """
    text = f"{value:.2f}"
    return "0.00" if text == "-0.00" else text


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


# ---------------------------------------------------------------------------
# ticks
# ---------------------------------------------------------------------------


def nice_step(span: float, target: int) -> float:
    """A round tick interval - 1, 2 or 5 times a power of ten.

    Round numbers are not decoration: an axis reading 0, 0.2, 0.4 is read at a
    glance and one reading 0, 0.173, 0.346 is not, and the guidance makes ticks
    the channel that carries every value not directly labelled.
    """
    if target <= 0:
        raise ValueError(f"target must be positive, got {target!r}")
    if span <= 0.0:
        return 1.0
    rough = span / target
    magnitude = 10.0 ** math.floor(math.log10(rough))
    for multiple in (1.0, 2.0, 5.0, 10.0):
        if rough <= multiple * magnitude:
            return multiple * magnitude
    return 10.0 * magnitude  # pragma: no cover - the loop above is exhaustive


def nice_bounds(low: float, high: float, target: int = 5) -> tuple[float, float, float]:
    """``(lower, upper, step)`` - the axis extended outwards to round numbers."""
    if not (math.isfinite(low) and math.isfinite(high)):
        raise ValueError(f"axis bounds must be finite, got ({low!r}, {high!r})")
    if high < low:
        raise ValueError(f"upper bound {high!r} is below lower bound {low!r}")
    if high == low:
        high = low + 1.0
    step = nice_step(high - low, target)
    lower = math.floor(low / step) * step
    upper = math.ceil(high / step) * step
    return lower, upper, step


def ticks_between(lower: float, upper: float, step: float) -> tuple[float, ...]:
    """Every tick from ``lower`` to ``upper`` inclusive, free of drift.

    Multiplying the index by the step rather than accumulating it keeps
    ``0.30000000000000004`` out of an axis label, and keeps two runs identical.
    """
    if step <= 0.0:
        raise ValueError(f"step must be positive, got {step!r}")
    count = int(round((upper - lower) / step))
    return tuple(lower + step * index for index in range(count + 1))


# ---------------------------------------------------------------------------
# the pieces a chart is made of
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Series:
    """One line, and optionally the band around it.

    ``band`` is ``(lower, upper)`` - the interquartile range, drawn as a 10%
    wash of the series hue.  It is not an error bar and is not a confidence
    interval: it is where the middle half of the candidates are, which is the
    quantity Day 13 showed to be several times larger than the gaps between the
    policies.
    """

    label: str
    values: tuple[float, ...]
    slot: int
    dashed: bool = False
    band: tuple[tuple[float, ...], tuple[float, ...]] | None = None
    #: Draw the final value beside the line end.  Suppressed automatically when
    #: two labels in the same panel would collide.
    end_label: str | None = None

    def __post_init__(self) -> None:
        if self.slot not in SERIES_SLOTS and self.slot != REFERENCE_SLOT:
            raise ValueError(f"no colour for slot {self.slot!r}")
        if not self.values:
            raise ValueError(f"series {self.label!r} has no values")
        if self.band is not None:
            lower, upper = self.band
            if len(lower) != len(self.values) or len(upper) != len(self.values):
                raise ValueError(f"series {self.label!r} has a band of a different length")


@dataclass(frozen=True, slots=True)
class Panel:
    """One set of axes.  Several panels of the same shape are small multiples."""

    x: tuple[float, ...]
    series: tuple[Series, ...]
    title: str = ""

    def __post_init__(self) -> None:
        if not self.x:
            raise ValueError("a panel needs at least one x value")
        for entry in self.series:
            if len(entry.values) != len(self.x):
                raise ValueError(
                    f"series {entry.label!r} has {len(entry.values)} values "
                    f"for {len(self.x)} x positions"
                )


@dataclass(frozen=True, slots=True)
class Stack:
    """One segment of a stacked column - a category that has a count per bucket."""

    label: str
    values: tuple[float, ...]
    slot: int


@dataclass(frozen=True, slots=True)
class ColumnPanel:
    """A stacked column chart's axes: one column per bucket."""

    categories: tuple[str, ...]
    stacks: tuple[Stack, ...]
    title: str = ""

    def __post_init__(self) -> None:
        for stack in self.stacks:
            if len(stack.values) != len(self.categories):
                raise ValueError(
                    f"stack {stack.label!r} has {len(stack.values)} values "
                    f"for {len(self.categories)} categories"
                )


@dataclass(frozen=True, slots=True)
class _Rect:
    x: float
    y: float
    width: float
    height: float


# ---------------------------------------------------------------------------
# the document
# ---------------------------------------------------------------------------


def _style() -> str:
    """The palette, declared once as custom properties and swapped for dark.

    Written into every file rather than shared, because each ``.svg`` is opened
    standalone (and through an ``<img>`` in Markdown, which does not inherit the
    host page's stylesheet).
    """

    def block(index: int) -> str:
        parts = [
            f"--surface:{_SURFACE[index]}",
            f"--ink:{_INK[index]}",
            f"--ink-2:{_INK_2[index]}",
            f"--muted:{_MUTED[index]}",
            f"--grid:{_GRID[index]}",
            f"--axis:{_AXIS[index]}",
        ]
        parts += [f"--s{slot}:{hexes[index]}" for slot, hexes in sorted(SERIES_SLOTS.items())]
        return ";".join(parts) + ";"

    return (
        "<style>\n"
        f":root{{{block(0)}}}\n"
        f"@media (prefers-color-scheme:dark){{:root{{{block(1)}}}}}\n"
        f"text{{font-family:{_FONT};fill:var(--ink-2)}}\n"
        ".t{font-size:15px;font-weight:600;fill:var(--ink)}\n"
        ".s{font-size:12px;fill:var(--muted)}\n"
        ".p{font-size:12px;font-weight:600;fill:var(--ink)}\n"
        ".a{font-size:11px;fill:var(--muted);font-variant-numeric:tabular-nums}\n"
        ".l{font-size:12px;fill:var(--ink-2)}\n"
        ".v{font-size:11px;font-weight:600;fill:var(--ink-2);"
        "font-variant-numeric:tabular-nums}\n"
        ".g{stroke:var(--grid);stroke-width:1;fill:none}\n"
        ".x{stroke:var(--axis);stroke-width:1;fill:none}\n"
        ".ln{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}\n"
        "</style>"
    )


def _colour(slot: int) -> str:
    return "var(--muted)" if slot == REFERENCE_SLOT else f"var(--s{slot})"


def _legend(
    series: Sequence[Series],
    *,
    x: float,
    y: float,
    width: float,
) -> tuple[list[str], float]:
    """Line-key plus text, wrapped into rows.  Returns the parts and the height.

    Always drawn for two or more series.  The key is a short stroke in the
    series colour and the text is ink, which is the whole point: identity
    survives greyscale, a colour-vision deficiency and a printout.
    """
    if len(series) < 2:
        return [], 0.0
    # A rough per-character advance is enough: the labels are short and the
    # only thing being avoided is two keys overlapping.
    widths = [_LEGEND_KEY_W + 8.0 + 6.6 * len(entry.label) + 22.0 for entry in series]
    parts: list[str] = []
    cursor_x, cursor_y, rows = x, y, 1
    for entry, item_width in zip(series, widths, strict=True):
        if cursor_x > x and cursor_x + item_width > x + width:
            cursor_x, rows = x, rows + 1
            cursor_y += _LEGEND_ROW_H
        mid = cursor_y + 4.0
        dash = ' stroke-dasharray="5 4"' if entry.dashed else ""
        parts.append(
            f'<line class="ln" x1="{_f(cursor_x)}" y1="{_f(mid)}" '
            f'x2="{_f(cursor_x + _LEGEND_KEY_W)}" y2="{_f(mid)}" '
            f'stroke="{_colour(entry.slot)}"{dash}/>'
        )
        parts.append(
            f'<text class="l" x="{_f(cursor_x + _LEGEND_KEY_W + 8.0)}" '
            f'y="{_f(mid + 4.0)}">{_escape(entry.label)}</text>'
        )
        cursor_x += item_width
    return parts, rows * _LEGEND_ROW_H


def _header(title: str, subtitle: str, y_label: str) -> tuple[list[str], float]:
    """Title, optional subtitle, optional y-axis caption - and where the plot starts.

    Stacked, and the plot top is *returned* rather than assumed.  An axis
    caption placed at a fixed offset lands on top of the subtitle exactly when
    a subtitle is present, which is most of the time; making the caller ask
    where the header ended is what makes that impossible rather than merely
    unlikely.
    """
    parts = [f'<text class="t" x="{_f(_PAD)}" y="{_f(_TITLE_Y)}">{_escape(title)}</text>']
    cursor = float(_TITLE_Y)
    for line in (subtitle, y_label):
        if not line:
            continue
        cursor += _HEADER_LINE_H
        parts.append(f'<text class="s" x="{_f(_PAD)}" y="{_f(cursor)}">{_escape(line)}</text>')
    return parts, cursor + _HEADER_GAP


def _document(
    *,
    width: float,
    height: float,
    title: str,
    body: Sequence[str],
) -> str:
    opening = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_f(width)} {_f(height)}" '
        f'width="{_f(width)}" height="{_f(height)}" role="img" '
        f'aria-label="{_escape(title)}">',
        f"<title>{_escape(title)}</title>",
        _style(),
        f'<rect width="{_f(width)}" height="{_f(height)}" fill="var(--surface)"/>',
    ]
    return "\n".join([*opening, *body, "</svg>"]) + "\n"


# ---------------------------------------------------------------------------
# line charts
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LineChart:
    """Everything one line chart (or one grid of them) needs.

    A value object rather than a pile of keyword arguments so that a figure can
    be built, asserted on, and only then rendered - the tests check the geometry
    of a chart without parsing its SVG.
    """

    title: str
    panels: tuple[Panel, ...]
    x_label: str = ""
    y_label: str = ""
    subtitle: str = ""
    columns: int = 1
    panel_width: float = 660.0
    panel_height: float = 300.0
    #: Forced axis bounds.  Small multiples must share a scale or the reader
    #: compares shapes that are not comparable, so this is set explicitly rather
    #: than derived per panel.
    y_min: float | None = None
    y_max: float | None = None
    y_format: str = "{:.2f}"
    x_tick_step: int = 5
    reference: tuple[float, str] | None = None
    #: Legend entries, in slot order.  Taken from the first panel when omitted;
    #: given explicitly when panels hold different subsets of the same series.
    legend: tuple[Series, ...] = field(default=())

    def __post_init__(self) -> None:
        if not self.panels:
            raise ValueError("a chart needs at least one panel")
        if self.columns < 1:
            raise ValueError(f"columns must be positive, got {self.columns!r}")


def _bounds(chart: LineChart) -> tuple[float, float, float]:
    lows: list[float] = []
    highs: list[float] = []
    for panel in chart.panels:
        for entry in panel.series:
            lows.append(min(entry.values))
            highs.append(max(entry.values))
            if entry.band is not None:
                lows.append(min(entry.band[0]))
                highs.append(max(entry.band[1]))
    if chart.reference is not None:
        lows.append(chart.reference[0])
        highs.append(chart.reference[0])
    low = min(lows) if chart.y_min is None else chart.y_min
    high = max(highs) if chart.y_max is None else chart.y_max
    lower, upper, step = nice_bounds(low, high)
    if chart.y_min is not None:
        lower = chart.y_min
    if chart.y_max is not None:
        upper = chart.y_max
    return lower, upper, step


def _plot_panel(
    panel: Panel,
    rect: _Rect,
    *,
    chart: LineChart,
    lower: float,
    upper: float,
    step: float,
) -> list[str]:
    parts: list[str] = []
    if panel.title:
        parts.append(
            f'<text class="p" x="{_f(rect.x)}" y="{_f(rect.y - 8.0)}">'
            f"{_escape(panel.title)}</text>"
        )

    span = upper - lower or 1.0
    x_low, x_high = panel.x[0], panel.x[-1]
    x_span = (x_high - x_low) or 1.0

    def sx(value: float) -> float:
        return rect.x + rect.width * (value - x_low) / x_span

    def sy(value: float) -> float:
        return rect.y + rect.height * (1.0 - (value - lower) / span)

    # gridlines and y ticks
    for tick in ticks_between(lower, upper, step):
        y = sy(tick)
        parts.append(
            f'<line class="g" x1="{_f(rect.x)}" y1="{_f(y)}" '
            f'x2="{_f(rect.x + rect.width)}" y2="{_f(y)}"/>'
        )
        parts.append(
            f'<text class="a" x="{_f(rect.x - 8.0)}" y="{_f(y + 3.5)}" '
            f'text-anchor="end">{_escape(chart.y_format.format(tick))}</text>'
        )

    # x axis
    parts.append(
        f'<line class="x" x1="{_f(rect.x)}" y1="{_f(rect.y + rect.height)}" '
        f'x2="{_f(rect.x + rect.width)}" y2="{_f(rect.y + rect.height)}"/>'
    )
    tick = int(x_low)
    while tick <= int(x_high):
        x = sx(tick)
        parts.append(
            f'<text class="a" x="{_f(x)}" y="{_f(rect.y + rect.height + 16.0)}" '
            f'text-anchor="middle">{tick}</text>'
        )
        tick += chart.x_tick_step

    if chart.reference is not None:
        value, label = chart.reference
        y = sy(value)
        parts.append(
            f'<line x1="{_f(rect.x)}" y1="{_f(y)}" x2="{_f(rect.x + rect.width)}" '
            f'y2="{_f(y)}" stroke="var(--muted)" stroke-width="1" stroke-dasharray="4 4"/>'
        )
        parts.append(
            f'<text class="a" x="{_f(rect.x + rect.width - 2.0)}" y="{_f(y - 5.0)}" '
            f'text-anchor="end">{_escape(label)}</text>'
        )

    # bands first, so lines sit on top of every wash
    for entry in panel.series:
        if entry.band is None:
            continue
        low_values, high_values = entry.band
        forward = " ".join(
            f"{_f(sx(x))},{_f(sy(v))}" for x, v in zip(panel.x, high_values, strict=True)
        )
        backward = " ".join(
            f"{_f(sx(x))},{_f(sy(v))}"
            for x, v in zip(reversed(panel.x), reversed(low_values), strict=True)
        )
        parts.append(
            f'<polygon points="{forward} {backward}" fill="{_colour(entry.slot)}" '
            f'fill-opacity="0.10" stroke="none"/>'
        )

    for entry in panel.series:
        points = " ".join(
            f"{_f(sx(x))},{_f(sy(v))}" for x, v in zip(panel.x, entry.values, strict=True)
        )
        dash = ' stroke-dasharray="5 4"' if entry.dashed else ""
        parts.append(
            f'<polyline class="ln" points="{points}" stroke="{_colour(entry.slot)}"{dash}/>'
        )

    parts.extend(_end_marks(panel, sx=sx, sy=sy, right=rect.x + rect.width))
    return parts


def _end_marks(
    panel: Panel,
    *,
    sx: Callable[[float], float],
    sy: Callable[[float], float],
    right: float,
) -> list[str]:
    """End dots, and end labels when they do not collide.

    The guidance is explicit that converging end labels must not be nudged apart
    - a label detached from its line is worse than no label - so when any two
    would land within :data:`_MIN_LABEL_GAP` the whole panel falls back to the
    legend.  Deciding per panel rather than per label keeps a chart from showing
    two of three values, which reads as a claim about those two.
    """
    labelled = [entry for entry in panel.series if entry.end_label is not None]
    positions = sorted(sy(entry.values[-1]) for entry in labelled)
    crowded = any(
        later - earlier < _MIN_LABEL_GAP
        for earlier, later in zip(positions, positions[1:], strict=False)
    )

    parts: list[str] = []
    for entry in panel.series:
        if entry.slot == REFERENCE_SLOT:
            continue
        x, y = sx(panel.x[-1]), sy(entry.values[-1])
        # A 2px ring in the surface colour, so a dot stays legible where two
        # lines cross - the guidance's surface ring, not a border.
        parts.append(
            f'<circle cx="{_f(x)}" cy="{_f(y)}" r="4" fill="{_colour(entry.slot)}" '
            f'stroke="var(--surface)" stroke-width="2"/>'
        )
    if not crowded:
        for entry in labelled:
            x, y = sx(panel.x[-1]), sy(entry.values[-1])
            anchor = "end" if x > right - 34.0 else "start"
            offset = -9.0 if anchor == "end" else 9.0
            parts.append(
                f'<text class="v" x="{_f(x + offset)}" y="{_f(y - 9.0)}" '
                f'text-anchor="{anchor}">{_escape(str(entry.end_label))}</text>'
            )
    return parts


def render_line_chart(chart: LineChart) -> str:
    """One chart, or a grid of small multiples sharing one scale."""
    lower, upper, step = _bounds(chart)
    rows = math.ceil(len(chart.panels) / chart.columns)
    has_panel_titles = any(panel.title for panel in chart.panels)
    panel_title_h = _PANEL_TITLE_H if has_panel_titles else 0.0

    cell_w = _Y_GUTTER + chart.panel_width
    cell_h = panel_title_h + chart.panel_height + _X_AXIS_H
    grid_w = chart.columns * cell_w + (chart.columns - 1) * _PANEL_GAP_X
    width = grid_w + 2 * _PAD
    grid_h = rows * cell_h + (rows - 1) * _PANEL_GAP_Y

    body, body_top = _header(chart.title, chart.subtitle, chart.y_label)
    for index, panel in enumerate(chart.panels):
        row, column = divmod(index, chart.columns)
        origin_x = _PAD + column * (cell_w + _PANEL_GAP_X)
        origin_y = body_top + row * (cell_h + _PANEL_GAP_Y)
        rect = _Rect(
            x=origin_x + _Y_GUTTER,
            y=origin_y + panel_title_h,
            width=chart.panel_width,
            height=chart.panel_height,
        )
        body.extend(_plot_panel(panel, rect, chart=chart, lower=lower, upper=upper, step=step))

    cursor = body_top + grid_h + _AXIS_LABEL_H
    if chart.x_label:
        body.append(
            f'<text class="s" x="{_f(_PAD + grid_w / 2.0)}" y="{_f(cursor)}" '
            f'text-anchor="middle">{_escape(chart.x_label)}</text>'
        )
        cursor += _AXIS_LABEL_H

    entries = chart.legend if chart.legend else chart.panels[0].series
    legend, legend_h = _legend(entries, x=_PAD, y=cursor, width=grid_w)
    body.extend(legend)

    return _document(
        width=width,
        height=cursor + legend_h + _PAD,
        title=chart.title,
        body=body,
    )


# ---------------------------------------------------------------------------
# stacked column chart
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ColumnChart:
    """Stacked columns - counts split by a category, across buckets."""

    title: str
    panels: tuple[ColumnPanel, ...]
    x_label: str = ""
    y_label: str = ""
    subtitle: str = ""
    columns: int = 1
    panel_width: float = 660.0
    panel_height: float = 200.0
    y_max: float | None = None
    #: Show every ``n``-th category label, so a 21-bucket axis stays readable.
    label_every: int = 1

    def __post_init__(self) -> None:
        if not self.panels:
            raise ValueError("a chart needs at least one panel")
        if self.label_every < 1:
            raise ValueError(f"label_every must be positive, got {self.label_every!r}")


#: Column bodies are capped rather than filling their slot; the leftover is air.
_MAX_BAR_W = 24.0
#: A 2px gap in the surface colour separates touching stacked segments.
_SEGMENT_GAP = 2.0


def render_column_chart(chart: ColumnChart) -> str:
    """Stacked columns, one panel per group, sharing one scale."""
    totals = [
        sum(stack.values[index] for stack in panel.stacks)
        for panel in chart.panels
        for index in range(len(panel.categories))
    ]
    lower, upper, step = nice_bounds(0.0, max(totals) if totals else 1.0)
    if chart.y_max is not None:
        upper = chart.y_max
    span = upper - lower or 1.0

    rows = math.ceil(len(chart.panels) / chart.columns)
    has_panel_titles = any(panel.title for panel in chart.panels)
    panel_title_h = _PANEL_TITLE_H if has_panel_titles else 0.0
    cell_w = _Y_GUTTER + chart.panel_width
    cell_h = panel_title_h + chart.panel_height + _X_AXIS_H
    grid_w = chart.columns * cell_w + (chart.columns - 1) * _PANEL_GAP_X
    grid_h = rows * cell_h + (rows - 1) * _PANEL_GAP_Y

    body, body_top = _header(chart.title, chart.subtitle, chart.y_label)
    for index, panel in enumerate(chart.panels):
        row, column = divmod(index, chart.columns)
        rect = _Rect(
            x=_PAD + column * (cell_w + _PANEL_GAP_X) + _Y_GUTTER,
            y=body_top + row * (cell_h + _PANEL_GAP_Y) + panel_title_h,
            width=chart.panel_width,
            height=chart.panel_height,
        )
        body.extend(
            _plot_columns(panel, rect, chart=chart, lower=lower, upper=upper, step=step, span=span)
        )

    cursor = body_top + grid_h + _AXIS_LABEL_H
    if chart.x_label:
        body.append(
            f'<text class="s" x="{_f(_PAD + grid_w / 2.0)}" y="{_f(cursor)}" '
            f'text-anchor="middle">{_escape(chart.x_label)}</text>'
        )
        cursor += _AXIS_LABEL_H

    keys = tuple(
        Series(label=stack.label, values=(0.0,), slot=stack.slot)
        for stack in chart.panels[0].stacks
    )
    legend, legend_h = _legend(keys, x=_PAD, y=cursor, width=grid_w)
    body.extend(legend)

    return _document(
        width=grid_w + 2 * _PAD,
        height=cursor + legend_h + _PAD,
        title=chart.title,
        body=body,
    )


def _plot_columns(
    panel: ColumnPanel,
    rect: _Rect,
    *,
    chart: ColumnChart,
    lower: float,
    upper: float,
    step: float,
    span: float,
) -> list[str]:
    parts: list[str] = []
    if panel.title:
        parts.append(
            f'<text class="p" x="{_f(rect.x)}" y="{_f(rect.y - 8.0)}">'
            f"{_escape(panel.title)}</text>"
        )
    for tick in ticks_between(lower, upper, step):
        y = rect.y + rect.height * (1.0 - (tick - lower) / span)
        parts.append(
            f'<line class="g" x1="{_f(rect.x)}" y1="{_f(y)}" '
            f'x2="{_f(rect.x + rect.width)}" y2="{_f(y)}"/>'
        )
        parts.append(
            f'<text class="a" x="{_f(rect.x - 8.0)}" y="{_f(y + 3.5)}" '
            f'text-anchor="end">{tick:.0f}</text>'
        )
    parts.append(
        f'<line class="x" x1="{_f(rect.x)}" y1="{_f(rect.y + rect.height)}" '
        f'x2="{_f(rect.x + rect.width)}" y2="{_f(rect.y + rect.height)}"/>'
    )

    slot_w = rect.width / max(1, len(panel.categories))
    bar_w = min(_MAX_BAR_W, slot_w * 0.7)
    baseline = rect.y + rect.height
    for index, category in enumerate(panel.categories):
        centre = rect.x + slot_w * (index + 0.5)
        cursor = baseline
        for stack in panel.stacks:
            value = stack.values[index]
            if value <= 0.0:
                continue
            height = rect.height * value / span
            top = cursor - height
            # The 2px surface gap belongs *between* segments, so it comes off
            # the bottom of every segment except the one sitting on the axis.
            gap = 0.0 if cursor >= baseline else _SEGMENT_GAP
            drawn = max(0.0, height - gap)
            parts.append(
                f'<rect x="{_f(centre - bar_w / 2.0)}" y="{_f(top)}" '
                f'width="{_f(bar_w)}" height="{_f(drawn)}" '
                f'fill="{_colour(stack.slot)}"/>'
            )
            cursor = top
        if index % chart.label_every == 0:
            parts.append(
                f'<text class="a" x="{_f(centre)}" y="{_f(baseline + 16.0)}" '
                f'text-anchor="middle">{_escape(category)}</text>'
            )
    return parts


__all__ = [
    "REFERENCE_SLOT",
    "SERIES_SLOTS",
    "ColumnChart",
    "ColumnPanel",
    "LineChart",
    "Panel",
    "Series",
    "Stack",
    "nice_bounds",
    "nice_step",
    "render_column_chart",
    "render_line_chart",
    "ticks_between",
]
