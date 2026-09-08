"""The SVG writer.

A chart cannot be tested by looking at it, so what is asserted here is the set
of properties that make looking at it unnecessary:

* the output is **well-formed XML** and parses;
* every mark and every label lands **inside the canvas** - the one failure a
  reader would notice immediately and the one this project cannot catch by eye
  in CI;
* the data really is **on the axes it claims** - a known value maps to the
  pixel the scale says it should;
* the **accessibility rules hold**: a legend whenever there are two or more
  series, text in ink rather than in the data colour, colliding end labels
  dropped rather than nudged;
* the same input produces **byte-identical output**, which is what makes the
  committed figures reviewable and the experiment reproducible.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import pytest

from app.simulation.charts import (
    REFERENCE_SLOT,
    SERIES_SLOTS,
    ColumnChart,
    ColumnPanel,
    LineChart,
    Panel,
    Series,
    Stack,
    nice_bounds,
    nice_step,
    render_column_chart,
    render_line_chart,
    ticks_between,
)

SVG = "{http://www.w3.org/2000/svg}"


def parse(svg: str) -> ET.Element:
    # S314 is about parsing hostile input. The input here is a string this test
    # generated three lines earlier from literals in this file; there is no
    # untrusted document and no entity for a billion-laughs to expand.
    return ET.fromstring(svg)  # noqa: S314


def viewbox(root: ET.Element) -> tuple[float, float]:
    _, _, width, height = (float(v) for v in str(root.get("viewBox")).split())
    return width, height


def bars(root: ET.Element) -> list[ET.Element]:
    """Only the data rectangles - not the surface, which is also a rect."""
    return [
        node
        for node in root.iter(f"{SVG}rect")
        if re.fullmatch(r"var\(--s\d\)", str(node.get("fill", "")))
    ]


def category_labels(root: ET.Element) -> list[str]:
    """The x-axis labels.  Y ticks share their class and are right-anchored."""
    return [
        str(node.text)
        for node in root.iter(f"{SVG}text")
        if node.get("class") == "a" and node.get("text-anchor") == "middle"
    ]


def numbers(text: str) -> list[float]:
    return [float(match) for match in re.findall(r"-?\d+\.?\d*", text)]


def simple_chart(**kwargs) -> LineChart:
    defaults = dict(
        title="A chart",
        panels=(
            Panel(
                x=(0.0, 1.0, 2.0),
                series=(
                    Series(label="one", values=(1.0, 0.5, 0.0), slot=1),
                    Series(label="two", values=(0.0, 0.5, 1.0), slot=2),
                ),
            ),
        ),
        x_label="questions",
        y_label="error",
    )
    return LineChart(**{**defaults, **kwargs})


class TestTicks:
    @pytest.mark.parametrize(
        ("span", "target", "expected"),
        [
            (1.0, 5, 0.2),
            (0.9, 5, 0.2),
            (10.0, 5, 2.0),
            (100.0, 5, 20.0),
            (7.0, 5, 2.0),
            (4.0, 5, 1.0),
        ],
    )
    def test_the_step_is_a_round_number(self, span, target, expected):
        assert nice_step(span, target) == pytest.approx(expected)

    def test_a_degenerate_span_still_produces_a_usable_step(self):
        assert nice_step(0.0, 5) == 1.0

    def test_the_axis_is_widened_outwards_to_round_numbers(self):
        lower, upper, step = nice_bounds(0.13, 0.87)
        assert lower <= 0.13 and upper >= 0.87
        assert lower == pytest.approx(0.0)
        assert upper == pytest.approx(1.0)
        assert step == pytest.approx(0.2)

    def test_equal_bounds_are_given_a_range(self):
        lower, upper, _ = nice_bounds(2.0, 2.0)
        assert upper > lower

    def test_ticks_do_not_drift(self):
        """Accumulating the step would put 0.30000000000000004 on an axis."""
        assert ticks_between(0.0, 1.0, 0.1) == pytest.approx(
            (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
        )
        assert all(f"{value:.1f}" != "-0.0" for value in ticks_between(-1.0, 1.0, 0.5))

    def test_invalid_arguments_are_rejected(self):
        with pytest.raises(ValueError, match="target must be positive"):
            nice_step(1.0, 0)
        with pytest.raises(ValueError, match="step must be positive"):
            ticks_between(0.0, 1.0, 0.0)
        with pytest.raises(ValueError, match="below lower bound"):
            nice_bounds(1.0, 0.0)
        with pytest.raises(ValueError, match="must be finite"):
            nice_bounds(0.0, float("inf"))


class TestDocument:
    def test_it_is_well_formed_xml(self):
        root = parse(render_line_chart(simple_chart()))
        assert root.tag == f"{SVG}svg"

    def test_it_names_itself_for_a_screen_reader(self):
        root = parse(render_line_chart(simple_chart(title="Mean error")))
        assert root.get("role") == "img"
        assert root.get("aria-label") == "Mean error"
        assert root.find(f"{SVG}title").text == "Mean error"

    def test_it_paints_its_own_surface(self):
        """The file is opened standalone and through an <img>; a transparent
        background would take the host page's colour and break the palette."""
        root = parse(render_line_chart(simple_chart()))
        first = root.find(f"{SVG}rect")
        assert first.get("fill") == "var(--surface)"

    def test_it_declares_both_palettes(self):
        style = parse(render_line_chart(simple_chart())).find(f"{SVG}style").text
        assert "prefers-color-scheme:dark" in style
        for slot, (light, dark) in SERIES_SLOTS.items():
            assert f"--s{slot}:{light}" in style
            assert f"--s{slot}:{dark}" in style

    def test_markup_in_a_label_is_escaped(self):
        svg = render_line_chart(simple_chart(title='A <b> & "quoted" title'))
        assert "<b>" not in svg.replace("<b>", "", 0) or "&lt;b&gt;" in svg
        assert "&amp;" in svg
        parse(svg)  # still parses


class TestLayout:
    def charts(self):
        return [
            simple_chart(),
            simple_chart(subtitle="a subtitle"),
            simple_chart(y_label=""),
            simple_chart(subtitle="", y_label=""),
            simple_chart(reference=(0.5, "threshold")),
            simple_chart(
                panels=tuple(
                    Panel(
                        title=f"panel {index}",
                        x=(0.0, 1.0, 2.0),
                        series=(Series(label="one", values=(1.0, 0.5, 0.0), slot=1),),
                    )
                    for index in range(6)
                ),
                columns=3,
                panel_width=200.0,
                panel_height=140.0,
                legend=(
                    Series(label="one", values=(0.0,), slot=1),
                    Series(label="two", values=(0.0,), slot=2),
                ),
            ),
        ]

    def test_nothing_is_drawn_outside_the_canvas(self):
        for chart in self.charts():
            svg = render_line_chart(chart)
            root = parse(svg)
            width, height = viewbox(root)
            for element in root.iter():
                for attribute in ("x", "y", "x1", "y1", "x2", "y2", "cx", "cy"):
                    value = element.get(attribute)
                    if value is None:
                        continue
                    limit = width if attribute[0] in "xc" and attribute != "cy" else height
                    assert -1.0 <= float(value) <= limit + 1.0, (attribute, value, chart.title)
                for attribute in ("points",):
                    if element.get(attribute) is None:
                        continue
                    values = numbers(str(element.get(attribute)))
                    assert all(-1.0 <= v <= max(width, height) + 1.0 for v in values)

    def test_the_header_lines_never_overlap(self):
        """A y-axis caption at a fixed offset lands on the subtitle."""
        root = parse(render_line_chart(simple_chart(subtitle="a subtitle")))
        baselines = sorted(
            float(str(node.get("y")))
            for node in root.iter(f"{SVG}text")
            if node.get("class") in ("t", "s") and float(str(node.get("y"))) < 120.0
        )
        assert all(
            later - earlier >= 14.0
            for earlier, later in zip(baselines, baselines[1:], strict=False)
        )

    def test_a_value_lands_on_the_pixel_the_scale_says(self):
        chart = simple_chart(y_min=0.0, y_max=1.0, panel_height=200.0)
        root = parse(render_line_chart(chart))
        polylines = [n for n in root.iter(f"{SVG}polyline") if n.get("class") == "ln"]
        points = numbers(str(polylines[0].get("points")))
        top_y, mid_y, bottom_y = points[1], points[3], points[5]
        # 1.0 at the top of a 200px plot, 0.5 halfway down, 0.0 at the baseline.
        assert mid_y - top_y == pytest.approx(100.0)
        assert bottom_y - mid_y == pytest.approx(100.0)

    def test_more_panels_make_a_taller_chart_not_a_denser_one(self):
        one = parse(render_line_chart(simple_chart()))
        many = parse(render_line_chart(self.charts()[-1]))
        assert viewbox(many)[1] > viewbox(one)[1]


class TestAccessibility:
    def test_two_series_always_get_a_legend(self):
        root = parse(render_line_chart(simple_chart()))
        labels = [n.text for n in root.iter(f"{SVG}text") if n.get("class") == "l"]
        assert labels == ["one", "two"]

    def test_a_single_series_gets_no_legend_box(self):
        """One colour; the title already says what is plotted."""
        chart = simple_chart(
            panels=(Panel(x=(0.0, 1.0), series=(Series(label="only", values=(1.0, 0.0), slot=1),)),)
        )
        root = parse(render_line_chart(chart))
        assert [n for n in root.iter(f"{SVG}text") if n.get("class") == "l"] == []

    def test_text_never_wears_the_data_colour(self):
        root = parse(render_line_chart(simple_chart()))
        for node in root.iter(f"{SVG}text"):
            assert node.get("fill") is None
            assert "--s" not in str(node.get("style") or "")

    def test_the_reference_channel_is_muted_and_dashed(self):
        """Ground truth must never look like one of the things being compared."""
        chart = simple_chart(
            panels=(
                Panel(
                    x=(0.0, 1.0),
                    series=(
                        Series(label="policy", values=(1.0, 0.0), slot=1),
                        Series(label="truth", values=(0.4, 0.4), slot=REFERENCE_SLOT, dashed=True),
                    ),
                ),
            )
        )
        root = parse(render_line_chart(chart))
        lines = [n for n in root.iter(f"{SVG}polyline")]
        reference = [n for n in lines if n.get("stroke") == "var(--muted)"]
        assert len(reference) == 1
        assert reference[0].get("stroke-dasharray")

    def test_the_reference_series_gets_no_end_dot(self):
        chart = simple_chart(
            panels=(
                Panel(
                    x=(0.0, 1.0),
                    series=(
                        Series(label="policy", values=(1.0, 0.0), slot=1),
                        Series(label="truth", values=(0.4, 0.4), slot=REFERENCE_SLOT, dashed=True),
                    ),
                ),
            )
        )
        root = parse(render_line_chart(chart))
        assert len(list(root.iter(f"{SVG}circle"))) == 1

    def test_end_dots_carry_a_surface_ring(self):
        root = parse(render_line_chart(simple_chart()))
        for circle in root.iter(f"{SVG}circle"):
            assert circle.get("stroke") == "var(--surface)"
            assert circle.get("stroke-width") == "2"
            assert float(str(circle.get("r"))) >= 4.0

    def test_end_labels_are_dropped_rather_than_nudged_when_they_collide(self):
        """Two labels a pixel apart would have to be moved off their own lines."""
        crowded = simple_chart(
            panels=(
                Panel(
                    x=(0.0, 1.0),
                    series=(
                        Series(label="a", values=(1.0, 0.500), slot=1, end_label="0.500"),
                        Series(label="b", values=(0.0, 0.502), slot=2, end_label="0.502"),
                    ),
                ),
            )
        )
        separated = simple_chart(
            panels=(
                Panel(
                    x=(0.0, 1.0),
                    series=(
                        Series(label="a", values=(1.0, 1.0), slot=1, end_label="1.000"),
                        Series(label="b", values=(0.0, 0.0), slot=2, end_label="0.000"),
                    ),
                ),
            )
        )
        assert "0.502" not in render_line_chart(crowded)
        assert "1.000" in render_line_chart(separated)

    def test_a_band_is_a_wash_rather_than_a_block(self):
        chart = simple_chart(
            panels=(
                Panel(
                    x=(0.0, 1.0),
                    series=(
                        Series(
                            label="median",
                            values=(0.5, 0.4),
                            slot=1,
                            band=((0.3, 0.2), (0.7, 0.6)),
                        ),
                        Series(label="other", values=(0.1, 0.1), slot=2),
                    ),
                ),
            )
        )
        root = parse(render_line_chart(chart))
        polygons = list(root.iter(f"{SVG}polygon"))
        assert len(polygons) == 1
        assert polygons[0].get("fill-opacity") == "0.10"


class TestValidation:
    def test_a_series_must_have_a_value_for_every_x(self):
        with pytest.raises(ValueError, match="has 2 values for 3 x positions"):
            Panel(x=(0.0, 1.0, 2.0), series=(Series(label="a", values=(1.0, 0.0), slot=1),))

    def test_a_band_must_match_its_series(self):
        with pytest.raises(ValueError, match="band of a different length"):
            Series(label="a", values=(1.0, 0.0), slot=1, band=((0.0,), (1.0,)))

    def test_an_unknown_colour_slot_is_rejected(self):
        with pytest.raises(ValueError, match="no colour for slot"):
            Series(label="a", values=(1.0,), slot=99)

    def test_an_empty_series_is_rejected(self):
        with pytest.raises(ValueError, match="has no values"):
            Series(label="a", values=(), slot=1)

    def test_an_empty_panel_is_rejected(self):
        with pytest.raises(ValueError, match="at least one x value"):
            Panel(x=(), series=())

    def test_a_chart_needs_a_panel(self):
        with pytest.raises(ValueError, match="at least one panel"):
            LineChart(title="t", panels=())
        with pytest.raises(ValueError, match="at least one panel"):
            ColumnChart(title="t", panels=())

    def test_a_non_positive_column_count_is_rejected(self):
        with pytest.raises(ValueError, match="columns must be positive"):
            simple_chart(columns=0)
        with pytest.raises(ValueError, match="label_every must be positive"):
            ColumnChart(
                title="t",
                panels=(ColumnPanel(categories=("1",), stacks=()),),
                label_every=0,
            )

    def test_a_stack_must_match_its_categories(self):
        with pytest.raises(ValueError, match="has 1 values for 2 categories"):
            ColumnPanel(categories=("1", "2"), stacks=(Stack(label="a", values=(1.0,), slot=4),))


class TestColumnChart:
    def chart(self) -> ColumnChart:
        return ColumnChart(
            title="Stopping",
            subtitle="why and when",
            panels=(
                ColumnPanel(
                    title="adaptive",
                    categories=("3", "4", "5"),
                    stacks=(
                        Stack(label="budget", values=(0.0, 2.0, 6.0), slot=4),
                        Stack(label="quiet", values=(3.0, 1.0, 0.0), slot=5),
                    ),
                ),
                ColumnPanel(
                    title="random",
                    categories=("3", "4", "5"),
                    stacks=(
                        Stack(label="budget", values=(1.0, 1.0, 5.0), slot=4),
                        Stack(label="quiet", values=(2.0, 2.0, 1.0), slot=5),
                    ),
                ),
            ),
            x_label="questions",
            y_label="sessions",
        )

    def test_it_is_well_formed_and_inside_its_canvas(self):
        root = parse(render_column_chart(self.chart()))
        width, height = viewbox(root)
        for rect in root.iter(f"{SVG}rect"):
            x, y = float(str(rect.get("x", 0))), float(str(rect.get("y", 0)))
            w, h = float(str(rect.get("width"))), float(str(rect.get("height")))
            assert x >= -1.0 and y >= -1.0
            assert x + w <= width + 1.0
            assert y + h <= height + 1.0

    def test_a_zero_segment_draws_nothing(self):
        """An empty category would otherwise get a hairline of pure ink."""
        root = parse(render_column_chart(self.chart()))
        # 4 non-zero segments in the first panel (one budget and one quiet
        # value are zero), 6 in the second, out of 12 declared.
        assert len(bars(root)) == 4 + 6

    def test_the_bar_body_is_capped_rather_than_filling_its_slot(self):
        root = parse(render_column_chart(self.chart()))
        for rect in bars(root):
            assert float(str(rect.get("width"))) <= 24.0

    def test_every_stack_appears_in_the_legend(self):
        root = parse(render_column_chart(self.chart()))
        labels = [n.text for n in root.iter(f"{SVG}text") if n.get("class") == "l"]
        assert labels == ["budget", "quiet"]

    def test_a_column_carries_its_category_label(self):
        assert category_labels(parse(render_column_chart(self.chart()))) == ["3", "4", "5"] * 2

    def test_sparse_axes_can_thin_their_labels(self):
        chart = ColumnChart(
            title="t",
            panels=(
                ColumnPanel(
                    categories=tuple(str(n) for n in range(20)),
                    stacks=(Stack(label="a", values=(1.0,) * 20, slot=4),),
                ),
            ),
            label_every=5,
        )
        root = parse(render_column_chart(chart))
        shown = category_labels(root)
        assert shown == ["0", "5", "10", "15"]


class TestReproducibility:
    def test_the_same_chart_renders_byte_identically(self):
        chart = simple_chart(subtitle="s", reference=(0.5, "half"))
        assert render_line_chart(chart) == render_line_chart(chart)

    def test_no_negative_zero_reaches_the_output(self):
        """The sign of a rounded-away negative is a property of the floating
        point unit, not of the data, and would show up in a diff."""
        chart = simple_chart(
            panels=(
                Panel(
                    x=(0.0, 1.0),
                    series=(
                        Series(label="a", values=(-1e-9, 1.0), slot=1),
                        Series(label="b", values=(0.0, 1.0), slot=2),
                    ),
                ),
            )
        )
        assert "-0.00" not in render_line_chart(chart)

    def test_coordinates_are_written_at_a_fixed_precision(self):
        svg = render_line_chart(simple_chart())
        for value in numbers(str(parse(svg).get("viewBox"))):
            assert value == round(value, 2)
