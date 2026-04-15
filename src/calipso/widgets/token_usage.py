"""Token usage widget — displays input/output tokens per request as a bar chart."""

from dataclasses import dataclass, field

from calipso.cmd import Cmd, none
from calipso.widget import WidgetHandle, create_widget

# --- Model ---

MAX_VISIBLE = 20


@dataclass
class TokenRecord:
    request_number: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int


@dataclass
class TokenUsageModel:
    records: list[TokenRecord] = field(default_factory=list)


# --- Messages ---


@dataclass(frozen=True)
class UsageRecorded:
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int


TokenUsageMsg = UsageRecorded


# --- Update (pure) ---


def update(model: TokenUsageModel, msg: TokenUsageMsg) -> tuple[TokenUsageModel, Cmd]:
    match msg:
        case UsageRecorded(
            input_tokens=inp,
            output_tokens=out,
            cache_read_tokens=cr,
            cache_write_tokens=cw,
        ):
            model.records.append(
                TokenRecord(
                    request_number=len(model.records) + 1,
                    input_tokens=inp,
                    output_tokens=out,
                    cache_read_tokens=cr,
                    cache_write_tokens=cw,
                )
            )
            return model, none


# --- Views ---


def _format_tokens(n: int) -> str:
    """Format token count with K/M suffix."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def view_html(model: TokenUsageModel) -> str:
    if not model.records:
        return (
            '<div id="widget-token-usage" class="widget">'
            "<h3>Token Usage</h3>"
            "<em>No requests yet</em>"
            "</div>"
        )

    visible = model.records[-MAX_VISIBLE:]

    # Totals across all records
    total_in = sum(r.input_tokens for r in model.records)
    total_out = sum(r.output_tokens for r in model.records)
    total_cache_r = sum(r.cache_read_tokens for r in model.records)
    total_cache_w = sum(r.cache_write_tokens for r in model.records)

    # Chart dimensions
    chart_w = 280
    chart_h = 100
    n = len(visible)
    bar_w = max(4, chart_w // n - 2)
    gap = 2

    max_total = max(r.input_tokens + r.output_tokens for r in visible)
    if max_total == 0:
        max_total = 1

    bars = []
    for i, rec in enumerate(visible):
        x = i * (bar_w + gap)
        inp_h = rec.input_tokens / max_total * chart_h
        out_h = rec.output_tokens / max_total * chart_h
        inp_y = chart_h - inp_h - out_h
        out_y = chart_h - out_h

        tooltip = (
            f"#{rec.request_number}: "
            f"in={_format_tokens(rec.input_tokens)} "
            f"out={_format_tokens(rec.output_tokens)}"
        )
        bars.append(
            f'<rect x="{x}" y="{inp_y:.1f}" width="{bar_w}" '
            f'height="{inp_h:.1f}" fill="#5b9bd5" rx="1">'
            f"<title>{tooltip}</title></rect>"
            f'<rect x="{x}" y="{out_y:.1f}" width="{bar_w}" '
            f'height="{out_h:.1f}" fill="#ed7d31" rx="1">'
            f"<title>{tooltip}</title></rect>"
        )

    svg_w = n * (bar_w + gap)
    svg = (
        f'<svg width="{svg_w}" height="{chart_h}" '
        f'viewBox="0 0 {svg_w} {chart_h}" '
        'style="display:block;margin:8px 0">'
        f"{''.join(bars)}"
        "</svg>"
    )

    legend = (
        '<div style="font-size:0.75em;color:#888;margin-top:4px">'
        '<span style="color:#5b9bd5">&#9632;</span> input '
        '<span style="color:#ed7d31">&#9632;</span> output'
        "</div>"
    )

    summary = (
        '<div style="font-size:0.8em;margin-top:4px">'
        f"<strong>{len(model.records)}</strong> requests &middot; "
        f"in: {_format_tokens(total_in)} &middot; "
        f"out: {_format_tokens(total_out)}"
    )
    if total_cache_r or total_cache_w:
        summary += (
            f" &middot; cache r: {_format_tokens(total_cache_r)}"
            f" / w: {_format_tokens(total_cache_w)}"
        )
    summary += "</div>"

    return (
        '<div id="widget-token-usage" class="widget">'
        "<h3>Token Usage</h3>"
        f"{svg}{legend}{summary}"
        "</div>"
    )


# --- Factory ---


def create_token_usage() -> WidgetHandle:
    return create_widget(
        id="widget-token-usage",
        model=TokenUsageModel(),
        update=update,
        view_html=view_html,
    )
