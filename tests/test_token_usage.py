"""Tests for the TokenUsage widget."""

import pytest
from pydantic_ai import models

from calipso.cmd import CmdNone
from calipso.widgets.token_usage import (
    TokenUsageModel,
    UsageRecorded,
    create_token_usage,
    update,
)

models.ALLOW_MODEL_REQUESTS = False
pytestmark = pytest.mark.anyio


class TestTokenUsageUpdate:
    def test_appends_record(self):
        model = TokenUsageModel()
        msg = UsageRecorded(
            input_tokens=100,
            output_tokens=50,
            cache_read_tokens=10,
            cache_write_tokens=5,
        )
        model, cmd = update(model, msg)
        assert len(model.records) == 1
        assert model.records[0].input_tokens == 100
        assert model.records[0].output_tokens == 50
        assert model.records[0].cache_read_tokens == 10
        assert model.records[0].cache_write_tokens == 5

    def test_increments_request_number(self):
        model = TokenUsageModel()
        msg = UsageRecorded(
            input_tokens=100,
            output_tokens=50,
            cache_read_tokens=0,
            cache_write_tokens=0,
        )
        model, _ = update(model, msg)
        model, _ = update(model, msg)
        assert model.records[0].request_number == 1
        assert model.records[1].request_number == 2

    def test_returns_cmd_none(self):
        model = TokenUsageModel()
        msg = UsageRecorded(
            input_tokens=100,
            output_tokens=50,
            cache_read_tokens=0,
            cache_write_tokens=0,
        )
        _, cmd = update(model, msg)
        assert isinstance(cmd, CmdNone)


class TestTokenUsageViewHtml:
    def test_empty_model(self):
        w = create_token_usage()
        html = w.view_html()
        assert 'id="widget-token-usage"' in html
        assert "No requests yet" in html

    def test_contains_svg_after_data(self):
        w = create_token_usage()
        w.send(
            UsageRecorded(
                input_tokens=500,
                output_tokens=200,
                cache_read_tokens=0,
                cache_write_tokens=0,
            )
        )
        html = w.view_html()
        assert "<svg" in html
        assert "#5b9bd5" in html  # input color
        assert "#ed7d31" in html  # output color

    def test_shows_totals(self):
        w = create_token_usage()
        w.send(
            UsageRecorded(
                input_tokens=1500,
                output_tokens=300,
                cache_read_tokens=0,
                cache_write_tokens=0,
            )
        )
        html = w.view_html()
        assert "1.5K" in html
        assert "1</strong> requests" in html

    def test_shows_cache_stats_when_present(self):
        w = create_token_usage()
        w.send(
            UsageRecorded(
                input_tokens=1000,
                output_tokens=200,
                cache_read_tokens=500,
                cache_write_tokens=100,
            )
        )
        html = w.view_html()
        assert "cache" in html.lower()

    def test_no_cache_stats_when_zero(self):
        w = create_token_usage()
        w.send(
            UsageRecorded(
                input_tokens=1000,
                output_tokens=200,
                cache_read_tokens=0,
                cache_write_tokens=0,
            )
        )
        html = w.view_html()
        assert "cache" not in html.lower()


class TestTokenUsageHandle:
    def test_send_updates_model(self):
        w = create_token_usage()
        w.send(
            UsageRecorded(
                input_tokens=100,
                output_tokens=50,
                cache_read_tokens=0,
                cache_write_tokens=0,
            )
        )
        assert len(w.model.records) == 1
        assert w.model.records[0].request_number == 1

    def test_no_tools(self):
        w = create_token_usage()
        assert list(w.view_tools()) == []

    def test_no_messages(self):
        w = create_token_usage()
        assert list(w.view_messages()) == []
