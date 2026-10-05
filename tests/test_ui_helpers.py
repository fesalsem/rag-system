"""
tests/test_ui_helpers.py - Tests for the pure UI helpers.

safe_markdown is the security boundary in this app. A model's answer and an
uploaded file's name are interpolated into markup that Streamlit renders with
raw HTML enabled, and Streamlit ships no sanitiser. The security tests here
assert the output never carries an attacker-controlled tag, and prove a stronger
global invariant: every '<' in any output begins one of the small set of tags
this module emits itself, so no attribute (an event handler included) can ride
along on one.
"""

import re

import pytest

from ui_helpers import (
    GENERIC_ERROR,
    escape,
    safe_markdown,
    source_label,
    source_labels,
)

DASH = "—"  # the separator source_label emits between file name and page

# The only tags this module ever produces. safe_markdown escapes its input first,
# so any other tag in the output would mean the escaping failed.
_ALLOWED_TAGS = ("strong", "em", "code", "ul", "li", "br")
_ALLOWED_TAG_RE = re.compile(r"</?(?:%s)>" % "|".join(_ALLOWED_TAGS))
_ANY_TAG_RE = re.compile(r"<[^>]*>")


def assert_only_allowlisted_markup(output: str) -> None:
    """Every tag, and every literal '<', must be one this module emits."""
    for tag in _ANY_TAG_RE.findall(output):
        assert _ALLOWED_TAG_RE.fullmatch(tag), (
            f"output contains markup this module does not emit: {tag!r} "
            f"in {output!r}"
        )
    for index, char in enumerate(output):
        if char == "<":
            assert _ALLOWED_TAG_RE.match(output, index), (
                f"unescaped '<' at {index} in {output!r}"
            )


class TestEscape:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("<", "&lt;"),
            (">", "&gt;"),
            ("&", "&amp;"),
            ('"', "&quot;"),
            ("'", "&#x27;"),
        ],
    )
    def test_neutralises_each_dangerous_character(self, raw, expected):
        assert escape(raw) == expected

    def test_none_becomes_empty_string(self):
        assert escape(None) == ""

    def test_stringifies_non_strings(self):
        assert escape(42) == "42"

    def test_whole_tag_is_neutralised(self):
        out = escape('<script>alert("x") & \'y\'</script>')
        assert "<script" not in out
        assert "&lt;script&gt;" in out


class TestSafeMarkdownSecurity:
    HOSTILE = [
        '<iframe srcdoc="<script>alert(1)</script>"></iframe>',
        '<img src=x onerror="alert(1)">',
        "<script>alert(1)</script>",
        '<iframe src="javascript:alert(1)"></iframe>',
        '"><script>alert(document.cookie)</script>',
        "<svg/onload=alert(1)>",
        "**<script>alert(1)</script>**",
        "- <iframe srcdoc='<script>x</script>'></iframe>",
        "<a href='javascript:alert(1)'>click</a>",
    ]

    @pytest.mark.parametrize("payload", HOSTILE)
    def test_no_raw_dangerous_tag_survives(self, payload):
        out = safe_markdown(payload)
        assert "<iframe" not in out.lower()
        assert "<script" not in out.lower()
        assert "<img" not in out.lower()
        # A '<' that starts anything but an allowlisted tag would be a hole.
        assert_only_allowlisted_markup(out)

    def test_onerror_cannot_survive_as_an_attribute(self):
        # The specific payload from the report. Escaping turns the tag opener
        # into "&lt;img", so no <img> element exists and the handler has nothing
        # to attach to. The literal text "onerror=" does remain, but only as
        # inert escaped text; asserting its raw absence would be asserting
        # something false. What must hold is that no TAG carries it.
        payload = '<img src=x onerror="alert(1)">'
        assert "onerror=" in payload

        out = safe_markdown(payload)

        assert "&lt;img" in out
        assert "<img" not in out
        assert only_allowlisted_markup(out)
        for tag in _ANY_TAG_RE.findall(out):
            assert "onerror" not in tag.lower()

    @pytest.mark.parametrize(
        "text",
        [
            "Hello world",
            "**bold** and `code` and *italic*",
            "1 < 2 and 3 > 2",
            "- a bullet\n- another bullet",
            "report*v2*final.pdf",
            'quotes "double" and \'single\' and & ampersand',
            "line with <b>not allowed</b> markup",
            "<iframe srcdoc=\"<script>alert(1)</script>\"></iframe>",
        ],
    )
    def test_all_output_markup_is_allowlisted(self, text):
        assert_only_allowlisted_markup(safe_markdown(text))


def only_allowlisted_markup(output: str) -> bool:
    """Non-raising form of the invariant, for use inside assertions."""
    for tag in _ANY_TAG_RE.findall(output):
        if not _ALLOWED_TAG_RE.fullmatch(tag):
            return False
    for index, char in enumerate(output):
        if char == "<" and not _ALLOWED_TAG_RE.match(output, index):
            return False
    return True


class TestSafeMarkdownFormatting:
    def test_bold(self):
        assert safe_markdown("**bold**") == "<strong>bold</strong>"

    def test_bullets(self):
        assert safe_markdown("- one\n- two") == "<ul><li>one</li><li>two</li></ul>"

    def test_filename_star_is_not_italicised(self):
        # A single '*' after a word character must not open emphasis, so a
        # versioned file name renders literally instead of half-converted.
        out = safe_markdown("report*v2*final.pdf")
        assert "<em>" not in out
        assert out == "report*v2*final.pdf"

    def test_inline_code(self):
        assert safe_markdown("`code`") == "<code>code</code>"

    def test_paragraph_lines_joined_with_break(self):
        assert safe_markdown("a\nb") == "a<br>b"

    def test_none_renders_empty(self):
        assert safe_markdown(None) == ""


class TestSourceLabel:
    def test_with_page(self):
        meta = {"file_name": "report.pdf", "page_label": "Page 2"}
        assert source_label(meta) == f"report.pdf {DASH} Page 2"

    def test_without_page(self):
        assert source_label({"file_name": "report.pdf"}) == "report.pdf"

    def test_empty_metadata(self):
        assert source_label({}) == "Unknown document"

    def test_none(self):
        assert source_label(None) == "Unknown document"

    def test_blank_file_name_falls_back(self):
        # An empty name falls back to the placeholder; the page, which is real,
        # is still shown beside it.
        meta = {"file_name": "", "page_label": "Page 2"}
        assert source_label(meta) == f"Unknown document {DASH} Page 2"


class TestSourceLabels:
    def _doc(self, meta):
        return type("D", (), {"metadata": meta})()

    def test_deduplicates_and_preserves_order(self):
        a = self._doc({"file_name": "a.pdf", "page_label": "Page 1"})
        b = self._doc({"file_name": "b.pdf", "page_label": "Page 2"})
        labels = source_labels([a, b, a, b])
        assert labels == [
            f"a.pdf {DASH} Page 1",
            f"b.pdf {DASH} Page 2",
        ]

    def test_empty_iterable(self):
        assert source_labels([]) == []

    def test_none_iterable(self):
        assert source_labels(None) == []

    def test_document_without_metadata(self):
        doc = type("D", (), {})()
        assert source_labels([doc]) == ["Unknown document"]


class TestGenericError:
    def test_contains_no_exception_text(self):
        # The app shows this constant instead of str(exc); exception text used
        # to be rendered straight into the UI and leaked paths and internals.
        for marker in ("Traceback", "Exception", "Errno", 'File "', "Error:"):
            assert marker not in GENERIC_ERROR
        assert GENERIC_ERROR.strip()
