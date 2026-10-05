"""
ui_helpers.py — Pure helpers shared by the Streamlit UI.

These live apart from app.py because importing app.py runs the entire
Streamlit script, which cannot be imported under pytest. Everything here is
side-effect free so it can be tested directly.

The escaping matters more than it looks. Streamlit renders markdown with
`rehype-raw` when `unsafe_allow_html=True`, and ships no sanitiser, so an f-string
that interpolates an uploaded file name or a model answer into markup is an HTML
injection sink. React neutralises `<script>` and inline `onerror` handlers, but
`<iframe srcdoc="...">` executes with same-origin access to the page, so a
crafted file name reached the browser of whoever queried that document.
Escaping every interpolated value closes it.
"""

import html
import logging
import re
from typing import Any, Iterable, Mapping

logger = logging.getLogger(__name__)

# Shown to the visitor; the real exception goes to the log. Exception text used
# to be rendered straight into the UI, which leaked filesystem paths and library
# internals to anyone who could trigger a failure.
GENERIC_ERROR = (
    "Something went wrong handling that request. "
    "The details have been written to the server log."
)


def escape(value: Any) -> str:
    """HTML-escape a value for interpolation into markup."""
    if value is None:
        return ""
    return html.escape(str(value), quote=True)


def source_label(metadata: Mapping[str, Any] | None) -> str:
    """
    Build a human-readable citation label from chunk metadata.

    The page is omitted rather than guessed when the loader did not supply one,
    so a missing page number never reads as "Page 1".
    """
    meta = metadata or {}
    name = meta.get("file_name") or "Unknown document"
    page = meta.get("page_label")
    return f"{name} — {page}" if page else str(name)


def source_labels(documents: Iterable[Any]) -> list[str]:
    """Deduplicated citation labels, in first-seen order."""
    seen: set[str] = set()
    labels: list[str] = []
    for doc in documents or []:
        label = source_label(getattr(doc, "metadata", None))
        if label not in seen:
            seen.add(label)
            labels.append(label)
    return labels


# ---------------------------------------------------------------------------
# Minimal safe markdown
# ---------------------------------------------------------------------------

_BULLET_RE = re.compile(r"^\s*[-*•]\s+(.*)$")

# Applied in order, to already-escaped text. The emphasis pattern refuses to
# match when the opening asterisk follows a word character or another asterisk,
# so a filename like "report*v2*final.pdf" or a bullet "--x--" is left alone
# rather than half-converted.
_INLINE_RULES = (
    (re.compile(r"\*\*(.+?)\*\*", re.DOTALL), r"<strong>\1</strong>"),
    (re.compile(r"`([^`]+?)`"), r"<code>\1</code>"),
    (re.compile(r"(?<![*\w])\*([^*\n]+?)\*(?!\*)"), r"<em>\1</em>"),
)


def _inline(text: str) -> str:
    for pattern, replacement in _INLINE_RULES:
        text = pattern.sub(replacement, text)
    return text


def safe_markdown(text: Any) -> str:
    """
    Render untrusted text as a small, closed subset of markdown.

    Escaping the text and stopping there is safe but ugly: the model's answers
    are full of `**bold**` and bullet lists, and every one of them would show up
    literally. Passing the text through unescaped is the injection sink described
    in this module's docstring. So the text is escaped first, after which no
    attacker-controlled angle bracket remains, and only then are a few constructs
    rewritten into tags this module emits itself.

    This is deliberately not a markdown implementation. Bold, italic, inline
    code and `-`/`*` bullets are recognised; every other construct, including
    headings, tables and links, stays literal. Nothing is accepted from the
    input except plain text.
    """
    escaped = escape(text)

    blocks: list[str] = []
    paragraph: list[str] = []
    bullets: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            blocks.append("<br>".join(_inline(line) for line in paragraph))
            paragraph.clear()

    def flush_bullets() -> None:
        if bullets:
            items = "".join(f"<li>{_inline(item)}</li>" for item in bullets)
            blocks.append(f"<ul>{items}</ul>")
            bullets.clear()

    for line in escaped.split("\n"):
        match = _BULLET_RE.match(line)
        if match:
            flush_paragraph()
            bullets.append(match.group(1))
        elif not line.strip():
            flush_paragraph()
            flush_bullets()
        else:
            flush_bullets()
            paragraph.append(line)

    flush_paragraph()
    flush_bullets()
    return "".join(blocks)
