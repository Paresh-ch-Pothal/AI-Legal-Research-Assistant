"""
pdf_export.py
─────────────
Utility for exporting the Legal AI chat transcript to a downloadable,
shareable PDF — styled like an actual chat conversation (coloured,
right/left-aligned message bubbles, a branded header, page numbers,
and a "source chip" panel under each answer) rather than a plain
document dump.

Requires:
    pip install reportlab

Usage (from app.py):
    from pdf_export import generate_chat_pdf

    pdf_bytes = generate_chat_pdf(
        st.session_state.messages,
        session_id=st.session_state.session_id,
    )
    st.download_button(
        "📥 Download Chat as PDF",
        data=pdf_bytes,
        file_name="legal_chat.pdf",
        mime="application/pdf",
    )
"""

import io
import re
from datetime import datetime
from typing import List, Dict, Optional

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_LEFT, TA_RIGHT, TA_CENTER
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas as pdfcanvas
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether,
)

# ── Palette ──────────────────────────────────────────────────────
INDIGO       = colors.HexColor("#4F46E5")
INDIGO_LIGHT = colors.HexColor("#C7D2FE")
GREY_BG      = colors.HexColor("#F4F4F5")
GREY_BORDER  = colors.HexColor("#E4E4E7")
CHIP_BG      = colors.HexColor("#FAFAFA")
TEXT_DARK    = colors.HexColor("#1F2937")
TEXT_MUTED   = colors.HexColor("#6B7280")
WHITE        = colors.white

# ── Layout constants ─────────────────────────────────────────────
PAGE_W, PAGE_H = A4
MARGIN        = 2 * cm
BANNER_H      = 2.5 * cm
TOP_INSET     = 3.6 * cm   # frame top = below banner + meta line
BOTTOM_INSET  = 1.8 * cm   # frame bottom = above footer
CONTENT_W     = PAGE_W - 2 * MARGIN
MAX_BUBBLE_W  = CONTENT_W * 0.78
MIN_BUBBLE_W  = 3.5 * cm


# ── Markdown → ReportLab mini-XML ───────────────────────────────
def _clean_markdown(text: str) -> str:
    """
    Convert the lightweight markdown used in chat answers into ReportLab's
    mini-XML markup, safely escaping anything the user/model typed first so
    it can never be misread as a tag.
    """
    if not text:
        return ""

    # 1) Escape special characters FIRST so nothing in the source text can
    #    be interpreted as markup once we inject our own tags below.
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # 2) Markdown headings -> bold line
    text = re.sub(r"^#{1,6}\s*(.+)$", r"<b>\1</b>", text, flags=re.MULTILINE)

    # 3) Bold / italic markdown -> reportlab inline tags
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)

    # 4) Bullet points -> indented unicode bullet (avoid named HTML entities)
    text = re.sub(r"^[-•]\s+", "    \u2022 ", text, flags=re.MULTILINE)

    # 5) Line breaks
    text = text.replace("\n", "<br/>")
    return text


def _build_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="UserMsg", parent=styles["Normal"], fontSize=10.5, leading=15,
        textColor=WHITE, alignment=TA_LEFT,
    ))
    styles.add(ParagraphStyle(
        name="AssistantMsg", parent=styles["Normal"], fontSize=10.5, leading=15,
        textColor=TEXT_DARK, alignment=TA_LEFT,
    ))
    styles.add(ParagraphStyle(
        name="RoleLabelUser", parent=styles["Normal"], fontSize=8,
        textColor=TEXT_MUTED, alignment=TA_RIGHT, spaceAfter=3,
    ))
    styles.add(ParagraphStyle(
        name="RoleLabelAssistant", parent=styles["Normal"], fontSize=8,
        textColor=TEXT_MUTED, alignment=TA_LEFT, spaceAfter=3,
    ))
    styles.add(ParagraphStyle(
        name="SourceTitle", parent=styles["Normal"], fontSize=7.5,
        textColor=TEXT_MUTED, spaceAfter=3, fontName="Helvetica-Bold",
    ))
    styles.add(ParagraphStyle(
        name="SourceChip", parent=styles["Normal"], fontSize=8, leading=12,
        textColor=TEXT_DARK,
    ))
    styles.add(ParagraphStyle(
        name="EmptyState", parent=styles["Normal"], fontSize=11,
        textColor=TEXT_MUTED, alignment=TA_CENTER,
    ))
    return styles


# ── Bubble sizing & construction ────────────────────────────────
def _first_line_plain(raw: str) -> str:
    """Strip markdown noise so short one-liners can be width-measured."""
    return re.sub(r"[#*_`]", "", raw or "").split("\n")[0]


def _bubble_width(raw_text: str, font_name: str = "Helvetica", font_size: int = 10.5) -> float:
    """Shrink-wrap short one-line messages; cap long/multi-line ones."""
    if not raw_text or "\n" in raw_text or len(raw_text) > 140:
        return MAX_BUBBLE_W
    w = stringWidth(_first_line_plain(raw_text), font_name, font_size) + 26  # padding
    return max(MIN_BUBBLE_W, min(MAX_BUBBLE_W, w))


def _make_bubble(content_html: str, raw_text: str, role: str, styles):
    """Build a single rounded, coloured chat-bubble as a Table flowable."""
    is_user = role == "user"
    width = _bubble_width(raw_text)

    msg_style = styles["UserMsg"] if is_user else styles["AssistantMsg"]
    bg = INDIGO if is_user else GREY_BG

    bubble = Table([[Paragraph(content_html, msg_style)]], colWidths=[width])
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROUNDEDCORNERS", [10, 10, 10, 10]),
    ]
    if not is_user:
        style_cmds.append(("BOX", (0, 0), (-1, -1), 0.6, GREY_BORDER))
    bubble.setStyle(TableStyle(style_cmds))
    bubble.hAlign = "RIGHT" if is_user else "LEFT"
    return bubble


def _make_sources_block(sources, styles):
    """A small bordered 'chip' panel listing cited sources under an answer."""
    if not sources:
        return None
    rows = [Paragraph("REFERENCED SOURCES", styles["SourceTitle"])]
    for idx, src in enumerate(sources, 1):
        fname = src.get("filename", "Unknown File")
        court = src.get("court") or "—"
        case_no = src.get("case_number") or "—"
        rows.append(Paragraph(
            f"<b>{idx}. {fname}</b>  ·  {court}  ·  {case_no}",
            styles["SourceChip"],
        ))
    box = Table([[rows]], colWidths=[MAX_BUBBLE_W])
    box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), CHIP_BG),
        ("BOX", (0, 0), (-1, -1), 0.5, GREY_BORDER),
        ("ROUNDEDCORNERS", [8, 8, 8, 8]),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    box.hAlign = "LEFT"
    return box


# ── Page-numbering canvas (adds "Page X of Y" after layout finishes) ──
class _NumberedCanvas(pdfcanvas.Canvas):
    def __init__(self, *args, **kwargs):
        pdfcanvas.Canvas.__init__(self, *args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self._draw_footer(total)
            pdfcanvas.Canvas.showPage(self)
        pdfcanvas.Canvas.save(self)

    def _draw_footer(self, total):
        self.saveState()
        self.setStrokeColor(GREY_BORDER)
        self.line(MARGIN, BOTTOM_INSET - 0.3 * cm, PAGE_W - MARGIN, BOTTOM_INSET - 0.3 * cm)
        self.setFont("Helvetica", 8)
        self.setFillColor(TEXT_MUTED)
        self.drawCentredString(PAGE_W / 2, BOTTOM_INSET - 0.7 * cm,
                                f"Page {self._pageNumber} of {total}")
        self.setFont("Helvetica-Oblique", 7.5)
        self.drawString(MARGIN, BOTTOM_INSET - 0.7 * cm, "Legal AI Research Assistant")
        self.restoreState()


# ── Main entry point ─────────────────────────────────────────────
def generate_chat_pdf(
    messages: List[Dict],
    session_id: Optional[str] = None,
    title: str = "Legal AI Research Assistant",
) -> bytes:
    """
    Render the chat history into a chat-styled PDF and return it as raw
    bytes, ready to hand to st.download_button(data=...).
    """
    buffer = io.BytesIO()
    styles = _build_styles()

    meta_bits = [f"Generated {datetime.now().strftime('%d %b %Y, %H:%M')}"]
    if session_id:
        meta_bits.append(f"Session {session_id}")
    meta_bits.append(f"{len(messages)} message{'s' if len(messages) != 1 else ''}")
    meta_line = "    ·    ".join(meta_bits)

    def _draw_header(c, continued: bool):
        c.saveState()
        c.setFillColor(INDIGO)
        c.rect(0, PAGE_H - BANNER_H, PAGE_W, BANNER_H, fill=1, stroke=0)

        # Small badge icon (drawn with vector shapes, not an emoji glyph --
        # base-14 PDF fonts don't carry emoji, so emoji render as blank boxes)
        badge = 0.85 * cm
        badge_x = MARGIN
        badge_y = PAGE_H - BANNER_H / 2 - badge / 2
        c.setFillColor(WHITE)
        c.roundRect(badge_x, badge_y, badge, badge, 4, fill=1, stroke=0)
        c.setFillColor(INDIGO)
        c.setFont("Helvetica-Bold", 13)
        c.drawCentredString(badge_x + badge / 2, badge_y + badge / 2 - 4.5, "\u00A7")

        text_x = badge_x + badge + 0.3 * cm
        c.setFillColor(WHITE)
        c.setFont("Helvetica-Bold", 15 if continued else 17)
        c.drawString(text_x, PAGE_H - BANNER_H + (0.8 if continued else 0.9) * cm,
                     "Legal AI Research Assistant")

        c.setFont("Helvetica", 9)
        c.setFillColor(INDIGO_LIGHT)
        sub = "Chat Transcript  ·  continued" if continued else "Chat Transcript"
        c.drawString(text_x, PAGE_H - BANNER_H + (0.35 if continued else 0.4) * cm, sub)

        if not continued:
            c.setFont("Helvetica", 8)
            c.setFillColor(TEXT_MUTED)
            c.drawString(MARGIN, PAGE_H - BANNER_H - 0.55 * cm, meta_line)
        c.restoreState()

    def _first_page(c, doc):
        _draw_header(c, continued=False)

    def _later_pages(c, doc):
        _draw_header(c, continued=True)

    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        topMargin=TOP_INSET, bottomMargin=BOTTOM_INSET,
        leftMargin=MARGIN, rightMargin=MARGIN,
        title=title,
    )

    story = []

    if not messages:
        story.append(Spacer(1, 80))
        story.append(Paragraph("No conversation recorded yet.", styles["EmptyState"]))
    else:
        for msg in messages:
            role = msg.get("role", "user")
            is_user = role == "user"
            raw_content = msg.get("content", "")
            content_html = _clean_markdown(raw_content)

            label_style = styles["RoleLabelUser"] if is_user else styles["RoleLabelAssistant"]
            label_text = "You" if is_user else "Legal AI Assistant"
            story.append(Paragraph(label_text, label_style))

            block = [_make_bubble(content_html, raw_content, role, styles)]

            sources = msg.get("sources") or []
            if sources and not is_user:
                src_block = _make_sources_block(sources, styles)
                if src_block:
                    block.append(Spacer(1, 4))
                    block.append(src_block)

            story.append(KeepTogether(block))
            story.append(Spacer(1, 16))

    doc.build(
        story,
        onFirstPage=_first_page,
        onLaterPages=_later_pages,
        canvasmaker=_NumberedCanvas,
    )
    buffer.seek(0)
    return buffer.getvalue()