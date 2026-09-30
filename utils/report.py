"""A self-contained HTML report, and the one function every figure is saved through."""

import base64
import html
import io


class Report:
    """Collects the run's figures, tables and log into one self-contained HTML file.

    Blocks are appended in the order they are produced, so the report reads in the
    order the analysis ran. Figures are embedded as base64 PNGs rather than linked:
    a report that stops rendering once the results directory is reorganised is not
    worth writing, and the .pdf/.svg files stay on disk for the manuscript.
    """

    def __init__(self):
        """Start an empty report. Takes no arguments."""
        self.blocks = []

    def heading(self, text, level=2):
        """Append a heading.

        Args:
            text: Heading text, without a number; sections and subsections are
                numbered when the report is rendered.
            level: HTML heading level, 2 for a section and 3 for a subsection.
        """
        self.blocks.append(("heading", (level, text)))

    def paragraph(self, text):
        """Append a paragraph of explanatory prose.

        Args:
            text: Plain text; escaped, so it may contain < and &. Inline maths
                goes between \\( and \\), in LaTeX.
        """
        self.blocks.append(("paragraph", text))

    def table(self, frame, caption=None, float_format=None, formatters=None):
        """Append a table.

        Args:
            frame: DataFrame to render; nothing is appended if it is empty.
            caption: Line printed above the table.
            float_format: Callable applied to every float cell.
            formatters: {column: callable} taking precedence over `float_format`,
                so a p-value column can keep its own notation.
        """
        if frame is None or frame.empty:
            return
        self.blocks.append(("table", (frame.copy(), caption, float_format, formatters)))

    def figure(self, fig, caption=None):
        """Append a figure, captured as it stands.

        Args:
            fig: Matplotlib figure. It is rendered to PNG immediately, so this
                must be called before the figure is closed.
            caption: Line printed under the figure.
        """
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        self.blocks.append(
            ("figure", (base64.b64encode(buffer.getvalue()).decode("ascii"), caption)))

    def callout(self, text):
        """Append the one sentence a reader must not miss.

        Args:
            text: The verdict, in plain words.
        """
        self.blocks.append(("callout", text))

    def equation(self, tex):
        """Append a displayed equation.

        Args:
            tex: LaTeX source, without the surrounding $$. Rendered by KaTeX when
                the report is opened; offline, the source itself is shown, which
                is still readable.
        """
        self.blocks.append(("equation", tex))

    def references(self, items):
        """Append a numbered reference list.

        Args:
            items: Citations in order; the numbering matches the [n] markers
                used in the prose above.
        """
        self.blocks.append(("references", list(items)))

    def code(self, text, caption=None):
        """Append a block of preformatted text: a log, or a JSON document.

        Args:
            text: The text, kept verbatim.
            caption: Line printed above it.
        """
        self.blocks.append(("code", (text, caption)))

    def render(self, title, subtitle=""):
        """The complete HTML document as a string.

        Args:
            title: Document title and top heading.
            subtitle: Line under the title, typically the command that was run.
        """
        parts = [_REPORT_HEAD.format(title=html.escape(title))]
        parts.append(f"<h1>{html.escape(title)}</h1>")
        if subtitle:
            parts.append(f'<p class="subtitle">{html.escape(subtitle)}</p>')
        # Numbered here rather than by the callers, so a section that only some
        # runs produce (--pairwise) renumbers everything after it consistently
        section = subsection = 0
        for kind, payload in self.blocks:
            if kind == "heading":
                level, text = payload
                if level == 2:
                    section, subsection = section + 1, 0
                    number, anchor = f"{section}.", f"sec-{section}"
                else:
                    subsection += 1
                    number, anchor = f"{section}.{subsection}", f"sec-{section}-{subsection}"
                parts.append(f'<h{level} id="{anchor}">{number} '
                             f"{html.escape(text)}</h{level}>")
            elif kind == "paragraph":
                parts.append(f"<p>{html.escape(payload)}</p>")
            elif kind == "callout":
                parts.append(f'<p class="verdict">{html.escape(payload)}</p>')
            elif kind == "equation":
                parts.append(f'<div class="eq">$${html.escape(payload)}$$</div>')
            elif kind == "references":
                parts.append("<ol class=\"refs\">" + "".join(
                    f"<li>{html.escape(item)}</li>" for item in payload) + "</ol>")
            elif kind == "code":
                text, caption = payload
                if caption:
                    parts.append(f'<p class="caption">{html.escape(caption)}</p>')
                parts.append(f"<pre>{html.escape(text)}</pre>")
            elif kind == "figure":
                encoded, caption = payload
                parts.append('<figure>'
                             f'<img src="data:image/png;base64,{encoded}" alt="'
                             f'{html.escape(caption or "figure")}">')
                if caption:
                    parts.append(f"<figcaption>{html.escape(caption)}</figcaption>")
                parts.append("</figure>")
            elif kind == "table":
                frame, caption, float_format, formatters = payload
                if caption:
                    parts.append(f'<p class="caption">{html.escape(caption)}</p>')
                parts.append('<div class="scroll">' + frame.to_html(
                    index=False, na_rep="n/a", border=0, escape=True,
                    float_format=float_format, formatters=formatters or {},
                ) + "</div>")
        parts.append("</main></body></html>")
        return "\n".join(parts)

    def write(self, path, title, subtitle=""):
        """Write the report and say where it went.

        Args:
            path: File to write.
            title: Document title and top heading.
            subtitle: Line under the title.
        """
        with open(path, "w") as handle:
            handle.write(self.render(title, subtitle))
        print(f"Report written to {path}")
        return path


# One per run. A module-level collector rather than a parameter threaded through
# every plotting helper: `save_figure` is the single point every figure passes
# through, so registering there catches all of them without touching signatures.
REPORT = Report()


_REPORT_HEAD = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{ color-scheme: light dark;
           --ink: #1a1a1a; --bg: #ffffff; --muted: #5a5a5a;
           --rule: #d9d9d9; --band: #f5f5f5; --accent: #5b2d8e; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --ink: #e8e8e8; --bg: #16181c; --muted: #a0a0a0;
             --rule: #33363d; --band: #1e2127; --accent: #c4a7e7; }}
  }}
  html {{ background: var(--bg); }}
  body {{ margin: 0; padding: 0 16px 64px; color: var(--ink); background: var(--bg);
          font: 15px/1.6 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}
  main, h1, p.subtitle {{ max-width: 980px; margin-left: auto; margin-right: auto; }}
  h1 {{ font-size: 1.7rem; padding-top: 32px; margin-bottom: 4px; }}
  h2 {{ font-size: 1.25rem; margin-top: 40px; padding-top: 12px;
        border-top: 2px solid var(--accent); }}
  h3 {{ font-size: 1.05rem; margin-top: 28px; color: var(--muted); }}
  p.subtitle {{ color: var(--muted); margin-top: 0; font-family: ui-monospace,
                SFMono-Regular, Menlo, monospace; font-size: 0.85rem;
                word-break: break-all; }}
  p.caption {{ color: var(--muted); font-size: 0.9rem; margin-bottom: 6px; }}
  p.verdict {{ border-left: 4px solid var(--accent); background: var(--band);
               padding: 12px 16px; margin: 16px 0; font-size: 1.02rem; }}
  figure {{ margin: 20px 0; }}
  img {{ max-width: 100%; height: auto; display: block; background: #fff;
         border: 1px solid var(--rule); border-radius: 6px; padding: 8px; }}
  figcaption {{ color: var(--muted); font-size: 0.85rem; margin-top: 8px; }}
  .scroll {{ overflow-x: auto; }}
  table {{ border-collapse: collapse; font-variant-numeric: tabular-nums;
           font-size: 0.87rem; margin-bottom: 8px; }}
  th, td {{ padding: 5px 12px; text-align: right; white-space: nowrap;
            border-bottom: 1px solid var(--rule); }}
  th {{ text-align: right; font-weight: 600; }}
  td:first-child, th:first-child, td:nth-child(2), th:nth-child(2) {{ text-align: left; }}
  tbody tr:nth-child(even) {{ background: var(--band); }}
  ol.refs {{ font-size: 0.86rem; color: var(--ink); padding-left: 24px; }}
  ol.refs li {{ margin-bottom: 7px; }}
  div.eq {{ overflow-x: auto; overflow-y: hidden; margin: 10px 0 14px; }}
  pre {{ background: var(--band); border: 1px solid var(--rule); border-radius: 6px;
         padding: 12px; overflow-x: auto; font-size: 0.8rem; line-height: 1.45; }}
</style>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css">
<script defer src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.js"></script>
<script defer src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body, {{delimiters: [
    {{left: '$$', right: '$$', display: true}},
    {{left: '\\\\(', right: '\\\\)', display: false}}], throwOnError: false}});"></script>
</head><body><main>"""


def save_figure(fig, results, stem, formats):
    """Write one figure to `results`/OS-stats/ once per requested format.

    Args:
        fig: Matplotlib figure to write.
        results: Results directory; its OS-stats/ subdirectory must already exist.
        stem: File name without extension.
        formats: Extensions to write, e.g. ("pdf", "svg").

    The figure is also captured for the HTML report, which is why this is the only
    place figures are written: a figure saved past it would be missing there.
    """
    for fmt in formats:
        fig.savefig(f"{results}/OS-stats/{stem}.{fmt}", dpi=200, format=fmt)
    REPORT.figure(fig, caption=stem)
