# ruff: noqa: E501, RUF001
"""Turn docs/report/FormulaOneRL_Report.md into an IEEE Access LaTeX project.

usage: python scripts/report_to_latex.py   (needs pandoc on PATH)

Writes docs/report/latex/main.tex next to the IEEE Access class files and copies
the figures, so the folder can be zipped and uploaded to Overleaf as is. Tables,
figures, the pseudocode and the reference list are converted here; pandoc
converts the prose and maths.
"""

import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs" / "report"
OUT = REPORT / "latex"

TITLE = (
    "Safety-Shielded Hierarchical Reinforcement Learning for Formula~1: "
    "A Move-Level Racing Driver and a Lap-Level PPO Race Strategist"
)
PREAMBLE = r"""\documentclass{ieeeaccess}
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{tabularx}
\usepackage{array}
\usepackage{url}
\sloppy
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
"""
UNICODE = {
    "≈": "\\ensuremath{\\approx}",
    "→": "\\ensuremath{\\rightarrow}",
    "±": "\\ensuremath{\\pm}",
    "×": "\\ensuremath{\\times}",
    "↑": "\\ensuremath{\\uparrow}",
    "↓": "\\ensuremath{\\downarrow}",
    "≥": "\\ensuremath{\\geq}",
    "≤": "\\ensuremath{\\leq}",
    "·": "\\textperiodcentered{}",
    "–": "--",
    "—": "---",
}
# Tables with at least this many columns span both columns; so do these captions.
WIDE_COLUMNS = 5
WIDE_TABLES = {"Ablation study", "Development stages and what each changed"}
WIDE_FIGS = {
    "fig2_pipeline",
    "fig3_sac_learning_curve",
    "fig4_tactical_reward_ablation",
    "fig10_track_line",
    "fig11_tyre_models",
    "fig13_race_trace",
}


def raw(tex: str) -> str:
    return "\n```{=latex}\n" + tex + "\n```\n"


def inline_md(text: str) -> str:
    """Markdown inline text -> LaTeX through pandoc (keeps maths and emphasis)."""
    out = subprocess.run(
        ["pandoc", "-f", "markdown", "-t", "latex", "--wrap=none"],
        input=text,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return cite(out)


def cite(tex: str) -> str:
    def repl(m: re.Match[str]) -> str:
        nums = re.findall(r"\d+", m.group(0))
        return "\\cite{" + ",".join(f"b{n}" for n in nums) + "}"

    # [3] or [4], [14] -- after pandoc, brackets come through as {[}3{]}
    tex = re.sub(r"\{\[\}(\d+)\{\]\}(?:,\s*\{\[\}(\d+)\{\]\})*", repl, tex)
    return re.sub(r"\[(\d+)\](?:,\s*\[(\d+)\])*", repl, tex)


def table(lines: list[str], idx: int, caption: str) -> str:
    rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines]
    head, body = rows[0], rows[2:]
    n = len(head)
    wide = n >= WIDE_COLUMNS or caption in WIDE_TABLES
    env = "table*" if wide else "table"
    width = "\\textwidth" if wide else "\\columnwidth"
    spec = ">{\\raggedright\\arraybackslash}X" * n
    out = [
        f"\\begin{{{env}}}[!t]",
        f"\\caption{{{inline_md(caption)}}}",
        f"\\label{{tab:{idx + 1}}}",
        "\\centering\\footnotesize",
        f"\\begin{{tabularx}}{{{width}}}{{{spec}}}",
        "\\hline",
        " & ".join(f"\\textbf{{{inline_md(c)}}}" for c in head) + " \\\\",
        "\\hline",
    ]
    for r in body:
        out.append(" & ".join(inline_md(c) for c in r) + " \\\\")
    out += ["\\hline", "\\end{tabularx}", f"\\end{{{env}}}"]
    return raw("\n".join(out))


def main() -> None:
    md = (REPORT / "FormulaOneRL_Report.md").read_text(encoding="utf-8")
    md = md.split("---", 2)[2]  # drop YAML header
    body, refs = md.split("# References", 1)

    # abstract + keywords
    abstract = body.split("# Abstract", 1)[1].split("**Keywords:**", 1)[0].strip()
    keywords = body.split("**Keywords:**", 1)[1].split("\n\n", 1)[0].strip()
    body = body.split("**Keywords:**", 1)[1].split("\n\n", 1)[1]

    # figures with the caption paragraph that follows
    def fig(m: re.Match[str]) -> str:
        path, caption = m.group(1), m.group(2)
        name = Path(path).stem
        caption = re.sub(r"^Fig\. \d+\.\s*", "", caption.replace("\n", " "))
        # The class's \\Figure macro: wider than a column -> a two-column figure.
        width = "\\textwidth" if name in WIDE_FIGS else "0.98\\columnwidth"
        return raw(
            "\\Figure[t!](topskip=0pt, botskip=0pt, midskip=0pt)"
            f"[width={width}]{{{name}.png}}\n"
            + (
                f"{{\\parbox{{0.92\\textwidth}}{{{inline_md(caption)}}}"
                if name in WIDE_FIGS
                else f"{{{inline_md(caption)}"
            )
            + f"\\label{{fig:{name}}}}}"
        )

    body = re.sub(
        r"!\[[^\]]*\]\(([^)]+)\)\{[^}]*\}\s*\n\s*\n\*(Fig\. .*?)\*\n",
        fig,
        body,
        flags=re.S,
    )
    # figure numbers written in the text -> \ref
    order = re.findall(r"\\label\{fig:([^}]+)\}", body)
    for i, name in enumerate(order, 1):
        body = re.sub(rf"\(Fig\. {i}\)", f"(Fig.~\\\\ref{{fig:{name}}})", body)
    body = re.sub(
        r"Fig\. (\d+)",
        lambda m: f"Fig.~\\ref{{fig:{order[int(m.group(1)) - 1]}}}",
        body,
    )

    # pseudocode: ```Algorithm N: title ...``` blocks
    def algo(m: re.Match[str]) -> str:
        code = m.group(0)[4:-3].strip("\n")
        num, title = re.match(r"Algorithm (\d+): (.*)", code).groups()  # type: ignore[union-attr]
        return raw(
            "\\begin{figure*}[!t]\n\\footnotesize\n\\begin{verbatim}\n"
            + code
            + f"\n\\end{{verbatim}}\n\\caption{{{title}.}}\n\\label{{alg:{num}}}\n\\end{{figure*}}"
        )

    body = re.sub(r"```\nAlgorithm \d+:.*?```", algo, body, flags=re.S)

    # tables, each preceded by a "Table: caption" paragraph
    lines = body.split("\n")
    out: list[str] = []
    i = idx = 0
    caption = ""
    while i < len(lines):
        if lines[i].startswith("Table: "):
            caption = lines[i][len("Table: ") :].strip()
            i += 1
        elif lines[i].startswith("|"):
            j = i
            while j < len(lines) and lines[j].startswith("|"):
                j += 1
            out.append(table(lines[i:j], idx, caption))
            idx += 1
            i = j
        else:
            out.append(lines[i])
            i += 1
    body = "\n".join(out)
    # table numbers written in the text -> \ref
    body = re.sub(r"Table (\d+)", r"Table~\\ref{tab:\1}", body)

    # headings: drop manual numbers; intro subsections go under \section{Introduction}
    body = re.sub(r"^(#+) (\d+(\.\d+)*\.?) ", r"\1 ", body, flags=re.M)
    body = body.replace(
        "## Background and Context", "# Introduction\n\n## Background and Context", 1
    )

    tex = subprocess.run(
        ["pandoc", "-f", "markdown-auto_identifiers", "-t", "latex", "--wrap=preserve"],
        input=body,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    tex = cite(tex)
    tex = re.sub(r"\\label\{[a-z0-9-]+\}\n", "\n", tex)  # pandoc's heading labels
    tex = tex.replace(
        "\\section{Introduction}",
        "\\section{Introduction}\n\\label{sec:introduction}",
        1,
    )

    # references
    items = re.findall(r"^\[(\d+)\] (.+)$", refs, flags=re.M)
    bib = "\n\n".join(f"\\bibitem{{b{n}}} {inline_md(t)}" for n, t in items)

    pre = PREAMBLE
    doc = f"""{pre}\\begin{{document}}
\\history{{Date of publication xxxx 00, 0000, date of current version xxxx 00, 0000.}}
\\doi{{10.1109/ACCESS.2017.DOI}}

\\title{{{TITLE}}}
\\author{{\\uppercase{{Member One}}\\authorrefmark{{1}}, \\uppercase{{Member Two}}\\authorrefmark{{1}}, and \\uppercase{{Member Three}}\\authorrefmark{{1}}}}
\\address[1]{{Department, Institution, City, Country (e-mail: name@example.edu)}}
\\tfootnote{{Code, configurations and logged results: https://github.com/ADI-KRISH/Hierarchical-Multi-Agent-RL-for-Formula-1}}

\\markboth
{{Members \\headeretal: Safety-Shielded Hierarchical RL for Formula 1}}
{{Members \\headeretal: Safety-Shielded Hierarchical RL for Formula 1}}

\\corresp{{Corresponding author: Member One (e-mail: name@example.edu).}}

\\begin{{abstract}}
{inline_md(abstract)}
\\end{{abstract}}

\\begin{{keywords}}
{inline_md(keywords)}
\\end{{keywords}}

\\titlepgskip=-15pt

\\maketitle

{tex}

\\begin{{thebibliography}}{{00}}

{bib}

\\end{{thebibliography}}

\\EOD

\\end{{document}}
"""
    for ch, rep in UNICODE.items():
        doc = doc.replace(ch, rep)
    (OUT / "main.tex").write_text(doc, encoding="utf-8")
    for f in (REPORT / "figures").glob("*.png"):
        shutil.copy(f, OUT / f.name)
    print("wrote", OUT / "main.tex")


if __name__ == "__main__":
    main()
