#!/usr/bin/env python3
"""Generate a LaTeX / Markdown / CSV comparison table from ShapeNetParts runs.

Reads every `<exp_dir>/*/seg_summary.csv` and emits a grouped table with the
same style as `classification_comparison_report.py`: Category | Model |
Strategy | Acc | Class mIoU | Instance mIoU, with \\multirow for category
and SSL base model, best values bolded per column.

Usage:
    python scripts/reports/partseg_comparison_report.py --exp_dir experiments/ShapeNetParts
    python scripts/reports/partseg_comparison_report.py --exp_dir experiments/ShapeNetParts \\
        --use_citation_in_tables

By default, outputs are written to `<exp_dir>/latex/`.
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _citations import (
    canonicalize,
    render_grouped_table,
    build_citation_paragraph,
)


METRICS = ["best_accuracy", "best_class_miou", "best_instance_miou"]
LATEX_HEADERS = {
    "best_accuracy":       r"Acc (\%)",
    "best_class_miou":     r"Class mIoU (\%)",
    "best_instance_miou":  r"Instance mIoU (\%)",
}
MD_HEADERS = {
    "best_accuracy":       "Acc (%)",
    "best_class_miou":     "Class mIoU (%)",
    "best_instance_miou":  "Instance mIoU (%)",
}


def clean_label(folder: str) -> str:
    name = re.sub(r"^smoke_", "", folder)
    name = re.sub(r"_partseg$", "", name)
    return name


def collect(exp_dir: Path) -> pd.DataFrame:
    rows = []
    for summary in sorted(exp_dir.glob("*/seg_summary.csv")):
        df = pd.read_csv(summary)
        if df.empty:
            continue
        row = df.iloc[-1].to_dict()
        raw = clean_label(summary.parent.name)
        base, strategy = canonicalize(raw)
        row["raw_label"] = raw
        row["base"] = base
        row["strategy"] = strategy
        rows.append(row)
    if not rows:
        raise SystemExit(f"No seg_summary.csv found under {exp_dir}")
    return pd.DataFrame(rows)


def render_latex(df: pd.DataFrame, cite_in_tables: bool = False) -> str:
    data_rows = []
    for _, r in df.iterrows():
        row = {'base': r['base'], 'strategy': r['strategy']}
        for m in METRICS:
            row[m] = float(r[m])
        data_rows.append(row)

    table = render_grouped_table(
        data_rows,
        metric_cols=METRICS,
        metric_headers=[LATEX_HEADERS[m] for m in METRICS],
        caption="ShapeNetParts part segmentation results. "
                "Best value per column in \\textbf{bold}.",
        label="tab:shapenetparts",
        cite_in_tables=cite_in_tables,
    )

    preamble = [
        r"% Auto-generated LaTeX document — ShapeNetParts part segmentation",
        r"% Compile with: pdflatex <filename>.tex  (then bibtex if using \bibliography)",
        r"\documentclass[a4paper,11pt]{article}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{booktabs}",
        r"\usepackage{multirow}",
        r"\usepackage{adjustbox}",
        r"\usepackage[numbers,sort&compress]{natbib}",
        r"\usepackage[margin=1in]{geometry}",
        r"\bibliographystyle{unsrtnat}",
        r"\begin{document}",
    ]
    body = []
    if not cite_in_tables:
        names = df['base'].tolist() + [s for s in df['strategy'].tolist()
                                       if s not in ('-', 'FF', 'Full', 'Full Finetuning')]
        paragraph = build_citation_paragraph(names)
        if paragraph:
            body += [paragraph, ""]
    closing = [
        "",
        r"\bibliography{references}",
        r"\end{document}",
    ]
    return "\n".join(preamble + [""] + body + [table] + closing)


def render_markdown(df: pd.DataFrame) -> str:
    cols = ["Category", "Model", "Strategy"] + [MD_HEADERS[m] for m in METRICS]
    out = ["| " + " | ".join(cols) + " |",
           "|" + "|".join(["---"] * len(cols)) + "|"]
    from _citations import get_category
    best = {m: df[m].astype(float).max() for m in METRICS}
    df = df.copy()
    df['category'] = df['base'].apply(get_category)
    for _, r in df.sort_values(['category', 'base', 'strategy']).iterrows():
        cells = [r['category'], r['base'], r['strategy']]
        for m in METRICS:
            v = float(r[m])
            s = f"{v:.2f}"
            if v == best[m]:
                s = f"**{s}**"
            cells.append(s)
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", default="experiments/ShapeNetParts", type=Path)
    ap.add_argument("--out_dir", default=None, type=Path,
                    help="Output dir (default: <exp_dir>/latex)")
    ap.add_argument("--use_citation_in_tables", action="store_true",
                    help="Put \\citep{} next to each model name in the table. "
                         "Default: cite methods once in a paragraph above the table.")
    args = ap.parse_args()

    df = collect(args.exp_dir)

    if args.out_dir is None:
        args.out_dir = args.exp_dir / "latex"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "shapenetparts_comparison.csv"
    tex_path = args.out_dir / "shapenetparts_comparison.tex"
    md_path = args.out_dir / "shapenetparts_comparison.md"

    df[["raw_label", "base", "strategy"] + METRICS].to_csv(csv_path, index=False)
    tex_path.write_text(render_latex(df, args.use_citation_in_tables) + "\n")
    md_path.write_text(render_markdown(df) + "\n")

    from _citations import copy_references_bib
    copy_references_bib(args.out_dir)

    print(render_markdown(df))
    print()
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {tex_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
