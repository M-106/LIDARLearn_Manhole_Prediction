#!/usr/bin/env python3
"""Generate a LaTeX / Markdown / CSV comparison report for S3DIS semantic
segmentation runs.

Reads every `<exp_dir>/*/seg_summary.csv` and emits two grouped tables in the
same style as `classification_comparison_report.py` / `partseg_comparison_report.py`:

  1. **Summary table** — Accuracy (OA), Class mIoU, Instance mIoU per model.
  2. **Per-class IoU table** — one column per S3DIS class (13 classes), optional
     via `--per_class`.

Both tables follow the shared layout from `_citations.render_grouped_table`
(Category multirow, SSL grouping via \\cmidrule, best values bolded), and the
output is a standalone LaTeX document ready for `pdflatex`.

Usage:
    python scripts/reports/semseg_comparison_report.py --exp_dir experiments/S3DIS
    python scripts/reports/semseg_comparison_report.py --exp_dir experiments/S3DIS \\
        --per_class --use_citation_in_tables

By default, outputs land in `<exp_dir>/latex/`.
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
    "best_accuracy":       r"OA (\%)",
    "best_class_miou":     r"Class mIoU (\%)",
    "best_instance_miou":  r"Instance mIoU (\%)",
}
MD_HEADERS = {
    "best_accuracy":       "OA (%)",
    "best_class_miou":     "Class mIoU (%)",
    "best_instance_miou":  "Instance mIoU (%)",
}

# S3DIS class names (13 classes, order matches iou_0..iou_12 in seg_summary.csv).
S3DIS_CLASSES = [
    "ceiling", "floor", "wall", "beam", "column", "window", "door",
    "table", "chair", "sofa", "bookcase", "board", "clutter",
]


def clean_label(folder: str) -> str:
    name = re.sub(r"^smoke_", "", folder)
    name = re.sub(r"_s3dis$", "", name)
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


def _data_rows(df: pd.DataFrame, metric_cols: list[str]) -> list[dict]:
    out = []
    for _, r in df.iterrows():
        row = {"base": r["base"], "strategy": r["strategy"]}
        for m in metric_cols:
            row[m] = float(r[m]) if m in r and pd.notna(r[m]) else None
        out.append(row)
    return out


def render_latex(df: pd.DataFrame,
                 cite_in_tables: bool = False,
                 per_class: bool = False) -> str:
    # ── Main summary table ──
    summary_table = render_grouped_table(
        _data_rows(df, METRICS),
        metric_cols=METRICS,
        metric_headers=[LATEX_HEADERS[m] for m in METRICS],
        caption="S3DIS semantic segmentation results. "
                "Best value per column in \\textbf{bold}.",
        label="tab:s3dis_summary",
        cite_in_tables=cite_in_tables,
    )

    # ── Optional per-class IoU table ──
    per_class_block = ""
    if per_class:
        iou_cols = [f"iou_{i}" for i in range(len(S3DIS_CLASSES))]
        pc_rows = _data_rows(df, iou_cols)
        per_class_block = "\n\n" + render_grouped_table(
            pc_rows,
            metric_cols=iou_cols,
            metric_headers=S3DIS_CLASSES,
            caption="S3DIS per-class IoU (\\%). "
                    "Best value per class in \\textbf{bold}.",
            label="tab:s3dis_perclass",
            cite_in_tables=cite_in_tables,
            metric_width="0.85cm",
        )

    preamble = [
        r"% Auto-generated LaTeX document — S3DIS semantic segmentation",
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
        names = df["base"].tolist() + [s for s in df["strategy"].tolist()
                                       if s not in ("-", "FF", "Full", "Full Finetuning")]
        paragraph = build_citation_paragraph(names)
        if paragraph:
            body += [paragraph, ""]
    closing = [
        "",
        r"\bibliography{references}",
        r"\end{document}",
    ]
    return "\n".join(preamble + [""] + body + [summary_table] + [per_class_block] + closing)


def render_markdown(df: pd.DataFrame, per_class: bool = False) -> str:
    from _citations import get_category

    df = df.copy()
    df["category"] = df["base"].apply(get_category)
    df = df.sort_values(["category", "base", "strategy"])

    # Summary
    cols = ["Category", "Model", "Strategy"] + [MD_HEADERS[m] for m in METRICS]
    out = ["| " + " | ".join(cols) + " |",
           "|" + "|".join(["---"] * len(cols)) + "|"]
    best = {m: df[m].astype(float).max() for m in METRICS}
    for _, r in df.iterrows():
        cells = [r["category"], r["base"], r["strategy"]]
        for m in METRICS:
            v = float(r[m])
            s = f"{v:.2f}"
            if v == best[m]:
                s = f"**{s}**"
            cells.append(s)
        out.append("| " + " | ".join(cells) + " |")

    if per_class:
        iou_cols = [f"iou_{i}" for i in range(len(S3DIS_CLASSES))]
        best_iou = {c: df[c].astype(float).max() for c in iou_cols}
        out.append("")
        out.append("### Per-class IoU (%)")
        out.append("")
        cols = ["Model", "Strategy"] + S3DIS_CLASSES
        out.append("| " + " | ".join(cols) + " |")
        out.append("|" + "|".join(["---"] * len(cols)) + "|")
        for _, r in df.iterrows():
            cells = [r["base"], r["strategy"]]
            for c in iou_cols:
                v = float(r[c])
                s = f"{v:.2f}"
                if v == best_iou[c]:
                    s = f"**{s}**"
                cells.append(s)
            out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", default="experiments/S3DIS", type=Path)
    ap.add_argument("--out_dir", default=None, type=Path,
                    help="Output dir (default: <exp_dir>/latex)")
    ap.add_argument("--per_class", action="store_true",
                    help="Also emit the per-class IoU table (13 columns, S3DIS classes).")
    ap.add_argument("--use_citation_in_tables", action="store_true",
                    help="Put \\citep{} next to each model name in the tables. "
                         "Default: cite methods once in a paragraph above the tables.")
    args = ap.parse_args()

    df = collect(args.exp_dir)

    if args.out_dir is None:
        args.out_dir = args.exp_dir / "latex"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "s3dis_comparison.csv"
    tex_path = args.out_dir / "s3dis_comparison.tex"
    md_path = args.out_dir / "s3dis_comparison.md"

    iou_cols = [f"iou_{i}" for i in range(len(S3DIS_CLASSES))]
    keep_cols = ["raw_label", "base", "strategy"] + METRICS + iou_cols
    keep_cols = [c for c in keep_cols if c in df.columns]
    df[keep_cols].to_csv(csv_path, index=False)
    tex_path.write_text(render_latex(df, args.use_citation_in_tables,
                                     per_class=args.per_class) + "\n")
    md_path.write_text(render_markdown(df, per_class=args.per_class) + "\n")

    from _citations import copy_references_bib
    copy_references_bib(args.out_dir)

    print(render_markdown(df, per_class=args.per_class))
    print()
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {tex_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
