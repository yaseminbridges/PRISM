"""Command-line interface for PRISM.

Three subcommands:

  prism run <phenopacket> <exomiser>
    Re-ranks a single case and writes output as JSON, plain-text summary, or HTML.
    Use --llm ollama to run with a real model; default is mock (deterministic, no model).

  prism batch <phenopacket_dir> --exomiser-dir <dir> --output-dir <dir>
    Re-ranks all phenopackets in a directory, writing one .html and one .json per case.
    Add --skip-existing to skip cases that already have output files — useful for
    rerunning after partial failures without reprocessing completed cases.

  prism pheval-gene-result <results_dir> --output-dir <dir> --phenopacket-dir <dir>
    Converts PRISM JSON report files into PhEval gene result parquet files for
    standardised benchmarking against other tools.

Output formats for `run`:
  json    — full RankedReport as JSON (default; machine-readable)
  summary — plain-text table (human-readable; good for quick inspection)
  html    — interactive HTML report with colour-coded scores and expandable feature details
"""
import argparse
import html as _html_escape
import sys
from pathlib import Path

from pheval.post_processing.post_processing import SortOrder

from prism.pheval_export import IDENTIFIER_TYPES, export_pheval_gene_results
from prism._paths import HPO_OBO, HPOA
from prism.pipeline import PRISMConfig, PRISMResources, run, run_with_resources
from prism.reasoning.llm import LLMClient, MockLLMClient


# ---------------------------------------------------------------------------
# Plain-text summary
# ---------------------------------------------------------------------------

def _summary(report) -> str:
    lines: list[str] = []
    lines.append(f"PRISM report — case: {report.case_id}")
    lines.append("=" * 70)

    if report.callable_diagnoses:
        lines.append("\nCALLABLE DIAGNOSIS:")
        for rc in report.callable_diagnoses:
            gene = rc.candidate.gene_symbol or "?"
            did  = rc.candidate.disease_id or "?"
            name = rc.candidate.disease_name or did
            score = f"{rc.fit.fit_score:.3f}" if rc.fit.fit_score is not None else "?"
            mech = rc.mechanism.verdict if rc.mechanism else "?"
            lines.append(f"  Gene: {gene}  ID: {did}  fit={score}  mechanism={mech}")
            lines.append(f"  {name}")
            if did in report.narratives:
                lines.append(f"  Narrative: {report.narratives[did]}")
    else:
        lines.append("\nNo callable diagnosis (insufficient evidence).")

    if report.disambiguation:
        d = report.disambiguation
        lines.append(f"\nDISAMBIGUATION:")
        lines.append(f"  Candidates: {', '.join(d.candidate_disease_ids)}")
        lines.append(f"  Best supported: {d.best_supported or 'none'}")
        lines.append(f"  {d.rationale}")

    n = min(10, len(report.reranked))
    lines.append(f"\nTOP {n} RE-RANKED CANDIDATES:")
    header = f"  {'#':>3}  {'Was':>3}  {'Gene':<10}  {'DiseaseID':<14}  {'Disease Name':<30}  {'Fit':>6}  {'Mech':<12}  Move"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for rc in report.reranked[:n]:
        gene  = (rc.candidate.gene_symbol or "")[:9]
        did   = (rc.candidate.disease_id or "")[:13]
        name  = (rc.candidate.disease_name or did)[:29]
        score = f"{rc.fit.fit_score:.3f}" if rc.fit.fit_score is not None else "     ?"
        mech  = (rc.mechanism.verdict if rc.mechanism else "unknown")[:11]
        delta = rc.old_rank - rc.new_rank
        arrow = f"↑{delta}" if delta > 0 else (f"↓{abs(delta)}" if delta < 0 else "=")
        lines.append(
            f"  {rc.new_rank:>3}  {rc.old_rank:>3}  {gene:<10}  {did:<14}  {name:<30}  {score:>6}  {mech:<12}  {arrow}"
        )
        fit = rc.fit
        if fit.matched:
            terms = ", ".join(fm.disease_feature.hpo_label for fm in fit.matched[:4])
            suffix = f" (+{len(fit.matched)-4} more)" if len(fit.matched) > 4 else ""
            lines.append(f"       matched   : {terms}{suffix}")
        if fit.partial:
            for fm in fit.partial:
                lines.append(f"       LLM partial: {fm.patient_term.label} ≈ {fm.disease_feature.hpo_label}")
                lines.append(f"                  provenance: {fm.provenance[:100]}")
        if fit.unexplained:
            terms = ", ".join(t.label for t in fit.unexplained[:3])
            suffix = f" (+{len(fit.unexplained)-3} more)" if len(fit.unexplained) > 3 else ""
            lines.append(f"       unexplained: {terms}{suffix}")
        if fit.expected_absent:
            terms = ", ".join(f.hpo_label for f in fit.expected_absent[:3])
            lines.append(f"       absent    : {terms}")
        if fit.contradictions:
            terms = ", ".join(f.hpo_label for f in fit.contradictions)
            lines.append(f"       contradiction: {terms}")
        if rc.candidate.variants:
            for v in rc.candidate.variants:
                lines.append(f"       variant: {v.variant_id}  [{v.acmg or '?'}]  {(v.consequence or '').lower()}")
        candidate_did = rc.candidate.disease_id or rc.candidate.gene_symbol
        if candidate_did in report.narratives:
            import textwrap
            for ln in textwrap.wrap(report.narratives[candidate_did], width=70):
                lines.append(f"       narrative : {ln}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def _e(text: str) -> str:
    """HTML-escape a string."""
    return _html_escape.escape(str(text))


def _fit_class(score: float | None) -> str:
    if score is None:
        return "fit-na"
    if score >= 0.5:
        return "fit-high"
    if score >= 0:
        return "fit-mid"
    return "fit-low"


def _mech_class(verdict: str) -> str:
    return {"congruent": "mech-congruent", "incongruent": "mech-incongruent"}.get(
        verdict, "mech-uncertain"
    )


def _html_report(report) -> str:
    h: list[str] = []

    # ---- head ----
    h.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>PRISM — {_e(report.case_id)}</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:system-ui,sans-serif;font-size:14px;color:#1a1a2e;background:#f4f6f9;padding:2rem}}
  h1{{font-size:1.6rem;color:#0f3460;margin-bottom:.25rem}}
  h2{{font-size:1.1rem;color:#0f3460;margin:1.5rem 0 .5rem}}
  .meta{{color:#555;font-size:.85rem;margin-bottom:1.5rem}}
  /* callable banner */
  .callable{{background:#e8f5e9;border-left:4px solid #43a047;border-radius:4px;padding:.75rem 1rem;margin-bottom:1rem}}
  .callable h3{{color:#2e7d32;margin-bottom:.4rem}}
  .callable .narrative{{font-style:italic;color:#444;margin-top:.4rem;font-size:.9rem}}
  /* no-call banner */
  .no-call{{background:#fff8e1;border-left:4px solid #ffa000;border-radius:4px;padding:.6rem 1rem;color:#5d4037;margin-bottom:1rem}}
  /* disambiguation */
  .disambig{{background:#e3f2fd;border-left:4px solid #1e88e5;border-radius:4px;padding:.75rem 1rem;margin-bottom:1rem;font-size:.9rem}}
  .disambig strong{{color:#0d47a1}}
  /* main table */
  table{{border-collapse:collapse;width:100%;background:#fff;border-radius:6px;overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,.1)}}
  thead tr{{background:#0f3460;color:#fff}}
  th{{padding:9px 12px;text-align:left;font-weight:600;font-size:.85rem;white-space:nowrap}}
  tbody tr:nth-child(even){{background:#f8f9fb}}
  tbody tr:hover{{background:#e8eef7}}
  td{{padding:7px 12px;vertical-align:top;border-bottom:1px solid #e0e0e0}}
  td.rank{{font-weight:700;color:#0f3460;text-align:center;width:42px}}
  td.was{{text-align:center;color:#888;width:42px}}
  td.gene{{font-weight:600}}
  td.did{{font-family:monospace;font-size:.82rem;color:#333}}
  td.score{{text-align:right;font-weight:700;font-family:monospace}}
  td.move{{text-align:center;font-weight:700}}
  /* score colours */
  .fit-high{{color:#2e7d32}}
  .fit-mid {{color:#e65100}}
  .fit-low {{color:#c62828}}
  .fit-na  {{color:#9e9e9e}}
  /* mechanism badges */
  .badge{{display:inline-block;border-radius:3px;padding:1px 7px;font-size:.78rem;font-weight:600;white-space:nowrap}}
  .mech-congruent  {{background:#c8e6c9;color:#1b5e20}}
  .mech-uncertain  {{background:#fff3e0;color:#bf360c}}
  .mech-incongruent{{background:#ffcdd2;color:#b71c1c}}
  /* movement */
  .up  {{color:#2e7d32}}
  .dn  {{color:#c62828}}
  .eq  {{color:#9e9e9e}}
  /* feature detail section */
  details{{margin-top:5px}}
  details summary{{cursor:pointer;color:#1565c0;font-size:.82rem;user-select:none}}
  details summary:hover{{text-decoration:underline}}
  .feat-grid{{display:grid;grid-template-columns:130px 1fr;gap:3px 10px;font-size:.82rem;margin-top:6px;padding:6px 0 2px 4px}}
  .feat-label{{color:#757575;font-weight:600}}
  .feat-val{{color:#212121}}
  .tag{{display:inline-block;background:#e8eaf6;color:#283593;border-radius:3px;padding:0 5px;font-size:.75rem;margin-right:3px;white-space:nowrap}}
  .tag-llm{{background:#fce4ec;color:#880e4f}}
  .tag-absent{{background:#fbe9e7;color:#bf360c}}
  .tag-contra{{background:#fce4ec;color:#b71c1c}}
  .tag-unexp{{background:#f3e5f5;color:#6a1b9a}}
  /* narrative box */
  .narrative-box{{background:#e3f2fd;border-radius:4px;padding:7px 10px;margin-top:6px;font-style:italic;font-size:.87rem;color:#0d47a1;line-height:1.5}}
  .narrative-box .llm-credit{{font-size:.75rem;color:#555;font-style:normal;margin-top:3px}}
  /* callable highlight row */
  tr.callable-row{{outline:2px solid #43a047;outline-offset:-1px}}
</style>
</head>
<body>
""")

    # ---- header ----
    h.append(f"<h1>PRISM Report</h1>")
    h.append(f'<p class="meta">Case: <strong>{_e(report.case_id)}</strong></p>')

    # ---- callable banner ----
    callable_ids = {rc.candidate.disease_id for rc in report.callable_diagnoses}
    if report.callable_diagnoses:
        for rc in report.callable_diagnoses:
            did = rc.candidate.disease_id or rc.candidate.gene_symbol or ""
            name = rc.candidate.disease_name or did
            gene = rc.candidate.gene_symbol or "?"
            score = f"{rc.fit.fit_score:.3f}" if rc.fit.fit_score is not None else "?"
            mech = rc.mechanism.verdict if rc.mechanism else "?"
            h.append(f"""<div class="callable">
  <h3>&#10003; Callable Diagnosis</h3>
  <strong>{_e(name)}</strong> &nbsp;
  <span class="tag">{_e(gene)}</span>
  <span class="tag">{_e(did)}</span>
  <span class="badge {_mech_class(mech)}">{_e(mech)}</span>
  &nbsp; fit&nbsp;=&nbsp;<span class="{_fit_class(rc.fit.fit_score)}">{_e(score)}</span>""")
            if did in report.narratives:
                h.append(f'  <p class="narrative">{_e(report.narratives[did])}</p>')
            h.append("</div>")
    else:
        h.append('<div class="no-call">&#9888; No callable diagnosis — insufficient evidence to make a high-confidence call.</div>')

    # ---- disambiguation ----
    if report.disambiguation:
        d = report.disambiguation
        ids_str = ", ".join(_e(i) for i in d.candidate_disease_ids)
        best = _e(d.best_supported) if d.best_supported else "<em>none</em>"
        h.append(f"""<div class="disambig">
  <strong>Disambiguation:</strong> overlapping candidates: {ids_str}<br>
  Best supported: <strong>{best}</strong><br>
  <span style="color:#333">{_e(d.rationale)}</span>
</div>""")

    # ---- main table ----
    n = len(report.reranked)
    h.append(f"<h2>Re-ranked Candidates ({n} total)</h2>")
    h.append("""<table>
<thead><tr>
  <th>#</th><th>Was</th><th>Gene</th><th>Disease ID</th><th>Disease Name</th>
  <th>Fit &#x25BC;</th><th>Mechanism</th><th>Move</th>
</tr></thead>
<tbody>""")

    for rc in report.reranked:
        gene = rc.candidate.gene_symbol or ""
        did  = rc.candidate.disease_id or ""
        name = rc.candidate.disease_name or did
        score_val = rc.fit.fit_score
        score_str = f"{score_val:.3f}" if score_val is not None else "—"
        mech_v = rc.mechanism.verdict if rc.mechanism else "unknown"
        delta = rc.old_rank - rc.new_rank
        if delta > 0:
            arrow = f'<span class="up">&#8593;{delta}</span>'
        elif delta < 0:
            arrow = f'<span class="dn">&#8595;{abs(delta)}</span>'
        else:
            arrow = '<span class="eq">=</span>'

        row_class = ' class="callable-row"' if did in callable_ids else ""
        h.append(f'<tr{row_class}>')
        h.append(f'  <td class="rank">{rc.new_rank}</td>')
        h.append(f'  <td class="was">{rc.old_rank}</td>')
        h.append(f'  <td class="gene">{_e(gene)}</td>')
        h.append(f'  <td class="did">{_e(did)}</td>')

        # Disease name + expandable feature detail
        detail_html = _feature_detail(rc, report.narratives)
        h.append(f'  <td>{_e(name)}{detail_html}</td>')

        h.append(f'  <td class="score {_fit_class(score_val)}">{_e(score_str)}</td>')
        h.append(f'  <td><span class="badge {_mech_class(mech_v)}">{_e(mech_v)}</span>')
        if rc.mechanism:
            h.append(f'    <br><span style="font-size:.75rem;color:#555">{_e(rc.mechanism.reason[:80])}</span>')
        h.append('  </td>')
        h.append(f'  <td class="move">{arrow}</td>')
        h.append('</tr>')

    h.append("</tbody></table>")
    h.append("</body></html>")
    return "\n".join(h)


def _feature_detail(rc, narratives: dict) -> str:
    """Build the expandable <details> block for one candidate row."""
    fit = rc.fit
    did = rc.candidate.disease_id or rc.candidate.gene_symbol or ""
    parts: list[str] = []

    if fit.matched:
        tags = "".join(
            f'<span class="tag">{_e(fm.disease_feature.hpo_label)}</span>'
            for fm in fit.matched
        )
        parts.append(f'<div class="feat-label">Matched</div><div class="feat-val">{tags}</div>')

    if fit.partial:
        rows = []
        for fm in fit.partial:
            rows.append(
                f'<span class="tag tag-llm">LLM</span> '
                f'{_e(fm.patient_term.label)} &#8776; {_e(fm.disease_feature.hpo_label)}'
                f'<br><span style="font-size:.75rem;color:#555">{_e(fm.provenance[:120])}</span>'
            )
        parts.append(f'<div class="feat-label">LLM partial</div><div class="feat-val">{"<br>".join(rows)}</div>')

    if fit.unexplained:
        tags = "".join(
            f'<span class="tag tag-unexp">{_e(t.label)}</span>'
            for t in fit.unexplained
        )
        parts.append(f'<div class="feat-label">Unexplained</div><div class="feat-val">{tags}</div>')

    if fit.expected_absent:
        tags = "".join(
            f'<span class="tag tag-absent">{_e(f.hpo_label)}</span>'
            for f in fit.expected_absent
        )
        parts.append(f'<div class="feat-label">Expected absent</div><div class="feat-val">{tags}</div>')

    if fit.age_excused:
        tags = "".join(
            f'<span class="tag" style="background:#e0f7fa;color:#006064">{_e(f.hpo_label)}</span>'
            for f in fit.age_excused
        )
        parts.append(f'<div class="feat-label">Age excused</div><div class="feat-label">{tags}</div>')

    if fit.contradictions:
        tags = "".join(
            f'<span class="tag tag-contra">{_e(f.hpo_label)}</span>'
            for f in fit.contradictions
        )
        parts.append(f'<div class="feat-label">Contradiction</div><div class="feat-val">{tags}</div>')

    if rc.candidate.variants:
        rows = []
        for v in rc.candidate.variants:
            acmg_style = {
                "PATHOGENIC": "color:#b71c1c;font-weight:bold",
                "LIKELY_PATHOGENIC": "color:#c62828",
                "UNCERTAIN_SIGNIFICANCE": "color:#f57f17",
                "VUS": "color:#f57f17",
                "LIKELY_BENIGN": "color:#2e7d32",
                "BENIGN": "color:#1b5e20",
            }.get((v.acmg or "").upper(), "color:#555")
            acmg_str = f'<span style="{acmg_style}">{_e(v.acmg or "?")}</span>'
            cons_str = _e((v.consequence or "").replace("_", " ").lower())
            rows.append(f'<span style="font-family:monospace;font-size:.8rem">{_e(v.variant_id)}</span> {acmg_str} {cons_str}')
        parts.append(f'<div class="feat-label">Variants</div><div class="feat-val">{"<br>".join(rows)}</div>')

    narrative_html = ""
    if did in narratives:
        narrative_html = (
            f'<div class="narrative-box">{_e(narratives[did])}'
            f'<div class="llm-credit">&#129302; LLM narrative</div></div>'
        )

    if not parts and not narrative_html:
        return ""

    grid = f'<div class="feat-grid">{"".join(parts)}</div>' if parts else ""
    return (
        f"<details><summary>details &amp; evidence</summary>"
        f"{grid}{narrative_html}</details>"
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_llm(args) -> LLMClient:
    if args.llm == "ollama":
        from prism.reasoning.llm import OllamaLLMClient
        print(
            f"[PRISM] LLM backend: Ollama  model={args.llm_model}  url={args.llm_url}",
            file=sys.stderr,
        )
        return OllamaLLMClient(
            model=args.llm_model,
            base_url=args.llm_url,
            verbose=args.verbose,
        )
    if args.verbose:
        print("[PRISM] LLM backend: Mock (deterministic)", file=sys.stderr)
    return MockLLMClient()


def _build_config(args) -> PRISMConfig:
    return PRISMConfig(
        hpo_path=args.hpo,
        hpoa_path=args.hpoa,
        top_n=args.top_n,
        rescore_mode=args.mode,
        narrative_top_n=args.narrative_top_n,
    )


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--hpo",  type=Path, default=HPO_OBO)
    p.add_argument("--hpoa", type=Path, default=HPOA)
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument(
        "--narrative-top-n", type=int, default=3,
        help="Generate LLM narrative for this many top candidates (default: 3)",
    )
    p.add_argument("--mode", choices=["conservative", "aggressive"], default="conservative")
    p.add_argument("--llm", choices=["mock", "ollama"], default="mock")
    p.add_argument("--llm-model", default="qwen2.5:7b")
    p.add_argument("--llm-url", default="http://localhost:11434")
    p.add_argument("--verbose", "-v", action="store_true")


# ---------------------------------------------------------------------------
# 'run' subcommand — single case
# ---------------------------------------------------------------------------

def _cmd_run(args) -> None:
    llm = _build_llm(args)
    config = _build_config(args)
    report = run(args.phenopacket, args.exomiser, config=config, llm=llm)

    if args.format == "summary":
        output = _summary(report)
    elif args.format == "html":
        output = _html_report(report)
    else:
        output = report.model_dump_json(indent=2)

    if args.output:
        args.output.write_text(output)
        print(f"Written to {args.output}")
    else:
        print(output)


# ---------------------------------------------------------------------------
# 'batch' subcommand — directory of cases
# ---------------------------------------------------------------------------

def _find_exomiser(phenopacket: Path, exomiser_dir: Path) -> Path | None:
    """Locate the Exomiser parquet paired with a phenopacket.

    Convention: {stem}-exomiser.parquet lives in exomiser_dir.
    The phenopacket stem may already end in '-phenopacket'; strip that too.
    """
    stem = phenopacket.stem
    # Strip a trailing '-phenopacket' suffix if present
    if stem.endswith("-phenopacket"):
        stem = stem[: -len("-phenopacket")]
    candidate = exomiser_dir / f"{stem}-exomiser.parquet"
    return candidate if candidate.exists() else None


def _cmd_batch(args) -> None:
    phenopacket_dir: Path = args.phenopacket_dir
    exomiser_dir: Path = args.exomiser_dir or phenopacket_dir
    output_dir: Path = args.output_dir
    raw_dir = output_dir / "raw_results"
    raw_dir.mkdir(parents=True, exist_ok=True)

    phenopackets = sorted(phenopacket_dir.glob("*.json"))
    if not phenopackets:
        print(f"No *.json files found in {phenopacket_dir}", file=sys.stderr)
        sys.exit(1)

    llm = _build_llm(args)
    config = _build_config(args)
    resources = PRISMResources(config)

    ok = skipped = failed = 0
    for pp in phenopackets:
        exomiser_path = _find_exomiser(pp, exomiser_dir)
        if exomiser_path is None:
            print(f"  [skip]  {pp.name}  — no matching exomiser parquet in {exomiser_dir}", file=sys.stderr)
            skipped += 1
            continue

        stem = pp.stem
        if stem.endswith("-phenopacket"):
            stem = stem[: -len("-phenopacket")]

        if args.skip_existing:
            json_path = raw_dir / f"{stem}.json"
            html_path = raw_dir / f"{stem}.html"
            if json_path.exists() or html_path.exists():
                print(f"  [skip]  {pp.name}  — output already exists", file=sys.stderr)
                skipped += 1
                continue

        print(f"  [run]   {pp.name}", file=sys.stderr, end="", flush=True)
        try:
            report = run_with_resources(pp, exomiser_path, resources, llm=llm)
        except Exception as exc:
            print(f"  FAILED: {exc}", file=sys.stderr)
            failed += 1
            continue

        html_path = raw_dir / f"{stem}.html"
        json_path = raw_dir / f"{stem}.json"
        html_path.write_text(_html_report(report))
        json_path.write_text(report.model_dump_json(indent=2))

        callable_label = (
            report.callable_diagnoses[0].candidate.gene_symbol
            if report.callable_diagnoses else "no-call"
        )
        print(f"  →  {callable_label}  [{html_path.name}, {json_path.name}]", file=sys.stderr)
        ok += 1

    print(
        f"\nBatch complete: {ok} succeeded, {skipped} skipped (no parquet), {failed} failed.",
        file=sys.stderr,
    )
    if failed:
        sys.exit(1)

    if ok > 0:
        print("\n[PRISM] Running PhEval post-processing...", file=sys.stderr)
        sort_order = SortOrder.DESCENDING if args.sort_order == "descending" else SortOrder.ASCENDING
        export_pheval_gene_results(
            results_dir=raw_dir,
            output_dir=output_dir,
            phenopacket_dir=phenopacket_dir,
            suffix=args.suffix,
            identifier_type=args.identifier_type,
            sort_order=sort_order,
        )
        print(f"[PRISM] PhEval results written to {output_dir / 'pheval_gene_results'}", file=sys.stderr)


# ---------------------------------------------------------------------------
# 'pheval-gene-result' subcommand — export PRISM reports to PhEval format
# ---------------------------------------------------------------------------

def _cmd_pheval_gene_result(args) -> None:
    sort_order = SortOrder.DESCENDING if args.sort_order == "descending" else SortOrder.ASCENDING
    export_pheval_gene_results(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        phenopacket_dir=args.phenopacket_dir,
        suffix=args.suffix,
        identifier_type=args.identifier_type,
        sort_order=sort_order,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="prism",
        description="PRISM — phenotype re-ranking layer on top of Exomiser",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # ---- run ----
    run_p = sub.add_parser("run", help="Re-rank a single case")
    run_p.add_argument("phenopacket", type=Path, help="GA4GH phenopacket JSON")
    run_p.add_argument("exomiser", type=Path, help="Exomiser results parquet")
    run_p.add_argument("--output", type=Path, default=None, help="Write output to file")
    run_p.add_argument(
        "--format", choices=["json", "summary", "html"], default="json",
        help="Output format (default: json)",
    )
    _add_common_args(run_p)
    run_p.set_defaults(func=_cmd_run)

    # ---- batch ----
    batch_p = sub.add_parser(
        "batch",
        help="Re-rank all cases in a directory, writing .html and .json per case",
    )
    batch_p.add_argument(
        "phenopacket_dir", type=Path,
        help="Directory containing GA4GH phenopacket JSON files",
    )
    batch_p.add_argument(
        "--exomiser-dir", type=Path, default=None,
        help="Directory containing Exomiser parquet files "
             "(default: same as phenopacket_dir)",
    )
    batch_p.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory to write output files ({stem}.html and {stem}.json per case)",
    )
    batch_p.add_argument(
        "--skip-existing", action="store_true",
        help="Skip cases that already have .json and .html output (useful for rerunning after failures)",
    )
    batch_p.add_argument(
        "--identifier-type", choices=IDENTIFIER_TYPES, default="ensembl_id",
        help="Gene identifier type for PhEval export (default: ensembl_id)",
    )
    batch_p.add_argument(
        "--suffix", default=".json",
        help="File suffix for PhEval export (default: .json)",
    )
    batch_p.add_argument(
        "--sort-order", choices=["ascending", "descending"], default="descending",
        help="Score sort order for PhEval export (default: descending)",
    )
    _add_common_args(batch_p)
    batch_p.set_defaults(func=_cmd_batch)

    # ---- pheval-gene-result ----
    pheval_p = sub.add_parser(
        "pheval-gene-result",
        help="Convert PRISM re-ranked JSON reports into PhEval gene result parquet files",
    )
    pheval_p.add_argument(
        "results_dir", type=Path,
        help="Directory containing PRISM report JSON files (each with a 'reranked' field)",
    )
    pheval_p.add_argument(
        "--output-dir", type=Path, required=True,
        help="PhEval output directory (parquet files are written under "
             "{output-dir}/pheval_gene_results/)",
    )
    pheval_p.add_argument(
        "--phenopacket-dir", type=Path, required=True,
        help="Directory containing the corresponding GA4GH phenopackets",
    )
    pheval_p.add_argument(
        "--suffix", default=".json",
        help="File suffix used to find report files in results_dir (default: .json)",
    )
    pheval_p.add_argument(
        "--identifier-type", choices=IDENTIFIER_TYPES, default="ensembl_id",
        help="Gene identifier type to resolve (default: ensembl_id)",
    )
    pheval_p.add_argument(
        "--sort-order", choices=["ascending", "descending"], default="descending",
        help="Whether higher (descending) or lower (ascending) scores rank first (default: descending)",
    )
    pheval_p.set_defaults(func=_cmd_pheval_gene_result)

    args = parser.parse_args()
    args.func(args)