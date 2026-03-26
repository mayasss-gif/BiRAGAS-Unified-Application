
# report/ipaa_report_case_studies.py
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .ipaa_report_data import (
    CohortData,
    get_entities_for_pathway,
    total_hits_for_pathway,
    safe_mkdir,
    write_text,
)


CATEGORIES = ["tf", "metabolite", "epigenetic"]


def _neglog10(p: Optional[float]) -> float:
    if p is None:
        return 0.0
    try:
        if p > 0:
            return -math.log10(p)
    except Exception:
        pass
    return 0.0


def pathway_activity_score(cohort: CohortData, pathway: str) -> float:
    """
    activity = |t| else |delta|
    """
    df = cohort.pathway_df
    cols = cohort.cols
    if df is None or df.empty or "pathway" not in df.columns:
        return 0.0

    row = df.loc[df["pathway"] == pathway]
    if row.empty:
        return 0.0

    if cols.t and cols.t in row.columns:
        v = row.iloc[0][cols.t]
        if pd.notna(v):
            return float(abs(v))

    if cols.delta and cols.delta in row.columns:
        v = row.iloc[0][cols.delta]
        if pd.notna(v):
            return float(abs(v))

    return 0.0


def pathway_significance_score(cohort: CohortData, pathway: str) -> float:
    """
    sig = -log10(FDR) else -log10(p)
    """
    df = cohort.pathway_df
    cols = cohort.cols
    if df is None or df.empty or "pathway" not in df.columns:
        return 0.0

    row = df.loc[df["pathway"] == pathway]
    if row.empty:
        return 0.0

    if cols.fdr and cols.fdr in row.columns:
        v = row.iloc[0][cols.fdr]
        if pd.notna(v):
            return _neglog10(float(v))

    if cols.p and cols.p in row.columns:
        v = row.iloc[0][cols.p]
        if pd.notna(v):
            return _neglog10(float(v))

    return 0.0


def pathway_evidence_score(cohort: CohortData, pathway: str) -> float:
    """
    evidence = log10(1 + total_entity_hits)
    """
    hits = total_hits_for_pathway(cohort.overlap_canon, pathway)
    return math.log10(1.0 + float(max(0, hits)))


def combined_pathway_score(cohort: CohortData, pathway: str) -> float:
    """
    score = activity * (1 + sig) * (1 + evidence)
    """
    a = pathway_activity_score(cohort, pathway)
    s = pathway_significance_score(cohort, pathway)
    e = pathway_evidence_score(cohort, pathway)
    return float(a * (1.0 + s) * (1.0 + e))


def significant_set(cohort: CohortData, sig_fdr: float = 0.05, top_n: int = 200) -> pd.DataFrame:
    df = cohort.pathway_df.copy()
    cols = cohort.cols
    if df is None or df.empty or "pathway" not in df.columns:
        return pd.DataFrame()

    # filter by FDR if present, else by p
    if cols.fdr and cols.fdr in df.columns:
        sdf = df.loc[df[cols.fdr].notna() & (df[cols.fdr] <= float(sig_fdr))].copy()
    elif cols.p and cols.p in df.columns:
        sdf = df.loc[df[cols.p].notna() & (df[cols.p] <= float(sig_fdr))].copy()
    else:
        sdf = df.copy()

    # cap by abs(t) if present
    if top_n and top_n > 0 and not sdf.empty:
        if cols.t and cols.t in sdf.columns:
            sdf["_abs_t"] = sdf[cols.t].abs()
            sdf = sdf.sort_values("_abs_t", ascending=False).head(int(top_n)).drop(columns=["_abs_t"])
        else:
            sdf = sdf.head(int(top_n))

    return sdf


def rank_shared_pathways(
    a: CohortData,
    b: CohortData,
    sig_fdr: float = 0.05,
    top_n: int = 200,
    limit: int = 50,
) -> List[str]:
    """
    Shared pathways between the two significant sets, ranked by mean combined score.
    """
    sa = significant_set(a, sig_fdr=sig_fdr, top_n=top_n)
    sb = significant_set(b, sig_fdr=sig_fdr, top_n=top_n)

    if sa.empty or sb.empty:
        return []

    shared = sorted(set(sa["pathway"].astype(str)).intersection(set(sb["pathway"].astype(str))))
    if not shared:
        return []

    scored = []
    for p in shared:
        s1 = combined_pathway_score(a, p)
        s2 = combined_pathway_score(b, p)
        scored.append((p, (s1 + s2) / 2.0))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [p for p, _ in scored[: max(1, int(limit))]]


def compare_entities_for_pathway(
    a: CohortData,
    b: CohortData,
    pathway: str,
    entity_limit: int = 200,
) -> Dict[str, Dict[str, List[str]]]:
    """
    Returns for each category:
      shared, a_only, b_only (lists of entity names)
    Entity sets are aggregated across directions (direction="ALL") for robustness.
    """
    out: Dict[str, Dict[str, List[str]]] = {}

    for cat in CATEGORIES:
        ea = set(get_entities_for_pathway(a.overlap_canon, pathway, cat, direction="ALL", entity_limit=entity_limit))
        eb = set(get_entities_for_pathway(b.overlap_canon, pathway, cat, direction="ALL", entity_limit=entity_limit))

        shared = sorted(ea.intersection(eb))
        a_only = sorted(ea - eb)
        b_only = sorted(eb - ea)

        out[cat] = {"shared": shared, "a_only": a_only, "b_only": b_only}

    return out


def _sample_list(x: List[str], k: int = 12) -> str:
    if not x:
        return "—"
    return ", ".join(x[: max(1, int(k))]) + (" …" if len(x) > k else "")


def build_case_studies_html(
    out_html: Path,
    cohorts: Dict[str, CohortData],
    sig_fdr: float,
    top_n: int,
    top_limit: int = 50,
    entity_limit: int = 200,
) -> None:
    """
    Generates a self-contained local HTML file under OUT_ROOT/report/.
    The pathway dropdown updates based on selected disease pair (shared top 50).
    """
    names = sorted(cohorts.keys())
    default_a = names[0] if names else ""
    default_b = names[1] if len(names) > 1 else (names[0] if names else "")

    # Precompute:
    # - significant pathway lists per cohort (for pairwise shared)
    # - combined scores per cohort for pathways in its significant set
    sig_paths: Dict[str, List[str]] = {}
    scores: Dict[str, Dict[str, float]] = {}
    entities: Dict[str, Dict[str, Dict[str, List[str]]]] = {}
    # entities[disease][pathway][category] = list of entities sorted by hit count

    for cname, cdata in cohorts.items():
        sdf = significant_set(cdata, sig_fdr=sig_fdr, top_n=top_n)
        sig_paths[cname] = sdf["pathway"].astype(str).tolist() if not sdf.empty else []
        scores[cname] = {}
        entities[cname] = {}

        # compute scores & entity lists only for pathways in significant set (caps size)
        for p in sig_paths[cname]:
            scores[cname][p] = combined_pathway_score(cdata, p)
            entities[cname][p] = {}
            for cat in CATEGORIES:
                entities[cname][p][cat] = get_entities_for_pathway(
                    cdata.overlap_canon, p, cat, direction="ALL", entity_limit=entity_limit
                )

    payload = {
        "cohorts": names,
        "sig_fdr": sig_fdr,
        "top_n": top_n,
        "top_limit": top_limit,
        "categories": CATEGORIES,
        "sig_paths": sig_paths,
        "scores": scores,
        "entities": entities,
        "default_a": default_a,
        "default_b": default_b,
    }

    # Simple SVG bar chart rendering in JS, no external libs.
        # Simple SVG bar chart rendering in JS, no external libs.
    # IMPORTANT: Do NOT use f-strings here because the JS contains `${...}` template literals.
    html = """<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>IPAA Mechanistic Case Studies</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; }
    .row { display: flex; gap: 16px; flex-wrap: wrap; }
    .card { border: 1px solid #ddd; border-radius: 12px; padding: 16px; min-width: 320px; flex: 1; }
    h1 { margin-top: 0; }
    select { padding: 8px; border-radius: 8px; border: 1px solid #bbb; }
    .small { color: #555; font-size: 13px; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border-bottom: 1px solid #eee; padding: 8px; text-align: left; vertical-align: top; }
    code { background: #f6f6f6; padding: 2px 6px; border-radius: 6px; }
    .btns button { margin-right: 8px; padding: 6px 10px; border-radius: 10px; border: 1px solid #bbb; background: #fafafa; cursor: pointer; }
    .btns button:hover { background: #f0f0f0; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }
    .warn { color: #a33; }
  </style>
</head>
<body>
  <h1>Mechanistic case studies (interactive compare)</h1>
  <p class="small">
    Pick any two diseases and a shared pathway. Shared vs unique entities update live per layer.
    Pathway dropdown is limited to the <b>__TOP_LIMIT__</b> shared pathways for the selected disease pair,
    ranked by <code>|t|</code> + <code>FDR</code> + entity-hit evidence.
  </p>

  <div class="row">
    <div class="card">
      <h3>Compare diseases</h3>
      <div class="row">
        <div>
          <div class="small">Disease A</div>
          <select id="diseaseA"></select>
        </div>
        <div>
          <div class="small">Disease B</div>
          <select id="diseaseB"></select>
        </div>
      </div>

      <div style="height:12px;"></div>

      <div class="small">Pathway (top shared)</div>
      <select id="pathwaySel" style="width:100%;"></select>

      <div style="height:12px;"></div>
      <div class="small">Quick pick: Top 3 (for current disease pair)</div>
      <div class="btns" id="quickBtns"></div>

      <div style="height:12px;"></div>
      <div class="small">Default: <span class="mono" id="defaultLabel"></span></div>
    </div>

    <div class="card">
      <h3>Case study plot</h3>
      <div class="small">Shared vs unique entity counts (updates live).</div>
      <div id="plotArea"></div>
      <div class="small warn" id="emptyWarn" style="margin-top:10px;"></div>
    </div>
  </div>

  <div class="card" style="margin-top:16px;">
    <h3>Entity lists (live)</h3>
    <table>
      <thead>
        <tr>
          <th>Entity type</th>
          <th>Shared (sample)</th>
          <th id="aOnlyHdr">A only (sample)</th>
          <th id="bOnlyHdr">B only (sample)</th>
        </tr>
      </thead>
      <tbody id="entityTable"></tbody>
    </table>

    <div style="height:12px;"></div>
    <div class="small"><b>Full lists</b></div>
    <div id="fullLists"></div>
  </div>

<script id="DATA" type="application/json">__PAYLOAD_JSON__</script>
<script>
const DATA = JSON.parse(document.getElementById("DATA").textContent);

function uniq(arr) {
  return Array.from(new Set(arr));
}

function intersect(a, b) {
  const B = new Set(b);
  return a.filter(x => B.has(x));
}

function diff(a, b) {
  const B = new Set(b);
  return a.filter(x => !B.has(x));
}

function sample(arr, k=12) {
  if (!arr || arr.length === 0) return "—";
  const s = arr.slice(0, k).join(", ");
  return arr.length > k ? (s + " …") : s;
}

function scoreFor(disease, pathway) {
  const m = DATA.scores[disease] || {};
  const v = m[pathway];
  return (typeof v === "number") ? v : 0;
}

function sharedTopPathways(a, b) {
  const pa = DATA.sig_paths[a] || [];
  const pb = DATA.sig_paths[b] || [];
  const shared = intersect(pa, pb);
  // rank by mean combined score
  const ranked = shared
    .map(p => [p, (scoreFor(a,p) + scoreFor(b,p)) / 2.0])
    .sort((x,y) => y[1] - x[1])
    .slice(0, DATA.top_limit)
    .map(x => x[0]);
  return ranked;
}

function setOptions(sel, options, selected) {
  sel.innerHTML = "";
  options.forEach(o => {
    const opt = document.createElement("option");
    opt.value = o;
    opt.textContent = o;
    if (o === selected) opt.selected = true;
    sel.appendChild(opt);
  });
}

function entityList(disease, pathway, cat) {
  const d = DATA.entities[disease] || {};
  const p = d[pathway] || {};
  return (p[cat] || []);
}

function renderPlot(counts) {
  // counts: {cat: {shared: n, a_only: n, b_only: n}}
  const width = 520;
  const barH = 18;
  const gap = 18;
  const cats = DATA.categories;

  const maxV = Math.max(1, ...cats.flatMap(c => [counts[c].shared, counts[c].a_only, counts[c].b_only]));
  const scale = (v) => Math.round((v / maxV) * 420);

  let svg = `<svg width="${width}" height="${cats.length*(barH*3+gap)+20}" xmlns="http://www.w3.org/2000/svg">`;
  svg += `<text x="0" y="14" font-size="12" fill="#555">Counts (scaled)</text>`;

  let y = 28;
  cats.forEach(cat => {
    const s = counts[cat].shared;
    const ao = counts[cat].a_only;
    const bo = counts[cat].b_only;

    svg += `<text x="0" y="${y-6}" font-size="12" fill="#111">${cat.toUpperCase()}</text>`;

    svg += `<rect x="0" y="${y}" width="${scale(s)}" height="${barH}" fill="#444"/><text x="${scale(s)+6}" y="${y+13}" font-size="12" fill="#111">shared: ${s}</text>`;
    y += barH + 4;
    svg += `<rect x="0" y="${y}" width="${scale(ao)}" height="${barH}" fill="#888"/><text x="${scale(ao)+6}" y="${y+13}" font-size="12" fill="#111">A only: ${ao}</text>`;
    y += barH + 4;
    svg += `<rect x="0" y="${y}" width="${scale(bo)}" height="${barH}" fill="#bbb"/><text x="${scale(bo)+6}" y="${y+13}" font-size="12" fill="#111">B only: ${bo}</text>`;
    y += barH + gap;
  });

  svg += `</svg>`;
  document.getElementById("plotArea").innerHTML = svg;
}

function renderEntities(a, b, pathway) {
  document.getElementById("aOnlyHdr").textContent = `${a} only (sample)`;
  document.getElementById("bOnlyHdr").textContent = `${b} only (sample)`;
  document.getElementById("defaultLabel").textContent = `${a} vs ${b} :: ${pathway}`;

  const tbody = document.getElementById("entityTable");
  tbody.innerHTML = "";

  const full = document.getElementById("fullLists");
  full.innerHTML = "";

  let anyNonEmpty = false;

  const counts = {};

  DATA.categories.forEach(cat => {
    const ea = uniq(entityList(a, pathway, cat));
    const eb = uniq(entityList(b, pathway, cat));

    const sh = intersect(ea, eb);
    const ao = diff(ea, eb);
    const bo = diff(eb, ea);

    if (sh.length || ao.length || bo.length) anyNonEmpty = true;

    counts[cat] = {
      shared: sh.length,
      a_only: ao.length,
      b_only: bo.length
    };

    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><b>${cat.toUpperCase()}</b></td>
      <td>${sample(sh)}</td>
      <td>${sample(ao)}</td>
      <td>${sample(bo)}</td>
    `;
    tbody.appendChild(tr);

    const block = document.createElement("div");
    block.style.marginTop = "10px";
    block.innerHTML = `
      <div class="small"><b>${cat.toUpperCase()}</b></div>
      <div class="small">Shared:</div>
      <div class="mono">${sh.length ? sh.join(", ") : "—"}</div>
      <div class="small">Unique to ${a}:</div>
      <div class="mono">${ao.length ? ao.join(", ") : "—"}</div>
      <div class="small">Unique to ${b}:</div>
      <div class="mono">${bo.length ? bo.join(", ") : "—"}</div>
    `;
    full.appendChild(block);
  });

  renderPlot(counts);

  const warn = document.getElementById("emptyWarn");
  if (!anyNonEmpty) {
    warn.textContent =
      "No entities found for this pathway in one or both diseases. This usually means: (1) pathway missing in overlap JSON, or (2) naming mismatch. Try another pathway from the dropdown.";
  } else {
    warn.textContent = "";
  }
}

function renderQuickButtons(pathways) {
  const box = document.getElementById("quickBtns");
  box.innerHTML = "";
  pathways.slice(0,3).forEach(p => {
    const btn = document.createElement("button");
    btn.textContent = p;
    btn.onclick = () => {
      setOptions(document.getElementById("pathwaySel"), pathways, p);
      onSelectionChanged();
    };
    box.appendChild(btn);
  });
}

function onDiseaseChanged() {
  const a = document.getElementById("diseaseA").value;
  const b = document.getElementById("diseaseB").value;
  const top = sharedTopPathways(a,b);
  setOptions(document.getElementById("pathwaySel"), top, top[0] || "");
  renderQuickButtons(top);
  onSelectionChanged();
}

function onSelectionChanged() {
  const a = document.getElementById("diseaseA").value;
  const b = document.getElementById("diseaseB").value;
  const p = document.getElementById("pathwaySel").value;
  if (!a || !b || !p) return;
  renderEntities(a,b,p);
}

(function init() {
  const selA = document.getElementById("diseaseA");
  const selB = document.getElementById("diseaseB");
  setOptions(selA, DATA.cohorts, DATA.default_a);
  setOptions(selB, DATA.cohorts, DATA.default_b);

  selA.onchange = onDiseaseChanged;
  selB.onchange = onDiseaseChanged;
  document.getElementById("pathwaySel").onchange = onSelectionChanged;

  onDiseaseChanged();
})();
</script>
</body>
</html>
"""

    # Inject only what we need safely:
    html = html.replace("__TOP_LIMIT__", str(int(top_limit)))
    html = html.replace("__PAYLOAD_JSON__", json.dumps(payload))

    safe_mkdir(out_html.parent)
    write_text(out_html, html)

    safe_mkdir(out_html.parent)
    write_text(out_html, html)


@dataclass
class StaticCaseStudy:
    pathway: str
    table_md: str
    full_md: str


def build_static_case_studies_markdown(
    a: CohortData,
    b: CohortData,
    pathways: List[str],
    sample_k: int = 12,
    entity_limit: int = 200,
) -> List[StaticCaseStudy]:
    """
    Static blocks for Markdown/PDF (top 3 pathways).
    """
    out: List[StaticCaseStudy] = []
    a_name = a.name
    b_name = b.name

    for p in pathways:
        comp = compare_entities_for_pathway(a, b, p, entity_limit=entity_limit)

        # Build compact counts table with samples
        rows = []
        for cat in CATEGORIES:
            sh = comp[cat]["shared"]
            ao = comp[cat]["a_only"]
            bo = comp[cat]["b_only"]
            rows.append({
                "Entity type": cat.upper(),
                "Shared (sample)": _sample_list(sh, sample_k),
                f"{a_name} only (sample)": _sample_list(ao, sample_k),
                f"{b_name} only (sample)": _sample_list(bo, sample_k),
            })
        tdf = pd.DataFrame(rows)
        table_md = tdf.to_markdown(index=False)

        # Full lists (still text)
        lines = []
        lines.append(f"#### {p}\n")
        lines.append(f"Default: {a_name} vs {b_name}\n")
        lines.append("Shared vs unique entity counts (static in PDF; updates in HTML).\n\n")
        lines.append(table_md + "\n\n")
        lines.append("Below are lists (HTML is live; here is static text).\n\n")

        for cat in CATEGORIES:
            sh = comp[cat]["shared"]
            ao = comp[cat]["a_only"]
            bo = comp[cat]["b_only"]
            lines.append(f"**{cat.upper()}**\n")
            lines.append(f"- Shared: {', '.join(sh) if sh else '—'}\n")
            lines.append(f"- {a_name} only: {', '.join(ao) if ao else '—'}\n")
            lines.append(f"- {b_name} only: {', '.join(bo) if bo else '—'}\n\n")

        out.append(StaticCaseStudy(pathway=p, table_md=table_md, full_md="\n".join(lines)))

    return out
