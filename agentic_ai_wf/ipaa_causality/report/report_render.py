#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ayass Bioscience — Cross-Disease Report (RENDER + PDF + UI)
==========================================================

File: report_render.py  (CODE 3 / 3)

This module is responsible for:
- Rendering the HTML report (Jinja2 template embedded here)
- Writing JS/CSS assets for interactivity (case-study compare + dropdowns)
- Ensuring DEFAULT case-study panels auto-populate on page load (fix for "empty" lists)
- Exporting PDF (Chrome/Edge headless preferred; WeasyPrint fallback)
- Writing report_artifact.json next to the report

Key feature requested:
- Show 3 top pathways by default as 3 panels
- Each panel has a dropdown that allows choosing any of the TOP 50 pathways
- Entity lists/counts must be populated for those top pathways (via CASE_DATA)

NOTE:
- This module does NOT decide which pathways are "top" or build CASE_DATA.
  It expects caller (core_report.py) to pass:
    - top_pathways (len<=50)  (for dropdown options)
    - default_case_pathways (len==3)  (initial panels)
    - CASE_DATA (pathway->disease->layer->entities)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import weasyprint  # type: ignore
import re

from jinja2 import Template  # type: ignore
_HAVE_JINJA2 = True


# -------------------------
# HTML helpers
# -------------------------
def _html_escape(s: Any) -> str:
    s = "" if s is None else str(s)
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def df_to_interactive_table(
    df: pd.DataFrame,
    table_id: str,
    title: str = "",
    max_rows: int = 5000,
) -> str:
    """
    Builds a lightweight searchable/sortable table.
    Sorting/search are handled by report_ui.js (no external libs).

    Caller should keep df not huge; we cap rows anyway.
    """
    if df is None or df.empty:
        return f"<div class='card'><h3>{_html_escape(title)}</h3><p class='muted'>No data.</p></div>"

    df2 = df.copy()
    if len(df2) > int(max_rows):
        df2 = df2.head(int(max_rows))

    cols = [str(c) for c in df2.columns]
    head = "".join(f"<th data-col='{_html_escape(c)}'>{_html_escape(c)}</th>" for c in cols)

    rows_html = []
    for _, r in df2.iterrows():
        tds = []
        for c in cols:
            v = r.get(c, "")
            tds.append(f"<td>{_html_escape(v)}</td>")
        rows_html.append("<tr>" + "".join(tds) + "</tr>")

    title_html = f"<h3>{_html_escape(title)}</h3>" if title else ""
    return f"""
    <div class="card">
      {title_html}
      <div class="table-tools">
        <input class="table-search" type="text" placeholder="Search..." data-target="{_html_escape(table_id)}"/>
      </div>
      <div class="table-wrap">
        <table id="{_html_escape(table_id)}" class="data-table">
          <thead><tr>{head}</tr></thead>
          <tbody>
            {''.join(rows_html)}
          </tbody>
        </table>
      </div>
      <p class="muted small">Tip: click a column header to sort.</p>
    </div>
    """


# -------------------------
# Assets (CSS + JS)
# -------------------------
CSS_TEXT = """
:root{
  --bg:#0b0f14; --card:#0f1621; --muted:#9bb0c4; --text:#e7eef7;
  --accent:#6ea8fe; --accent2:#7ee0b8; --danger:#ff6b6b; --line:#1f2a3a;
  --chip:#131c28;
  --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  --sans: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji","Segoe UI Emoji";
}
*{box-sizing:border-box}
body{margin:0;font-family:var(--sans);background:var(--bg);color:var(--text)}
a{color:var(--accent);text-decoration:none}
a:hover{text-decoration:underline}
.container{max-width:1180px;margin:0 auto;padding:22px}
.header{display:flex;gap:16px;align-items:flex-start;justify-content:space-between;margin-bottom:18px}
.h-title{font-size:22px;margin:0 0 6px 0}
.h-sub{color:var(--muted);margin:0}
.grid{display:grid;grid-template-columns:1fr;gap:14px}
@media(min-width:980px){.grid-2{grid-template-columns:1fr 1fr}.grid-3{grid-template-columns:1fr 1fr 1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:16px;box-shadow:0 1px 0 rgba(255,255,255,0.03)}
h2{margin:0 0 10px 0;font-size:18px}
h3{margin:0 0 10px 0;font-size:16px}
.muted{color:var(--muted)}
.small{font-size:12px}
.kpi{display:flex;gap:10px;flex-wrap:wrap}
.chip{background:var(--chip);border:1px solid var(--line);border-radius:999px;padding:6px 10px;font-size:12px;color:var(--muted)}
hr.sep{border:none;border-top:1px solid var(--line);margin:16px 0}
.table-wrap{overflow:auto;border-radius:12px;border:1px solid var(--line)}
table{border-collapse:collapse;width:100%;min-width:540px}
th,td{padding:10px 10px;border-bottom:1px solid var(--line);font-size:13px}
th{position:sticky;top:0;background:#101a28;cursor:pointer;user-select:none}
tr:hover td{background:#0e1723}
.table-tools{display:flex;justify-content:flex-end;margin:8px 0 10px 0}
.table-search{width:260px;max-width:100%;padding:8px 10px;border-radius:10px;border:1px solid var(--line);
  background:#0b121c;color:var(--text)}
label{font-size:12px;color:var(--muted);display:block;margin-bottom:6px}
select{width:100%;padding:9px 10px;border-radius:10px;border:1px solid var(--line);background:#0b121c;color:var(--text)}
.row{display:grid;grid-template-columns:1fr;gap:10px}
@media(min-width:980px){.row-2{grid-template-columns:1fr 1fr}.row-3{grid-template-columns:1fr 1fr 1fr}}
.badge{display:inline-block;padding:2px 8px;border-radius:999px;background:#101a28;border:1px solid var(--line);color:var(--muted);font-size:12px}
.listgrid{display:grid;grid-template-columns:1fr;gap:10px}
@media(min-width:980px){.listgrid{grid-template-columns:1fr 1fr 1fr}}
.entitybox{border:1px solid var(--line);border-radius:14px;padding:12px;background:#0b121c}
.entitybox h4{margin:0 0 8px 0;font-size:13px}
.entitybox .counts{display:flex;gap:10px;flex-wrap:wrap;margin:0 0 8px 0}
.entitybox ul{margin:0;padding-left:18px;color:var(--muted);font-size:12px}
.entitybox li{margin:4px 0}
.plotbox{border:1px dashed var(--line);border-radius:14px;padding:10px;background:#0b121c}
.plotrow{display:flex;gap:12px;align-items:stretch}
.plotcol{flex:1}
svg{max-width:100%;height:auto}
.footer{margin-top:16px;color:var(--muted);font-size:12px}
.code{font-family:var(--mono);font-size:12px;background:#0b121c;border:1px solid var(--line);border-radius:12px;padding:10px;white-space:pre-wrap}
"""


JS_TEXT = r"""
(function(){
  function $(sel, root){ return (root||document).querySelector(sel); }
  function $all(sel, root){ return Array.from((root||document).querySelectorAll(sel)); }

  // ---------- Table search + sort ----------
  function normalizeText(s){ return String(s||'').toLowerCase(); }

  function installTableSearch(){
    $all('input.table-search').forEach(inp=>{
      inp.addEventListener('input', ()=>{
        const tid = inp.getAttribute('data-target');
        const table = document.getElementById(tid);
        if(!table) return;
        const q = normalizeText(inp.value);
        const rows = Array.from(table.tBodies[0].rows);
        rows.forEach(tr=>{
          const text = normalizeText(tr.innerText);
          tr.style.display = (q==='' || text.indexOf(q)>=0) ? '' : 'none';
        });
      });
    });
  }

  function installTableSort(){
    $all('table.data-table thead th').forEach(th=>{
      th.addEventListener('click', ()=>{
        const table = th.closest('table');
        if(!table) return;
        const idx = Array.from(th.parentNode.children).indexOf(th);
        const tbody = table.tBodies[0];
        const rows = Array.from(tbody.rows).filter(r=>r.style.display !== 'none');

        const cur = th.getAttribute('data-sort') || 'none';
        const next = (cur==='asc') ? 'desc' : 'asc';
        $all('th', table.tHead).forEach(x=>x.removeAttribute('data-sort'));
        th.setAttribute('data-sort', next);

        rows.sort((a,b)=>{
          const A = a.cells[idx] ? a.cells[idx].innerText.trim() : '';
          const B = b.cells[idx] ? b.cells[idx].innerText.trim() : '';
          const aNum = parseFloat(A.replace(/,/g,'')); const bNum = parseFloat(B.replace(/,/g,''));
          const aIsNum = !isNaN(aNum) && A !== ''; const bIsNum = !isNaN(bNum) && B !== '';
          if(aIsNum && bIsNum){
            return next==='asc' ? (aNum-bNum) : (bNum-aNum);
          }
          const cmp = A.localeCompare(B);
          return next==='asc' ? cmp : -cmp;
        });

        rows.forEach(r=>tbody.appendChild(r));
      });
    });
  }

  // ---------- Case study logic ----------
  function getCaseData(){
    try{
      const el = document.getElementById('CASE_DATA_JSON');
      if(!el) return {};
      return JSON.parse(el.textContent || '{}');
    }catch(e){ return {}; }
  }
  function getMeta(){
    try{
      const el = document.getElementById('REPORT_META_JSON');
      if(!el) return {};
      return JSON.parse(el.textContent || '{}');
    }catch(e){ return {}; }
  }

  function unique(list){
    const out=[]; const seen=new Set();
    (list||[]).forEach(x=>{
      const v = String(x||'').trim();
      if(!v) return;
      if(seen.has(v)) return;
      seen.add(v); out.push(v);
    });
    return out;
  }

  function toNameList(items){
    // items can be ["STAT1"] OR [{name:"STAT1",score:..}]
    if(!items) return [];
    if(Array.isArray(items)){
      return unique(items.map(it=>{
        if(typeof it === 'string') return it;
        if(it && typeof it === 'object') return it.name || it.entity || it.id || '';
        return '';
      }));
    }
    return [];
  }

  function setOps(aList, bList){
    const A = new Set(aList||[]);
    const B = new Set(bList||[]);
    const shared = []; const aOnly=[]; const bOnly=[];
    A.forEach(x=>{ if(B.has(x)) shared.push(x); else aOnly.push(x); });
    B.forEach(x=>{ if(!A.has(x)) bOnly.push(x); });
    shared.sort(); aOnly.sort(); bOnly.sort();
    return {shared, aOnly, bOnly};
  }

  function renderList(ul, items, limit){
    if(!ul) return;
    const L = (limit==null) ? items.length : Math.min(items.length, limit);
    ul.innerHTML = '';
    if(L===0){
      const li = document.createElement('li'); li.textContent = '—';
      ul.appendChild(li); return;
    }
    for(let i=0;i<L;i++){
      const li = document.createElement('li'); li.textContent = items[i];
      ul.appendChild(li);
    }
    if(items.length > L){
      const li = document.createElement('li'); li.textContent = `… (+${items.length-L} more)`;
      ul.appendChild(li);
    }
  }

  function renderCounts(panel, typ, sharedN, aN, bN){
    const el = panel.querySelector(`[data-count="${typ}"]`);
    if(!el) return;
    el.textContent = `${sharedN} shared • ${aN} A-only • ${bN} B-only`;
  }

  function renderMiniBar(svg, sharedN, aN, bN){
    if(!svg) return;
    const W = 320, H = 40;
    const total = sharedN + aN + bN;
    const sW = total ? Math.round(W*(sharedN/total)) : 0;
    const aW = total ? Math.round(W*(aN/total)) : 0;
    const bW = Math.max(0, W - sW - aW);

    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    svg.innerHTML = `
      <rect x="0" y="8" width="${W}" height="24" rx="10" ry="10" fill="#101a28" stroke="#1f2a3a"></rect>
      <rect x="0" y="8" width="${sW}" height="24" rx="10" ry="10" fill="#6ea8fe"></rect>
      <rect x="${sW}" y="8" width="${aW}" height="24" fill="#7ee0b8"></rect>
      <rect x="${sW+aW}" y="8" width="${bW}" height="24" rx="10" ry="10" fill="#ff6b6b"></rect>
    `;
  }

  function updatePanel(panel){
    const caseData = getCaseData();
    const meta = getMeta();

    const pathwaySel = panel.querySelector('select[data-role="pathway"]');
    const aSel = panel.querySelector('select[data-role="diseaseA"]');
    const bSel = panel.querySelector('select[data-role="diseaseB"]');

    const pathway = pathwaySel ? pathwaySel.value : '';
    const A = aSel ? aSel.value : '';
    const B = bSel ? bSel.value : '';

    const title = panel.querySelector('[data-role="panelTitle"]');
    if(title) title.textContent = pathway || '—';

    const node = (caseData && pathway && caseData[pathway]) ? caseData[pathway] : {};
    const aNode = node && A ? (node[A] || {}) : {};
    const bNode = node && B ? (node[B] || {}) : {};

    // Ensure we always get lists
    const aTF = toNameList(aNode.tf); const bTF = toNameList(bNode.tf);
    const aMET = toNameList(aNode.metabolite); const bMET = toNameList(bNode.metabolite);
    const aEPI = toNameList(aNode.epigenetic); const bEPI = toNameList(bNode.epigenetic);

    const tf = setOps(aTF, bTF);
    const met = setOps(aMET, bMET);
    const epi = setOps(aEPI, bEPI);

    // counts badges + mini plot
    renderCounts(panel, 'tf', tf.shared.length, tf.aOnly.length, tf.bOnly.length);
    renderCounts(panel, 'metabolite', met.shared.length, met.aOnly.length, met.bOnly.length);
    renderCounts(panel, 'epigenetic', epi.shared.length, epi.aOnly.length, epi.bOnly.length);

    // mini plot for each type
    renderMiniBar(panel.querySelector('svg[data-plot="tf"]'), tf.shared.length, tf.aOnly.length, tf.bOnly.length);
    renderMiniBar(panel.querySelector('svg[data-plot="metabolite"]'), met.shared.length, met.aOnly.length, met.bOnly.length);
    renderMiniBar(panel.querySelector('svg[data-plot="epigenetic"]'), epi.shared.length, epi.aOnly.length, epi.bOnly.length);

    // lists
    renderList(panel.querySelector('ul[data-list="tf_shared"]'), tf.shared, 18);
    renderList(panel.querySelector('ul[data-list="tf_aonly"]'), tf.aOnly, 18);
    renderList(panel.querySelector('ul[data-list="tf_bonly"]'), tf.bOnly, 18);

    renderList(panel.querySelector('ul[data-list="met_shared"]'), met.shared, 18);
    renderList(panel.querySelector('ul[data-list="met_aonly"]'), met.aOnly, 18);
    renderList(panel.querySelector('ul[data-list="met_bonly"]'), met.bOnly, 18);

    renderList(panel.querySelector('ul[data-list="epi_shared"]'), epi.shared, 18);
    renderList(panel.querySelector('ul[data-list="epi_aonly"]'), epi.aOnly, 18);
    renderList(panel.querySelector('ul[data-list="epi_bonly"]'), epi.bOnly, 18);

    // status note
    const note = panel.querySelector('[data-role="sourceNote"]');
    if(note){
      if(!pathway || !A || !B){
        note.textContent = 'Select a pathway and two diseases.';
      }else{
        const ok = (aTF.length+aMET.length+aEPI.length+bTF.length+bMET.length+bEPI.length) > 0;
        note.textContent = ok ? 'Entities loaded from CASE_DATA.' : 'No entities found for this pathway/disease combo (CASE_DATA empty).';
      }
    }
  }

  function installPanels(){
    $all('.case-panel').forEach(panel=>{
      $all('select', panel).forEach(sel=>{
        sel.addEventListener('change', ()=>updatePanel(panel));
      });
    });
  }

  function populateSelect(sel, values, selected){
    if(!sel) return;
    sel.innerHTML = '';
    (values||[]).forEach(v=>{
      const opt = document.createElement('option');
      opt.value = v; opt.textContent = v;
      sel.appendChild(opt);
    });
    if(selected && values && values.indexOf(selected)>=0){
      sel.value = selected;
    }
  }

  function boot(){
    installTableSearch();
    installTableSort();

    const meta = getMeta();
    const diseases = meta.diseases || [];
    const topPathways = meta.top_pathways || [];
    const defaults = meta.default_case_pathways || [];

    // Fill each panel dropdowns
    const panels = $all('.case-panel');
    panels.forEach((panel, idx)=>{
      const pSel = panel.querySelector('select[data-role="pathway"]');
      const aSel = panel.querySelector('select[data-role="diseaseA"]');
      const bSel = panel.querySelector('select[data-role="diseaseB"]');

      populateSelect(pSel, topPathways, defaults[idx] || topPathways[idx] || '');
      populateSelect(aSel, diseases, diseases[0] || '');
      populateSelect(bSel, diseases, diseases[1] || diseases[0] || '');

      updatePanel(panel); // IMPORTANT: populate immediately (fix "empty")
    });

    installPanels();
  }

  document.addEventListener('DOMContentLoaded', boot);
})();
"""


# -------------------------
# Template
# -------------------------
HTML_TEMPLATE = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>{{ title }}</title>
  <style>{{ css_text }}</style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div>
        <h1 class="h-title">{{ title }}</h1>
        <p class="h-sub">{{ subtitle }}</p>
        <div class="kpi">
          <span class="chip">Diseases: {{ diseases|length }}</span>
          <span class="chip">q cutoff: {{ q_cutoff }}</span>
          <span class="chip">Top pathways dropdown: {{ top_pathways|length }} (cap 50)</span>
          <span class="chip">Default case panels: {{ default_case_pathways|length }}</span>
        </div>
      </div>
      <div class="card" style="max-width:360px">
        <h3>Downloads</h3>
        <div class="kpi">
          <span class="chip"><a href="report_artifact.json">report_artifact.json</a></span>
          {% if pdf_name %}
          <span class="chip"><a href="{{ pdf_name }}">{{ pdf_name }}</a></span>
          {% endif %}
        </div>
        <p class="muted small">Tip: open this folder locally; links work best from the Report directory.</p>
      </div>
    </div>

    <div class="grid grid-2">
      <div>
        <div class="card">
          <h2>Key takeaways</h2>
          {% if key_takeaways and key_takeaways|length>0 %}
            <ul>
              {% for item in key_takeaways %}
                <li>{{ item }}</li>
              {% endfor %}
            </ul>
          {% else %}
            <p class="muted">No takeaways generated.</p>
          {% endif %}
        </div>

        <div class="card">
          <h2>Minimum validation checklist</h2>
          {% if validation_checklist and validation_checklist|length>0 %}
            <ul>
              {% for item in validation_checklist %}
                <li>{{ item }}</li>
              {% endfor %}
            </ul>
          {% else %}
            <p class="muted">No checklist generated.</p>
          {% endif %}
        </div>
      </div>

      <div>
        <div class="card">
          <h2>Methods snapshot</h2>
          <div class="code">{{ methods_snapshot }}</div>
          <hr class="sep"/>
          <p class="muted small">
            Entity source preference: <span class="badge">OUT_ROOT/jsons_all*</span> →
            <span class="badge">disease overlap json</span> →
            <span class="badge">ALL_COMBINED.csv</span>.
          </p>
        </div>
      </div>
    </div>

    <hr class="sep"/>

    <div class="grid grid-2">
      <div>
        {{ shared_themes_table|safe }}
      </div>
      <div>
        {{ shared_pathways_table|safe }}
      </div>
    </div>

    <div class="grid">
      {{ discordance_table|safe }}
    </div>

    <hr class="sep"/>

    <div class="card">
      <h2>Mechanistic case studies (interactive compare)</h2>
      <p class="muted">
        Three panels default to the top-3 pathways. Each panel can switch to any of the top-50 pathways.
        The PDF shows the default comparison; HTML updates live.
      </p>

      <div class="grid grid-3">
        {% for i in range(3) %}
        <div class="card case-panel" data-panel="{{ i }}">
          <h3 data-role="panelTitle">—</h3>

          <div class="row row-3">
            <div>
              <label>Pathway (top 50)</label>
              <select data-role="pathway"></select>
            </div>
            <div>
              <label>Disease A</label>
              <select data-role="diseaseA"></select>
            </div>
            <div>
              <label>Disease B</label>
              <select data-role="diseaseB"></select>
            </div>
          </div>

          <p class="muted small" data-role="sourceNote">—</p>

          <div class="listgrid">
            <div class="entitybox">
              <h4>TF</h4>
              <p class="counts"><span class="badge" data-count="tf">—</span></p>
              <div class="plotbox"><svg data-plot="tf"></svg></div>
              <p class="muted small">Shared</p><ul data-list="tf_shared"></ul>
              <p class="muted small">A-only</p><ul data-list="tf_aonly"></ul>
              <p class="muted small">B-only</p><ul data-list="tf_bonly"></ul>
            </div>

            <div class="entitybox">
              <h4>METABOLITE</h4>
              <p class="counts"><span class="badge" data-count="metabolite">—</span></p>
              <div class="plotbox"><svg data-plot="metabolite"></svg></div>
              <p class="muted small">Shared</p><ul data-list="met_shared"></ul>
              <p class="muted small">A-only</p><ul data-list="met_aonly"></ul>
              <p class="muted small">B-only</p><ul data-list="met_bonly"></ul>
            </div>

            <div class="entitybox">
              <h4>EPIGENETIC</h4>
              <p class="counts"><span class="badge" data-count="epigenetic">—</span></p>
              <div class="plotbox"><svg data-plot="epigenetic"></svg></div>
              <p class="muted small">Shared</p><ul data-list="epi_shared"></ul>
              <p class="muted small">A-only</p><ul data-list="epi_aonly"></ul>
              <p class="muted small">B-only</p><ul data-list="epi_bonly"></ul>
            </div>
          </div>
        </div>
        {% endfor %}
      </div>

      <p class="muted small" style="margin-top:10px">
        If a pathway shows “No entities found…”, that means CASE_DATA does not contain entities for that pathway/disease.
        Fix is upstream: ensure overlap JSONs are being merged into CASE_DATA for those top pathways.
      </p>
    </div>

    <div class="footer">
      <p>Generated by Ayass Bioscience report pipeline.</p>
    </div>

  </div>

  <!-- Embedded JSON for offline use -->
  <script id="REPORT_META_JSON" type="application/json">{{ report_meta_json|safe }}</script>
  <script id="CASE_DATA_JSON" type="application/json">{{ case_data_json|safe }}</script>
  <script>{{ js_text }}</script>
</body>
</html>
"""


# -------------------------
# Public API
# -------------------------
@dataclass
class RenderInputs:
    title: str
    subtitle: str
    diseases: List[str]
    q_cutoff: float
    top_pathways: List[str]                 # <= 50 expected
    default_case_pathways: List[str]        # len==3 expected (pad upstream if fewer)
    case_data: Dict[str, Any]               # pathway->disease->layer->entities
    shared_themes_table_html: str = ""
    shared_pathways_table_html: str = ""
    discordance_table_html: str = ""
    key_takeaways: Optional[List[str]] = None
    validation_checklist: Optional[List[str]] = None
    methods_snapshot: str = ""
    pdf_name: Optional[str] = None
    report_artifact: Optional[Dict[str, Any]] = None


def render_report_html(inputs: RenderInputs) -> str:
    """
    Render the HTML string for the report.
    """
    report_meta = {
        "diseases": inputs.diseases,
        "q_cutoff": inputs.q_cutoff,
        "top_pathways": inputs.top_pathways,
        "default_case_pathways": inputs.default_case_pathways,
    }

    ctx = {
        "title": inputs.title,
        "subtitle": inputs.subtitle,
        "q_cutoff": inputs.q_cutoff,
        "diseases": inputs.diseases,
        "top_pathways": inputs.top_pathways,
        "default_case_pathways": inputs.default_case_pathways,
        "shared_themes_table": inputs.shared_themes_table_html or "<div class='card'><h3>Shared themes</h3><p class='muted'>No data.</p></div>",
        "shared_pathways_table": inputs.shared_pathways_table_html or "<div class='card'><h3>Shared pathways</h3><p class='muted'>No data.</p></div>",
        "discordance_table": inputs.discordance_table_html or "<div class='card'><h3>Discordance</h3><p class='muted'>No data.</p></div>",
        "key_takeaways": inputs.key_takeaways or [],
        "validation_checklist": inputs.validation_checklist or [],
        "methods_snapshot": inputs.methods_snapshot or "",
        "css_text": CSS_TEXT,
        "js_text": JS_TEXT,
        "pdf_name": inputs.pdf_name or "",
        "report_meta_json": json.dumps(report_meta, ensure_ascii=False),
        "case_data_json": json.dumps(inputs.case_data or {}, ensure_ascii=False),
    }

    if _HAVE_JINJA2:
        tmpl = Template(HTML_TEMPLATE)
        return tmpl.render(**ctx)

    # Fallback if Jinja2 absent: ultra-minimal replace (keeps core functionality)
    html = HTML_TEMPLATE
    # crude replacements for the handful we need; tables will not render loops properly without Jinja2.
    # In practice you should install jinja2; we keep this as a safety net.
    html = html.replace("{{ title }}", _html_escape(ctx["title"]))
    html = html.replace("{{ subtitle }}", _html_escape(ctx["subtitle"]))
    html = html.replace("{{ q_cutoff }}", str(ctx["q_cutoff"]))
    html = html.replace("{{ css_text }}", ctx["css_text"])
    html = html.replace("{{ js_text }}", ctx["js_text"])
    html = html.replace("{{ report_meta_json|safe }}", ctx["report_meta_json"])
    html = html.replace("{{ case_data_json|safe }}", ctx["case_data_json"])
    html = html.replace("{{ shared_themes_table|safe }}", ctx["shared_themes_table"])
    html = html.replace("{{ shared_pathways_table|safe }}", ctx["shared_pathways_table"])
    html = html.replace("{{ discordance_table|safe }}", ctx["discordance_table"])
    # strip remaining template markers
    html = re_strip_jinja(html)
    return html


def re_strip_jinja(s: str) -> str:
    """
    Emergency cleanup for non-jinja environments.
    """

    s = re.sub(r"\{\{.*?\}\}", "", s, flags=re.DOTALL)
    s = re.sub(r"\{%.*?%\}", "", s, flags=re.DOTALL)
    return s


def write_report(
    report_dir: Path,
    html_text: str,
    report_artifact: Optional[Dict[str, Any]] = None,
    html_name: str = "index.html",
) -> Path:
    """
    Writes index.html + report_artifact.json.
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    html_path = report_dir / html_name
    html_path.write_text(html_text, encoding="utf-8")

    artifact = report_artifact or {}
    artifact_path = report_dir / "report_artifact.json"
    try:
        artifact_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        # always create something
        artifact_path.write_text("{}", encoding="utf-8")

    return html_path


# -------------------------
# PDF export
# -------------------------
def _which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def _try_chrome_like() -> Optional[str]:
    # Prefer explicit binaries when available
    candidates = [
        "google-chrome", "google-chrome-stable",
        "chromium", "chromium-browser",
        "msedge", "microsoft-edge",
    ]
    for c in candidates:
        p = _which(c)
        if p:
            return p
    return None


def write_pdf(
    html_path: Path,
    pdf_path: Path,
    timeout_sec: int = 120,
    prefer_chrome: bool = True,
) -> Tuple[bool, str]:
    """
    Export PDF. Returns (ok, message).

    Chrome/Edge headless is preferred because it runs JS, so the PDF should NOT be empty.
    """
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    if prefer_chrome:
        chrome = _try_chrome_like()
        if chrome:
            # Use file:// URL
            url = html_path.resolve().as_uri()
            cmd = [
                chrome,
                "--headless=new",
                "--disable-gpu",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--run-all-compositor-stages-before-draw",
                "--virtual-time-budget=7000",  # give JS time to populate
                f"--print-to-pdf={str(pdf_path)}",
                url,
            ]
            try:
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=int(timeout_sec))
                if p.returncode == 0 and pdf_path.exists() and pdf_path.stat().st_size > 10_000:
                    return True, f"PDF generated via {Path(chrome).name}"
                # If it created tiny/empty pdf, still return diagnostics
                msg = (p.stderr or p.stdout or "").strip()
                if pdf_path.exists() and pdf_path.stat().st_size <= 10_000:
                    return False, f"Chrome created tiny PDF; likely render timing issue. stderr/stdout: {msg[:400]}"
                return False, f"Chrome/Edge export failed (rc={p.returncode}). stderr/stdout: {msg[:400]}"
            except Exception as e:
                return False, f"Chrome/Edge export exception: {e}"

    # WeasyPrint fallback (may not run JS → not recommended for your interactive sections)
    try:
        
        weasyprint.HTML(filename=str(html_path)).write_pdf(str(pdf_path))
        if pdf_path.exists() and pdf_path.stat().st_size > 10_000:
            return True, "PDF generated via WeasyPrint (JS may not execute)"
        return False, "WeasyPrint produced missing/tiny PDF"
    except Exception as e:
        return False, f"WeasyPrint unavailable/failed: {e}"


# -------------------------
# Convenience end-to-end writer
# -------------------------
def build_and_write_report(
    report_dir: Path,
    inputs: RenderInputs,
    export_pdf: bool = True,
    pdf_name: str = "report.pdf",
) -> Dict[str, Any]:
    """
    Creates index.html + report_artifact.json, and optionally report.pdf.
    Returns artifact dict enriched with paths + pdf status.
    """
    if len(inputs.top_pathways) > 50:
        inputs.top_pathways = inputs.top_pathways[:50]

    # Ensure exactly 3 default panels (pad with top_pathways if needed)
    dflt = list(inputs.default_case_pathways or [])
    while len(dflt) < 3 and len(inputs.top_pathways) > len(dflt):
        dflt.append(inputs.top_pathways[len(dflt)])
    inputs.default_case_pathways = dflt[:3]

    # Set pdf name into template context so link appears
    inputs.pdf_name = pdf_name if export_pdf else None

    html = render_report_html(inputs)
    html_path = write_report(report_dir, html, report_artifact=inputs.report_artifact or {})

    artifact = dict(inputs.report_artifact or {})
    artifact["report_dir"] = str(report_dir)
    artifact["html"] = str(html_path)

    if export_pdf:
        pdf_path = report_dir / pdf_name
        ok, msg = write_pdf(html_path, pdf_path, prefer_chrome=True)
        artifact["pdf"] = str(pdf_path) if pdf_path.exists() else ""
        artifact["pdf_ok"] = bool(ok)
        artifact["pdf_message"] = msg
        # Update artifact file after pdf
        (report_dir / "report_artifact.json").write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")

    return artifact
