# BiRAGAS Unified Application

**Autonomous Bioinformatics Discovery Engine** — Ayass Bioscience LLC

839 Python modules · 17 API endpoints · 11 CRISPR engines · 28 causality modules · 88.9B combinations

## Live Demo

**[Launch BiRAGAS App](https://mayasss-gif.github.io/BiRAGAS-Unified-Application/BiRAGAS_Unified_App.html)**

## Deploy Backend (One-Click)

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/mayasss-gif/BiRAGAS-Unified-Application)

## Run Locally

```bash
git clone https://github.com/mayasss-gif/BiRAGAS-Unified-Application.git
cd BiRAGAS-Unified-Application
python3 start.py
```

## Platform

| Module | Files | Engines |
|--------|-------|---------|
| BiRAGAS CRISPR | 139 | EditingEngine, KnockoutEngine, CombinationEngine, ACEScoringEngine, ScreeningEngine, RNABaseEditEngine, TranscriptomeEngine, NonCodingEngine, FullCausalityIntegrator, SelfCorrector, MegaScaleEngine |
| Agentic AI Workflow | 697 | DEG, Pathway, Drug Discovery, Deconvolution, Single-Cell, Multi-Omics, GWAS/MR, Perturbation, IPAA Causality |
| Fallback Engines | 7 | DEGEngine, PathwayEngine, DrugDiscoveryEngine, DeconvolutionEngine, SingleCellEngine, MultiOmicsEngine, GWASEngine |

## API Endpoints (17)

| Endpoint | Engine | Plots |
|----------|--------|-------|
| `/api/guides` | EditingEngine | Bar + Doughnut |
| `/api/knockout` | KnockoutEngine | Bar + Radar |
| `/api/combinations` | CombinationEngine | Bar + Pie |
| `/api/screening` | ACEScoringEngine | Bar + Heatmap |
| `/api/rna` | RNA Engines | Bar |
| `/api/causality` | FullCausalityIntegrator | Stacked Bar + Pie |
| `/api/deg` | DEGEngine | Volcano + Heatmap |
| `/api/pathway` | PathwayEngine | Bar + Pie |
| `/api/drug` | DrugDiscoveryEngine | Bar + Pie |
| `/api/deconv` | DeconvolutionEngine | Stacked Bar + Pie |
| `/api/singlecell` | SingleCellEngine | UMAP + Bar |
| `/api/multiomics` | MultiOmicsEngine | Heatmap + Bar |
| `/api/gwas` | GWASEngine | Manhattan + Bar |
| `/api/perturbation` | CombinationEngine | Heatmap + Bar |
| `/api/pipeline` | Full Pipeline | Bar + Rankings |

## License

Proprietary — Ayass Bioscience LLC
