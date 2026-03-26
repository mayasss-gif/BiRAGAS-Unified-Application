"""
BiRAGAS CRISPR Complete v3.0 — CLI Entry Point (DNA + RNA)
"""
import argparse, json, logging, sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(name)s | %(message)s', datefmt='%H:%M:%S')

def main():
    p = argparse.ArgumentParser(description="BiRAGAS CRISPR Complete v3.0 — DNA + RNA Analysis Platform")
    p.add_argument('--serve', action='store_true', help='Start web server')
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--analyze', type=str, help='Run pipeline on CRISPR directory')
    p.add_argument('--disease', type=str, default='Disease')
    p.add_argument('--output', type=str, default='./biragas_output')
    p.add_argument('--design', type=str, help='Design guides (DNA or RNA)')
    p.add_argument('--knockout', type=str, help='Knockout/knockdown strategy')
    p.add_argument('--nuclease', type=str, default='NGG', help='NGG/NNGRRT/TTTV/Cas13a/Cas13b/Cas13d/dCas13')
    p.add_argument('--target-type', type=str, default='auto', choices=['DNA','RNA','auto'])
    p.add_argument('--n-guides', type=int, default=4)
    p.add_argument('--capabilities', action='store_true')
    p.add_argument('--base-edit', type=str, help='RNA base editing: A-to-I or C-to-U')
    p.add_argument('--crispri', type=str, help='Design CRISPRi guides for gene')
    p.add_argument('--crispra', type=str, help='Design CRISPRa guides for gene')
    args = p.parse_args()

    if args.serve:
        from .api.server import run_server
        run_server(port=args.port)
    elif args.analyze:
        from .pipeline.unified_orchestrator import UnifiedOrchestrator
        orch = UnifiedOrchestrator({'verbose': True})
        report = orch.run(crispr_dir=args.analyze, disease_name=args.disease, output_dir=args.output)
        print(json.dumps(report, indent=2, default=str))
    elif args.design:
        from .core.editing_engine import EditingEngine
        e = EditingEngine()
        guides = e.design_guides(args.design, n_guides=args.n_guides,
                                  nuclease=args.nuclease, target_type=args.target_type)
        tt = guides[0].target_type if guides else args.target_type
        print(f"\n{'='*60}")
        print(f"  {'RNA' if tt == 'RNA' else 'DNA'} Guide Design: {args.design} ({args.nuclease})")
        print(f"{'='*60}")
        for i, g in enumerate(guides, 1):
            w = ', '.join(g.warnings) if g.warnings else 'Clean'
            extra = f"  KD: {g.knockdown_efficiency:.0f}%" if tt == "RNA" else ""
            print(f"  {i}. {g.sequence}  Score: {g.composite_score:.2f}  GC: {g.gc_content:.0%}{extra}  {w}")
    elif args.knockout:
        from .core.editing_engine import EditingEngine
        e = EditingEngine()
        s = e.design_knockout_strategy(args.knockout, n_guides=args.n_guides,
                                        nuclease=args.nuclease, target_type=args.target_type)
        label = "Knockdown" if s.target_type == "RNA" else "Knockout"
        print(f"\n{'='*60}")
        print(f"  {label} Strategy: {s.gene} ({s.target_type}) | {s.n_configs} configs | Max: {s.expected_efficiency:.0%}")
        print(f"{'='*60}")
        for c in s.configs:
            print(f"  {c['config_id']:<20} {c['type']:<15} Eff: {c['expected_efficiency']:.1%}  KO: {c['ko_probability']:.1%}")
    elif args.crispri:
        from .rna.transcriptome_engine import TranscriptomeEngine
        t = TranscriptomeEngine()
        guides = t.design_crispri_guides(args.crispri, n_guides=args.n_guides)
        print(f"\nCRISPRi guides for {args.crispri}:")
        for g in guides:
            print(f"  {g.guide_sequence}  Score: {g.guide_score:.2f}  FC: {g.expected_fold_change:+.1f}x  TSS dist: {g.distance_to_tss}")
    elif args.crispra:
        from .rna.transcriptome_engine import TranscriptomeEngine
        t = TranscriptomeEngine()
        guides = t.design_crispra_guides(args.crispra, n_guides=args.n_guides)
        print(f"\nCRISPRa guides for {args.crispra}:")
        for g in guides:
            print(f"  {g.guide_sequence}  Score: {g.guide_score:.2f}  FC: {g.expected_fold_change:+.1f}x  TSS dist: {g.distance_to_tss}")
    elif args.capabilities:
        from .pipeline.unified_orchestrator import UnifiedOrchestrator
        print(json.dumps(UnifiedOrchestrator().get_capabilities(), indent=2, default=str))
    else:
        p.print_help()
        print("\n  Quick start: python -m BiRAGAS_CRISPR_Complete --serve")

if __name__ == '__main__':
    main()
