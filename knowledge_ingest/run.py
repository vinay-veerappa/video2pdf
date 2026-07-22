"""
Entry point. Edit config/config.py (or override below), then:
    python -m knowledge_ingest.run                # ingest transcripts -> units
    python -m knowledge_ingest.run --build-vectors # load units -> LanceDB
"""
import argparse
from .config.config import PipelineConfig
from .pipeline.ingest import IngestPipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="override input dir")
    ap.add_argument("--output", help="override output dir")
    ap.add_argument("--source-type", default=None, help="transcript|pdf|own_doc")
    ap.add_argument("--max-files", type=int, default=None, help="process only first N files")
    ap.add_argument("--file-filter", default=None, help="glob to pick files, e.g. '*Review*'")
    ap.add_argument("--no-skip", action="store_true", help="force re-run (skip_existing=False)")
    ap.add_argument("--build-vectors", action="store_true",
                    help="skip ingest; load existing units into LanceDB")
    ap.add_argument("--units", nargs="+", default=None,
                    help="units dir(s) for --build-vectors (default: <output>/units)")
    ap.add_argument("--db", default="./knowledge.lancedb")
    ap.add_argument("--ict-aware", action="store_true",
                    help="use ICT-aware prompts (embeds ICT domain knowledge)")
    args = ap.parse_args()

    cfg = PipelineConfig()
    if args.input:
        cfg.input_dir = args.input
    if args.output:
        cfg.output_dir = args.output
    if args.source_type:
        cfg.source_type = args.source_type
    if args.max_files is not None:
        cfg.max_files = args.max_files
    if args.file_filter:
        cfg.file_filter = args.file_filter
    if args.no_skip:
        cfg.skip_existing = False
    if args.ict_aware:
        cfg.ict_aware = True

    if args.build_vectors:
        from .pipeline.vector_store import build_lancedb
        units = args.units or [f"{cfg.output_dir}/units"]
        if len(units) > 1:
            from .multidir import assert_no_collisions
            assert_no_collisions(units)
        build_lancedb(units, db_path=args.db,
                      embed_model=cfg.ollama.embed_model)
        return

    IngestPipeline(cfg).run()


if __name__ == "__main__":
    main()
