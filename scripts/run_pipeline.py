"""Run the full experiment pipeline.

Usage:
    python scripts/run_pipeline.py --no-heavy-grid
"""
import argparse

from patient_complaints_ai.pipeline import run_full_pipeline, save_json, REPORTS_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-geo-reviews", action="store_true", help="Use demo/synthetic complaints instead of Yandex Geo Reviews.")
    parser.add_argument("--no-rulaw", action="store_true", help="Use demo law docs instead of RuLaw. Not recommended for final runs.")
    parser.add_argument("--heavy-grid", action="store_true", help="Run heavy RuModernBERT grid search.")
    parser.add_argument("--use-llm", action="store_true", help="Use LLM generation if OPENAI_API_KEY is set.")
    args = parser.parse_args()

    summary = run_full_pipeline(
        use_geo_reviews=not args.no_geo_reviews,
        use_rulaw=not args.no_rulaw,
        run_heavy_grid=args.heavy_grid,
        use_llm=args.use_llm,
    )
    save_json(summary, REPORTS_DIR / "experiment_summary.json")


if __name__ == "__main__":
    main()
