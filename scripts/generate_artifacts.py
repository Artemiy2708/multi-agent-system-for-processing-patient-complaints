"""Generate report figures, tables, MLOps manifest and functional test outputs.

Run after scripts/run_pipeline.py has created reports/experiment_summary.json.
"""
from patient_complaints_ai.pipeline import (
    create_all_report_artifacts,
    load_experiment_summary,
    run_functional_tests,
    save_mlops_manifest,
)


def main() -> None:
    summary = load_experiment_summary()
    artifacts = create_all_report_artifacts(build_confusion_matrix=False)
    manifest = save_mlops_manifest(summary)
    tests = run_functional_tests(summary)
    print("Artifacts:")
    print(artifacts)
    print("MLOps manifest:", manifest)
    print("Functional tests:")
    print(tests)


if __name__ == "__main__":
    main()
