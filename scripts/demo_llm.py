"""Run LLM-based demo answer generation using existing final_law_chroma.

Set environment variables before running:
    export OPENAI_API_KEY="sk-or-..."
    export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
    export OPENAI_MODEL="openai/gpt-oss-120b:free"

The script does not retrain models.
"""
import json
import os

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from patient_complaints_ai.pipeline import (
    PatientComplaintMultiAgentSystem,
    VECTOR_DIR,
    REPORTS_DIR,
    load_experiment_summary,
    save_json,
)


DEMO_COMPLAINT = """
Я была на приеме в клинике Альфа 20.05.2026. Врач почти ничего не объяснил,
администратор разговаривал грубо, я ждала больше часа, а потом мне выставили
счет выше, чем обещали по телефону. Прошу разобраться.
"""


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Create an OpenRouter/OpenAI key and set it as an environment variable.")

    summary = load_experiment_summary()
    best_model = summary.get("best_rag_embedding_model", "sentence-transformers/LaBSE")

    final_dir = VECTOR_DIR / "final_law_chroma"
    if not final_dir.exists():
        raise FileNotFoundError(f"{final_dir} does not exist. Run the full pipeline first.")

    vectordb = Chroma(
        persist_directory=str(final_dir),
        embedding_function=HuggingFaceEmbeddings(model_name=best_model),
        collection_name="law_docs",
    )

    system = PatientComplaintMultiAgentSystem(vectordb, use_llm=True)
    result = system.run(DEMO_COMPLAINT)
    save_json(result, REPORTS_DIR / "example_multiagent_result_llm.json")

    print(json.dumps(result, ensure_ascii=False, indent=2)[:8000])
    print("\nGenerated response:\n")
    print(result["response"])


if __name__ == "__main__":
    main()
