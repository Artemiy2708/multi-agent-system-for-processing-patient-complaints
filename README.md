# Multi-agent Patient Complaint Response System

Репозиторий содержит код ВКР/магистерской работы по разработке мультиагентной системы для обработки жалоб пациентов.

## Что делает система

Pipeline выполняет следующие этапы:

1. загрузка и фильтрация русскоязычных отзывов из Yandex Geo Reviews;
2. классификация жалоб с помощью `deepvk/RuModernBERT-base`;
3. baseline-классификация через `intfloat/multilingual-e5-base + LogisticRegression`;
4. загрузка и парсинг RuLaw XML;
5. построение RAG-индекса в Chroma;
6. сравнение embedding-моделей для RAG;
7. NER и анонимизация;
8. генерация ответа через rule-based fallback или LLM через OpenRouter/OpenAI-compatible API;
9. сохранение графиков, таблиц, MLOps manifest и functional tests.

## Основные результаты финального эксперимента

- dataset: около 29 876 негативных отзывов;
- train/test: около 23 900 / 5 976;
- RuModernBERT: Accuracy около 0.94, Macro F1 около 0.923;
- E5 + LogisticRegression baseline: Accuracy около 0.49;
- RuLaw: 3000 нормативных документов;
- RAG: лучшая embedding-модель `sentence-transformers/LaBSE`;
- LLM demo: OpenRouter-compatible generation.

## Установка

```bash
git clone <your-repo-url>
cd <repo>
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Для Colab можно установить зависимости так:

```bash
pip install -r requirements.txt
```

## Запуск полного эксперимента

```bash
python scripts/run_pipeline.py
```

Полный запуск обучает модели и строит RAG-индексы, поэтому может занимать долгое время и требует GPU.

## Генерация графиков и отчетных артефактов

После полного эксперимента:

```bash
python scripts/generate_artifacts.py
```

Артефакты сохраняются в `reports/`.

## LLM demo

Для OpenRouter:

```bash
export OPENAI_API_KEY="sk-or-..."
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
export OPENAI_MODEL="openai/gpt-oss-120b:free"
python scripts/demo_llm.py
```


## Структура

```text
patient_complaints_ai/
  pipeline.py                 # основная логика pipeline
scripts/
  run_pipeline.py             # запуск полного эксперимента
  generate_artifacts.py       # графики, таблицы, тесты, manifest
  demo_llm.py                 # LLM demo без переобучения
notebooks/
  final_with_artifacts_tests_llm.ipynb
requirements.txt
.env.example
.gitignore
```
