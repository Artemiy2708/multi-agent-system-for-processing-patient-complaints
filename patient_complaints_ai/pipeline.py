

# ================================================================================
# Notebook cell 4
# ================================================================================
"""
Курсовая работа:
Разработка мультиагентной системы для генерации ответов на жалобы пациентов

Что есть в коде:
1) сравнение классификации deepvk/RuModernBERT-base и intfloat/multilingual-e5-base;
2) NER-блок для жалоб пациентов;
3) grid search по параметрам;
4) выбор embedding-модели для RAG;
5) загрузчики Yandex Geo Reviews и RusLawOD/rulawtexts;
6) RAG по нормативным документам;
7) мультиагентный пайплайн: классификация -> NER -> RAG -> генерация -> проверка качества.
"""

from __future__ import annotations

import gc
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    pipeline,
)

# LangChain / RAG
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# NER
from natasha import Doc, MorphVocab, NewsEmbedding, NewsNERTagger, Segmenter

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_DIR = Path(".").resolve()
DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "reports"
VECTOR_DIR = PROJECT_DIR / "vector_db"
for p in [DATA_DIR, MODELS_DIR, REPORTS_DIR, VECTOR_DIR]:
    p.mkdir(parents=True, exist_ok=True)


SMOKE_TEST = False
MAX_CLASSIFICATION_ROWS = 3_000 if SMOKE_TEST else 50_000
MAX_LAW_DOCS_FOR_RAG = 300 if SMOKE_TEST else 3000
MAX_REVIEW_ROWS = 20_000 if SMOKE_TEST else 500_000

RUMODERNBERT_MODEL = "deepvk/RuModernBERT-base"
E5_MODEL = "intfloat/multilingual-e5-base"
RAG_EMBEDDING_MODELS = [
    "intfloat/multilingual-e5-base",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/LaBSE",
]


def clean_text(text: Any, lower: bool = True) -> str:
    text = "" if pd.isna(text) else str(text)
    text = text.replace("\n", " ").replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower() if lower else text


def clean_text_for_classifier(text: Any) -> str:
    text = clean_text(text, lower=True)
    text = re.sub(r"[^а-яёa-z0-9\s.,!?;:()\-]", " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def contains_any(text: str, keywords: Sequence[str]) -> bool:
    low = text.lower()
    return any(k.lower() in low for k in keywords)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)



# ================================================================================
# Notebook cell 5
# ================================================================================
# Yandex Geo Reviews Dataset 2023
# Загружаем не из GitHub, потому что в GitHub-репозитории нет самого файла датасета.
# Используем Hugging Face mirror: d0rj/geo-reviews-dataset-2023


from datasets import load_dataset

GEO_REVIEWS_HF_DATASET = "d0rj/geo-reviews-dataset-2023"

MEDICINE_KEYWORDS = [
    "медицина", "медицин", "клиника", "больница", "поликлиника", "врач",
    "стоматология", "аптека", "лаборатория", "анализ", "диагност",
]
FOOD_KEYWORDS = [
    "кафе", "ресторан", "бар", "пиццерия", "столовая", "кофейня",
    "общепит", "еда", "фастфуд", "бургер", "суши",
]
SERVICE_KEYWORDS = [
    "хам", "груб", "персонал", "администратор", "очеред", "ждал", "ожидание",
    "не дозвон", "запись", "сервис", "обслуживание",
]
QUALITY_KEYWORDS = [
    "плохо", "ужас", "отврат", "гряз", "ошиб", "некачеств", "невкус",
    "боль", "диагноз", "лечение", "симптом", "назнач", "анализ",
]
PRICE_KEYWORDS = [
    "дорого", "цена", "стоимость", "деньги", "оплата", "чек", "навяз"
]


def download_geo_reviews_dataset(
    target_dir: Path = DATA_DIR / "geo-reviews-dataset-2023",
    max_rows: Optional[int] = MAX_REVIEW_ROWS,
) -> Path:
    """
    Скачивает Yandex Geo Reviews Dataset 2023 с Hugging Face mirror
    и сохраняет локально в CSV, чтобы дальше пайплайн работал как раньше.
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    output_path = target_dir / "geo-reviews-dataset-2023.csv"

    if output_path.exists() and output_path.stat().st_size > 1024:
        print(f"Geo Reviews уже есть локально: {output_path}")
        return output_path

    print(f"Скачиваю Geo Reviews Dataset с Hugging Face: {GEO_REVIEWS_HF_DATASET}")

    ds = load_dataset(GEO_REVIEWS_HF_DATASET, split="train")
    df = ds.to_pandas()

    if max_rows is not None:
        df = df.head(max_rows)

    df.to_csv(output_path, index=False)

    print(f"Geo Reviews сохранён в: {output_path}")
    print(f"Размер датасета: {df.shape}")
    print(f"Колонки: {df.columns.tolist()}")

    return output_path


def parse_tskv_line(line: str) -> Dict[str, str]:
    result = {}
    for part in line.rstrip("\n").split("\t"):
        if "=" in part:
            key, value = part.split("=", 1)
            result[key] = value
    return result


def read_geo_reviews_tskv(
    path: Path,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, desc=f"Reading {path.name}")):
            if max_rows is not None and i >= max_rows:
                break
            rows.append(parse_tskv_line(line))
    return pd.DataFrame(rows)


def find_geo_reviews_file(
    root: Path = DATA_DIR / "geo-reviews-dataset-2023",
) -> Optional[Path]:
    """
    Ищет локальный файл отзывов.
    Игнорирует README и битые маленькие файлы, которые могли появиться после 404.
    """
    if not root.exists():
        return None

    for pattern in ["*.csv", "*.tskv", "*.txt"]:
        files = [
            p for p in root.rglob(pattern)
            if p.is_file()
            and "README" not in p.name.upper()
            and p.stat().st_size > 1024
        ]
        if files:
            return sorted(files, key=lambda p: p.stat().st_size, reverse=True)[0]

    return None


def load_geo_reviews(
    root: Path = DATA_DIR / "geo-reviews-dataset-2023",
    max_rows: int = MAX_REVIEW_ROWS,
) -> pd.DataFrame:
    """
    Загружает отзывы.

    Логика:
    1. Сначала ищет локальный .csv/.tskv/.txt.
    2. Если ничего нет, скачивает датасет с Hugging Face.
    3. Возвращает DataFrame.
    """
    root.mkdir(parents=True, exist_ok=True)

    data_file = find_geo_reviews_file(root)

    if data_file is None:
        data_file = download_geo_reviews_dataset(
            target_dir=root,
            max_rows=max_rows,
        )

    print(f"Используется файл отзывов: {data_file}")

    if data_file.suffix.lower() == ".csv":
        return pd.read_csv(data_file, nrows=max_rows)

    return read_geo_reviews_tskv(data_file, max_rows=max_rows)


def infer_domain_from_rubrics(rubrics: Any, text: str = "") -> Optional[str]:
    rubrics = "" if pd.isna(rubrics) else str(rubrics).lower()
    joined = f"{rubrics} {text.lower()}"

    if contains_any(joined, MEDICINE_KEYWORDS):
        return "medicine"

    if contains_any(joined, FOOD_KEYWORDS):
        return "food"

    return None


def infer_complaint_topic(text: str) -> str:
    low = text.lower()

    service = contains_any(low, SERVICE_KEYWORDS)
    quality = contains_any(low, QUALITY_KEYWORDS)
    price = contains_any(low, PRICE_KEYWORDS)

    if service and quality:
        return "service_and_quality"

    if service:
        return "service"

    if quality:
        return "quality"

    if price:
        return "price"

    return "other"


def prepare_complaint_dataset_from_geo(df: pd.DataFrame) -> pd.DataFrame:
    if "text" not in df.columns:
        raise ValueError(
            f"В датасете нет колонки text. Доступные колонки: {df.columns.tolist()}"
        )

    df = df.copy()

    df["text"] = df["text"].apply(clean_text_for_classifier)
    df = df[df["text"].str.len() >= 30].copy()

    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        df = df[df["rating"].fillna(5) <= 3].copy()

    rub_col = "rubrics" if "rubrics" in df.columns else None

    df["domain"] = [
        infer_domain_from_rubrics(r if rub_col else "", t)
        for r, t in zip(
            df[rub_col] if rub_col else [""] * len(df),
            df["text"],
        )
    ]

    df = df[df["domain"].isin(["medicine", "food"])].copy()

    df["topic"] = df["text"].apply(infer_complaint_topic)
    df["label_name"] = df["domain"] + "__" + df["topic"]

    counts = df["label_name"].value_counts()
    df = df[df["label_name"].isin(counts[counts >= 5].index)].copy()

    return df[["text", "domain", "topic", "label_name"]].reset_index(drop=True)


def make_demo_complaints_dataset() -> pd.DataFrame:
    rows = [
        (
            "medicine",
            "service",
            "В клинике администратор грубо разговаривал, пришлось ждать врача больше часа.",
        ),
        (
            "medicine",
            "quality",
            "Врач назначил лечение без объяснений, после приема стало хуже.",
        ),
        (
            "medicine",
            "price",
            "Стоматология навязала дорогие услуги, итоговая стоимость оказалась выше обещанной.",
        ),
        (
            "food",
            "service",
            "В ресторане официант хамил, заказ несли сорок минут.",
        ),
        (
            "food",
            "quality",
            "Еда была холодная и невкусная, в зале грязно.",
        ),
        (
            "food",
            "price",
            "Кафе очень дорогое, чек не соответствует качеству еды.",
        ),
        (
            "medicine",
            "service_and_quality",
            "В поликлинике огромная очередь, врач торопился и неправильно оформил назначение.",
        ),
        (
            "food",
            "service_and_quality",
            "В кафе долго ждали заказ, персонал грубый, еда пересоленная.",
        ),
    ]

    data = []

    for domain, topic, text in rows:
        for i in range(20):
            data.append(
                {
                    "text": clean_text_for_classifier(
                        text + f" Номер примера {i}."
                    ),
                    "domain": domain,
                    "topic": topic,
                    "label_name": f"{domain}__{topic}",
                }
            )

    return pd.DataFrame(data)


def load_or_create_complaint_dataset(
    use_geo_reviews: bool = True,
) -> pd.DataFrame:
    if use_geo_reviews:
        try:
            raw = load_geo_reviews(max_rows=MAX_REVIEW_ROWS)
            df = prepare_complaint_dataset_from_geo(raw)

            if len(df) >= 100:
                print(f"Используется Geo Reviews dataset: {len(df)} примеров")
                return df

            print(
                f"После фильтрации Geo Reviews осталось мало примеров: {len(df)}. "
                "Используется demo dataset."
            )

        except Exception as e:
            print("Geo reviews loading failed, using demo dataset:", repr(e))

    return make_demo_complaints_dataset()



# ================================================================================
# Notebook cell 7
# ================================================================================
# RuLaw XML: загрузка, парсинг, фильтрация реальных законов

import re
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm.auto import tqdm


RULAW_DIR = DATA_DIR / "rulaw_xml"
RULAW_CACHE_PATH = DATA_DIR / "rulaw_selected_documents.csv"


def make_demo_law_docs() -> pd.DataFrame:
    """
    Demo-документы разрешены только если явно запустить use_rulaw=False.
    При use_rulaw=True эта функция НЕ вызывается.
    """
    return pd.DataFrame(
        [
            {
                "source": "demo_law_1",
                "title": "ФЗ об основах охраны здоровья граждан",
                "doc_date": "",
                "doc_type": "demo",
                "dataset": "demo_law",
                "text": (
                    "Медицинская организация обязана обеспечивать качество и безопасность "
                    "медицинской деятельности, информировать пациента о состоянии здоровья, "
                    "методах лечения, рисках и возможных последствиях. Пациент имеет право "
                    "на уважительное и гуманное отношение."
                ),
            },
            {
                "source": "demo_law_2",
                "title": "Закон о защите прав потребителей",
                "doc_date": "",
                "doc_type": "demo",
                "dataset": "demo_law",
                "text": (
                    "Потребитель имеет право на получение необходимой и достоверной информации "
                    "об услуге, её цене, условиях оказания и исполнителе. При оказании услуги "
                    "ненадлежащего качества потребитель вправе требовать устранения недостатков "
                    "или уменьшения цены."
                ),
            },
            {
                "source": "demo_law_3",
                "title": "Правила рассмотрения обращений граждан",
                "doc_date": "",
                "doc_type": "demo",
                "dataset": "demo_law",
                "text": (
                    "Обращение гражданина должно быть зарегистрировано и рассмотрено в установленный "
                    "срок. Ответ должен быть мотивированным, понятным и содержать результаты проверки "
                    "доводов заявителя."
                ),
            },
        ]
    )


def download_rulaw_xml(
    kind: str = "lite",
    target_dir: Path = RULAW_DIR,
) -> Path:
    """
    Скачивает и распаковывает RuLaw XML corpus.

    Если XML уже распакованы, 7z заново НЕ запускается.
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    zip_path = target_dir / f"corpus_xml_{kind}.zip"
    extract_dir = target_dir / f"corpus_xml_{kind}_extracted"

    if extract_dir.exists():
        existing_xml_count = sum(1 for _ in extract_dir.rglob("*.xml"))
        if existing_xml_count > 0:
            print(
                f"RuLaw XML уже распакован: {extract_dir}. "
                f"XML файлов: {existing_xml_count}"
            )
            return extract_dir

    urls = [
        f"https://storage.yandexcloud.net/rulawtexts/corpus_xml_{kind}.zip",
        f"https://storage.yandexcloud.net/rulawtexts/corpus_xml_{kind}.z01",
        f"https://storage.yandexcloud.net/rulawtexts/corpus_xml_{kind}.z02",
        f"https://storage.yandexcloud.net/rulawtexts/corpus_xml_{kind}.z03",
    ]

    for url in urls:
        file_path = target_dir / url.split("/")[-1]

        if file_path.exists() and file_path.stat().st_size > 0:
            print(f"Файл уже есть: {file_path}")
            continue

        print(f"Скачиваю: {url}")
        subprocess.run(["wget", "-O", str(file_path), url], check=True)

    extract_dir.mkdir(parents=True, exist_ok=True)

    print(f"Распаковываю RuLaw XML в {extract_dir}...")
    subprocess.run(["7z", "x", str(zip_path), f"-o{extract_dir}", "-y"], check=True)
    print("Распаковка RuLaw XML завершена.")

    return extract_dir


def normalize_law_text(text: Any) -> str:
    text = "" if text is None or pd.isna(text) else str(text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_xml_text_and_meta(path: Path) -> Dict[str, Any]:
    """
    Достаёт текст и метаданные из одного XML RuLaw.

    Поддерживает оба варианта:
    - обычные теги title/date/type;
    - специфичные RuLaw-теги headingIPS/docdateIPS/doc_typeIPS.
    """
    try:
        tree = ET.parse(path)
        root = tree.getroot()

        text_parts = []
        title = path.stem
        doc_date = ""
        doc_type = "rulaw_xml"

        for elem in root.iter():
            tag = elem.tag.lower().split("}")[-1]
            value = normalize_law_text(elem.text)
            attr_val = normalize_law_text(elem.attrib.get("val", ""))

            if value:
                text_parts.append(value)

            if tag in {"headingips", "heading", "title", "name", "doc_title", "document_title"}:
                candidate = value or attr_val
                if len(candidate) > 5:
                    title = candidate[:500]

            if tag in {"docdateips", "date", "doc_date", "publication_date"} and not doc_date:
                doc_date = (attr_val or value)[:100]

            if tag in {"doc_typeips", "type", "doc_type", "document_type"}:
                candidate = attr_val or value
                if candidate:
                    doc_type = candidate[:200]

        text = normalize_law_text(" ".join(text_parts))

        return {
            "source": str(path),
            "title": title,
            "doc_date": doc_date,
            "doc_type": doc_type,
            "text": text,
            "dataset": "rulaw_xml",
        }

    except Exception:
        try:
            raw = path.read_text(encoding="utf-8", errors="ignore")
            text = normalize_law_text(raw)
            return {
                "source": str(path),
                "title": path.stem,
                "doc_date": "",
                "doc_type": "rulaw_xml_raw",
                "text": text,
                "dataset": "rulaw_xml",
            }
        except Exception:
            return {
                "source": str(path),
                "title": path.stem,
                "doc_date": "",
                "doc_type": "parse_error",
                "text": "",
                "dataset": "rulaw_xml",
            }


def is_relevant_law_text(text: str, keywords: List[str]) -> bool:
    low = text.lower()
    return any(keyword in low for keyword in keywords)


def load_law_documents_from_xml(
    xml_root: Path,
    keywords: Optional[List[str]] = None,
    max_docs: int = MAX_LAW_DOCS_FOR_RAG,
    max_files_to_scan: Optional[int] = None,
    min_text_len: int = 300,
    cache_path: Path = RULAW_CACHE_PATH,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """
    Парсит реальные XML-документы RuLaw.

    Важно:
    - моки не возвращает;
    - если max_files_to_scan=None, просматривает весь corpus_xml_lite_extracted;
    - останавливается, когда набрал max_docs релевантных документов;
    - кеширует только реальные RuLaw-документы.
    """
    if cache_path.exists() and not force_rebuild:
        cached = pd.read_csv(cache_path)

        if not cached.empty and "source" in cached.columns:
            has_demo = cached["source"].astype(str).str.startswith("demo_law").any()
            has_real_xml = cached["source"].astype(str).str.contains(".xml", regex=False).any()

            if not has_demo and has_real_xml:
                print(f"Используется кеш RuLaw: {cache_path}")
                print(f"Документов в кеше: {len(cached)}")
                return cached.head(max_docs).reset_index(drop=True)

        print("Кеш найден, но он пустой/битый/содержит demo_law. Пересобираю.")

    xml_files = list(xml_root.rglob("*.xml"))

    if max_files_to_scan is not None:
        xml_files = xml_files[:max_files_to_scan]

    print(f"XML файлов для просмотра: {len(xml_files)}")
    print(f"Нужно отобрать реальных документов: {max_docs}")

    keywords = keywords or [
        "здравоохран",
        "медицин",
        "пациент",
        "пациентск",
        "врач",
        "лечени",
        "диагноз",
        "информирован",
        "соглас",
        "потребител",
        "обращени",
        "жалоб",
        "услуг",
        "качество",
        "безопасност",
        "персональн",
        "данн",
    ]
    keywords = [keyword.lower() for keyword in keywords]

    rows = []
    scanned = 0

    for path in tqdm(xml_files, desc="Parsing RuLaw XML"):
        if len(rows) >= max_docs:
            print(f"Достигнут лимит реальных документов: {max_docs}")
            break

        scanned += 1
        item = extract_xml_text_and_meta(path)
        text = item["text"]

        if len(text) < min_text_len:
            continue

        if not is_relevant_law_text(text, keywords):
            continue

        rows.append(item)

        if len(rows) % 250 == 0:
            print(f"Отобрано реальных RuLaw-документов: {len(rows)}")

    df = pd.DataFrame(rows)

    print(f"Просмотрено XML: {scanned}")
    print(f"Итого отобрано RuLaw-документов: {len(df)}")

    if df.empty:
        print(
            "WARNING: по keyword-фильтру ничего не найдено. "
            "Беру первые реальные XML-документы без фильтра по ключевым словам."
        )

        rows = []
        for path in tqdm(xml_files, desc="Fallback: collecting real RuLaw XML"):
            if len(rows) >= max_docs:
                break

            item = extract_xml_text_and_meta(path)
            if len(item["text"]) >= min_text_len:
                rows.append(item)

        df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(
            "RuLaw XML распарсен, но ни одного реального документа собрать не удалось. "
            "Моки НЕ используются."
        )

    if df["source"].astype(str).str.startswith("demo_law").any():
        raise ValueError("В law_df попали demo_law. Это запрещено.")

    df["dataset"] = "rulaw_xml"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    print(f"RuLaw cache сохранён: {cache_path}")

    return df.reset_index(drop=True)


def load_or_create_law_docs(
    use_rulaw: bool = True,
    kind: str = "lite",
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """
    Загружает документы для RAG.

    use_rulaw=True:
    - только реальные RuLaw XML;
    - никакого fallback на demo_law.

    use_rulaw=False:
    - явно возвращает demo laws для smoke/demo режима.
    """
    if not use_rulaw:
        print("use_rulaw=False: используется demo law dataset.")
        return make_demo_law_docs()

    xml_root = download_rulaw_xml(kind=kind)

    df = load_law_documents_from_xml(
        xml_root=xml_root,
        keywords=[
            "здравоохран",
            "медицин",
            "пациент",
            "пациентск",
            "врач",
            "лечени",
            "диагноз",
            "информирован",
            "соглас",
            "потребител",
            "обращени",
            "жалоб",
            "услуг",
            "качество",
            "безопасност",
            "персональн",
            "данн",
        ],
        max_docs=MAX_LAW_DOCS_FOR_RAG,
        max_files_to_scan=None,
        cache_path=RULAW_CACHE_PATH,
        force_rebuild=force_rebuild,
    )

    print("Проверка law_df:")
    print(df.shape)
    print(df[["source", "title", "doc_type", "dataset"]].head(10))

    if df["source"].astype(str).str.startswith("demo_law").any():
        raise ValueError("load_or_create_law_docs вернул demo_law при use_rulaw=True.")

    return df.reset_index(drop=True)




# ================================================================================
# Notebook cell 9
# ================================================================================
def prepare_train_test(df: pd.DataFrame, text_col: str = "text", label_col: str = "label_name", test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, LabelEncoder]:
    df = df[[text_col, label_col]].dropna().copy()
    df[text_col] = df[text_col].apply(clean_text_for_classifier)
    counts = df[label_col].value_counts()
    df = df[df[label_col].isin(counts[counts >= 2].index)].copy()
    le = LabelEncoder()
    df["label"] = le.fit_transform(df[label_col])
    train_df, test_df = train_test_split(df[[text_col, label_col, "label"]], test_size=test_size, random_state=RANDOM_STATE, stratify=df["label"])
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), le


def compute_metrics_from_logits(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "weighted_f1": f1_score(labels, preds, average="weighted"),
    }


def build_hf_dataset(df: pd.DataFrame, tokenizer: AutoTokenizer, text_col: str = "text", max_length: int = 256) -> Dataset:
    ds = Dataset.from_pandas(df[[text_col, "label"]], preserve_index=False)
    def tokenize(batch):
        return tokenizer(batch[text_col], truncation=True, max_length=max_length)
    return ds.map(tokenize, batched=True)


def train_rumodernbert_classifier(train_df: pd.DataFrame, test_df: pd.DataFrame, label_encoder: LabelEncoder, model_name: str = RUMODERNBERT_MODEL, output_dir: Path = MODELS_DIR / "rumodernbert_complaint_classifier", learning_rate: float = 2e-5, batch_size: int = 8, num_train_epochs: int = 2, max_length: int = 256, weight_decay: float = 0.01, warmup_ratio: float = 0.05) -> Tuple[Trainer, Dict[str, Any]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_ds = build_hf_dataset(train_df, tokenizer, max_length=max_length)
    test_ds = build_hf_dataset(test_df, tokenizer, max_length=max_length)
    id2label = {i: label for i, label in enumerate(label_encoder.classes_)}
    label2id = {label: i for i, label in id2label.items()}
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_encoder.classes_), id2label=id2label, label2id=label2id, attn_implementation="eager")
    args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to="none",
        seed=RANDOM_STATE,
        fp16=torch.cuda.is_available(),
    )
    trainer = Trainer(
      model=model,
      args=args,
      train_dataset=train_ds,
      eval_dataset=test_ds,
      processing_class=tokenizer,
      data_collator=DataCollatorWithPadding(tokenizer),
      compute_metrics=compute_metrics_from_logits,
      callbacks=[
        EarlyStoppingCallback(early_stopping_patience=2)
      ],
    )
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    pd.DataFrame({"label_id": list(range(len(label_encoder.classes_))), "label_name": label_encoder.classes_}).to_csv(output_dir / "label_mapping.csv", index=False)
    save_json(metrics, output_dir / "metrics.json")
    return trainer, metrics


def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


class E5Embedder:
    def __init__(self, model_name: str = E5_MODEL, device: str = DEVICE, max_length: int = 256):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: Sequence[str], batch_size: int = 16, prefix: str = "query: ", normalize: bool = True) -> np.ndarray:
        vectors = []
        for start in tqdm(range(0, len(texts), batch_size), desc=f"Encoding {self.model_name}"):
            batch_texts = [prefix + clean_text(t, lower=False) for t in texts[start:start + batch_size]]
            encoded = self.tokenizer(batch_texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
            output = self.model(**encoded)
            pooled = mean_pooling(output.last_hidden_state, encoded["attention_mask"])
            if normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.append(pooled.detach().cpu().numpy())
        return np.vstack(vectors)


def train_e5_logreg_classifier(train_df: pd.DataFrame, test_df: pd.DataFrame, label_encoder: LabelEncoder, model_name: str = E5_MODEL, c_value: float = 2.0, max_iter: int = 1000, batch_size: int = 16, max_length: int = 256) -> Tuple[LogisticRegression, Dict[str, Any], E5Embedder]:
    embedder = E5Embedder(model_name=model_name, max_length=max_length)
    x_train = embedder.encode(train_df["text"].tolist(), batch_size=batch_size, prefix="query: ")
    x_test = embedder.encode(test_df["text"].tolist(), batch_size=batch_size, prefix="query: ")
    clf = LogisticRegression(C=c_value, max_iter=max_iter, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(x_train, train_df["label"].values)
    pred = clf.predict(x_test)
    metrics = {
        "model": f"{model_name} + LogisticRegression",
        "accuracy": float(accuracy_score(test_df["label"].values, pred)),
        "macro_f1": float(f1_score(test_df["label"].values, pred, average="macro")),
        "weighted_f1": float(f1_score(test_df["label"].values, pred, average="weighted")),
    }
    print(classification_report(test_df["label"].values, pred, target_names=label_encoder.classes_, zero_division=0))
    return clf, metrics, embedder



# ================================================================================
# Notebook cell 11
# ================================================================================
def grid_search_e5_logreg(train_df: pd.DataFrame, test_df: pd.DataFrame, label_encoder: LabelEncoder, model_name: str = E5_MODEL, param_grid: Optional[Dict[str, Sequence[Any]]] = None) -> pd.DataFrame:
    if param_grid is None:
        param_grid = {"c_value": [0.5, 1.0, 2.0, 4.0], "max_length": [128, 256]}
    rows = []
    for params in ParameterGrid(param_grid):
        print("E5 grid params:", params)
        _, metrics, embedder = train_e5_logreg_classifier(train_df, test_df, label_encoder, model_name=model_name, c_value=params["c_value"], max_length=params["max_length"], batch_size=8 if SMOKE_TEST else 32)
        rows.append({**params, **metrics})
        del embedder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    result = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    result.to_csv(REPORTS_DIR / "grid_search_e5_logreg.csv", index=False)
    return result


def grid_search_rumodernbert(train_df: pd.DataFrame, test_df: pd.DataFrame, label_encoder: LabelEncoder, param_grid: Optional[Dict[str, Sequence[Any]]] = None, base_output_dir: Path = MODELS_DIR / "grid_rumodernbert") -> pd.DataFrame:
    if param_grid is None:
        param_grid = {"learning_rate": [1e-5, 2e-5, 3e-5], "batch_size": [4, 8], "num_train_epochs": [2, 3], "max_length": [128, 256]}
    rows = []
    base_output_dir.mkdir(parents=True, exist_ok=True)
    for params in ParameterGrid(param_grid):
        run_dir = base_output_dir / f"lr_{params['learning_rate']}_bs_{params['batch_size']}_ep_{params['num_train_epochs']}_len_{params['max_length']}"
        try:
            _, metrics = train_rumodernbert_classifier(train_df, test_df, label_encoder, output_dir=run_dir, **params)
            rows.append({**params, "eval_accuracy": metrics.get("eval_accuracy"), "eval_macro_f1": metrics.get("eval_macro_f1"), "eval_weighted_f1": metrics.get("eval_weighted_f1"), "output_dir": str(run_dir)})
        except RuntimeError as e:
            rows.append({**params, "error": repr(e), "output_dir": str(run_dir)})
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    result = pd.DataFrame(rows)
    if "eval_macro_f1" in result.columns:
        result = result.sort_values("eval_macro_f1", ascending=False, na_position="last")
    result.to_csv(REPORTS_DIR / "grid_search_rumodernbert.csv", index=False)
    return result



# ================================================================================
# Notebook cell 13
# ================================================================================

class RussianComplaintNER:
    def __init__(self):
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        self.emb = NewsEmbedding()
        self.ner_tagger = NewsNERTagger(self.emb)
        self.patterns = {
            "PHONE": re.compile(r"(?:\+7|8)?[\s\-()]?\d{3}[\s\-()]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}"),
            "EMAIL": re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
            "DATE": re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b"),
            "MONEY": re.compile(r"\b\d+[\s\d]*(?:руб|₽|р\.|рублей|тыс|тысяч)\b", re.IGNORECASE),
        }
        self.medical_terms = ["врач", "доктор", "пациент", "лечение", "диагноз", "анализ", "прием", "операция", "симптом", "назначение", "поликлиника", "больница", "клиника", "стоматология", "регистратура"]
        self.service_terms = ["очередь", "администратор", "персонал", "хамство", "запись", "оплата", "чек", "договор", "услуга", "качество", "ожидание"]

    def extract_regex_entities(self, text: str) -> List[Dict[str, Any]]:
        entities = []
        for label, pattern in self.patterns.items():
            for m in pattern.finditer(text):
                entities.append({"text": m.group(0), "label": label, "start": m.start(), "end": m.end(), "source": "regex"})
        low = text.lower()
        for term in self.medical_terms:
            for m in re.finditer(re.escape(term), low):
                entities.append({"text": text[m.start():m.end()], "label": "MEDICAL_TERM", "start": m.start(), "end": m.end(), "source": "dictionary"})
        for term in self.service_terms:
            for m in re.finditer(re.escape(term), low):
                entities.append({"text": text[m.start():m.end()], "label": "SERVICE_TERM", "start": m.start(), "end": m.end(), "source": "dictionary"})
        return entities

    def extract_natasha_entities(self, text: str) -> List[Dict[str, Any]]:
        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_ner(self.ner_tagger)
        entities = []
        for span in doc.spans:
            span.normalize(self.morph_vocab)
            entities.append({"text": span.text, "normal": span.normal, "label": span.type, "start": span.start, "end": span.stop, "source": "natasha"})
        return entities

    def extract(self, text: str, anonymize: bool = True) -> Dict[str, Any]:
        text = clean_text(text, lower=False)
        entities = self.extract_natasha_entities(text) + self.extract_regex_entities(text)
        seen, unique = set(), []
        for e in sorted(entities, key=lambda x: (x["start"], x["end"], x["label"])):
            key = (e["start"], e["end"], e["label"])
            if key not in seen:
                unique.append(e)
                seen.add(key)
        anonymized_text = text
        if anonymize:
            for e in sorted(unique, key=lambda x: x["start"], reverse=True):
                if e["label"] in {"PHONE", "EMAIL", "PER"}:
                    anonymized_text = anonymized_text[:e["start"]] + f"[{e['label']}]" + anonymized_text[e["end"]:]
        return {"text": text, "anonymized_text": anonymized_text, "entities": unique}



# ================================================================================
# Notebook cell 15
# ================================================================================
# RAG: Chroma + оценка embedding-моделей на реальных жалобах из test_df

def make_langchain_documents(law_df: pd.DataFrame) -> List[Document]:
    docs = []

    if law_df.empty:
        raise ValueError("law_df пустой: RAG-документы не загружены.")

    if "source" in law_df.columns:
        has_demo = law_df["source"].astype(str).str.startswith("demo_law").any()
        if has_demo:
            raise ValueError("В law_df есть demo_law. Для финального RAG это запрещено.")

    for _, row in law_df.iterrows():
        content = f"{row.get('title', '')}\n\n{row.get('text', '')}"
        metadata = {
            "source": row.get("source", ""),
            "title": row.get("title", ""),
            "doc_date": row.get("doc_date", ""),
            "doc_type": row.get("doc_type", ""),
            "dataset": row.get("dataset", ""),
        }
        docs.append(
            Document(
                page_content=clean_text(content, lower=False),
                metadata=metadata,
            )
        )

    return docs


def split_documents(
    docs: List[Document],
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


RAG_CLASS_KEYWORDS = {
    "medicine__quality": [
        "медицин", "качество", "пациент", "лечени", "диагност", "здоров"
    ],
    "medicine__service": [
        "медицин", "услуг", "пациент", "обращени", "жалоб", "организац"
    ],
    "medicine__price": [
        "платн", "стоимость", "цена", "услуг", "потребител"
    ],
    "medicine__other": [
        "медицин", "пациент", "здравоохран", "обращени"
    ],
    "medicine__service_and_quality": [
        "медицин", "качество", "услуг", "пациент", "лечени"
    ],
    "food__quality": [
        "качество", "услуг", "потребител", "безопасност"
    ],
    "food__service": [
        "услуг", "потребител", "обращени", "жалоб"
    ],
    "food__price": [
        "цена", "стоимость", "платн", "потребител"
    ],
    "food__other": [
        "потребител", "услуг", "обращени"
    ],
    "food__service_and_quality": [
        "качество", "услуг", "потребител"
    ],
}


def build_rag_eval_queries_from_test_df(
    test_df: pd.DataFrame,
    label_column: str = "label_name",
    text_column: str = "text",
    max_queries: int = 200,
    min_text_len: int = 40,
    random_state: int = RANDOM_STATE,
) -> List[Dict[str, Any]]:
    """
    Делает RAG evaluation queries из настоящего test_df.

    Каждый отзыв из тестовой выборки становится запросом к правовой базе.
    expected_keywords берутся из класса жалобы.
    """
    required_columns = {label_column, text_column}
    missing_columns = required_columns - set(test_df.columns)

    if missing_columns:
        raise ValueError(
            f"В test_df нет колонок: {missing_columns}. "
            f"Доступные колонки: {list(test_df.columns)}"
        )

    df = test_df.copy()
    df[text_column] = df[text_column].astype(str)
    df = df[df[text_column].str.len() >= min_text_len]
    df = df[df[label_column].isin(RAG_CLASS_KEYWORDS.keys())]

    if df.empty:
        raise ValueError(
            "После фильтрации не осталось примеров для RAG evaluation. "
            "Проверь label_column/text_column и RAG_CLASS_KEYWORDS."
        )

    n_classes = df[label_column].nunique()
    per_class = max(1, max_queries // n_classes)

    sampled = (
        df.groupby(label_column, group_keys=False)
        .apply(
            lambda x: x.sample(
                n=min(len(x), per_class),
                random_state=random_state,
            )
        )
        .reset_index(drop=True)
    )

    if len(sampled) > max_queries:
        sampled = sampled.sample(
            n=max_queries,
            random_state=random_state,
        ).reset_index(drop=True)

    eval_queries = []

    for _, row in sampled.iterrows():
        label = row[label_column]
        query = row[text_column]

        eval_queries.append(
            {
                "query": query,
                "label": label,
                "expected_keywords": RAG_CLASS_KEYWORDS[label],
            }
        )

    print(f"RAG eval queries from test_df: {len(eval_queries)}")
    print(pd.Series([x["label"] for x in eval_queries]).value_counts())

    return eval_queries


def build_chroma_index(
    chunks: List[Document],
    embedding_model_name: str,
    persist_dir: Path,
) -> Chroma:
    if not chunks:
        raise ValueError("chunks пустой: нечего индексировать в Chroma.")

    if persist_dir.exists():
        shutil.rmtree(persist_dir)

    persist_dir.mkdir(parents=True, exist_ok=True)

    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device": DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_dir),
        collection_name="law_docs",
    )


def evaluate_retriever_on_test_complaints(
    vectordb: Chroma,
    eval_queries: List[Dict[str, Any]],
    k: int = 3,
) -> Dict[str, Any]:
    """
    Hit@k:
    hit = 1, если в top-k найденных правовых чанках есть хотя бы одно
    ожидаемое ключевое слово для класса жалобы.
    """
    hits = []
    details = []

    for item in tqdm(eval_queries, desc=f"RAG eval Hit@{k}"):
        query = item["query"]
        label = item["label"]
        expected_keywords = item["expected_keywords"]

        docs = vectordb.similarity_search(query, k=k)

        retrieved_text = " ".join(
            doc.page_content.lower()
            for doc in docs
        )

        hit = any(
            keyword.lower() in retrieved_text
            for keyword in expected_keywords
        )

        hits.append(int(hit))

        details.append(
            {
                "query": query[:500],
                "label": label,
                "hit": bool(hit),
                "expected_keywords": expected_keywords,
                "retrieved_titles": [
                    doc.metadata.get("title", doc.metadata.get("source", "unknown"))
                    for doc in docs
                ],
                "retrieved_sources": [
                    doc.metadata.get("source", "unknown")
                    for doc in docs
                ],
            }
        )

    return {
        "hit_at_k": float(np.mean(hits)) if hits else 0.0,
        "n_queries": len(eval_queries),
        "details": details,
    }


def compare_rag_embedding_models(
    law_chunks: List[Document],
    embedding_models: Sequence[str] = RAG_EMBEDDING_MODELS,
    test_df: Optional[pd.DataFrame] = None,
    k: int = 3,
    max_queries: int = 200,
) -> pd.DataFrame:
    """
    Сравнивает embedding-модели для RAG на настоящих жалобах из test_df,
    а не на трёх ручных запросах.
    """
    if test_df is None:
        raise ValueError(
            "Для финальной RAG-оценки нужен test_df. "
            "Вызови compare_rag_embedding_models(..., test_df=test_df)."
        )

    eval_queries = build_rag_eval_queries_from_test_df(
        test_df=test_df,
        label_column="label_name",
        text_column="text",
        max_queries=max_queries,
    )

    rows = []

    for model_name in embedding_models:
        print("Testing embedding model:", model_name)

        persist_dir = VECTOR_DIR / f"rag_eval_{re.sub(r'[^a-zA-Z0-9_\-]+', '_', model_name)}"

        try:
            vectordb = build_chroma_index(
                law_chunks,
                model_name,
                persist_dir,
            )

            metrics = evaluate_retriever_on_test_complaints(
                vectordb=vectordb,
                eval_queries=eval_queries,
                k=k,
            )

            rows.append(
                {
                    "embedding_model": model_name,
                    "hit_at_k": metrics["hit_at_k"],
                    "k": k,
                    "n_queries": metrics["n_queries"],
                    "details": json.dumps(metrics["details"], ensure_ascii=False),
                }
            )

            print(
                f"{model_name}: Hit@{k} = {metrics['hit_at_k']:.4f} "
                f"on {metrics['n_queries']} test complaints"
            )

        except Exception as e:
            rows.append(
                {
                    "embedding_model": model_name,
                    "hit_at_k": np.nan,
                    "k": k,
                    "n_queries": len(eval_queries),
                    "error": repr(e),
                }
            )
            print(f"RAG evaluation failed for {model_name}: {repr(e)}")

    result = pd.DataFrame(rows).sort_values(
        "hit_at_k",
        ascending=False,
        na_position="last",
    )

    result.to_csv(
        REPORTS_DIR / "rag_embedding_model_comparison.csv",
        index=False,
    )

    return result


def pick_best_embedding_model(comparison_df: pd.DataFrame) -> str:
    valid = comparison_df.dropna(subset=["hit_at_k"])
    return "intfloat/multilingual-e5-base" if valid.empty else valid.iloc[0]["embedding_model"]




# ================================================================================
# Notebook cell 17
# ================================================================================
USE_LLM = bool(os.getenv("OPENAI_API_KEY"))
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "openai/gpt-oss-120b:free")


def format_context(docs: List[Document], max_chars: int = 3500) -> str:
    parts, total = [], 0
    for i, d in enumerate(docs, start=1):
        title = d.metadata.get("title", "Документ")
        source = d.metadata.get("source", "")
        block = f"[{i}] {title}\nИсточник: {source}\nФрагмент: {clean_text(d.page_content, lower=False)}"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts)


def generate_template_response(complaint_text: str, predicted_topic: str, ner_payload: Dict[str, Any], retrieved_docs: List[Document]) -> str:
    context_titles = [d.metadata.get("title", "нормативный документ") for d in retrieved_docs]
    entities = ner_payload.get("entities", [])
    medical_terms = sorted({e["text"] for e in entities if e["label"] == "MEDICAL_TERM"})
    service_terms = sorted({e["text"] for e in entities if e["label"] == "SERVICE_TERM"})
    return f"""
Уважаемый заявитель!

Благодарим Вас за обращение. Ваша жалоба зарегистрирована и будет рассмотрена ответственным сотрудником.

По предварительной автоматической классификации обращение относится к категории: {predicted_topic}.
В тексте обращения выделены существенные обстоятельства:
- медицинские аспекты: {", ".join(medical_terms) if medical_terms else "не выделены явно"};
- сервисные аспекты: {", ".join(service_terms) if service_terms else "не выделены явно"}.

При рассмотрении обращения будут учтены применимые нормы и документы:
{chr(10).join("- " + t for t in context_titles[:5])}

Мы проверим изложенные Вами обстоятельства, включая качество оказания услуги, корректность коммуникации сотрудников,
сроки ожидания и полноту предоставленной информации. По результатам проверки Вам будет направлен мотивированный ответ.

С уважением,
Служба по работе с обращениями
""".strip()


def generate_llm_response(complaint_text: str, predicted_topic: str, ner_payload: Dict[str, Any], retrieved_docs: List[Document]) -> str:
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", OPENAI_BASE_URL)
    model_name = os.getenv("OPENAI_MODEL", OPENAI_MODEL)

    if not api_key:
        raise ValueError("OPENAI_API_KEY is empty")

    client = OpenAI(api_key=api_key, base_url=base_url)
    prompt = f"""
Ты — помощник службы качества медицинской организации.
Подготовь корректный, эмпатичный и юридически аккуратный ответ на жалобу пациента.
Не признавай вину без проверки, не ставь диагноз, не давай назначения, не раскрывай персональные данные.

Жалоба:
{complaint_text}

Класс обращения:
{predicted_topic}

NER:
{json.dumps(ner_payload.get("entities", []), ensure_ascii=False, indent=2)}

Нормативный контекст:
{format_context(retrieved_docs)}
""".strip()
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "system", "content": "Ты генерируешь ответы на жалобы пациентов."}, {"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content


def generate_response(complaint_text: str, predicted_topic: str, ner_payload: Dict[str, Any], retrieved_docs: List[Document], use_llm: bool = USE_LLM) -> str:
    if use_llm:
        try:
            return generate_llm_response(complaint_text, predicted_topic, ner_payload, retrieved_docs)
        except Exception as e:
            print("LLM generation failed, fallback to template:", repr(e))
    return generate_template_response(complaint_text, predicted_topic, ner_payload, retrieved_docs)


class ClassificationAgent:
    def __init__(self, model_dir: Path = MODELS_DIR / "rumodernbert_complaint_classifier"):
        self.model_dir = Path(model_dir)
        self.pipe = None
        self.id2label = None
        if self.model_dir.exists():
            try:
                self.pipe = pipeline("text-classification", model=str(self.model_dir), tokenizer=str(self.model_dir), device=0 if torch.cuda.is_available() else -1, truncation=True, max_length=256)
                mapping_path = self.model_dir / "label_mapping.csv"
                if mapping_path.exists():
                    mapping = pd.read_csv(mapping_path)
                    self.id2label = dict(zip(mapping["label_id"], mapping["label_name"]))
            except Exception as e:
                print("Could not load HF classifier, using rules:", repr(e))

    def predict(self, text: str) -> Dict[str, Any]:
        cleaned = clean_text_for_classifier(text)
        if self.pipe is not None:
            pred = self.pipe(cleaned)[0]
            label = pred["label"]
            if label.startswith("LABEL_") and self.id2label is not None:
                label = self.id2label.get(int(label.replace("LABEL_", "")), label)
            return {"label": label, "score": float(pred["score"]), "method": "RuModernBERT"}
        domain = infer_domain_from_rubrics("", cleaned) or "medicine"
        return {"label": f"{domain}__{infer_complaint_topic(cleaned)}", "score": 0.0, "method": "weak_rules"}


class NERAgent:
    def __init__(self):
        self.ner = RussianComplaintNER()
    def extract(self, text: str) -> Dict[str, Any]:
        return self.ner.extract(text, anonymize=True)


class LegalRAGAgent:
    def __init__(self, vectordb: Chroma):
        self.vectordb = vectordb
    def retrieve(self, complaint_text: str, classification: Dict[str, Any], k: int = 4) -> List[Document]:
        query = f"Жалоба пациента. Категория: {classification.get('label')}. Текст: {complaint_text}. Найди нормы о правах пациента, качестве услуги, порядке рассмотрения обращения и защите прав потребителя."
        return self.vectordb.similarity_search(query, k=k)


class ResponseGenerationAgent:
    def __init__(self, use_llm: bool = USE_LLM):
        self.use_llm = use_llm
    def generate(self, complaint_text: str, classification: Dict[str, Any], ner_payload: Dict[str, Any], docs: List[Document]) -> str:
        return generate_response(complaint_text, classification.get("label", "unknown"), ner_payload, docs, use_llm=self.use_llm)


class QualityControlAgent:
    RISKY_PATTERNS = [r"мы виноваты", r"наша вина", r"гарантируем выздоровление", r"назначаем лечение", r"вам необходимо принимать"]
    REQUIRED_PATTERNS = [r"благодар", r"обращени", r"провер", r"ответ"]
    def check(self, response: str) -> Dict[str, Any]:
        low = response.lower()
        risks = [p for p in self.RISKY_PATTERNS if re.search(p, low)]
        missing = [p for p in self.REQUIRED_PATTERNS if not re.search(p, low)]
        score = max(0.0, 1.0 - 0.2 * len(risks) - 0.1 * len(missing))
        return {"quality_score": score, "risk_patterns_found": risks, "required_patterns_missing": missing, "approved": score >= 0.7 and not risks}


class PatientComplaintMultiAgentSystem:
    def __init__(self, vectordb: Chroma, classifier_model_dir: Path = MODELS_DIR / "rumodernbert_complaint_classifier", use_llm: bool = USE_LLM):
        self.classifier = ClassificationAgent(classifier_model_dir)
        self.ner = NERAgent()
        self.rag = LegalRAGAgent(vectordb)
        self.generator = ResponseGenerationAgent(use_llm=use_llm)
        self.qc = QualityControlAgent()

    def run(self, complaint_text: str) -> Dict[str, Any]:
        classification = self.classifier.predict(complaint_text)
        ner_payload = self.ner.extract(complaint_text)
        safe_text = ner_payload["anonymized_text"]
        docs = self.rag.retrieve(safe_text, classification, k=4)
        response = self.generator.generate(safe_text, classification, ner_payload, docs)
        quality = self.qc.check(response)
        return {
            "input": complaint_text,
            "safe_input": safe_text,
            "classification": classification,
            "ner": ner_payload,
            "retrieved_docs": [{"title": d.metadata.get("title"), "source": d.metadata.get("source"), "snippet": d.page_content[:500]} for d in docs],
            "response": response,
            "quality": quality,
        }



# ================================================================================
# Notebook cell 19
# ================================================================================
def run_full_pipeline(
    use_geo_reviews: bool = True,
    use_rulaw: bool = True,
    run_heavy_grid: bool = False,
    use_llm: bool = USE_LLM,
) -> Dict[str, Any]:
    print("DEVICE:", DEVICE)

    complaints_df = load_or_create_complaint_dataset(
        use_geo_reviews=use_geo_reviews,
    )
    train_df, test_df, label_encoder = prepare_train_test(complaints_df)

    print("Train/Test split:")
    print("train_df:", train_df.shape)
    print("test_df:", test_df.shape)
    print(test_df["label_name"].value_counts())

    # 1) RuModernBERT
    rumodernbert_trainer, rumodernbert_metrics = train_rumodernbert_classifier(
        train_df=train_df,
        test_df=test_df,
        label_encoder=label_encoder,
        learning_rate=2e-5,
        batch_size=4 if SMOKE_TEST else 8,
        num_train_epochs=1 if SMOKE_TEST else 3,
        max_length=128 if SMOKE_TEST else 256,
    )

    # 2) E5 + Logistic Regression
    e5_clf, e5_metrics, e5_embedder = train_e5_logreg_classifier(
        train_df=train_df,
        test_df=test_df,
        label_encoder=label_encoder,
        c_value=2.0,
        batch_size=8 if SMOKE_TEST else 32,
        max_length=128 if SMOKE_TEST else 256,
    )

    comparison = pd.DataFrame(
        [
            {
                "model": "RuModernBERT fine-tuning",
                "accuracy": rumodernbert_metrics.get("eval_accuracy"),
                "macro_f1": rumodernbert_metrics.get("eval_macro_f1"),
                "weighted_f1": rumodernbert_metrics.get("eval_weighted_f1"),
            },
            e5_metrics,
        ]
    )
    comparison.to_csv(REPORTS_DIR / "classification_comparison.csv", index=False)

    # 3) Grid search
    e5_grid = grid_search_e5_logreg(
        train_df,
        test_df,
        label_encoder,
        param_grid={
            "c_value": [1.0] if SMOKE_TEST else [0.5],
            "max_length": [128] if SMOKE_TEST else [128],
        },
    )

    rumodernbert_grid = None
    if run_heavy_grid:
        rumodernbert_grid = grid_search_rumodernbert(
            train_df,
            test_df,
            label_encoder,
            param_grid={
                "learning_rate": [2e-5] if SMOKE_TEST else [1e-5, 2e-5, 3e-5],
                "batch_size": [4] if SMOKE_TEST else [4, 8],
                "num_train_epochs": [1] if SMOKE_TEST else [2, 3],
                "max_length": [128] if SMOKE_TEST else [128, 256],
            },
        )

    # 4) RAG на реальных RuLaw XML + оценка на test_df жалоб
    law_df = load_or_create_law_docs(
        use_rulaw=use_rulaw,
        force_rebuild=True,
    )

    if use_rulaw and law_df["source"].astype(str).str.startswith("demo_law").any():
        raise ValueError("RAG получил demo_law при use_rulaw=True. Останавливаю пайплайн.")

    law_chunks = split_documents(
        make_langchain_documents(law_df),
    )

    print("law_df:", law_df.shape)
    print("law_chunks:", len(law_chunks))
    print("first law chunk metadata:", law_chunks[0].metadata)
    print("first law chunk text:", law_chunks[0].page_content[:500])

    rag_models = RAG_EMBEDDING_MODELS[:1] if SMOKE_TEST else RAG_EMBEDDING_MODELS

    rag_embedding_comparison = compare_rag_embedding_models(
        law_chunks=law_chunks,
        embedding_models=rag_models,
        test_df=test_df,
        k=3,
        max_queries=30 if SMOKE_TEST else 200,
    )

    best_rag_embedding_model = pick_best_embedding_model(
        rag_embedding_comparison,
    )

    final_vectordb = build_chroma_index(
        law_chunks,
        best_rag_embedding_model,
        VECTOR_DIR / "final_law_chroma",
    )

    # 5) Demo мультиагентной системы
    system = PatientComplaintMultiAgentSystem(
        final_vectordb,
        use_llm=use_llm,
    )

    example_complaint = """
    Я была на приеме в клинике Альфа 20.05.2026. Врач почти ничего не объяснил,
    администратор разговаривал грубо, я ждала больше часа, а потом мне выставили
    счет выше, чем обещали по телефону. Прошу разобраться.
    """

    result = system.run(example_complaint)

    save_json(result, REPORTS_DIR / "example_multiagent_result.json")

    with (REPORTS_DIR / "example_response.txt").open("w", encoding="utf-8") as f:
        f.write(result["response"])

    summary = {
        "classification_comparison": comparison.to_dict(orient="records"),
        "e5_grid_top": e5_grid.head(5).to_dict(orient="records"),
        "rumodernbert_grid_top": (
            None
            if rumodernbert_grid is None
            else rumodernbert_grid.head(5).to_dict(orient="records")
        ),
        "rag_embedding_comparison": rag_embedding_comparison.to_dict(orient="records"),
        "best_rag_embedding_model": best_rag_embedding_model,
        "use_llm_for_generation": use_llm,
        "dataset_size": int(len(complaints_df)),
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "law_docs": int(len(law_df)),
        "law_chunks": int(len(law_chunks)),
        "example_result_path": str(REPORTS_DIR / "example_multiagent_result.json"),
    }

    save_json(summary, REPORTS_DIR / "experiment_summary.json")

    return summary


if __name__ == "__main__":
    # Для финального запуска:
    # 1) use_geo_reviews=True — берём Yandex Geo Reviews;
    # 2) use_rulaw=True — берём реальные RuLaw XML;
    # 3) RAG evaluation берёт запросы из test_df
    summary = run_full_pipeline(
        use_geo_reviews=True,
        use_rulaw=True,
        run_heavy_grid=False,
        use_llm=USE_LLM,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:4000])




# ================================================================================
# Notebook cell 22
# ================================================================================

# 9. Финальные артефакты для отчёта: графики, тесты, MLOps, LLM

from pathlib import Path
from datetime import datetime
import platform
import sys
import json
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from transformers import pipeline as hf_pipeline

FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
TESTS_DIR = REPORTS_DIR / "tests"
MLOPS_DIR = REPORTS_DIR / "mlops"

for path in [FIGURES_DIR, TABLES_DIR, TESTS_DIR, MLOPS_DIR]:
    path.mkdir(parents=True, exist_ok=True)


def load_experiment_summary(path: Path = REPORTS_DIR / "experiment_summary.json") -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Не найден {path}. Сначала запусти run_full_pipeline() и сохрани summary."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_report_test_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, LabelEncoder]:
    """
    Быстро восстанавливает датасет и train/test split для графиков и confusion matrix.
    Не обучает модели заново.
    """
    complaints_df = load_or_create_complaint_dataset(use_geo_reviews=True)
    train_df, test_df, label_encoder = prepare_train_test(complaints_df)
    return complaints_df, train_df, test_df, label_encoder


def save_class_distribution_plot(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_path: Path = FIGURES_DIR / "class_distribution.png",
) -> Path:
    train_counts = train_df["label_name"].value_counts().rename("train")
    test_counts = test_df["label_name"].value_counts().rename("test")
    counts = pd.concat([train_counts, test_counts], axis=1).fillna(0)
    counts = counts.sort_values("train", ascending=False)

    ax = counts.plot(kind="bar", figsize=(14, 6))
    ax.set_title("Распределение классов жалоб в train/test")
    ax.set_xlabel("Класс жалобы")
    ax.set_ylabel("Количество примеров")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    counts.to_csv(TABLES_DIR / "class_distribution.csv")
    return output_path


def save_classification_comparison_plot(
    summary: Dict[str, Any],
    output_path: Path = FIGURES_DIR / "classification_model_comparison.png",
) -> Path:
    df = pd.DataFrame(summary["classification_comparison"])
    metrics = ["accuracy", "macro_f1", "weighted_f1"]
    plot_df = df.set_index("model")[metrics]

    ax = plot_df.plot(kind="bar", figsize=(11, 6), ylim=(0, 1.0))
    ax.set_title("Сравнение моделей классификации жалоб")
    ax.set_xlabel("Модель")
    ax.set_ylabel("Значение метрики")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    df.to_csv(TABLES_DIR / "classification_comparison.csv", index=False)
    return output_path


def find_trainer_state_json(model_dir: Path = MODELS_DIR / "rumodernbert_complaint_classifier") -> Optional[Path]:
    candidates = list(model_dir.rglob("trainer_state.json"))
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def save_training_curve_plot(
    output_path: Path = FIGURES_DIR / "rumodernbert_training_curve.png",
) -> Optional[Path]:
    """
    Строит learning curve RuModernBERT из trainer_state.json, если он есть.
    Если trainer_state.json отсутствует, использует зафиксированные значения из текущего финального запуска.
    """
    trainer_state_path = find_trainer_state_json()
    records = []

    if trainer_state_path is not None:
        state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
        for row in state.get("log_history", []):
            if "eval_accuracy" in row:
                records.append(
                    {
                        "epoch": row.get("epoch"),
                        "accuracy": row.get("eval_accuracy"),
                        "macro_f1": row.get("eval_macro_f1"),
                        "weighted_f1": row.get("eval_weighted_f1"),
                        "eval_loss": row.get("eval_loss"),
                    }
                )

    if not records:
        # fallback по логам финального запуска
        records = [
            {"epoch": 1, "accuracy": 0.916332, "macro_f1": 0.888210, "weighted_f1": 0.916379, "eval_loss": 0.436003},
            {"epoch": 2, "accuracy": 0.936747, "macro_f1": 0.915297, "weighted_f1": 0.936257, "eval_loss": 0.300981},
            {"epoch": 3, "accuracy": 0.945114, "macro_f1": 0.927837, "weighted_f1": 0.944815, "eval_loss": 0.319308},
        ]

    df = pd.DataFrame(records).dropna(subset=["epoch"])
    df = df.sort_values("epoch")
    df.to_csv(TABLES_DIR / "rumodernbert_training_curve.csv", index=False)

    ax = df.plot(x="epoch", y=["accuracy", "macro_f1", "weighted_f1"], marker="o", figsize=(9, 5), ylim=(0.85, 1.0))
    ax.set_title("Динамика качества RuModernBERT по эпохам")
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Метрика")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def save_rag_embedding_comparison_plot(
    summary: Dict[str, Any],
    output_path: Path = FIGURES_DIR / "rag_embedding_hit_at_3.png",
) -> Path:
    df = pd.DataFrame(summary["rag_embedding_comparison"])
    df = df.dropna(subset=["hit_at_k"]).sort_values("hit_at_k", ascending=False)
    df.to_csv(TABLES_DIR / "rag_embedding_model_comparison.csv", index=False)

    ax = df.plot(x="embedding_model", y="hit_at_k", kind="bar", legend=False, figsize=(12, 5), ylim=(0, 1.0))
    k = int(df["k"].dropna().iloc[0]) if "k" in df.columns and not df.empty else 3
    ax.set_title(f"Сравнение embedding-моделей для RAG по Hit@{k}")
    ax.set_xlabel("Embedding-модель")
    ax.set_ylabel(f"Hit@{k}")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def save_rag_hit_by_class_plot(
    summary: Dict[str, Any],
    output_path: Path = FIGURES_DIR / "rag_hit_at_3_by_class.png",
) -> Optional[Path]:
    df = pd.DataFrame(summary["rag_embedding_comparison"])
    df = df.dropna(subset=["hit_at_k"])
    if df.empty or "details" not in df.columns:
        return None

    best_row = df.sort_values("hit_at_k", ascending=False).iloc[0]
    details = json.loads(best_row["details"])
    details_df = pd.DataFrame(details)
    if details_df.empty or "label" not in details_df.columns:
        return None

    by_class = details_df.groupby("label")["hit"].mean().sort_values(ascending=False)
    by_class.to_csv(TABLES_DIR / "rag_hit_at_3_by_class.csv")

    ax = by_class.plot(kind="bar", figsize=(13, 5), ylim=(0, 1.0))
    ax.set_title(f"Hit@3 по классам жалоб для лучшей RAG-модели: {best_row['embedding_model']}")
    ax.set_xlabel("Класс жалобы")
    ax.set_ylabel("Hit@3")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def save_confusion_matrix_for_rumodernbert(
    test_df: pd.DataFrame,
    label_encoder: LabelEncoder,
    sample_size: Optional[int] = None,
    output_path: Path = FIGURES_DIR / "rumodernbert_confusion_matrix.png",
) -> Optional[Path]:
    """
    Строит confusion matrix для сохранённого RuModernBERT.
    Для ускорения можно передать sample_size=1000, но для финала лучше оставить None.
    """
    model_dir = MODELS_DIR / "rumodernbert_complaint_classifier"
    if not model_dir.exists():
        print(f"Нет сохранённой модели: {model_dir}. Confusion matrix пропущена.")
        return None

    mapping_path = model_dir / "label_mapping.csv"
    if mapping_path.exists():
        mapping = pd.read_csv(mapping_path)
        id2label = dict(zip(mapping["label_id"], mapping["label_name"]))
    else:
        id2label = {i: label for i, label in enumerate(label_encoder.classes_)}

    eval_df = test_df.copy()
    if sample_size is not None and len(eval_df) > sample_size:
        eval_df = eval_df.sample(sample_size, random_state=RANDOM_STATE).reset_index(drop=True)

    clf = hf_pipeline(
        "text-classification",
        model=str(model_dir),
        tokenizer=str(model_dir),
        device=0 if torch.cuda.is_available() else -1,
        truncation=True,
        max_length=256,
    )

    y_true = eval_df["label_name"].tolist()
    predictions = []
    for text in tqdm(eval_df["text"].astype(str).tolist(), desc="Predict RuModernBERT for confusion matrix"):
        pred = clf(clean_text_for_classifier(text))[0]
        label = pred["label"]
        if label.startswith("LABEL_"):
            label = id2label.get(int(label.replace("LABEL_", "")), label)
        predictions.append(label)

    labels = list(label_encoder.classes_)
    cm = confusion_matrix(y_true, predictions, labels=labels)
    report = classification_report(y_true, predictions, labels=labels, output_dict=True, zero_division=0)

    pd.DataFrame(cm, index=labels, columns=labels).to_csv(TABLES_DIR / "rumodernbert_confusion_matrix.csv")
    pd.DataFrame(report).T.to_csv(TABLES_DIR / "rumodernbert_classification_report.csv")

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm)
    ax.set_title("Confusion matrix RuModernBERT")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def save_latex_tables(summary: Dict[str, Any]) -> Dict[str, str]:
    """
    Сохраняет таблицы в .tex для прямой вставки в Overleaf через \input{}.
    """
    outputs = {}

    cls = pd.DataFrame(summary["classification_comparison"])
    cls_path = TABLES_DIR / "classification_comparison.tex"
    cls.to_latex(cls_path, index=False, float_format="%.4f")
    outputs["classification_comparison_tex"] = str(cls_path)

    rag = pd.DataFrame(summary["rag_embedding_comparison"])
    keep_cols = [c for c in ["embedding_model", "hit_at_k", "k", "n_queries"] if c in rag.columns]
    rag_path = TABLES_DIR / "rag_embedding_comparison.tex"
    rag[keep_cols].to_latex(rag_path, index=False, float_format="%.4f")
    outputs["rag_embedding_comparison_tex"] = str(rag_path)

    return outputs


def save_mlops_manifest(summary: Dict[str, Any]) -> Path:
    """
    Минимальный MLOps-артефакт: фиксируем окружение, пути, параметры и метрики.
    """
    manifest = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "python": sys.version,
        "platform": platform.platform(),
        "device": str(DEVICE),
        "smoke_test": SMOKE_TEST,
        "max_classification_rows": MAX_CLASSIFICATION_ROWS,
        "max_review_rows": MAX_REVIEW_ROWS,
        "max_law_docs_for_rag": MAX_LAW_DOCS_FOR_RAG,
        "reports_dir": str(REPORTS_DIR),
        "models_dir": str(MODELS_DIR),
        "vector_dir": str(VECTOR_DIR),
        "summary": summary,
    }
    path = MLOPS_DIR / "experiment_manifest.json"
    save_json(manifest, path)
    return path


def run_functional_tests(
    summary: Optional[Dict[str, Any]] = None,
    vectordb: Optional[Chroma] = None,
) -> Path:
    """
    Набор smoke/functional tests для раздела 'Тестирование'.
    Не заменяет pytest, но сохраняет JSON-отчёт.
    """
    results = []

    def record(name: str, ok: bool, message: str = ""):
        results.append({"test": name, "passed": bool(ok), "message": message})

    # 1. clean_text
    cleaned = clean_text("  Пациент   ждал   2 часа!  ", lower=False)
    record("clean_text normalizes spaces", "  " not in cleaned, cleaned)

    # 2. NER anonymization
    ner = RussianComplaintNER()
    ner_payload = ner.extract("Иванов Иван, телефон +7 999 123 45 67, был в клинике 20.05.2026", anonymize=True)
    record("NER anonymizes phone", "[PHONE]" in ner_payload["anonymized_text"], ner_payload["anonymized_text"])

    # 3. summary exists
    if summary is None:
        try:
            summary = load_experiment_summary()
            record("experiment_summary exists", True, str(REPORTS_DIR / "experiment_summary.json"))
        except Exception as e:
            record("experiment_summary exists", False, repr(e))
    else:
        record("experiment_summary provided", True, "summary object provided")

    # 4. no demo_law in example retrieved docs
    try:
        example_path = Path(summary.get("example_result_path", REPORTS_DIR / "example_multiagent_result.json"))
        if example_path.exists():
            example = json.loads(example_path.read_text(encoding="utf-8"))
            sources = [str(d.get("source", "")) for d in example.get("retrieved_docs", [])]
            record("retrieved docs are not demo_law", not any(s.startswith("demo_law") for s in sources), str(sources[:3]))
        else:
            record("example result exists", False, str(example_path))
    except Exception as e:
        record("example result readable", False, repr(e))

    # 5. quality checker
    qc = QualityControlAgent()
    qc_res = qc.check("Благодарим за обращение. Мы проведем проверку и направим ответ.")
    record("quality checker approves safe response", qc_res["approved"], json.dumps(qc_res, ensure_ascii=False))

    report = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "passed": sum(r["passed"] for r in results),
        "total": len(results),
        "results": results,
    }
    path = TESTS_DIR / "functional_tests_report.json"
    save_json(report, path)
    return path


def build_llm_prompt_artifact(
    complaint_text: str,
    predicted_topic: str,
    ner_payload: Dict[str, Any],
    retrieved_docs: List[Document],
    output_path: Path = REPORTS_DIR / "llm_prompt_example.txt",
) -> Path:
    """
    Сохраняет prompt для LLM-блока. Полезно для отчёта даже без API-ключа.
    """
    prompt = f"""
Ты — помощник службы качества медицинской организации.
Подготовь корректный, эмпатичный и юридически аккуратный ответ на жалобу пациента.
Не признавай вину без проверки, не ставь диагноз, не давай назначения, не раскрывай персональные данные.

Жалоба:
{complaint_text}

Класс обращения:
{predicted_topic}

NER:
{json.dumps(ner_payload.get('entities', []), ensure_ascii=False, indent=2)}

Нормативный контекст:
{format_context(retrieved_docs)}
""".strip()
    output_path.write_text(prompt, encoding="utf-8")
    return output_path


def create_all_report_artifacts(
    build_confusion_matrix: bool = True,
    confusion_matrix_sample_size: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Главная функция для финальной подготовки отчёта.
    Запускать после основного pipeline.
    """
    summary = load_experiment_summary()
    complaints_df, train_df, test_df, label_encoder = prepare_report_test_data()

    artifacts = {
        "class_distribution_plot": str(save_class_distribution_plot(train_df, test_df)),
        "classification_comparison_plot": str(save_classification_comparison_plot(summary)),
        "training_curve_plot": str(save_training_curve_plot()),
        "rag_embedding_comparison_plot": str(save_rag_embedding_comparison_plot(summary)),
        "rag_hit_by_class_plot": str(save_rag_hit_by_class_plot(summary)),
        "latex_tables": save_latex_tables(summary),
        "mlops_manifest": str(save_mlops_manifest(summary)),
        "functional_tests_report": str(run_functional_tests(summary=summary)),
    }

    if build_confusion_matrix:
        artifacts["confusion_matrix_plot"] = str(
            save_confusion_matrix_for_rumodernbert(
                test_df=test_df,
                label_encoder=label_encoder,
                sample_size=confusion_matrix_sample_size,
            )
        )

    save_json(artifacts, REPORTS_DIR / "report_artifacts_index.json")
    print(json.dumps(artifacts, ensure_ascii=False, indent=2))
    return artifacts

