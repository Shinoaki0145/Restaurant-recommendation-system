from __future__ import annotations

import inspect
import json
import re
import sys
import time
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
if not __package__ and str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.modules.setdefault("restaurant_ranker", sys.modules[__name__])

RANDOM_STATE = 42
MODEL_VERSION = "ridge-notebook-v3"

DEFAULT_LABELS_PATH = PROJECT_ROOT / "dataset" / "restaurant_dataset_ver1.csv"
DEFAULT_RESTAURANTS_PATH = PROJECT_ROOT / "dataset" / "foody_combined_data_final.csv"
DEFAULT_ARTIFACT_PATH = BACKEND_DIR / "artifacts" / "restaurant_ranker.joblib"
DEFAULT_METRICS_PATH = BACKEND_DIR / "artifacts" / "restaurant_ranker_metrics.json"
DEFAULT_IMAGE_URL = (
    "https://images.unsplash.com/photo-1517248135467-4c7edcad34c4"
    "?auto=format&fit=crop&w=1200&q=80"
)

STOPWORDS = {
    "toi",
    "muon",
    "hay",
    "cho",
    "biet",
    "tim",
    "kiem",
    "giup",
    "mot",
    "1",
    "nhung",
    "cac",
    "quan",
    "nha",
    "hang",
    "an",
    "o",
    "va",
    "co",
    "nao",
    "de",
    "di",
    "vao",
    "ban",
    "duoc",
    "goi",
    "y",
    "liet",
    "ke",
    "cua",
    "phu",
    "hop",
    "moi",
}

POSITION_COL = "vi_tri"
PRICE_VALUE_COL = "gia_ca"
QUALITY_COL = "chat_luong"
SERVICE_COL = "phuc_vu"
SPACE_COL = "khong_gian"
DELIVERY_COL = "delivery_flag"
BOOKING_COL = "booking_flag"
RATING_COLUMNS = [POSITION_COL, PRICE_VALUE_COL, QUALITY_COL, SERVICE_COL, SPACE_COL]

FEATURE_COLUMNS = [
    "text_match",
    "location_match",
    "price_fit",
    "rating_mean",
    "popularity_score",
    "service_match",
]

DAY_COLUMNS = [
    "chu_nhat",
    "thu_hai",
    "thu_ba",
    "thu_tu",
    "thu_nam",
    "thu_sau",
    "thu_bay",
]

RESPONSE_ALIAS_COLUMNS = [
    "restaurant_id",
    "restaurant_name_meta",
    "address_meta",
    "district_meta",
    "area_meta",
    "meta_keywords",
    "cuisines_meta",
    "target_audience_raw",
    "category_raw",
    "restaurant_url",
    "pricemin",
    "pricemax",
    *RATING_COLUMNS,
    "excellent",
    "good",
    "average",
    "bad",
    "totalview",
    "totalfavourite",
    "totalcheckins",
    DELIVERY_COL,
    BOOKING_COL,
    *DAY_COLUMNS,
    "rest_days_raw",
    "image",
]

MODEL_PIPELINE = Pipeline(
    [
        (
            "model",
            Ridge(random_state=RANDOM_STATE),
        )
    ]
)

MODEL_PARAMS = {
    "model__alpha": 10.0,
}


@dataclass
class FeatureArtifacts:
    word_vectorizer: TfidfVectorizer
    restaurant_ids: list[str]
    restaurant_word_matrix: Any
    location_pattern_map: dict[str, set[str]]
    location_candidates: list[tuple[str, tuple[str, ...]]]


def normalize_text(text: Any) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    text = str(text).replace("Đ", "D").replace("đ", "d")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _normalize_text_fixed(text: Any) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    text = str(text).replace("Đ", "D").replace("đ", "d")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


normalize_text = _normalize_text_fixed


def _normalize_text_unicode_safe(text: Any) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    text = str(text).replace("\u0110", "D").replace("\u0111", "d")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


normalize_text = _normalize_text_unicode_safe


def slugify_column_name(column_name: Any) -> str:
    text = normalize_text(column_name).replace("-", "_").replace(" ", "_")
    text = re.sub(r"[^a-z0-9_]", "", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def normalize_restaurant_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        return str(int(float(text)))
    except Exception:
        return text


def canonical_location(text: Any) -> str:
    location = normalize_text(text)
    location = re.sub(r"\b(thanh pho|tp|quan|q|huyen|h|thi xa|tx)\b", " ", location)
    return re.sub(r"\s+", " ", location).strip()


def contains_phrase(text: Any, phrase: Any) -> bool:
    text_norm = normalize_text(text)
    phrase_norm = normalize_text(phrase)
    return _contains_normalized_phrase(text_norm, phrase_norm)


def _contains_normalized_phrase(text_norm: str, phrase_norm: str) -> bool:
    if not text_norm or not phrase_norm:
        return False
    return f" {phrase_norm} " in f" {text_norm} "


def mentions_any(text: Any, phrases: tuple[str, ...] | list[str] | set[str]) -> bool:
    text_norm = normalize_text(text)
    return any(contains_phrase(text_norm, phrase) for phrase in phrases)


def _coalesce_column(df: pd.DataFrame, target: str, aliases: list[str], default: Any = "") -> None:
    if target in df.columns:
        return
    for alias in aliases:
        if alias in df.columns:
            df[target] = df[alias]
            return
    df[target] = default


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(value)
    if pd.isna(value):
        return None
    return value


def _json_safe_dict(record: dict[str, Any]) -> dict[str, Any]:
    return {str(key): _json_safe(value) for key, value in record.items()}


def prepare_restaurant_catalog(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df.columns = [slugify_column_name(column) for column in df.columns]

    _coalesce_column(df, "restaurant_id", ["restaurant_id", "restaurantid"], default="")
    df["restaurant_id"] = df["restaurant_id"].map(normalize_restaurant_id)

    _coalesce_column(df, "restaurant_name_meta", ["restaurant_name_meta", "name", "restaurant_name"], default="")
    _coalesce_column(df, "district_meta", ["district_meta", "district"], default="")
    _coalesce_column(df, "area_meta", ["area_meta", "area"], default="")
    _coalesce_column(df, "address_meta", ["address_meta", "address"], default="")
    _coalesce_column(df, "meta_keywords", ["meta_keywords", "metakeywords"], default="")
    _coalesce_column(df, "cuisines_meta", ["cuisines_meta", "cuisines"], default="")
    _coalesce_column(df, "target_audience_raw", ["target_audience_raw", "lsttargetaudience"], default="")
    _coalesce_column(df, "category_raw", ["category_raw", "lstcategory"], default="")
    _coalesce_column(df, "restaurant_url", ["restaurant_url", "restauranturl", "url"], default="")
    _coalesce_column(df, "rest_days_raw", ["rest_days_raw", "ngay_nghi"], default="")

    _coalesce_column(df, "pricemin", ["pricemin", "price_min"], default=0.0)
    _coalesce_column(df, "pricemax", ["pricemax", "price_max"], default=0.0)
    _coalesce_column(df, POSITION_COL, [POSITION_COL, "vi_tri"], default=0.0)
    _coalesce_column(df, PRICE_VALUE_COL, [PRICE_VALUE_COL, "gia_ca"], default=0.0)
    _coalesce_column(df, QUALITY_COL, [QUALITY_COL, "chat_luong"], default=0.0)
    _coalesce_column(df, SERVICE_COL, [SERVICE_COL, "phuc_vu"], default=0.0)
    _coalesce_column(df, SPACE_COL, [SPACE_COL, "khong_gian"], default=0.0)
    _coalesce_column(df, "excellent", ["excellent"], default=0.0)
    _coalesce_column(df, "good", ["good"], default=0.0)
    _coalesce_column(df, "average", ["average"], default=0.0)
    _coalesce_column(df, "bad", ["bad"], default=0.0)
    _coalesce_column(df, "totalview", ["totalview", "total_view"], default=0.0)
    _coalesce_column(df, "totalfavourite", ["totalfavourite", "total_favourite"], default=0.0)
    _coalesce_column(df, "totalcheckins", ["totalcheckins", "total_checkins"], default=0.0)
    _coalesce_column(df, DELIVERY_COL, [DELIVERY_COL, "giao_tan_noi"], default=0.0)
    _coalesce_column(df, BOOKING_COL, [BOOKING_COL, "dat_ban"], default=0.0)

    for day_column in DAY_COLUMNS:
        _coalesce_column(df, day_column, [day_column], default="")

    if "image" not in df.columns:
        if "image_url" in df.columns:
            df["image"] = df["image_url"]
        elif "thumbnail" in df.columns:
            df["image"] = df["thumbnail"]
        elif "thumbnail_url" in df.columns:
            df["image"] = df["thumbnail_url"]
        else:
            df["image"] = DEFAULT_IMAGE_URL
    df["image"] = df["image"].replace("", DEFAULT_IMAGE_URL).fillna(DEFAULT_IMAGE_URL)

    return df


def load_merged_restaurant_ranking(
    labels_path: str | Path = DEFAULT_LABELS_PATH,
    restaurants_path: str | Path = DEFAULT_RESTAURANTS_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels_df = pd.read_csv(labels_path).rename(
        columns={"district": "label_district", "cuisines": "label_cuisines", "source": "retrieval_source"}
    )
    restaurants_df = prepare_restaurant_catalog(pd.read_csv(restaurants_path))

    labels_df["restaurant_id"] = labels_df["restaurant_id"].map(normalize_restaurant_id)
    merged_df = labels_df.merge(restaurants_df, on="restaurant_id", how="left", validate="many_to_one")
    return merged_df, restaurants_df


def prepare_base_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    numeric_columns = [
        "pricemin",
        "pricemax",
        *RATING_COLUMNS,
        "totalview",
        "totalfavourite",
        "totalcheckins",
        DELIVERY_COL,
        BOOKING_COL,
    ]
    for column in numeric_columns:
        if column not in df.columns:
            df[column] = 0.0
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    text_columns = [
        "query",
        "restaurant_name_meta",
        "address_meta",
        "district_meta",
        "area_meta",
        "meta_keywords",
        "cuisines_meta",
        "category_raw",
        "label_district",
    ]
    for column in text_columns:
        if column not in df.columns:
            df[column] = ""
        df[column] = df[column].fillna("").astype(str)

    df["query_norm"] = df["query"].map(normalize_text)
    df["restaurant_text"] = df[
        [
            "restaurant_name_meta",
            "address_meta",
            "district_meta",
            "area_meta",
            "meta_keywords",
            "cuisines_meta",
            "category_raw",
        ]
    ].agg(" | ".join, axis=1)
    df["restaurant_text_norm"] = df["restaurant_text"].map(normalize_text)
    df["district_norm"] = df["district_meta"].map(canonical_location)
    df["area_norm"] = df["area_meta"].map(canonical_location)
    df["price_mid"] = np.where(
        (df["pricemin"] > 0) & (df["pricemax"] > 0),
        (df["pricemin"] + df["pricemax"]) / 2.0,
        np.where(df["pricemax"] > 0, df["pricemax"], df["pricemin"]),
    ).astype(float)
    return df


def build_location_pattern_map(values: pd.Series) -> dict[str, set[str]]:
    pattern_map: dict[str, set[str]] = defaultdict(set)
    for value in values.dropna().astype(str):
        original = normalize_text(value)
        canonical = canonical_location(value)
        if not canonical:
            continue
        pattern_map[canonical].update({original, canonical})
        match = re.match(r"quan\s+(\d+)$", original)
        if match:
            number = match.group(1)
            pattern_map[canonical].update({f"quan {number}", f"q {number}", f"q{number}"})
        if canonical == "thu duc":
            pattern_map[canonical].update({"thu duc", "tp thu duc", "thanh pho thu duc"})
    return pattern_map


def extract_location_target(query: str, location_pattern_map: dict[str, set[str]]) -> str:
    query_norm = normalize_text(query)
    candidates = sorted(
        location_pattern_map.items(),
        key=lambda item: max(len(pattern) for pattern in item[1]),
        reverse=True,
    )
    for canonical_name, patterns in candidates:
        if any(_contains_normalized_phrase(query_norm, pattern) for pattern in patterns):
            return canonical_name
    return ""


def build_feature_artifacts(restaurants_df: pd.DataFrame) -> FeatureArtifacts:
    prepared_df = prepare_base_frame(restaurants_df.assign(query=""))
    restaurants = prepared_df[["restaurant_id", "restaurant_text_norm"]].drop_duplicates("restaurant_id").reset_index(drop=True)

    min_df = 2 if len(restaurants) >= 2 else 1
    word_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=min_df)
    restaurant_word_matrix = word_vectorizer.fit_transform(restaurants["restaurant_text_norm"])

    location_candidates = pd.concat(
        [prepared_df["district_meta"], prepared_df["area_meta"]],
        ignore_index=True,
    )
    location_pattern_map = build_location_pattern_map(location_candidates)
    sorted_location_candidates = sorted(
        ((canonical_name, tuple(sorted(patterns, key=len, reverse=True))) for canonical_name, patterns in location_pattern_map.items()),
        key=lambda item: max(len(pattern) for pattern in item[1]),
        reverse=True,
    )

    return FeatureArtifacts(
        word_vectorizer=word_vectorizer,
        restaurant_ids=restaurants["restaurant_id"].tolist(),
        restaurant_word_matrix=restaurant_word_matrix,
        location_pattern_map=location_pattern_map,
        location_candidates=sorted_location_candidates,
    )


def add_text_match_feature(df: pd.DataFrame, feature_artifacts: FeatureArtifacts) -> pd.DataFrame:
    df = df.copy()
    restaurant_index = {restaurant_id: idx for idx, restaurant_id in enumerate(feature_artifacts.restaurant_ids)}
    query_word_matrix = feature_artifacts.word_vectorizer.transform(df["query_norm"])
    restaurant_indices = [restaurant_index.get(restaurant_id) for restaurant_id in df["restaurant_id"]]

    if all(index is not None for index in restaurant_indices):
        restaurant_word_matrix = feature_artifacts.restaurant_word_matrix[restaurant_indices]
    else:
        restaurant_word_matrix = feature_artifacts.word_vectorizer.transform(df["restaurant_text_norm"])

    # TF-IDF vectors are L2-normalized by default, so row-wise dot products equal cosine similarity.
    df["text_match"] = np.asarray(
        query_word_matrix.multiply(restaurant_word_matrix).sum(axis=1)
    ).reshape(-1).astype(float)
    return df


def add_location_match_feature(df: pd.DataFrame, feature_artifacts: FeatureArtifacts) -> pd.DataFrame:
    df = df.copy()
    normalized_queries = df["query"].map(normalize_text)
    df["district_target"] = normalized_queries.map(
        lambda value: next(
            (
                canonical_name
                for canonical_name, patterns in feature_artifacts.location_candidates
                if any(_contains_normalized_phrase(value, pattern) for pattern in patterns)
            ),
            "",
        )
    )
    df["location_match"] = [
        float(target != "" and (district_norm == target or area_norm == target))
        for target, district_norm, area_norm in zip(df["district_target"], df["district_norm"], df["area_norm"])
    ]
    return df


def parse_number_with_unit(value_text: str, unit_hint: str = "") -> float:
    fragment = normalize_text(f"{value_text} {unit_hint}".strip())
    match = re.search(r"(\d+(?:[\.,]\d+)?)", fragment)
    if not match:
        return np.nan
    number = float(match.group(1).replace(",", "."))
    if "trieu" in fragment:
        return number * 1_000_000
    if any(unit in fragment for unit in ("nghin", "ngan", "k")):
        return number * 1_000
    if number <= 500:
        return number * 1_000
    return number


def extract_price_bounds(query: str) -> tuple[float, float]:
    query_norm = normalize_text(query)
    match = re.search(
        r"(\d+(?:[\.,]\d+)?)\s*(trieu|nghin|ngan|k)?\s*[-?]\s*(\d+(?:[\.,]\d+)?)\s*(trieu|nghin|ngan|k)?",
        query_norm,
    )
    if match:
        left_value, left_unit, right_value, right_unit = match.groups()
        if not left_unit and right_unit:
            left_unit = right_unit
        if not right_unit and left_unit:
            right_unit = left_unit
        floor_value = parse_number_with_unit(left_value, left_unit or "")
        ceiling_value = parse_number_with_unit(right_value, right_unit or "")
        if floor_value > ceiling_value:
            floor_value, ceiling_value = ceiling_value, floor_value
        return float(floor_value), float(ceiling_value)

    for pattern in [
        r"gia\s*toi\s*da(?:\s*la)?\s*(\d+(?:[\.,]\d+)?)\s*(trieu|nghin|ngan|k)?",
        r"toi\s*da\s*(\d+(?:[\.,]\d+)?)\s*(trieu|nghin|ngan|k)?",
        r"khong\s*qua\s*(\d+(?:[\.,]\d+)?)\s*(trieu|nghin|ngan|k)?",
        r"duoi\s*(\d+(?:[\.,]\d+)?)\s*(trieu|nghin|ngan|k)?",
    ]:
        match = re.search(pattern, query_norm)
        if match:
            value, unit = match.groups()
            return np.nan, float(parse_number_with_unit(value, unit or ""))
    return np.nan, np.nan


def compute_price_fit(row: pd.Series) -> float:
    if row["price_requested"] <= 0 or row["price_mid"] <= 0:
        return 0.0
    if not pd.isna(row["price_floor"]) and row["price_mid"] < row["price_floor"]:
        gap = (row["price_floor"] - row["price_mid"]) / max(row["price_floor"], 1.0)
        return float(max(0.0, 1.0 - gap))
    if not pd.isna(row["price_ceiling"]) and row["price_mid"] > row["price_ceiling"]:
        gap = (row["price_mid"] - row["price_ceiling"]) / max(row["price_ceiling"], 1.0)
        return float(max(0.0, 1.0 - gap))
    return 1.0


def add_price_fit_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[["price_floor", "price_ceiling"]] = df["query"].apply(lambda value: pd.Series(extract_price_bounds(value)))
    df["price_requested"] = (~df["price_floor"].isna() | ~df["price_ceiling"].isna()).astype(float)
    df["price_fit"] = df.apply(compute_price_fit, axis=1)
    return df


def add_rating_mean_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rating_mean"] = df[RATING_COLUMNS].mean(axis=1) / 10.0
    return df


def compute_service_match(row: pd.Series) -> float:
    requested_scores: list[float] = []
    if row["delivery_required"] > 0:
        requested_scores.append(float(row[DELIVERY_COL] > 0))
    if row["booking_required"] > 0:
        requested_scores.append(float(row[BOOKING_COL] > 0))
    return float(np.mean(requested_scores)) if requested_scores else 0.0


def add_popularity_and_service_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["popularity_score"] = np.tanh(np.log1p(df["totalview"] + df["totalfavourite"] + df["totalcheckins"]) / 6.0)
    df["delivery_required"] = df["query"].map(
        lambda value: float(mentions_any(value, ("giao hang", "giao tan noi", "ship")))
    )
    df["booking_required"] = df["query"].map(
        lambda value: float(mentions_any(value, ("dat ban", "nhan dat ban")))
    )
    df["service_match"] = df.apply(compute_service_match, axis=1)
    return df


def build_restaurant_features(df: pd.DataFrame, feature_artifacts: FeatureArtifacts) -> pd.DataFrame:
    df = prepare_base_frame(df)
    df = add_text_match_feature(df, feature_artifacts)
    df = add_location_match_feature(df, feature_artifacts)
    df = add_price_fit_feature(df)
    df = add_rating_mean_feature(df)
    df = add_popularity_and_service_features(df)

    for feature in FEATURE_COLUMNS:
        df[feature] = pd.to_numeric(df[feature], errors="coerce").fillna(0.0).astype(float)
    return df


def infer_model_name(estimator: Any) -> str:
    model_step = estimator.named_steps["model"] if hasattr(estimator, "named_steps") else estimator
    return f"{model_step.__class__.__name__} Ranker"


def build_fit_kwargs(estimator: Any, train_df: pd.DataFrame) -> dict[str, Any]:
    model_step = estimator.named_steps["model"] if hasattr(estimator, "named_steps") else estimator
    fit_parameters = inspect.signature(model_step.fit).parameters
    if "group" in fit_parameters:
        return {"model__group": train_df.groupby("query").size().tolist()}
    return {}


def fit_selected_estimator(
    estimator: Any,
    model_params: dict[str, Any],
    full_df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[Any, float]:
    final_estimator = clone(estimator)
    final_estimator.set_params(**model_params)

    start_time = time.perf_counter()
    final_estimator.fit(
        full_df[feature_columns],
        full_df["label"],
        **build_fit_kwargs(final_estimator, full_df),
    )
    train_time = time.perf_counter() - start_time
    return final_estimator, float(train_time)


def build_training_summary(
    model_name: str,
    model_params: dict[str, Any],
    training_df: pd.DataFrame,
    training_time_s: float,
) -> dict[str, Any]:
    return {
        "model_version": MODEL_VERSION,
        "model_name": model_name,
        "feature_columns": FEATURE_COLUMNS,
        "training_rows": int(len(training_df)),
        "training_queries": int(training_df["query"].nunique()),
        "best_params": model_params,
        "training_time_s": float(training_time_s),
        "selection_source": "Selected from modelling.ipynb",
    }


def _build_rank_pairs(query: str, catalog_df: pd.DataFrame) -> pd.DataFrame:
    pairs_df = catalog_df.copy()
    pairs_df["query"] = str(query)
    return pairs_df


class RestaurantRankerService:
    def __init__(
        self,
        model: Any,
        feature_artifacts: FeatureArtifacts,
        restaurants_df: pd.DataFrame,
        metrics: dict[str, Any],
        artifact_path: Path | None = None,
    ) -> None:
        self.model = model
        self.feature_artifacts = feature_artifacts
        self.restaurants_df = prepare_restaurant_catalog(restaurants_df)
        self.metrics = metrics
        self.artifact_path = Path(artifact_path) if artifact_path else None
        self.feature_columns = list(FEATURE_COLUMNS)
        self.model_name = metrics.get("model_name") or infer_model_name(model)
        self.best_params = metrics.get("best_params", {})

    @classmethod
    def train(
        cls,
        labels_path: Path | str = DEFAULT_LABELS_PATH,
        restaurants_path: Path | str = DEFAULT_RESTAURANTS_PATH,
        artifact_path: Path | str | None = DEFAULT_ARTIFACT_PATH,
        metrics_path: Path | str | None = DEFAULT_METRICS_PATH,
    ) -> "RestaurantRankerService":
        merged_df, restaurants_df = load_merged_restaurant_ranking(labels_path, restaurants_path)
        feature_artifacts = build_feature_artifacts(restaurants_df)
        training_df = build_restaurant_features(merged_df, feature_artifacts)
        deployment_estimator, training_time_s = fit_selected_estimator(
            estimator=MODEL_PIPELINE,
            model_params=MODEL_PARAMS,
            full_df=training_df,
            feature_columns=FEATURE_COLUMNS,
        )
        metrics_payload = build_training_summary(
            model_name=infer_model_name(deployment_estimator),
            model_params=MODEL_PARAMS,
            training_df=training_df,
            training_time_s=training_time_s,
        )
        service = cls(
            model=deployment_estimator,
            feature_artifacts=feature_artifacts,
            restaurants_df=restaurants_df,
            metrics=metrics_payload,
            artifact_path=Path(artifact_path) if artifact_path else None,
        )

        if artifact_path:
            service.save(artifact_path)
        if metrics_path:
            metrics_target = Path(metrics_path)
            metrics_target.parent.mkdir(parents=True, exist_ok=True)
            metrics_target.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return service

    @classmethod
    def load(cls, artifact_path: Path | str = DEFAULT_ARTIFACT_PATH) -> "RestaurantRankerService":
        payload = joblib.load(artifact_path)
        if payload.get("model_version") != MODEL_VERSION:
            raise ValueError("Artifact version is outdated")
        return cls(
            model=payload["model"],
            feature_artifacts=payload["feature_artifacts"],
            restaurants_df=payload["restaurants_df"],
            metrics=payload["metrics"],
            artifact_path=Path(artifact_path),
        )

    @classmethod
    def load_or_train(
        cls,
        artifact_path: Path | str = DEFAULT_ARTIFACT_PATH,
        restaurants_path: Path | str = DEFAULT_RESTAURANTS_PATH,
        labels_path: Path | str = DEFAULT_LABELS_PATH,
        metrics_path: Path | str | None = DEFAULT_METRICS_PATH,
        force_retrain: bool = False,
    ) -> "RestaurantRankerService":
        artifact_target = Path(artifact_path)
        if not force_retrain and artifact_target.exists():
            try:
                return cls.load(artifact_target)
            except Exception:
                pass
        return cls.train(
            labels_path=labels_path,
            restaurants_path=restaurants_path,
            artifact_path=artifact_target,
            metrics_path=metrics_path,
        )

    def save(self, artifact_path: Path | str = DEFAULT_ARTIFACT_PATH) -> None:
        target = Path(artifact_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_version": MODEL_VERSION,
            "model": self.model,
            "feature_artifacts": self.feature_artifacts,
            "restaurants_df": self.restaurants_df,
            "metrics": self.metrics,
        }
        joblib.dump(payload, target)
        self.artifact_path = target

    def _build_feature_frame_for_query(
        self,
        query: str,
        candidate_restaurant_ids: list[str | int],
        candidate_restaurants: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
        normalized_ids = [normalize_restaurant_id(value) for value in candidate_restaurant_ids]
        normalized_ids = [value for value in normalized_ids if value]
        if not normalized_ids:
            raise ValueError("candidate_restaurant_ids must contain at least one valid restaurant id")

        if candidate_restaurants is None or candidate_restaurants.empty:
            raise ValueError("candidate_restaurants must be provided from the repository; CSV fallback is disabled")

        raw_df = candidate_restaurants.copy()
        raw_df.columns = [str(column) for column in raw_df.columns]
        raw_id_col = None
        for column in raw_df.columns:
            if slugify_column_name(column) in {"restaurant_id", "restaurantid"}:
                raw_id_col = column
                break
        if raw_id_col is None:
            raise ValueError("candidate_restaurants is missing a restaurant id column")

        raw_df["_restaurant_id_norm"] = raw_df[raw_id_col].map(normalize_restaurant_id)
        raw_df = raw_df[raw_df["_restaurant_id_norm"].isin(normalized_ids)].copy()
        raw_payload_by_id = {
            record["_restaurant_id_norm"]: _json_safe_dict(
                {key: value for key, value in record.items() if key != "_restaurant_id_norm"}
            )
            for record in raw_df.to_dict(orient="records")
        }
        raw_df = raw_df.drop(columns=["_restaurant_id_norm"])
        catalog_df = prepare_restaurant_catalog(raw_df)

        if catalog_df.empty:
            raise ValueError("No candidate restaurant metadata was available for scoring")

        catalog_df = catalog_df[catalog_df["restaurant_id"].isin(normalized_ids)].copy()
        if catalog_df.empty:
            raise ValueError("Candidate metadata does not overlap with candidate_restaurant_ids")

        pair_df = _build_rank_pairs(query=query, catalog_df=catalog_df)
        feature_df = build_restaurant_features(pair_df, self.feature_artifacts)
        return feature_df, raw_payload_by_id

    def _build_score_breakdown(self, row: pd.Series) -> dict[str, Any]:
        feature_values = {feature: float(row.get(feature, 0.0)) for feature in self.feature_columns}
        score_breakdown: dict[str, Any] = {
            "features": feature_values,
        }

        model_step = self.model.named_steps["model"] if hasattr(self.model, "named_steps") else self.model
        coefficients = getattr(model_step, "coef_", None)
        if coefficients is not None:
            weights = np.asarray(coefficients, dtype=float).reshape(-1)
            contributions = {
                feature: float(feature_values[feature] * weights[idx])
                for idx, feature in enumerate(self.feature_columns)
            }
            score_breakdown["contributions"] = contributions
            score_breakdown["intercept"] = float(np.asarray(getattr(model_step, "intercept_", 0.0)).reshape(-1)[0])

        return score_breakdown

    def rank(
        self,
        query: str,
        candidate_restaurant_ids: list[str | int],
        candidate_restaurants: pd.DataFrame | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        if not str(query).strip():
            raise ValueError("query must not be empty")

        feature_df, raw_payload_by_id = self._build_feature_frame_for_query(
            query=query,
            candidate_restaurant_ids=candidate_restaurant_ids,
            candidate_restaurants=candidate_restaurants,
        )

        y_pred = np.asarray(self.model.predict(feature_df[self.feature_columns]), dtype=float).reshape(-1)
        ranked_df = feature_df.copy()
        ranked_df["rank_score"] = y_pred
        ranked_df = ranked_df.sort_values("rank_score", ascending=False).head(top_k).reset_index(drop=True)

        results: list[dict[str, Any]] = []
        for _, row in ranked_df.iterrows():
            restaurant_id = normalize_restaurant_id(row["restaurant_id"])
            raw_payload = raw_payload_by_id.get(restaurant_id, {})
            compatibility_payload = {
                column: _json_safe(row[column])
                for column in RESPONSE_ALIAS_COLUMNS
                if column in row.index
            }
            compatibility_payload["restaurant_id"] = restaurant_id
            compatibility_payload["image"] = compatibility_payload.get("image") or DEFAULT_IMAGE_URL

            restaurant_payload = {**raw_payload, **compatibility_payload}
            restaurant_payload.setdefault("image", DEFAULT_IMAGE_URL)

            results.append(
                {
                    "restaurant_id": restaurant_id,
                    "rank_score": float(row["rank_score"]),
                    "score_breakdown": self._build_score_breakdown(row),
                    "restaurant": restaurant_payload,
                }
            )
        return results

    def health(self) -> dict[str, Any]:
        return {
            "model_version": MODEL_VERSION,
            "model_name": self.model_name,
            "feature_count": len(self.feature_columns),
            "feature_columns": self.feature_columns,
            "best_params": self.best_params,
            "artifact_path": str(self.artifact_path) if self.artifact_path else None,
            "metrics": self.metrics,
        }


def main() -> None:
    service = RestaurantRankerService.train(
        artifact_path=DEFAULT_ARTIFACT_PATH,
        metrics_path=DEFAULT_METRICS_PATH,
    )
    print("Training completed.")
    print(f"Artifact saved to: {DEFAULT_ARTIFACT_PATH}")
    print(f"Metrics saved to: {DEFAULT_METRICS_PATH}")
    print(service.health())


if __name__ == "__main__":
    main()
