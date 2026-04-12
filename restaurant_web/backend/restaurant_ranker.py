from __future__ import annotations

import json
import re
import sys
import time
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import ndcg_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRanker

RANDOM_STATE = 42

BACKEND_DIR = Path(__file__).resolve().parent
APP_DIR = BACKEND_DIR.parent
PROJECT_ROOT = APP_DIR.parent

DEFAULT_LABELS_PATH = PROJECT_ROOT / "dataset" / "restaurant_dataset_ver1.csv"
DEFAULT_RESTAURANTS_PATH = PROJECT_ROOT / "dataset" / "foody_combined_data_final.csv"
DEFAULT_ARTIFACT_PATH = APP_DIR / "artifacts" / "restaurant_ranker.joblib"
DEFAULT_METRICS_PATH = APP_DIR / "artifacts" / "restaurant_ranker_metrics.json"

# Backward compatibility for artifacts serialized before the backend package move.
sys.modules.setdefault("restaurant_ranker", sys.modules[__name__])

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

DAY_COLUMNS = ["thu_hai", "thu_ba", "thu_tu", "thu_nam", "thu_sau", "thu_bay", "chu_nhat"]
DAY_MAP = {
    "chu nhat": "chu_nhat",
    "thu hai": "thu_hai",
    "thu ba": "thu_ba",
    "thu tu": "thu_tu",
    "thu nam": "thu_nam",
    "thu sau": "thu_sau",
    "thu bay": "thu_bay",
}
TIME_SLOTS = {
    "breakfast": 7 * 60 + 30,
    "brunch": 10 * 60 + 30,
    "lunch": 12 * 60 + 30,
    "afternoon": 15 * 60 + 30,
    "dinner": 19 * 60 + 30,
    "late_night": 22 * 60 + 30,
}
GENERIC_QUERY_TOKENS = STOPWORDS | {
    "mon",
    "do",
    "am",
    "thuc",
    "gia",
    "ngon",
    "dep",
    "tot",
    "on",
    "chat",
    "luong",
    "phuc",
    "vu",
    "khong",
    "gian",
    "vi",
    "tri",
    "nhieu",
    "nguoi",
    "review",
    "danh",
    "cao",
    "re",
    "mem",
    "tuan",
    "ngay",
    "thu",
    "sang",
    "trua",
    "chieu",
    "toi",
    "giao",
    "hang",
    "tan",
    "noi",
    "ban",
    "bua",
    "mo",
    "cua",
    "rat",
    "qua",
    "chi",
    "can",
}

AUDIENCE_PATTERNS = {
    "sinh vien": ["sinh vien", "hoc sinh sinh vien"],
    "cap doi": ["cap doi", "hen ho", "lang man"],
    "gia dinh": ["gia dinh", "ba me"],
    "nhom hoi": ["nhom hoi", "nhom dong nguoi", "team cong ty", "hop nhom"],
    "nhom ban": ["nhom ban", "ban be", "tu tap ban be", "gap ban cu"],
    "dan van phong": ["dan van phong", "gioi van phong", "van phong", "com van phong"],
    "gioi manager": ["manager", "tiep khach", "khach hang", "doi tac", "doanh nghiep"],
    "khach du lich": ["khach du lich", "khach nuoc ngoai"],
    "tre em": ["tre em"],
    "nguoi lon tuoi": ["nguoi lon tuoi"],
}

CATEGORY_PATTERNS = {
    "an_vat": {"query": ["an vat", "do an vat"], "meta": ["an vat", "via he"]},
    "cafe_dessert": {
        "query": ["cafe", "ca phe", "dessert", "tra sua", "tra banh", "brunch", "tiem banh", "banh ngot"],
        "meta": ["cafe", "dessert", "tiem banh"],
    },
    "buffet": {"query": ["buffet"], "meta": ["buffet"]},
    "bar_pub": {"query": ["bar", "pub", "lounge"], "meta": ["bar pub", "beer club", "lounge"]},
    "quan_nhau": {"query": ["nhau", "beer garden", "beer club"], "meta": ["quan nhau", "beer club", "beer garden"]},
    "an_chay": {"query": ["chay"], "meta": ["an chay"]},
    "lau": {"query": ["lau", "lau bo", "lau ca", "lau nam"], "meta": ["lau"]},
    "nuong_bbq": {"query": ["nuong", "bbq"], "meta": ["nuong", "bbq"]},
}

CUISINE_PATTERNS = {
    "mon_viet": {"query": ["mon viet", "viet nam"], "meta": ["mon viet"]},
    "mon_han": {"query": ["han quoc", "mon han", "bbq han quoc"], "meta": ["mon han", "han quoc"]},
    "mon_nhat": {"query": ["mon nhat", "sushi", "ramen"], "meta": ["mon nhat", "nhat"]},
    "mon_thai": {"query": ["thai lan", "mon thai"], "meta": ["mon thai", "thai"]},
    "mon_au": {"query": ["mon au", "chau au", "steak"], "meta": ["mon au", "quoc te", "phap", "my"]},
    "mon_y": {"query": ["mon y", "mi y", "pizza", "italy", "italian"], "meta": ["mon y", "italy", "pizza"]},
    "mon_trung_hoa": {"query": ["trung hoa", "mon hoa", "dimsum", "dim sum"], "meta": ["trung hoa", "dimsum"]},
    "mon_hue": {"query": ["mon hue", "bun bo", "bun bo hue"], "meta": ["mon hue", "hue"]},
    "mon_bac": {"query": ["mon bac", "pho", "bun cha"], "meta": ["mon bac", "ha noi"]},
    "mon_mien_trung": {"query": ["mien trung"], "meta": ["mien trung"]},
    "mon_mien_nam": {"query": ["mien nam"], "meta": ["mien nam"]},
    "hai_san": {"query": ["hai san", "oc"], "meta": ["hai san", "oc"]},
}

FEATURE_COLUMNS = [
    "tfidf_cosine",
    "char_tfidf_cosine",
    "topic_overlap_ratio",
    "topic_exact_phrase_hit",
    "name_overlap_ratio",
    "meta_keyword_overlap_ratio",
    "cuisine_overlap_ratio",
    "category_overlap_ratio",
    "district_exact_match",
    "district_partial_match",
    "category_target_match",
    "cuisine_target_match",
    "audience_match_ratio",
    "business_audience_match",
    "price_ceiling_fit",
    "price_floor_fit",
    "price_range_overlap",
    "budget_gap_ratio",
    "delivery_match",
    "booking_match",
    "schedule_match_any",
    "schedule_match_mean",
    "early_open_match",
    "late_open_match",
    "midday_open_match",
    "all_week_open_match",
    "cheapness_score",
    "luxury_score",
    "price_mid",
    "vi_tri",
    "gia_ca",
    "chat_luong",
    "phuc_vu",
    "khong_gian",
    "quality_score_mean",
    "quality_pref_match",
    "service_pref_match",
    "space_pref_match",
    "position_pref_match",
    "price_value_pref_match",
    "log_totalview",
    "log_totalfavourite",
    "log_totalcheckins",
    "popularity_blend",
    "view_intent",
    "favourite_intent",
    "checkin_intent",
    "review_intent",
    "cheap_intent",
    "luxury_intent",
    "weekend_requested",
    "weekday_requested",
    "breakfast_requested",
    "brunch_requested",
    "lunch_requested",
    "afternoon_requested",
    "dinner_requested",
    "late_night_requested",
    "delivery_required",
    "booking_required",
    "business_intent",
]

RESTAURANT_RESPONSE_COLUMNS = [
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
    "vi_tri",
    "gia_ca",
    "chat_luong",
    "phuc_vu",
    "khong_gian",
    "excellent",
    "good",
    "average",
    "bad",
    "totalview",
    "totalfavourite",
    "totalcheckins",
    "delivery_flag",
    "booking_flag",
    "rest_days_raw",
    *DAY_COLUMNS,
]


@dataclass
class TextArtifacts:
    word_vectorizer: TfidfVectorizer
    char_vectorizer: TfidfVectorizer
    location_pattern_map: dict[str, set[str]]


def normalize_text(text: Any) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    text = str(text).replace("Đ", "D").replace("đ", "d")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def slugify_column_name(name: Any) -> str:
    text = normalize_text(name)
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


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


def tokenize(text: Any) -> list[str]:
    return [tok for tok in normalize_text(text).split() if tok and tok not in STOPWORDS]


def contains_phrase(text: Any, phrase: Any) -> bool:
    text_norm = normalize_text(text)
    phrase_norm = normalize_text(phrase)
    if not text_norm or not phrase_norm:
        return False
    pattern = rf"(?<![a-z0-9]){re.escape(phrase_norm)}(?![a-z0-9])"
    return re.search(pattern, text_norm) is not None


def mentions_any(text: Any, phrases: tuple[str, ...] | list[str] | set[str]) -> bool:
    text_norm = normalize_text(text)
    return any(contains_phrase(text_norm, phrase) for phrase in phrases)


def split_pipe_values(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    parts = re.split(r"\s*\|+\s*", str(value))
    clean: list[str] = []
    seen: set[str] = set()
    for part in parts:
        item = str(part).strip()
        norm = normalize_text(item)
        if not norm or norm in seen:
            continue
        clean.append(item)
        seen.add(norm)
    return clean


def join_values(values: list[str]) -> str:
    return " | ".join(values)


def canonical_location(text: Any) -> str:
    location = normalize_text(text)
    location = re.sub(r"\b(thanh pho|tp|quan|q|huyen|h|thi xa|tx)\b", " ", location)
    return re.sub(r"\s+", " ", location).strip()


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


def extract_location_target(query: str, pattern_map: dict[str, set[str]]) -> str:
    query_norm = normalize_text(query)
    candidates = sorted(pattern_map.items(), key=lambda item: max(len(pattern) for pattern in item[1]), reverse=True)
    for canonical, patterns in candidates:
        if any(contains_phrase(query_norm, pattern) for pattern in patterns):
            return canonical
    return ""


def extract_pattern_targets(query: str, mapping: dict[str, dict[str, list[str]]]) -> list[str]:
    query_norm = normalize_text(query)
    return [label for label, config in mapping.items() if any(contains_phrase(query_norm, phrase) for phrase in config["query"])]


def parse_day_list(value: Any) -> list[str]:
    items = split_pipe_values(value)
    result: list[str] = []
    for item in items:
        norm = normalize_text(item)
        if norm in DAY_MAP and DAY_MAP[norm] not in result:
            result.append(DAY_MAP[norm])
    return result


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
    match = re.search(r"(\d+(?:[\.,]\d+)?)\s*(trieu|nghin|ngan|k)?\s*[-–]\s*(\d+(?:[\.,]\d+)?)\s*(trieu|nghin|ngan|k)?", query_norm)
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


def extract_day_profile(query: str) -> dict[str, Any]:
    query_norm = normalize_text(query)
    day_targets = [slug for phrase, slug in DAY_MAP.items() if contains_phrase(query_norm, phrase)]
    weekend_requested = float(contains_phrase(query_norm, "cuoi tuan"))
    weekday_requested = float(contains_phrase(query_norm, "ngay thuong"))
    all_week_requested = float(any(contains_phrase(query_norm, phrase) for phrase in ("ca tuan", "mo cua ca tuan", "mo ban ca tuan")))
    if weekend_requested:
        day_targets.extend(["thu_bay", "chu_nhat"])
    if weekday_requested:
        day_targets.extend(["thu_hai", "thu_ba", "thu_tu", "thu_nam", "thu_sau"])
    if all_week_requested:
        day_targets.extend(DAY_COLUMNS)
    unique_days: list[str] = []
    for day in day_targets:
        if day not in unique_days:
            unique_days.append(day)
    return {
        "day_targets": unique_days,
        "weekend_requested": weekend_requested,
        "weekday_requested": weekday_requested,
        "all_week_requested": all_week_requested,
    }


def extract_time_preferences(query: str) -> dict[str, Any]:
    query_norm = normalize_text(query)
    profile = {
        "target_times": [],
        "breakfast_requested": 0.0,
        "brunch_requested": 0.0,
        "lunch_requested": 0.0,
        "afternoon_requested": 0.0,
        "dinner_requested": 0.0,
        "late_night_requested": 0.0,
        "early_required": 0.0,
        "late_required": 0.0,
        "midday_required": 0.0,
    }

    def add(flag_name: str, minute_value: int) -> None:
        profile[flag_name] = 1.0
        if minute_value not in profile["target_times"]:
            profile["target_times"].append(minute_value)

    if mentions_any(query_norm, ("an sang", "bua sang", "mo cua som", "mo buoi sang", "sang som")):
        add("breakfast_requested", TIME_SLOTS["breakfast"])
        profile["early_required"] = 1.0
    if contains_phrase(query_norm, "brunch"):
        add("brunch_requested", TIME_SLOTS["brunch"])
    if mentions_any(query_norm, ("an trua", "bua trua", "buoi trua", "trua", "xuyen trua")):
        add("lunch_requested", TIME_SLOTS["lunch"])
        profile["midday_required"] = 1.0
    if mentions_any(query_norm, ("buoi chieu", "chieu")):
        add("afternoon_requested", TIME_SLOTS["afternoon"])

    dinner_signal = mentions_any(query_norm, ("buoi toi", "bua toi", "an toi", "mo buoi toi", "sau gio lam"))
    dinner_signal = dinner_signal or re.search(r"(?<![a-z0-9])toi\s+(thu|chu nhat|cuoi tuan)", query_norm) is not None
    if dinner_signal:
        add("dinner_requested", TIME_SLOTS["dinner"])

    if mentions_any(query_norm, ("an khuya", "khuya", "mo muon", "sau 10 gio toi", "mo sau 10 gio toi")):
        add("late_night_requested", TIME_SLOTS["late_night"])
        profile["late_required"] = 1.0

    return profile


def build_query_phrase_candidates(query: str, cuisine_targets: list[str], category_targets: list[str]) -> list[str]:
    query_norm = normalize_text(query)
    phrases: list[str] = []
    for target in cuisine_targets:
        phrases.extend(CUISINE_PATTERNS[target]["query"])
    for target in category_targets:
        phrases.extend(CATEGORY_PATTERNS[target]["query"])

    tokens = [tok for tok in query_norm.split() if tok not in STOPWORDS]
    for n in (3, 2):
        if len(tokens) < n:
            continue
        for idx in range(len(tokens) - n + 1):
            gram_tokens = tokens[idx : idx + n]
            meaningful = [tok for tok in gram_tokens if tok not in GENERIC_QUERY_TOKENS and not tok.isdigit()]
            if not meaningful:
                continue
            phrase = " ".join(gram_tokens)
            if phrase not in phrases:
                phrases.append(phrase)

    for tok in tokens:
        if tok not in GENERIC_QUERY_TOKENS and not tok.isdigit() and len(tok) >= 3 and tok not in phrases:
            phrases.append(tok)

    return phrases[:12]


def extract_preference_targets(query: str) -> dict[str, float]:
    query_norm = normalize_text(query)

    quality_target = np.nan
    if mentions_any(query_norm, ("tuyet voi", "xuat sac")):
        quality_target = 5.0
    elif mentions_any(query_norm, ("khong can qua ngon", "chi can khong te", "khong te")):
        quality_target = 3.5
    elif mentions_any(query_norm, ("chat luong", "do an ngon", "ngon", "danh gia cao", "review tot")):
        quality_target = 4.5

    service_target = np.nan
    if mentions_any(query_norm, ("phuc vu on", "phuc vu khong te")):
        service_target = 3.5
    elif mentions_any(query_norm, ("phuc vu tot", "phuc vu nhanh", "nhiet tinh", "chi chu")):
        service_target = 4.5

    space_target = np.nan
    if contains_phrase(query_norm, "khong gian") and mentions_any(query_norm, ("te", "trung binh")):
        space_target = 2.5
    elif mentions_any(
        query_norm,
        (
            "khong gian dep",
            "khong gian xinh",
            "khong gian rong",
            "khong gian yen tinh",
            "khong gian de chiu",
            "khong gian thoai mai",
            "khong gian mo",
            "lang man",
            "rieng tu",
            "view dep",
            "rooftop",
            "acoustic",
            "thoang",
        ),
    ):
        space_target = 4.5

    position_target = np.nan
    if mentions_any(query_norm, ("vi tri dep", "vi tri tot", "vi tri duoc danh gia cao", "gan trung tam", "de tim")):
        position_target = 4.5

    price_value_target = np.nan
    if mentions_any(query_norm, ("gia hop ly", "gia mem", "gia re", "binh dan", "hop tui tien", "gia vua tam", "gia vua phai", "gia de chiu")):
        price_value_target = 4.5

    return {
        "quality_target": quality_target,
        "service_target": service_target,
        "space_target": space_target,
        "position_target": position_target,
        "price_value_target": price_value_target,
    }


def score_preference_match(value: float, target: float | None) -> float:
    if target is None or pd.isna(target):
        return 0.0
    return float(max(0.0, 1.0 - abs(value - target) / 3.0))


def compute_price_range_overlap(price_min: float, price_max: float, floor_value: float, ceiling_value: float) -> float:
    if price_min <= 0 and price_max <= 0:
        return 0.0
    if price_max <= 0:
        price_max = price_min
    if pd.isna(floor_value) and pd.isna(ceiling_value):
        return 0.0
    if pd.isna(floor_value):
        return float(price_min <= ceiling_value)
    if pd.isna(ceiling_value):
        return float(price_max >= floor_value)
    overlap = max(0.0, min(price_max, ceiling_value) - max(price_min, floor_value))
    span = max(1.0, ceiling_value - floor_value)
    return float(overlap / span)


def compute_price_gap_ratio(price_min: float, price_max: float, floor_value: float, ceiling_value: float) -> float:
    if price_min <= 0 and price_max <= 0:
        return 0.0
    if price_max <= 0:
        price_max = price_min
    if pd.isna(floor_value) and pd.isna(ceiling_value):
        return 0.0
    gap = 0.0
    if not pd.isna(floor_value) and price_max < floor_value:
        gap += floor_value - price_max
    if not pd.isna(ceiling_value) and price_min > ceiling_value:
        gap += price_min - ceiling_value
    scale = max(
        1.0,
        (0.0 if pd.isna(ceiling_value) else ceiling_value) - (0.0 if pd.isna(floor_value) else floor_value),
        floor_value if not pd.isna(floor_value) else 0.0,
        ceiling_value if not pd.isna(ceiling_value) else 0.0,
    )
    return float(gap / scale)


def minute_of_day(value: str) -> int:
    hour, minute = value.split(":")
    return int(hour) * 60 + int(minute)


def parse_day_schedule(value: Any) -> tuple[float, float]:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return (np.nan, np.nan)
    patterns = [
        r"\('([0-9]{2}:[0-9]{2})',\s*'([0-9]{2}:[0-9]{2})'\)",
        r"\(\('([0-9]{2}:[0-9]{2})',\s*'([0-9]{2}:[0-9]{2})'\),\s*(True|False)\)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            open_time, close_time = match.groups()[:2]
            return (minute_of_day(open_time), minute_of_day(close_time))
    return (np.nan, np.nan)


def is_open_for_slot(value: Any, slot: Any) -> float:
    open_min, close_min = parse_day_schedule(value)
    if pd.isna(open_min) or pd.isna(close_min):
        return 0.0
    target = TIME_SLOTS.get(slot) if isinstance(slot, str) else slot
    if target is None:
        return 1.0
    if close_min <= open_min:
        close_min += 24 * 60
        if target < open_min:
            target += 24 * 60
    return float(open_min <= target <= close_min)


def extract_audience_targets(query: str) -> list[str]:
    query_norm = normalize_text(query)
    return [
        label
        for label, phrases in AUDIENCE_PATTERNS.items()
        if any(contains_phrase(query_norm, phrase) for phrase in phrases)
    ]


def extract_query_flags(query: str) -> dict[str, float]:
    query_norm = normalize_text(query)
    cheap_intent = float(
        mentions_any(
            query_norm,
            ("gia hop ly", "gia mem", "gia re", "binh dan", "hop tui tien", "khong qua dat", "gia vua tam", "gia vua phai", "gia de chiu"),
        )
    )
    return {
        "cheap_intent": cheap_intent,
        "luxury_intent": float(mentions_any(query_norm, ("sang trong", "cao cap", "lang man"))),
        "view_intent": float(mentions_any(query_norm, ("luot xem", "nhieu nguoi xem"))),
        "favourite_intent": float(mentions_any(query_norm, ("yeu thich", "nhieu luot yeu thich", "duoc nhieu nguoi thich"))),
        "checkin_intent": float(mentions_any(query_norm, ("check in", "checkin", "check-in"))),
        "review_intent": float(mentions_any(query_norm, ("review", "danh gia", "nhieu review"))),
        "delivery_required": float(mentions_any(query_norm, ("giao hang", "giao tan noi", "ship"))),
        "booking_required": float(mentions_any(query_norm, ("dat ban", "nhan dat ban"))),
        "business_intent": float(mentions_any(query_norm, ("manager", "tiep khach", "khach hang", "doi tac", "doanh nghiep"))),
    }


def match_overlap_ratio(query_tokens: list[str], text: str) -> float:
    text_tokens = set(tokenize(text))
    content_tokens = {tok for tok in query_tokens if tok not in GENERIC_QUERY_TOKENS}
    if not content_tokens:
        return 0.0
    return len(content_tokens & text_tokens) / len(content_tokens)


def mapped_target_match(text: Any, targets: list[str], mapping: dict[str, dict[str, list[str]]]) -> float:
    if not targets:
        return 0.0
    text_norm = normalize_text(text)
    matched = 0
    for target in targets:
        if any(contains_phrase(text_norm, phrase) for phrase in mapping[target]["meta"]):
            matched += 1
    return matched / len(targets)


def audience_match_ratio(targets: list[str], audience_values: list[str]) -> float:
    if not targets:
        return 0.0
    audience_text = join_values(audience_values)
    matched = 0
    for target in targets:
        if any(contains_phrase(audience_text, phrase) for phrase in AUDIENCE_PATTERNS.get(target, [target])):
            matched += 1
    return matched / len(targets)


def default_ranker(random_state: int = RANDOM_STATE, n_estimators: int = 250) -> XGBRanker:
    return XGBRanker(
        objective="rank:pairwise",
        eval_metric=["ndcg@5", "ndcg@10"],
        learning_rate=0.05,
        n_estimators=n_estimators,
        max_depth=6,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        tree_method="hist",
        random_state=random_state,
    )


def compute_group_metrics(eval_df: pd.DataFrame, score_col: str = "pred_score", top_ks: tuple[int, ...] = (5, 10)) -> dict[str, float]:
    ndcg_store = defaultdict(list)
    mrr_store: list[float] = []
    hit_store = {k: [] for k in top_ks}

    for _, group in eval_df.groupby("query"):
        group = group.sort_values(score_col, ascending=False).reset_index(drop=True)
        y_true = group["label"].to_numpy(dtype=float)
        y_score = group[score_col].to_numpy(dtype=float)

        for k in top_ks:
            score = ndcg_score([y_true], [y_score], k=min(k, len(group)))
            ndcg_store[k].append(float(score))

        relevant_positions = np.where(group["label"].to_numpy(dtype=float) >= 4)[0]
        mrr_store.append(float(1.0 / (relevant_positions[0] + 1)) if len(relevant_positions) else 0.0)

        for k in top_ks:
            hit_store[k].append(float((group["label"].head(k) >= 4).any()))

    metrics = {f"NDCG@{k}": float(np.mean(values)) for k, values in ndcg_store.items()}
    metrics["MRR"] = float(np.mean(mrr_store))
    for k in top_ks:
        metrics[f"HIT@{k}"] = float(np.mean(hit_store[k]))
    return metrics


def load_labels_dataframe(labels_path: Path | str) -> pd.DataFrame:
    labels_df = pd.read_csv(labels_path, encoding="utf-8")
    required = {"query", "restaurant_id", "label"}
    missing = required - set(labels_df.columns)
    if missing:
        raise KeyError(f"Missing required label columns: {sorted(missing)}")
    labels_df = labels_df.copy()
    labels_df["restaurant_id"] = labels_df["restaurant_id"].map(normalize_restaurant_id)
    labels_df["label"] = pd.to_numeric(labels_df["label"], errors="coerce").fillna(0.0)
    labels_df["query"] = labels_df["query"].astype(str)
    return labels_df


def prepare_restaurant_catalog(restaurants_df: pd.DataFrame) -> pd.DataFrame:
    df = restaurants_df.copy()
    df = df.rename(columns={col: slugify_column_name(col) for col in df.columns})
    if "at_ban" in df.columns and "dat_ban" not in df.columns:
        df = df.rename(columns={"at_ban": "dat_ban"})

    df = df.rename(
        columns={
            "restaurantid": "restaurant_id",
            "name": "restaurant_name_meta",
            "district": "district_meta",
            "area": "area_meta",
            "address": "address_meta",
            "metakeywords": "meta_keywords",
            "cuisines": "cuisines_meta",
            "lsttargetaudience": "target_audience_raw",
            "lstcategory": "category_raw",
            "restauranturl": "restaurant_url",
            "ngay_nghi": "rest_days_raw",
            "giao_tan_noi": "delivery_flag",
            "dat_ban": "booking_flag",
        }
    )

    if "restaurant_id" not in df.columns:
        raise KeyError("Restaurant metadata must contain a RestaurantID/restaurant_id column")

    df["restaurant_id"] = df["restaurant_id"].map(normalize_restaurant_id)
    df = df[df["restaurant_id"] != ""].copy()
    df = df.drop_duplicates(subset=["restaurant_id"], keep="first").reset_index(drop=True)

    numeric_columns = [
        "pricemin",
        "pricemax",
        "vi_tri",
        "gia_ca",
        "chat_luong",
        "phuc_vu",
        "khong_gian",
        "excellent",
        "good",
        "average",
        "bad",
        "totalview",
        "totalfavourite",
        "totalcheckins",
        "delivery_flag",
        "booking_flag",
    ]
    for col in numeric_columns:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    text_columns = [
        "restaurant_name_meta",
        "address_meta",
        "district_meta",
        "area_meta",
        "meta_keywords",
        "cuisines_meta",
        "target_audience_raw",
        "category_raw",
        "restaurant_url",
        "rest_days_raw",
    ]
    for col in text_columns:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    for col in DAY_COLUMNS:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    df["target_audience_list"] = df["target_audience_raw"].map(split_pipe_values)
    df["category_list"] = df["category_raw"].map(split_pipe_values)
    df["cuisine_list"] = df["cuisines_meta"].map(split_pipe_values)
    df["rest_day_list"] = df["rest_days_raw"].map(parse_day_list)
    df["active_days"] = df["rest_day_list"].map(lambda rest_days: [day for day in DAY_COLUMNS if day not in rest_days])
    df["target_audience_text"] = df["target_audience_list"].map(join_values)
    df["category_text"] = df["category_list"].map(join_values)
    df["cuisine_text"] = df["cuisine_list"].map(join_values)
    df["restaurant_topic_text"] = df[
        ["restaurant_name_meta", "meta_keywords", "cuisine_text", "category_text", "target_audience_text"]
    ].agg(" | ".join, axis=1)
    df["restaurant_text"] = df[
        ["restaurant_name_meta", "address_meta", "district_meta", "area_meta", "meta_keywords", "cuisine_text", "category_text", "target_audience_text"]
    ].agg(" | ".join, axis=1)
    df["restaurant_text_norm"] = df["restaurant_text"].map(normalize_text)
    df["restaurant_topic_text_norm"] = df["restaurant_topic_text"].map(normalize_text)
    df["district_norm"] = df["district_meta"].map(canonical_location)
    df["area_norm"] = df["area_meta"].map(canonical_location)
    df["price_mid"] = np.where(
        (df["pricemin"] > 0) & (df["pricemax"] > 0),
        (df["pricemin"] + df["pricemax"]) / 2.0,
        np.where(df["pricemax"] > 0, df["pricemax"], df["pricemin"]),
    ).astype(float)
    df["log_totalview"] = np.log1p(df["totalview"])
    df["log_totalfavourite"] = np.log1p(df["totalfavourite"])
    df["log_totalcheckins"] = np.log1p(df["totalcheckins"])
    df["popularity_blend"] = 0.50 * df["log_totalview"] + 0.30 * df["log_totalfavourite"] + 0.20 * df["log_totalcheckins"]
    df["quality_score_mean"] = df[["vi_tri", "gia_ca", "chat_luong", "phuc_vu", "khong_gian"]].mean(axis=1)
    df["delivery_available"] = (df["delivery_flag"] > 0).astype(float)
    df["booking_available"] = (df["booking_flag"] > 0).astype(float)
    df["cheapness_score"] = 1.0 / (1.0 + np.log1p(df["price_mid"].clip(lower=0.0)))
    df["luxury_score"] = np.log1p(df["price_mid"].clip(lower=0.0))

    return df.sort_values("restaurant_id").reset_index(drop=True)


def load_restaurants_dataframe(restaurants_path: Path | str) -> pd.DataFrame:
    restaurants_df = pd.read_csv(restaurants_path, encoding="utf-8")
    return prepare_restaurant_catalog(restaurants_df)


def fit_text_artifacts(catalog_df: pd.DataFrame, extra_location_values: Optional[pd.Series] = None) -> TextArtifacts:
    word_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)
    word_vectorizer.fit(catalog_df["restaurant_text_norm"])
    char_vectorizer.fit(catalog_df["restaurant_text_norm"])

    location_values = [catalog_df["district_meta"], catalog_df["area_meta"]]
    if extra_location_values is not None:
        location_values.append(extra_location_values.astype(str))
    location_pattern_map = build_location_pattern_map(pd.concat(location_values, ignore_index=True))

    return TextArtifacts(
        word_vectorizer=word_vectorizer,
        char_vectorizer=char_vectorizer,
        location_pattern_map=location_pattern_map,
    )


def build_query_profiles(queries: list[str], location_pattern_map: dict[str, set[str]]) -> pd.DataFrame:
    query_rows: list[dict[str, Any]] = []
    for query in queries:
        floor_value, ceiling_value = extract_price_bounds(query)
        day_profile = extract_day_profile(query)
        time_profile = extract_time_preferences(query)
        category_targets = extract_pattern_targets(query, CATEGORY_PATTERNS)
        cuisine_targets = extract_pattern_targets(query, CUISINE_PATTERNS)
        row = {
            "query": query,
            "district_target": extract_location_target(query, location_pattern_map),
            "category_targets": category_targets,
            "cuisine_targets": cuisine_targets,
            "audience_targets": extract_audience_targets(query),
            "price_floor": floor_value,
            "price_ceiling": ceiling_value,
            "topic_phrases": build_query_phrase_candidates(query, cuisine_targets, category_targets),
        }
        row.update(day_profile)
        row.update(time_profile)
        row.update(extract_query_flags(query))
        row.update(extract_preference_targets(query))
        query_rows.append(row)
    return pd.DataFrame(query_rows)


def build_feature_frame(pairs_df: pd.DataFrame, catalog_df: pd.DataFrame, text_artifacts: TextArtifacts) -> pd.DataFrame:
    required = {"query", "restaurant_id"}
    missing = required - set(pairs_df.columns)
    if missing:
        raise KeyError(f"Missing required pair columns: {sorted(missing)}")

    frame = pairs_df.copy()
    frame["query"] = frame["query"].astype(str)
    frame["restaurant_id"] = frame["restaurant_id"].map(normalize_restaurant_id)

    merged = frame.merge(catalog_df, on="restaurant_id", how="left", validate="many_to_one")
    if merged["restaurant_name_meta"].isna().any():
        missing_ids = merged.loc[merged["restaurant_name_meta"].isna(), "restaurant_id"].dropna().unique().tolist()
        raise ValueError(f"Missing restaurant metadata for ids: {missing_ids[:10]}")

    query_profile_df = build_query_profiles(merged["query"].drop_duplicates().tolist(), text_artifacts.location_pattern_map)
    merged = merged.merge(query_profile_df, on="query", how="left", validate="many_to_one")
    merged["query_norm"] = merged["query"].map(normalize_text)
    merged["query_tokens"] = merged["query"].map(tokenize)

    merged["topic_overlap_ratio"] = [
        match_overlap_ratio(query_tokens, text)
        for query_tokens, text in zip(merged["query_tokens"], merged["restaurant_topic_text"])
    ]
    merged["name_overlap_ratio"] = [
        match_overlap_ratio(query_tokens, text)
        for query_tokens, text in zip(merged["query_tokens"], merged["restaurant_name_meta"])
    ]
    merged["meta_keyword_overlap_ratio"] = [
        match_overlap_ratio(query_tokens, text)
        for query_tokens, text in zip(merged["query_tokens"], merged["meta_keywords"])
    ]
    merged["cuisine_overlap_ratio"] = [
        match_overlap_ratio(query_tokens, text)
        for query_tokens, text in zip(merged["query_tokens"], merged["cuisine_text"] + " " + merged["meta_keywords"])
    ]
    merged["category_overlap_ratio"] = [
        match_overlap_ratio(query_tokens, text)
        for query_tokens, text in zip(merged["query_tokens"], merged["category_text"])
    ]
    merged["topic_exact_phrase_hit"] = [
        float(any(contains_phrase(text, phrase) for phrase in phrases))
        for phrases, text in zip(merged["topic_phrases"], merged["restaurant_topic_text_norm"])
    ]
    merged["district_exact_match"] = [
        float(target != "" and (district_norm == target or area_norm == target))
        for target, district_norm, area_norm in zip(merged["district_target"], merged["district_norm"], merged["area_norm"])
    ]
    merged["district_partial_match"] = [
        0.0
        if not target
        else float(
            target in f"{district_norm} {area_norm}".strip()
            or district_norm in target
            or area_norm in target
            or target == district_norm
            or target == area_norm
        )
        for target, district_norm, area_norm in zip(merged["district_target"], merged["district_norm"], merged["area_norm"])
    ]
    merged["category_target_match"] = [
        mapped_target_match(text, targets, CATEGORY_PATTERNS)
        for text, targets in zip(merged["category_text"], merged["category_targets"])
    ]
    merged["cuisine_target_match"] = [
        mapped_target_match(text, targets, CUISINE_PATTERNS)
        for text, targets in zip(merged["cuisine_text"] + " " + merged["meta_keywords"], merged["cuisine_targets"])
    ]
    merged["audience_match_ratio"] = [
        audience_match_ratio(targets, values)
        for targets, values in zip(merged["audience_targets"], merged["target_audience_list"])
    ]
    business_terms = AUDIENCE_PATTERNS["gioi manager"] + AUDIENCE_PATTERNS["dan van phong"]
    merged["business_audience_match"] = [
        0.0
        if business_intent <= 0
        else max(
            audience_score,
            float(any(any(contains_phrase(value, term) for term in business_terms) for value in audience_values)),
        )
        for business_intent, audience_score, audience_values in zip(
            merged["business_intent"],
            merged["audience_match_ratio"],
            merged["target_audience_list"],
        )
    ]
    merged["price_ceiling_fit"] = [
        0.0 if pd.isna(ceiling) else float(price_mid <= ceiling)
        for ceiling, price_mid in zip(merged["price_ceiling"], merged["price_mid"])
    ]
    merged["price_floor_fit"] = [
        0.0 if pd.isna(floor_value) else float(price_mid >= floor_value)
        for floor_value, price_mid in zip(merged["price_floor"], merged["price_mid"])
    ]
    merged["price_range_overlap"] = [
        compute_price_range_overlap(price_min, price_max, floor_value, ceiling_value)
        for price_min, price_max, floor_value, ceiling_value in zip(
            merged["pricemin"],
            merged["pricemax"],
            merged["price_floor"],
            merged["price_ceiling"],
        )
    ]
    merged["budget_gap_ratio"] = [
        compute_price_gap_ratio(price_min, price_max, floor_value, ceiling_value)
        for price_min, price_max, floor_value, ceiling_value in zip(
            merged["pricemin"],
            merged["pricemax"],
            merged["price_floor"],
            merged["price_ceiling"],
        )
    ]
    merged["delivery_match"] = np.where(merged["delivery_required"] > 0, merged["delivery_available"], 0.0)
    merged["booking_match"] = np.where(merged["booking_required"] > 0, merged["booking_available"], 0.0)

    def schedule_score(value: Any, day_slug: str, target_times: list[int], active_days: list[str]) -> float:
        if day_slug not in active_days:
            return 0.0
        if not target_times:
            open_min, close_min = parse_day_schedule(value)
            return float(not (pd.isna(open_min) or pd.isna(close_min)))
        return max(is_open_for_slot(value, target_time) for target_time in target_times)

    def schedule_summary(row: pd.Series) -> tuple[float, float]:
        target_days = row["day_targets"] if row["day_targets"] else (row["active_days"] if row["target_times"] else [])
        if not target_days:
            return 0.0, 0.0
        scores = [schedule_score(row[day], day, row["target_times"], row["active_days"]) for day in target_days]
        if not scores:
            return 0.0, 0.0
        return float(max(scores)), float(np.mean(scores))

    schedule_pairs = merged.apply(schedule_summary, axis=1)
    merged["schedule_match_any"] = schedule_pairs.map(lambda value: value[0])
    merged["schedule_match_mean"] = schedule_pairs.map(lambda value: value[1])

    def slot_match(row: pd.Series, minute_value: int, required_col: str) -> float:
        if row[required_col] <= 0:
            return 0.0
        target_days = row["day_targets"] if row["day_targets"] else row["active_days"]
        if not target_days:
            return 0.0
        return float(max(schedule_score(row[day], day, [minute_value], row["active_days"]) for day in target_days))

    merged["early_open_match"] = merged.apply(lambda row: slot_match(row, TIME_SLOTS["breakfast"], "early_required"), axis=1)
    merged["midday_open_match"] = merged.apply(lambda row: slot_match(row, TIME_SLOTS["lunch"], "midday_required"), axis=1)
    merged["late_open_match"] = merged.apply(lambda row: slot_match(row, TIME_SLOTS["late_night"], "late_required"), axis=1)
    merged["all_week_open_match"] = merged.apply(
        lambda row: 0.0
        if row["all_week_requested"] <= 0
        else float(
            all(
                day in row["active_days"] and schedule_score(row[day], day, row["target_times"], row["active_days"]) > 0
                for day in DAY_COLUMNS
            )
        ),
        axis=1,
    )

    merged["quality_pref_match"] = [
        score_preference_match(value, target)
        for value, target in zip(merged["chat_luong"], merged["quality_target"])
    ]
    merged["service_pref_match"] = [
        score_preference_match(value, target)
        for value, target in zip(merged["phuc_vu"], merged["service_target"])
    ]
    merged["space_pref_match"] = [
        score_preference_match(value, target)
        for value, target in zip(merged["khong_gian"], merged["space_target"])
    ]
    merged["position_pref_match"] = [
        score_preference_match(value, target)
        for value, target in zip(merged["vi_tri"], merged["position_target"])
    ]
    merged["price_value_pref_match"] = [
        score_preference_match(value, target)
        for value, target in zip(merged["gia_ca"], merged["price_value_target"])
    ]

    word_query_matrix = text_artifacts.word_vectorizer.transform(merged["query_norm"])
    word_restaurant_matrix = text_artifacts.word_vectorizer.transform(merged["restaurant_text_norm"])
    merged["tfidf_cosine"] = np.asarray(word_query_matrix.multiply(word_restaurant_matrix).sum(axis=1)).ravel().astype(float)

    char_query_matrix = text_artifacts.char_vectorizer.transform(merged["query_norm"])
    char_restaurant_matrix = text_artifacts.char_vectorizer.transform(merged["restaurant_text_norm"])
    merged["char_tfidf_cosine"] = np.asarray(char_query_matrix.multiply(char_restaurant_matrix).sum(axis=1)).ravel().astype(float)

    for col in FEATURE_COLUMNS:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)

    return merged


def split_by_query(df: pd.DataFrame, random_state: int = RANDOM_STATE) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    unique_queries = df["query"].drop_duplicates().to_numpy()
    train_queries, temp_queries = train_test_split(unique_queries, test_size=0.30, random_state=random_state)
    val_queries, test_queries = train_test_split(temp_queries, test_size=0.50, random_state=random_state)

    def subset(query_values: np.ndarray) -> pd.DataFrame:
        return df[df["query"].isin(query_values)].sort_values(["query", "restaurant_id"]).reset_index(drop=True)

    return subset(train_queries), subset(val_queries), subset(test_queries)


def group_sizes(df: pd.DataFrame) -> list[int]:
    return df.groupby("query").size().tolist()


def train_ranker(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    feature_columns: Optional[list[str]] = None,
    random_state: int = RANDOM_STATE,
    n_estimators: int = 250,
) -> tuple[XGBRanker, float]:
    feature_columns = feature_columns or FEATURE_COLUMNS
    ranker = default_ranker(random_state=random_state, n_estimators=n_estimators)

    fit_kwargs: dict[str, Any] = {"group": group_sizes(train_df), "verbose": False}
    if val_df is not None and not val_df.empty:
        fit_kwargs["eval_set"] = [(val_df[feature_columns], val_df["label"])]
        fit_kwargs["eval_group"] = [group_sizes(val_df)]

    start_time = time.time()
    ranker.fit(train_df[feature_columns], train_df["label"], **fit_kwargs)
    elapsed = time.time() - start_time
    return ranker, elapsed


def evaluate_ranker(
    ranker: XGBRanker,
    test_df: pd.DataFrame,
    feature_columns: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    feature_columns = feature_columns or FEATURE_COLUMNS
    eval_df = test_df.copy()
    eval_df["pred_score"] = ranker.predict(eval_df[feature_columns])
    metrics = compute_group_metrics(eval_df, score_col="pred_score")
    return eval_df, metrics


def feature_importance_frame(ranker: XGBRanker, feature_columns: Optional[list[str]] = None) -> pd.DataFrame:
    feature_columns = feature_columns or FEATURE_COLUMNS
    return (
        pd.DataFrame({"feature": feature_columns, "importance": ranker.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


class RestaurantRankerService:
    def __init__(
        self,
        model: XGBRanker,
        restaurant_catalog: pd.DataFrame,
        text_artifacts: TextArtifacts,
        feature_columns: list[str],
        evaluation_summary: dict[str, Any],
        artifact_path: Optional[Path] = None,
    ) -> None:
        self.model = model
        self.restaurant_catalog = restaurant_catalog.sort_values("restaurant_id").reset_index(drop=True)
        self.text_artifacts = text_artifacts
        self.feature_columns = feature_columns
        self.evaluation_summary = evaluation_summary
        self.artifact_path = artifact_path

    @classmethod
    def train_from_paths(
        cls,
        labels_path: Path | str = DEFAULT_LABELS_PATH,
        restaurants_path: Path | str = DEFAULT_RESTAURANTS_PATH,
        artifact_path: Optional[Path | str] = DEFAULT_ARTIFACT_PATH,
        metrics_path: Optional[Path | str] = DEFAULT_METRICS_PATH,
        n_estimators: int = 250,
    ) -> "RestaurantRankerService":
        labels_path = Path(labels_path)
        restaurants_path = Path(restaurants_path)
        artifact_path = Path(artifact_path) if artifact_path is not None else None
        metrics_path = Path(metrics_path) if metrics_path is not None else None

        labels_df = load_labels_dataframe(labels_path)
        restaurant_catalog = load_restaurants_dataframe(restaurants_path)
        labels_df = labels_df[labels_df["restaurant_id"].isin(restaurant_catalog["restaurant_id"])].copy()

        text_artifacts = fit_text_artifacts(
            restaurant_catalog,
            extra_location_values=labels_df["district"] if "district" in labels_df.columns else None,
        )
        labeled_pairs = labels_df[["query", "restaurant_id", "label"]].copy()
        feature_df = build_feature_frame(labeled_pairs, restaurant_catalog, text_artifacts)

        train_df, val_df, test_df = split_by_query(feature_df)
        eval_model, train_seconds = train_ranker(
            train_df=train_df,
            val_df=val_df,
            feature_columns=FEATURE_COLUMNS,
            n_estimators=n_estimators,
        )
        _, metrics = evaluate_ranker(eval_model, test_df, feature_columns=FEATURE_COLUMNS)

        final_model, final_train_seconds = train_ranker(
            train_df=feature_df.sort_values(["query", "restaurant_id"]).reset_index(drop=True),
            val_df=None,
            feature_columns=FEATURE_COLUMNS,
            n_estimators=n_estimators,
        )

        importance_df = feature_importance_frame(final_model, FEATURE_COLUMNS)
        evaluation_summary = {
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "labels_path": str(labels_path),
            "restaurants_path": str(restaurants_path),
            "catalog_size": int(len(restaurant_catalog)),
            "labeled_pair_count": int(len(feature_df)),
            "query_count": int(feature_df["query"].nunique()),
            "feature_count": len(FEATURE_COLUMNS),
            "split_summary": {
                "train_queries": int(train_df["query"].nunique()),
                "val_queries": int(val_df["query"].nunique()),
                "test_queries": int(test_df["query"].nunique()),
                "train_rows": int(len(train_df)),
                "val_rows": int(len(val_df)),
                "test_rows": int(len(test_df)),
            },
            "clean_test_metrics": metrics,
            "eval_training_seconds": float(train_seconds),
            "final_training_seconds": float(final_train_seconds),
            "top_feature_importance": importance_df.head(20).to_dict(orient="records"),
        }

        service = cls(
            model=final_model,
            restaurant_catalog=restaurant_catalog,
            text_artifacts=text_artifacts,
            feature_columns=FEATURE_COLUMNS,
            evaluation_summary=evaluation_summary,
            artifact_path=artifact_path,
        )

        if artifact_path is not None:
            service.save(artifact_path)
        if metrics_path is not None:
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(json.dumps(evaluation_summary, ensure_ascii=False, indent=2), encoding="utf-8")

        return service

    @classmethod
    def load(cls, artifact_path: Path | str) -> "RestaurantRankerService":
        loaded = joblib.load(artifact_path)
        required_attrs = ["model", "restaurant_catalog", "text_artifacts", "feature_columns", "evaluation_summary"]
        if not all(hasattr(loaded, attr) for attr in required_attrs):
            raise TypeError(f"Artifact at {artifact_path} is not compatible with {cls.__name__}")
        loaded.artifact_path = Path(artifact_path)
        return loaded

    @classmethod
    def load_existing(cls, artifact_path: Path | str = DEFAULT_ARTIFACT_PATH) -> "RestaurantRankerService":
        artifact_path = Path(artifact_path)
        if not artifact_path.exists():
            raise FileNotFoundError(
                f"Model artifact was not found at '{artifact_path}'. Train the ranker first before starting the API."
            )
        return cls.load(artifact_path)

    @classmethod
    def load_or_train(
        cls,
        artifact_path: Path | str = DEFAULT_ARTIFACT_PATH,
        labels_path: Path | str = DEFAULT_LABELS_PATH,
        restaurants_path: Path | str = DEFAULT_RESTAURANTS_PATH,
        metrics_path: Path | str = DEFAULT_METRICS_PATH,
        force_retrain: bool = False,
    ) -> "RestaurantRankerService":
        artifact_path = Path(artifact_path)
        labels_path = Path(labels_path)
        restaurants_path = Path(restaurants_path)
        artifact_is_fresh = (
            artifact_path.exists()
            and artifact_path.stat().st_mtime >= max(labels_path.stat().st_mtime, restaurants_path.stat().st_mtime)
        )
        if artifact_is_fresh and not force_retrain:
            return cls.load(artifact_path)

        return cls.train_from_paths(
            labels_path=labels_path,
            restaurants_path=restaurants_path,
            artifact_path=artifact_path,
            metrics_path=metrics_path,
        )

    def save(self, artifact_path: Path | str) -> None:
        artifact_path = Path(artifact_path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, artifact_path)
        self.artifact_path = artifact_path

    def rank(
        self,
        query: str,
        candidate_restaurant_ids: Optional[list[str | int]] = None,
        candidate_restaurants: Optional[pd.DataFrame] = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        if not str(query).strip():
            raise ValueError("query must not be empty")

        if candidate_restaurants is not None:
            candidate_catalog = prepare_restaurant_catalog(candidate_restaurants)
        else:
            candidate_catalog = self.restaurant_catalog

        if candidate_restaurant_ids:
            seen: set[str] = set()
            candidate_ids: list[str] = []
            for restaurant_id in candidate_restaurant_ids:
                normalized = normalize_restaurant_id(restaurant_id)
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    candidate_ids.append(normalized)
            candidate_catalog = candidate_catalog[candidate_catalog["restaurant_id"].isin(candidate_ids)].copy()

        if candidate_catalog.empty:
            raise ValueError("No candidate restaurants were available for ranking")

        candidate_pairs = pd.DataFrame(
            {"query": [query] * len(candidate_catalog), "restaurant_id": candidate_catalog["restaurant_id"].tolist()}
        )
        feature_df = build_feature_frame(candidate_pairs, candidate_catalog, self.text_artifacts)
        scores = self.model.predict(feature_df[self.feature_columns])

        ranked = feature_df.copy()
        ranked["rank_score"] = scores
        ranked = ranked.sort_values(["rank_score", "restaurant_id"], ascending=[False, True]).head(top_k).reset_index(drop=True)

        results: list[dict[str, Any]] = []
        for _, row in ranked.iterrows():
            restaurant_payload = {
                column: row[column]
                for column in RESTAURANT_RESPONSE_COLUMNS
                if column in row.index and not (isinstance(row[column], float) and pd.isna(row[column]))
            }
            results.append(
                {
                    "restaurant_id": row["restaurant_id"],
                    "rank_score": float(row["rank_score"]),
                    "restaurant": restaurant_payload,
                }
            )
        return results

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "artifact_path": str(self.artifact_path) if self.artifact_path else None,
            "catalog_size": int(len(self.restaurant_catalog)),
            "feature_count": len(self.feature_columns),
            "evaluation_summary": self.evaluation_summary,
        }
