import json
import random
import re
import time
from typing import Any, Dict, List, Tuple

import pandas as pd
from openai import OpenAI
from pinecone import Pinecone

QUERY_FILE = "restaurant_queries.txt"
CSV_PATH = "foody_combined_data_final.csv"
OUTPUT_CSV = "restaurant_dataset.csv"
DEBUG_CSV = "restaurant_dataset_debug.csv"
CHECKPOINT_JSONL = "restaurant_dataset_checkpoint.jsonl"

PINECONE_API_KEY = "xxx"
INDEX_NAME = "restaurant"
OPENAI_API_KEY = "xxx"
LLM_MODEL = "gpt-4.1-mini"

TOP_K = 15
RANDOM_K = 5
SEED = 42
PINECONE_OVERFETCH_FACTOR = 4
PINECONE_MAX_RETRIES = 4
LLM_MAX_RETRIES = 4
SAVE_EVERY = 10

from embed_model import embed_text

JSON_FALLBACK_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


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


FULL_DAYS = ["Chủ nhật", "Thứ hai", "Thứ ba", "Thứ tư", "Thứ năm", "Thứ sáu", "Thứ bảy"]
VIEW_BINS = [-float("inf"), 28, 58, 106, 235, 893, float("inf")]
VIEW_LABELS = [
    "rất ít lượt xem",
    "Ít lượt xem",
    "Lượt xem trung bình",
    "Nhiều lượt xem",
    "Rất nhiều lượt xem",
    "Thịnh hành",
]

TEXT_FIELDS = [
    "Name",
    "Address",
    "District",
    "Area",
    "PriceMin",
    "PriceMax",
    "MetaKeywords",
    "Cuisines",
    "LstTargetAudience",
    "LstCategory",
    "View Category",
    "Chất lượng",
    "Phục vụ",
    "Không gian",
    "DominantReview",
    "TotalFavourite",
    "TotalCheckins",
    "Giao tận nơi",
    "Đặt bàn",
    "Ngày bán",
    "Chủ nhật",
    "Thứ hai",
    "Thứ ba",
    "Thứ tư",
    "Thứ năm",
    "Thứ sáu",
    "Thứ bảy",
]


def get_selling_days(off_str: Any) -> str:
    if pd.isna(off_str):
        return ", ".join(FULL_DAYS)

    text = str(off_str).strip()
    if not text:
        return ", ".join(FULL_DAYS)

    off_list = [day.strip() for day in text.split("||") if day.strip()]
    selling_days = [day for day in FULL_DAYS if day not in off_list]
    return ", ".join(selling_days)


def preprocess_restaurant_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Ngày nghỉ" in df.columns:
        df["Ngày bán"] = df["Ngày nghỉ"].apply(get_selling_days)
    else:
        df["Ngày bán"] = ", ".join(FULL_DAYS)

    if "TotalView" in df.columns:
        total_view_numeric = pd.to_numeric(df["TotalView"], errors="coerce")
        df["View Category"] = pd.cut(
            total_view_numeric,
            bins=VIEW_BINS,
            labels=VIEW_LABELS,
            include_lowest=True,
        )
        df["View Category"] = df["View Category"].astype(str)
    else:
        df["View Category"] = ""

    for col in ("LstTargetAudience", "LstCategory"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(" || ", ", ", regex=False)
            df[col] = df[col].str.replace("||", ", ", regex=False)

    if all(col in df.columns for col in ("Excellent", "Good", "Average", "Bad")):
        df["DominantReview"] = df.apply(get_dominant_review_label, axis=1)
    else:
        df["DominantReview"] = ""

    return df


def load_restaurant_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = preprocess_restaurant_df(df)
    df = df.fillna("")
    for col in df.columns:
        df[col] = df[col].astype(str)
    for day in FULL_DAYS:
        if day in df.columns:
            df[day] = df[day].apply(format_opening_hours)
    if "RestaurantID" not in df.columns:
        raise ValueError("CSV phải có cột RestaurantID")
    df["RestaurantID"] = df["RestaurantID"].apply(normalize_restaurant_id)
    df = df[df["RestaurantID"] != ""].copy()
    df = df.drop_duplicates(subset=["RestaurantID"], keep="first").reset_index(drop=True)
    return df


def read_queries(query_file: str) -> List[str]:
    with open(query_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def is_truthy_flag(value: Any) -> bool:
    text = str(value).strip().lower()
    return text in {"1", "1.0", "true", "yes"}


def fallback_text(value: Any, default: str = "Không xác định") -> str:
    text = str(value).strip()
    return text if text else default


def to_float(value: Any) -> float:
    try:
        return float(str(value).strip())
    except Exception:
        return 0.0


def get_dominant_review_label(row: pd.Series) -> str:
    review_counts = {
        "Bad": to_float(row.get("Bad", 0)),
        "Average": to_float(row.get("Average", 0)),
        "Good": to_float(row.get("Good", 0)),
        "Excellent": to_float(row.get("Excellent", 0)),
    }
    if all(count == 0 for count in review_counts.values()):
        return "Chưa có review"
    dominant_label, dominant_count = max(review_counts.items(), key=lambda item: (item[1], 0))
    return f"{dominant_label} ({int(dominant_count) if dominant_count.is_integer() else dominant_count})"


def format_opening_hours(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return "Không rõ"

    match = re.match(r"^\('([^']*)',\s*'([^']*)'\)$", text)
    if match:
        open_time, close_time = match.groups()
        return f"{open_time} - {close_time}"

    return text


def row_to_text(row: pd.Series) -> str:
    name = row.get("Name", "")
    address = row.get("Address", "")
    district = row.get("District", "")
    area = row.get("Area", "")

    price_min = row.get("PriceMin", "")
    price_max = row.get("PriceMax", "")

    meta_keywords = row.get("MetaKeywords", "")
    cuisines = row.get("Cuisines", "")
    target_audience = row.get("LstTargetAudience", "")
    category = row.get("LstCategory", "")
    view_category = row.get("View Category", "")
    selling_days = row.get("Ngày bán", "")
    quality = row.get("Chất lượng", "")
    service = row.get("Phục vụ", "")
    space = row.get("Không gian", "")
    dominant_review = row.get("DominantReview", "")
    total_favourite = row.get("TotalFavourite", "")
    total_checkins = row.get("TotalCheckins", "")

    home_delivery = "Có giao hàng tận nơi" if is_truthy_flag(row.get("Giao tận nơi", "")) else "Không có giao hàng tận nơi"
    order_table = "Có đặt bàn" if is_truthy_flag(row.get("Đặt bàn", "")) else "Không có đặt bàn"
    opening_hours = [f"- {day}: {format_opening_hours(row.get(day, ''))}" for day in FULL_DAYS]

    parts = [
        f"Nhà hàng/Quán {name}",
        f"Địa chỉ: {address}",
        f"Nằm ở {district} trong khu vực {area}",
        "",
        f"Giá từ {price_min} đến {price_max} VND",
        "",
        f"Từ khóa: {meta_keywords}",
        f"Ẩm thực: {cuisines}",
        f"Đối tượng: {target_audience}",
        f"Danh mục: {category}",
        "",
        "Dịch vụ:",
        f"- Giao tận nơi: {home_delivery}",
        f"- Đặt bàn: {order_table}",
        "",
        f"Lượt xem: {view_category}",
        f"Ngày bán: {selling_days}",
        "",
        "Đánh giá thêm:",
        f"- Chất lượng: {quality}",
        f"- Phục vụ: {service}",
        f"- Không gian: {space}",
        f"- Mức review nổi trội: {dominant_review}",
        f"- TotalFavourite: {total_favourite}",
        f"- TotalCheckins: {total_checkins}",
        "",
        "Giờ mở cửa từng ngày:",
        *opening_hours,
    ]

    return "\n".join(parts)



def compact_restaurant_for_llm(row: pd.Series) -> Dict[str, Any]:
    data = {"restaurant_id": str(row.get("RestaurantID", ""))}
    for col in TEXT_FIELDS:
        if col in row.index:
            data[col] = row.get(col, "")
    return data



def embed_query(query: str) -> List[float]:
    emb = embed_text([query]).cpu().numpy()[0]
    return emb.tolist()



def query_pinecone_with_retry(index: Any, query_text: str, top_k: int) -> List[str]:
    vector = embed_query(query_text)
    last_error = None
    for attempt in range(1, PINECONE_MAX_RETRIES + 1):
        try:
            response = index.query(
                vector=vector,
                top_k=top_k * PINECONE_OVERFETCH_FACTOR,
                include_metadata=False,
            )
            matches = response.get("matches", []) if isinstance(response, dict) else response.matches
            ids: List[str] = []
            seen = set()
            for match in matches:
                rid = str(match["id"] if isinstance(match, dict) else match.id)
                if rid not in seen:
                    seen.add(rid)
                    ids.append(rid)
                if len(ids) >= top_k:
                    break
            return ids
        except Exception as exc:
            last_error = exc
            sleep_s = min(2 ** attempt, 10)
            print(f"[WARN] Pinecone lỗi lần {attempt}/{PINECONE_MAX_RETRIES}: {exc}. Sleep {sleep_s}s")
            time.sleep(sleep_s)
    raise RuntimeError(f"Pinecone query thất bại cho query: {query_text}\n{last_error}")



def sample_random_restaurant_ids(all_ids: List[str], excluded_ids: List[str], k: int, rng: random.Random) -> List[str]:
    excluded_set = set(excluded_ids)
    pool = [rid for rid in all_ids if rid not in excluded_set]
    if len(pool) < k:
        raise ValueError(f"Không đủ quán để random {k} quán không trùng. Pool còn {len(pool)} quán.")
    return rng.sample(pool, k)



def build_labeling_messages(query_text: str, restaurants: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    system_prompt = """
Bạn là chuyên gia gán nhãn relevance cho bài toán truy hồi quán ăn.

Mục tiêu:
Với 1 query và danh sách nhà hàng, hãy gán nhãn 0-4 cho TỪNG nhà hàng.

    Các thuộc tính quan trọng trong dữ liệu:
    - Name: tên quán.
    - Address, District, Area: vị trí quán.
    - PriceMin, PriceMax: khoảng giá.
    - MetaKeywords, Cuisines: từ khóa ngắn, loại ẩm thực, món nổi bật.
    - LstTargetAudience: nhóm khách phù hợp.
    - LstCategory: loại hình quán.
    - Giao tận nơi, Đặt bàn: cờ nhị phân cho khả năng giao hàng tận nơi / đặt bàn.
    - View Category: mức độ phổ biến theo lượt xem đã được chia nhóm.
    - Chất lượng, Phục vụ, Không gian: điểm đánh giá theo từng khía cạnh.
    - DominantReview: mức review trội nhất từ Excellent/Good/Average/Bad; nếu hòa thì đã chọn mức tệ hơn.
    - TotalFavourite, TotalCheckins: tín hiệu bổ sung về mức độ quan tâm và ghé quán.
    - Ngày bán: các ngày quán hoạt động.
    - Chủ nhật đến Thứ bảy: thời gian mở cửa từng ngày.

Ý nghĩa nhãn:
- 4 = Rất phù hợp: khớp gần như đầy đủ các điều kiện quan trọng trong query; là lựa chọn rất tốt.
- 3 = Khá phù hợp: khớp nhiều điều kiện chính nhưng còn lệch nhẹ hoặc thiếu một vài điều kiện phụ.
- 2 = Phù hợp trung bình: có liên quan rõ ràng nhưng chỉ khớp một phần; người dùng có thể cân nhắc.
- 1 = Ít phù hợp: chỉ khớp yếu hoặc chỉ dính 1 chi tiết nhỏ.
- 0 = Không phù hợp: hầu như không liên quan hoặc mâu thuẫn với nhu cầu.

Ưu tiên khi chấm:
1. Cuisines / món ăn / keyword chính.
2. District / Area / vị trí.
3. PriceMin-PriceMax nếu query nói về giá.
    4. LstTargetAudience nếu query nhắc nhóm đối tượng.
    5. LstCategory / loại hình quán.
    6. Chất lượng, Phục vụ, Không gian, DominantReview nếu query nhắc chất lượng/review.
    7. Ngày bán và giờ mở cửa từng ngày nếu query nhắc thời gian hoạt động.
    8. View Category, TotalFavourite, TotalCheckins nếu query nhắc độ nổi tiếng, đông khách, được yêu thích.
    9. Giao tận nơi, đặt bàn nếu query có nhắc.

Luật chấm:
- Không bịa thêm dữ kiện.
- Nếu thiếu dữ liệu cho một điều kiện, không được tự đoán là có.
    - Sai cuisine hoặc sai district là lỗi nặng, thường không được label cao.
    - Nếu query nhắc khoảng giá, hãy so theo PriceMin-PriceMax; không tự suy diễn quán "rẻ" hay "đắt" nếu thiếu ngưỡng rõ ràng.
    - Nếu query nhắc chất lượng hoặc review, ưu tiên xét Chất lượng, Phục vụ, Không gian và DominantReview.
    - Nếu query nhắc giao hàng, ưu tiên xét Giao tận nơi.
    - Nếu query nhắc đặt bàn, ưu tiên xét Đặt bàn.
    - Nếu query nhắc độ nổi tiếng, có thể dùng View Category, TotalFavourite, TotalCheckins như tín hiệu phụ, không thay thế món ăn/vị trí.
    - Nếu query nhắc ngày hoặc giờ mở cửa, chỉ dùng thông tin trong Ngày bán và các cột thời gian theo từng ngày.
- Chỉ gán 4 khi thực sự rất hợp.
- Không cố cân bằng label; chấm trung thực theo từng quán.
- Trả về đúng thứ tự đầu vào.
- reason phải ngắn, rõ, bằng tiếng Việt.
- Mỗi reason nên nêu 1-2 yếu tố chính khiến quán phù hợp hoặc không phù hợp.
- Chỉ trả về JSON đúng schema.
""".strip()

    user_payload = {
        "query": query_text,
        "restaurants": restaurants,
        "output_requirement": "Trả về restaurant_id, label, reason cho từng nhà hàng theo đúng thứ tự đầu vào.",
    }

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]



def call_llm_for_labels(client: OpenAI, model: str, query_text: str, restaurants: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    messages = build_labeling_messages(query_text, restaurants)
    schema = {
        "type": "object",
        "properties": {
            "labels": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "restaurant_id": {"type": "string"},
                        "label": {"type": "integer", "enum": [0, 1, 2, 3, 4]},
                        "reason": {"type": "string"},
                    },
                    "required": ["restaurant_id", "label", "reason"],
                    "additionalProperties": False,
                },
                "minItems": len(restaurants),
                "maxItems": len(restaurants),
            }
        },
        "required": ["labels"],
        "additionalProperties": False,
    }

    response = client.responses.create(
        model=model,
        input=messages,
        text={
            "format": {
                "type": "json_schema",
                "name": "restaurant_labels",
                "schema": schema,
                "strict": True,
            }
        },
    )
    data = json.loads(response.output_text)
    labels = data["labels"]
    validate_llm_output(restaurants, labels)
    return labels



def safe_call_llm_for_labels(client: OpenAI, model: str, query_text: str, restaurants: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    last_error = None
    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            return call_llm_for_labels(client, model, query_text, restaurants)
        except Exception as exc:
            last_error = exc
            print(f"[WARN] LLM structured output lỗi lần {attempt}/{LLM_MAX_RETRIES}: {exc}")
            time.sleep(min(2 ** attempt, 10))

    # fallback mềm
    messages = build_labeling_messages(query_text, restaurants)
    messages.append({
        "role": "user",
        "content": 'Chỉ trả về JSON thuần: {"labels":[{"restaurant_id":"...","label":0,"reason":"..."}]}'
    })
    response = client.responses.create(model=model, input=messages)
    text = response.output_text.strip()
    match = JSON_FALLBACK_PATTERN.search(text)
    if not match:
        raise RuntimeError(f"Không parse được JSON từ output. Lỗi trước đó: {last_error}\nOutput: {text[:1000]}")
    data = json.loads(match.group(0))
    labels = data["labels"]
    validate_llm_output(restaurants, labels)
    return labels

        
def validate_llm_output(restaurants, labels):
    input_ids = [str(item["restaurant_id"]) for item in restaurants]
    output_ids = [str(item["restaurant_id"]) for item in labels]

    if set(input_ids) != set(output_ids):
        raise ValueError(
            f"LLM trả về thiếu/thừa restaurant_id.\nInput: {input_ids}\nOutput: {output_ids}"
        )

    if len(output_ids) != len(set(output_ids)):
        raise ValueError(f"LLM trả về restaurant_id bị trùng: {output_ids}")

    for item in labels:
        if int(item["label"]) not in {0, 1, 2, 3, 4}:
            raise ValueError(f"Label không hợp lệ: {item}")



def build_rows(query_text: str, selected_rows: List[pd.Series], labels: List[Dict[str, Any]], source_map: Dict[str, str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    label_lookup = {str(x["restaurant_id"]): x for x in labels}
    final_rows: List[Dict[str, Any]] = []
    debug_rows: List[Dict[str, Any]] = []
    for row in selected_rows:
        rid = str(row["RestaurantID"])
        llm_item = label_lookup[rid]
        final_rows.append({
            "query": query_text,
            "restaurant": row_to_text(row),
            "label": int(llm_item["label"]),
        })
        debug_rows.append({
            "query": query_text,
            "restaurant_id": rid,
            "restaurant_name": row.get("Name", ""),
            "source": source_map.get(rid, ""),
            "label": int(llm_item["label"]),
            "reason": llm_item.get("reason", ""),
            "district": row.get("District", ""),
            "cuisines": row.get("Cuisines", ""),
        })
    return final_rows, debug_rows



def save_outputs(final_rows: List[Dict[str, Any]], debug_rows: List[Dict[str, Any]]) -> None:
    pd.DataFrame(final_rows).to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    pd.DataFrame(debug_rows).to_csv(DEBUG_CSV, index=False, encoding="utf-8-sig")



def append_checkpoint(record: Dict[str, Any]) -> None:
    with open(CHECKPOINT_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")



def main() -> None:
    if TOP_K <= 0 or RANDOM_K <= 0:
        raise ValueError("TOP_K và RANDOM_K phải > 0")

    rng = random.Random(SEED)

    print("[INFO] Đọc CSV...")
    df = load_restaurant_df(CSV_PATH)
    id_to_row: Dict[str, pd.Series] = {str(row["RestaurantID"]): row for _, row in df.iterrows()}
    all_ids = list(id_to_row.keys())

    print("[INFO] Đọc queries...")
    queries = read_queries(QUERY_FILE)
    if not queries:
        raise ValueError("queries.txt không có query nào")

    print("[INFO] Kết nối Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)

    print("[INFO] Kết nối OpenAI...")
    client = OpenAI(api_key=OPENAI_API_KEY)

    final_rows: List[Dict[str, Any]] = []
    debug_rows: List[Dict[str, Any]] = []

    for i, query_text in enumerate(queries, start=1):
        print(f"\n[INFO] Query {i}/{len(queries)}: {query_text}")

        top_ids = query_pinecone_with_retry(index=index, query_text=query_text, top_k=TOP_K)
        top_ids = [rid for rid in top_ids if rid in id_to_row]
        if len(top_ids) < TOP_K:
            raise ValueError(f"Query '{query_text}' chỉ lấy được {len(top_ids)} quán hợp lệ từ Pinecone, cần {TOP_K}.")

        random_ids = sample_random_restaurant_ids(all_ids=all_ids, excluded_ids=top_ids, k=RANDOM_K, rng=rng)
        selected_ids = top_ids + random_ids
        if len(selected_ids) != len(set(selected_ids)):
            raise ValueError(f"Bị trùng quán trong 20 quán của query: {query_text}")

        selected_rows = [id_to_row[rid] for rid in selected_ids]
        source_map = {rid: "pinecone" for rid in top_ids}
        source_map.update({rid: "random" for rid in random_ids})

        restaurants_for_llm = [compact_restaurant_for_llm(row) for row in selected_rows]
        labels = safe_call_llm_for_labels(
            client=client,
            model=LLM_MODEL,
            query_text=query_text,
            restaurants=restaurants_for_llm,
        )

        batch_final_rows, batch_debug_rows = build_rows(query_text, selected_rows, labels, source_map)
        final_rows.extend(batch_final_rows)
        debug_rows.extend(batch_debug_rows)

        append_checkpoint({
            "query": query_text,
            "selected_ids": selected_ids,
            "labels": labels,
        })

        if i % SAVE_EVERY == 0 or i == len(queries):
            save_outputs(final_rows, debug_rows)
            print(f"[INFO] Đã lưu tạm sau {i} query")

    save_outputs(final_rows, debug_rows)
    print(f"\n[DONE] Saved main dataset to: {OUTPUT_CSV}")
    print(f"[DONE] Saved debug dataset to: {DEBUG_CSV}")
    print(f"[DONE] Saved checkpoint to: {CHECKPOINT_JSONL}")


if __name__ == "__main__":
    main()
