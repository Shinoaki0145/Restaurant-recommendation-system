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
INDEX_NAME = "restaurant-recommendation"
OPENAI_API_KEY = "xxx"
LLM_MODEL = "gpt-4.1-mini"

TOP_K = 15
RANDOM_K = 5
SEED = 42
PINECONE_OVERFETCH_FACTOR = 4
PINECONE_MAX_RETRIES = 4
LLM_MAX_RETRIES = 4
SAVE_EVERY = 10

# Reuse embedding function giống notebook của bạn
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


# Các cột ưu tiên dùng để tóm tắt quán cho LLM
TEXT_FIELDS = [
    "Name", "Address", "District", "Area", "PriceMin", "PriceMax",
    "MetaKeywords", "Cuisines", "LstTargetAudience", "LstCategory", "AccessGuide",
    "Vị trí", "Giá cả", "Chất lượng", "Phục vụ", "Không gian",
    "Excellent", "Good", "Average", "Bad",
    "HasBooking", "HasDelivery", "HasPromotion",
    "TotalView", "TotalFavourite", "TotalCheckins",
    "Giao tận nơi", "Đặt bàn"
]


def load_restaurant_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.fillna("")
    for col in df.columns:
        df[col] = df[col].astype(str)
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


def row_to_text(row: pd.Series) -> str:
    name = row.get("Name", "")
    address = row.get("Address", "")
    district = row.get("District", "")
    area = row.get("Area", "")
    restaurant_id = row.get("RestaurantID", "")

    price_min = row.get("PriceMin", "")
    price_max = row.get("PriceMax", "")

    meta_keywords = row.get("MetaKeywords", "")
    cuisines = row.get("Cuisines", "")
    target_audience = row.get("LstTargetAudience", "")
    category = row.get("LstCategory", "")
    access_guide = fallback_text(row.get("AccessGuide", ""))

    location_rate = row.get("Vị trí", "")
    price_rate = row.get("Giá cả", "")
    quality_rate = row.get("Chất lượng", "")
    service_rate = row.get("Phục vụ", "")
    space_rate = row.get("Không gian", "")

    excellent = row.get("Excellent", "")
    good = row.get("Good", "")
    average = row.get("Average", "")
    bad = row.get("Bad", "")

    has_booking = fallback_text(row.get("HasBooking", ""))
    has_delivery = fallback_text(row.get("HasDelivery", ""))
    has_promotion = fallback_text(row.get("HasPromotion", ""))
    total_view = row.get("TotalView", "")
    total_favorite = row.get("TotalFavourite", "")
    total_checkin = row.get("TotalCheckins", "")

    home_delivery = "Có giao hàng tận nơi" if is_truthy_flag(row.get("Giao tận nơi", "")) else "Không có giao hàng tận nơi"
    order_table = "Có đặt bàn" if is_truthy_flag(row.get("Đặt bàn", "")) else "Không có đặt bàn"

    parts = [
        f"Nhà hàng: {name}",
        f"Địa chỉ: {address}, {district}, {area}",
        f"ID nhà hàng: {restaurant_id}",
        "",
        f"Giá: từ {price_min} đến {price_max} VND",
        "",
        f"Mô tả: {meta_keywords}",
        f"Ẩm thực: {cuisines}",
        f"Đối tượng mục tiêu: {target_audience}",
        f"Danh mục: {category}",
        f"Hướng dẫn đi lại: {access_guide}",
        "",
        "Đánh giá:",
        f"- Vị trí: {location_rate}",
        f"- Giá cả: {price_rate}",
        f"- Chất lượng: {quality_rate}",
        f"- Phục vụ: {service_rate}",
        f"- Không gian: {space_rate}",
        "",
        "Số lượng đánh giá:",
        f"- Excellent: {excellent}",
        f"- Good: {good}",
        f"- Average: {average}",
        f"- Bad: {bad}",
        "",
        "Dịch vụ:",
        f"- Giao tận nơi: {home_delivery}",
        f"- Đặt bàn: {order_table}",
        "",
        f"Tổng lượt xem: {total_view}",
        f"Tổng lượt yêu thích: {total_favorite}",
        f"Tổng lượt check-in: {total_checkin}",
        "",
        f"Có đặt bàn: {has_booking}",
        f"Có giao hàng: {has_delivery}",
        f"Có khuyến mãi: {has_promotion}",
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
- MetaKeywords, Cuisines: mô tả ngắn, loại ẩm thực, món nổi bật.
- LstTargetAudience: nhóm khách phù hợp.
- LstCategory: loại hình quán.
- AccessGuide: hướng dẫn đi lại.
- Vị trí, Giá cả, Chất lượng, Phục vụ, Không gian: điểm đánh giá từng mặt.
- Excellent, Good, Average, Bad: số lượng đánh giá theo mức độ.
- HasBooking, HasDelivery, HasPromotion: cờ hoặc trạng thái dịch vụ; có thể thiếu dữ liệu.
- Giao tận nơi, Đặt bàn: cờ nhị phân cho khả năng giao hàng tận nơi / đặt bàn.
- TotalView, TotalFavourite, TotalCheckins: tín hiệu phổ biến; chỉ dùng khi query nhắc độ nổi tiếng, đông khách, nhiều người biết đến.

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
6. Chất lượng, Phục vụ, Không gian, mức độ review nếu query có nhắc.
7. Booking, delivery, promotion nếu query có nhắc.
8. TotalView, TotalFavourite, TotalCheckins chỉ khi query nhắc độ nổi tiếng.

Luật chấm:
- Không bịa thêm dữ kiện.
- Nếu thiếu dữ liệu cho một điều kiện, không được tự đoán là có.
- Sai cuisine hoặc sai district là lỗi nặng, thường không được label cao.
- Nếu query nhắc khoảng giá, hãy so theo PriceMin-PriceMax; không tự suy diễn quán "rẻ" hay "đắt" nếu thiếu ngưỡng rõ ràng.
- Nếu query nhắc giao hàng, ưu tiên xét Giao tận nơi và HasDelivery.
- Nếu query nhắc đặt bàn, ưu tiên xét Đặt bàn và HasBooking.
- Nếu query nhắc khuyến mãi, chỉ dùng HasPromotion khi dữ liệu thật sự có.
- Các chỉ số TotalView, TotalFavourite, TotalCheckins không đủ để thay thế cho chất lượng món hoặc độ phù hợp chính.
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
            "price_min": row.get("PriceMin", ""),
            "price_max": row.get("PriceMax", ""),
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
