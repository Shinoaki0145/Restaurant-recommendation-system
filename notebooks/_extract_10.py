sample_queries = eval_df["query"].drop_duplicates().head(4).tolist()

diagnostic_cols = [
    "query", "restaurant_name_meta", "district_meta", "cuisines_meta", "category_raw", "target_audience_raw",
    "rest_day_list", "active_days", "label", "pred_score", "district_exact_match", "cuisine_target_match",
    "category_target_match", "price_range_overlap", "delivery_match", "booking_match", "schedule_match_mean",
    "quality_pref_match", "service_pref_match", "space_pref_match", "position_pref_match", "popularity_blend",
]

for sample_query in sample_queries:
    print("\nQUERY:", sample_query)
    display(
        eval_df.loc[eval_df["query"] == sample_query, diagnostic_cols]
        .sort_values("pred_score", ascending=False)
        .head(5)
    )
