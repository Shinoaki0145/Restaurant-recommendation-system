df = build_restaurant_features(raw_df)
print(f"Feature frame shape: {df.shape}")
print(f"Number of model features: {len(FEATURE_COLUMNS)}")
print(FEATURE_COLUMNS)

query_preview = df[query_profile_columns()].drop_duplicates("query").head(12).copy()
display(query_preview)

schema_preview_cols = [
    "restaurant_name_meta", "district_meta", "target_audience_raw", "target_audience_list",
    "category_raw", "category_list", "rest_days_raw", "rest_day_list", "active_days",
]
display(df[schema_preview_cols].head(12))

preview_cols = [
    "query", "restaurant_name_meta", "label", "district_meta", "tfidf_cosine", "char_tfidf_cosine",
    "district_exact_match", "cuisine_target_match", "category_target_match", "delivery_match",
    "booking_match", "schedule_match_mean", "quality_score_mean",
]
display(df[preview_cols].head(12))
