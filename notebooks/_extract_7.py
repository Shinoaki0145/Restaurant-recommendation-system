
train_df, val_df, test_df = split_by_query(df)
print(f"Train/Val/Test queries: {train_df['query'].nunique()} / {val_df['query'].nunique()} / {test_df['query'].nunique()}")
print(f"Train/Val/Test rows: {len(train_df)} / {len(val_df)} / {len(test_df)}")

ranker, train_seconds = train_ranker(
    train_df,
    val_df,
    feature_columns=FEATURE_COLUMNS,
    n_estimators=250,
)
print(f"Training time: {train_seconds:.2f}s")
