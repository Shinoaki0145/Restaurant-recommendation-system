
eval_df, metrics = evaluate_ranker(ranker, test_df, feature_columns=FEATURE_COLUMNS)
metrics_df = pd.DataFrame([metrics]).T.reset_index()
metrics_df.columns = ["metric", "value"]
display(metrics_df)
