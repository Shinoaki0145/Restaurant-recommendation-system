import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score
from sklearn.model_selection import train_test_split


LABEL_COL = "relevance_score"
QUERY_COL = "query_id"
CATEGORICAL_COLS = ["Quan_Huyen", "Phan_loai_mon"]
NUMERICAL_COLS = ["Gia_tien", "Thoi_gian_nau", "Pinecone_score", "BM25_score"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a LightGBM LGBMRanker model with rank_xendcg objective"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to CSV containing ranking data",
    )
    parser.add_argument(
        "--valid-size",
        type=float,
        default=0.2,
        help="Validation ratio based on unique query_id split",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--model-out",
        type=str,
        default="lgbm_ranker.txt",
        help="Output path for trained model",
    )
    return parser.parse_args()


def validate_columns(df: pd.DataFrame) -> None:
    required = [LABEL_COL, QUERY_COL, *CATEGORICAL_COLS, *NUMERICAL_COLS]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    validate_columns(df)

    # Ensure numeric columns are numeric for stable training behavior.
    for col in NUMERICAL_COLS + [LABEL_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("category")

    df = df.dropna(subset=[LABEL_COL, QUERY_COL, *NUMERICAL_COLS]).copy()

    labels = df[LABEL_COL].to_numpy()
    if np.any(labels < 0) or np.any(labels > 4):
        raise ValueError("relevance_score must be in [0, 4]")

    return df


def build_group_array(data: pd.DataFrame) -> np.ndarray:
    return data.groupby(QUERY_COL, sort=False).size().to_numpy(dtype=np.int32)


def split_by_query(df: pd.DataFrame, valid_size: float, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_queries = df[QUERY_COL].drop_duplicates().to_numpy()
    train_queries, valid_queries = train_test_split(
        unique_queries,
        test_size=valid_size,
        random_state=random_state,
        shuffle=True,
    )

    train_df = df[df[QUERY_COL].isin(train_queries)].copy()
    valid_df = df[df[QUERY_COL].isin(valid_queries)].copy()

    if train_df.empty or valid_df.empty:
        raise ValueError("Train/valid split is empty. Check valid_size and number of queries.")

    return train_df, valid_df


def compute_querywise_ndcg_at_k(df_valid: pd.DataFrame, preds: np.ndarray, k: int = 5) -> float:
    temp = df_valid[[QUERY_COL, LABEL_COL]].copy()
    temp["pred"] = preds

    scores = []
    for _, group in temp.groupby(QUERY_COL, sort=False):
        y_true = group[LABEL_COL].to_numpy().reshape(1, -1)
        y_pred = group["pred"].to_numpy().reshape(1, -1)
        scores.append(ndcg_score(y_true, y_pred, k=k))

    return float(np.mean(scores))


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.data_path)
    df = prepare_dataframe(df)

    train_df, valid_df = split_by_query(df, valid_size=args.valid_size, random_state=args.random_state)

    features = CATEGORICAL_COLS + NUMERICAL_COLS

    x_train = train_df[features]
    y_train = train_df[LABEL_COL].astype(int)
    g_train = build_group_array(train_df)

    x_valid = valid_df[features]
    y_valid = valid_df[LABEL_COL].astype(int)
    g_valid = build_group_array(valid_df)

    ranker = lgb.LGBMRanker(
        objective="rank_xendcg",
        n_estimators=3000,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=args.random_state,
    )

    ranker.fit(
        x_train,
        y_train,
        group=g_train,
        eval_set=[(x_valid, y_valid)],
        eval_group=[g_valid],
        eval_metric="ndcg",
        eval_at=[5],
        categorical_feature=CATEGORICAL_COLS,
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, first_metric_only=True, verbose=True),
            lgb.log_evaluation(period=50),
        ],
    )

    best_iter = ranker.best_iteration_
    preds_valid = ranker.predict(x_valid, num_iteration=best_iter)
    ndcg_at_5 = compute_querywise_ndcg_at_k(valid_df, preds_valid, k=5)

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    ranker.booster_.save_model(str(model_out))

    print("===== Training Summary =====")
    print(f"Train rows: {len(train_df):,} | Valid rows: {len(valid_df):,}")
    print(f"Train queries: {len(g_train):,} | Valid queries: {len(g_valid):,}")
    print(f"Best iteration: {best_iter}")

    lgb_val_ndcg5 = ranker.best_score_.get("valid_0", {}).get("ndcg@5", None)
    if lgb_val_ndcg5 is not None:
        print(f"LightGBM valid ndcg@5: {lgb_val_ndcg5:.6f}")

    print(f"Manual querywise mean ndcg@5: {ndcg_at_5:.6f}")
    print(f"Saved model to: {model_out.resolve()}")


if __name__ == "__main__":
    main()