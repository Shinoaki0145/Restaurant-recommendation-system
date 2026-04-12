
importance_df = feature_importance_frame(ranker, FEATURE_COLUMNS)
display(importance_df.head(20))

plt.figure(figsize=(10, 7))
top_importance = importance_df.head(20).iloc[::-1]
plt.barh(top_importance['feature'], top_importance['importance'])
plt.title('Top 20 Feature Importances')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()
