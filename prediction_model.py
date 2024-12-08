# prediction_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


def build_prediction_model(input_file):
    # Load data
    df = pd.read_csv(input_file)

    # Feature extraction
    tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    X = tfidf.fit_transform(df["cleaned_text"])

    # Create target variable (1 if score is above median, 0 otherwise)
    y = (df["score"] > df["score"].median()).astype(int)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate model
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("\nCross-validation scores:")
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"Mean accuracy: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")

    print("\nROC AUC Score:")
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f"ROC AUC: {roc_auc:.2f}")

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig("confusion_matrix.png")
    plt.close()

    # Feature importance
    feature_importance = (
        pd.DataFrame(
            {
                "feature": tfidf.get_feature_names_out(),
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .head(20)
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(x="importance", y="feature", data=feature_importance)
    plt.title("Top 20 Most Important Features")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.close()

    # Sentiment distribution
    df["sentiment"] = pd.cut(
        df["sentiment_score"],
        bins=[-1, -0.1, 0.1, 1],
        labels=["Negative", "Neutral", "Positive"],
    )

    plt.figure(figsize=(8, 6))
    df["sentiment"].value_counts().plot(kind="bar")
    plt.title("Distribution of Sentiment")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("sentiment_distribution.png")
    plt.close()

    print(
        "\nEvaluation completed. Check confusion_matrix.png, feature_importance.png, and sentiment_distribution.png for visualizations."
    )


if __name__ == "__main__":
    build_prediction_model("sentiment_stocks_data.csv")
