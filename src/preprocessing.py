import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

DATA_PATH = "/Users/rajsilwal/Documents/GitHub/Churn-Analysis/data/Dataset_ATS_v2.csv"


def main():
    # Load raw data
    df = pd.read_csv(DATA_PATH)

    # Replace empty strings with NaN
    df.replace(" ", np.nan, inplace=True)

    # Drop rows where target is missing
    df.dropna(subset=["Churn"], inplace=True)

    # Fill numeric columns with median
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)

    # Fill categorical columns with mode
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Label encoding
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # Split features and target
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # Save outputs
    df.to_csv("../data/cleaned_dataset.csv", index=False)
    X_scaled.to_csv("../data/x_scaled.csv", index=False)
    y.to_csv("../data/y.csv", index=False)

    print("Preprocessing complete.")


if __name__ == "__main__":
    main()