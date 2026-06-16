import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data():

    df = pd.read_csv("../data/loan_data.csv")

    df.drop("Loan_ID", axis=1, inplace=True)

    for col in df.columns:
        if df[col].dtype == "object":
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)

    le = LabelEncoder()

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = le.fit_transform(df[col])

    X = df.drop("Loan_Status", axis=1)
    y = df["Loan_Status"]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )