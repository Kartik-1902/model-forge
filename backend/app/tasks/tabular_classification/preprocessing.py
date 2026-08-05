import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_csv(data_path: str) -> pd.DataFrame:
    """Loads a CSV file into a pandas DataFrame."""
    return pd.read_csv(data_path)

def split_features_target(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    """Splits the DataFrame into features and target."""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Builds a scikit-learn ColumnTransformer that:
    - Imputes and scales numeric columns
    - Imputes and one-hot encodes categorical columns
    """
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(exclude="number").columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    return preprocessor
