import pandas as pd

def load_csv(train_path: str, valid_path: str, text_col: str, label_col: str):
    train = pd.read_csv(train_path)
    valid = pd.read_csv(valid_path)
    X_train, y_train = train[text_col].astype(str), train[label_col]
    X_valid, y_valid = valid[text_col].astype(str), valid[label_col]
    return (X_train, y_train), (X_valid, y_valid)
