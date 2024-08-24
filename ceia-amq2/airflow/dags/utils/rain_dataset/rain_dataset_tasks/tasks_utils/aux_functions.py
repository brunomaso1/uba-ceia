import awswrangler as wr
import pandas as pd


def map_bool(x):
    if isinstance(x, pd.DataFrame):
        return x.applymap(lambda y: {"Yes": 1, "No": 0}.get(y, y))
    else:
        return x.map({"Yes": 1, "No": 0})


def to_category(x):
    return x.astype("category")


def to_datetime(x):
    return x.astype("datetime64[ns]")


def save_to_csv(df, path):
    wr.s3.to_csv(df=df, path=path, index=False)


def eliminar_columnas(df, columnas_a_eliminar):
    return df.drop(columns=columnas_a_eliminar)


def fix_location(df):
    mapping_dict = {"Dartmoor": "DartmoorVillage", "Richmond": "RichmondSydney"}
    df_out = df.copy()
    df_out["Location"] = df_out["Location"].map(mapping_dict).fillna(df["Location"])
    return df_out


def upload_split_to_s3(
    X_train, X_test, y_train, y_test, train_test_split_path
):
    save_to_csv(X_train, train_test_split_path["X_train"])
    save_to_csv(X_test, train_test_split_path["X_test"])
    save_to_csv(y_train, train_test_split_path["y_train"])
    save_to_csv(y_test, train_test_split_path["y_test"])

def download_split_from_s3(train_test_split_path):
    X_train = wr.s3.read_csv(train_test_split_path["X_train"])
    X_test = wr.s3.read_csv(train_test_split_path["X_test"])
    y_train = wr.s3.read_csv(train_test_split_path["y_train"])
    y_test = wr.s3.read_csv(train_test_split_path["y_test"])

    return X_train, X_test, y_train, y_test
