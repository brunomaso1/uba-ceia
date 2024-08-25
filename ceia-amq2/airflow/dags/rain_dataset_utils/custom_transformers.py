from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FunctionTransformer, Pipeline
from . import aux_functions, enconding_functions


class ClapOutliersIRQTransformer(BaseEstimator, TransformerMixin):
    """
    Hace un clap de los outliers y guarda la información como un Transformer de Sklearn.
    O sea, se implementa un Transformer
    """
    def __init__(self, columns):
        self.IRQ_saved = {}
        self.columns = columns
        self.fitted = False

    def fit(self, X, y=None):
        for col in self.columns:
            # Rango itercuartílico
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IRQ = Q3 - Q1
            irq_lower_bound = Q1 - 3 * IRQ
            irq_upper_bound = Q3 + 3 * IRQ

            # Ajusto los valores al mínimo o máximo según corresponda.
            # Esto es para no pasarse de los valores mínimos o máximos con el IRQ.
            min_value = X[col].min()
            max_value = X[col].max()
            irq_lower_bound = max(irq_lower_bound, min_value)
            irq_upper_bound = min(irq_upper_bound, max_value)

            self.IRQ_saved[col + "irq_lower_bound"] = irq_lower_bound
            self.IRQ_saved[col + "irq_upper_bound"] = irq_upper_bound

        self.fitted = True

        return self

    def transform(self, X):
        if not self.fitted:
            raise ValueError("Fit the transformer first using fit().")

        X_transf = X.copy()

        for col in self.columns:
            irq_lower_bound = self.IRQ_saved[col + "irq_lower_bound"]
            irq_upper_bound = self.IRQ_saved[col + "irq_upper_bound"]
            X_transf[col] = X_transf[col].clip(
                upper=irq_upper_bound, lower=irq_lower_bound
            )

        return X_transf

# Todas las transformaciones aplicadas al conjunto.
def convertDataTypeTransformer(cat_columns, date_columns, bool_columns):
    return ColumnTransformer(
        [
            (
                "categories",
                FunctionTransformer(aux_functions.to_category),
                cat_columns,
            ),
            ("date", FunctionTransformer(aux_functions.to_datetime), date_columns),
            ("bool", FunctionTransformer(aux_functions.map_bool), bool_columns),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def missingValuesTransformer(cat_columns, bool_columns, cont_columns):
    cat_imputer = (
        "cat_missing_values_imputer",
        SimpleImputer(strategy="most_frequent"),
    )
    cont_imputer = ("cont_missing_values_imptuer", SimpleImputer(strategy="mean"))
    return ColumnTransformer(
        [
            ("cat_imputer", Pipeline([cat_imputer]), cat_columns + bool_columns),
            ("cont_imputer", Pipeline([cont_imputer]), cont_columns),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def cyclicalDateTransformer():
    return FunctionTransformer(
        func=enconding_functions.encode_cyclical_date,
        kw_args={"date_column": "Date"},
        validate=False,
    )


def fixLocationsTransformer():
    return FunctionTransformer(aux_functions.fix_location)


def encodeLocationTransformer(gdf_locations):
    return FunctionTransformer(
        func=enconding_functions.encode_location,
        kw_args={"gdf_locations": gdf_locations},
        validate=False,
    )


def encodeWindDirTransformer():
    return FunctionTransformer(enconding_functions.encode_wind_dir, validate=False)


def removeColumnsTransformer(columnas_codificadas):
    return FunctionTransformer(
        aux_functions.eliminar_columnas,
        kw_args={"columnas_a_eliminar": columnas_codificadas},
    )
