import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # Visualización de datos estadísticos
import statsmodels.api as sm  # Regresión lineal
from scipy.stats import chi2_contingency  # Test de chi-cuadrado
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score, recall_score)
from matplotlib.lines import Line2D

from sklearn.metrics import auc, roc_curve


def plot_roc(y_test, X_test_proba):
    fpr, tpr, _ = roc_curve(y_test, X_test_proba)

    plt.plot(fpr, tpr, label="Modelo")
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel("Tasa de falsos positivos")
    plt.ylabel("Tasa de verdaderos positivos")
    plt.title("Grafico curva ROC")
    plt.legend()
    plt.tight_layout()

    print(f"Area bajo la curva: {auc(fpr, tpr)}")


def fix_location(df):
    mapping_dict = {"Dartmoor": "DartmoorVillage",
                    "Richmond": "RichmondSydney"}
    df_out = df.copy()
    df_out['Location'] = df_out["Location"].map(
        mapping_dict).fillna(df["Location"])
    return df_out


def to_category(x):
    return x.astype('category')


def to_datetime(x):
    return x.astype('datetime64[ns]')


def map_bool(x):
    # Si el tipo es dataframe aplicar applymap sino aplicar map
    if isinstance(x, pd.DataFrame):
        return x.applymap(lambda y: {"Yes": 1, "No": 0}.get(y, y))
    else:
        return x.map({"Yes": 1, "No": 0})


def eliminar_columnas(df, columnas_a_eliminar):
    return df.drop(columns=columnas_a_eliminar)


def encode_cyclical_date(df, date_column='Date'):
    """
    Encodes a date column into cyclical features using sine and cosine transformations.

    Parameters:
    df (pandas.DataFrame): The dataframe containing the date column.
    date_column (str): The name of the date column. Default is 'Date'.

    Returns:
    pandas.DataFrame: The dataframe with new 'DayCos' and 'DaySin' columns added,
                      and intermediate columns removed.
    """
    # Make a copy to avoid modifying the original dataframe
    df = df.copy()

    # Ensure the date column is in datetime format
    df[date_column] = pd.to_datetime(df[date_column])

    # Calculate day of year
    df['DayOfYear'] = df[date_column].dt.dayofyear

    # Determine the number of days in the year for each date (taking leap years into account)
    df['DaysInYear'] = df[date_column].dt.is_leap_year.apply(
        lambda leap: 366 if leap else 365)

    # Convert day of the year to angle in radians, dividing by DaysInYear + 1
    df['Angle'] = 2 * np.pi * (df['DayOfYear'] - 1) / df['DaysInYear']

    # Calculate sine and cosine features
    df['DayCos'] = np.cos(df['Angle'])
    df['DaySin'] = np.sin(df['Angle'])

    # Remove intermediate columns
    df = df.drop(columns=["DayOfYear", "DaysInYear", "Angle"])

    return df


def encode_location(df: pd.DataFrame, gdf_locations) -> pd.DataFrame:
    return pd.merge(df, gdf_locations.drop(columns="geometry"), on="Location")


def encode_wind_dir(df):
    dirs = ["E", "ENE", "NE", "NNE", "N", "NNW", "NW", "WNW",
            "W", "WSW", "SW", "SSW", "S", "SSE", "SE", "ESE"]
    angles = np.radians(np.arange(0, 360, 22.5))
    mapping_dict = {d: a for (d, a) in zip(dirs, angles)}

    wind_dir_columns = ["WindGustDir", "WindDir9am", "WindDir3pm"]
    for column in wind_dir_columns:
        df[f"{column}Angle"] = df[column].map(mapping_dict)

        df[f"{column}Cos"] = np.cos(df[f"{column}Angle"].astype(float))
        df[f"{column}Sin"] = np.sin(df[f"{column}Angle"].astype(float))

        df = df.drop(columns=f"{column}Angle")

    return df


def plot_day_of_year_in_unit_circle():
    # Create a DataFrame to hold the values
    days = np.arange(1, 366, 2)
    days_in_year = 366

    angles = 2 * np.pi * (days - 1) / days_in_year
    cos_vals = np.cos(angles)
    sin_vals = np.sin(angles)

    df_days = pd.DataFrame({
        'Day': days,
        'Angle': angles,
        'DayCos': cos_vals,
        'DaySin': sin_vals
    })

    # Randomly select a day
    random_day = 37
    random_day_row = df_days[df_days['Day'] == random_day]

    # Plot the circle with 365 dots
    plt.figure(figsize=(5, 5))
    plt.plot(df_days['DayCos'], df_days['DaySin'],
             'bo', markersize=1)  # Circle with 365 dots

    # Highlight the random day
    plt.plot(random_day_row['DayCos'],
             random_day_row['DaySin'], 'ro', markersize=2)
    plt.text(random_day_row['DayCos'].values[0] + 0.02, random_day_row['DaySin'].values[0],
             f"""Day {random_day}\n({random_day_row['DayCos'].values[0]:.2f},{
        random_day_row['DaySin'].values[0]:.2f})""",
        fontsize=6, ha='left', va='bottom')

    # Draw x and y axes
    plt.axhline(0, color='grey', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='grey', linestyle='--', linewidth=0.5)

    # Draw the angle
    plt.plot([0, 1], [0, 0], 'k-', linewidth=1)
    plt.plot([0, random_day_row['DayCos'].values[0]], [
             0, random_day_row['DaySin'].values[0]], 'k-', linewidth=1)

    angle_text = f"{np.degrees(random_day_row['Angle'].values[0]):.2f}°"
    label_angle = random_day_row['Angle'].values[0] / 4
    plt.text(0.3 * np.cos(label_angle) + 0.05, 0.3 * np.sin(label_angle) +
             0.05, angle_text, color='k', fontsize=8, ha='center', va='center')

    # Mark and label the cosine value on the axes
    plt.plot([random_day_row['DayCos'].values[0], random_day_row['DayCos'].values[0]], [
             0, random_day_row['DaySin'].values[0]], 'k--', linewidth=0.4)
    plt.text(random_day_row['DayCos'].values[0], -0.05,
             f"{random_day_row['DayCos'].values[0]:.2f}",
             fontsize=7,
             ha='center',
             va='top')

    plt.plot([0, random_day_row['DayCos'].values[0]], [
             random_day_row['DaySin'].values[0], random_day_row['DaySin'].values[0]], 'k--', linewidth=0.4)
    plt.text(-0.05, random_day_row['DaySin'].values[0],
             f"{random_day_row['DaySin'].values[0]:.2f}",
             fontsize=7,
             ha='right',
             va='center')

    # Set equal aspect ratio
    plt.gca().set_aspect('equal', adjustable='box')

    # Labels and title
    plt.xlabel('DayCos')
    plt.ylabel('DaySin')
    plt.title('Representación del día del año en coordenadas polares', fontsize=10)

    plt.tick_params(axis='both', labelsize=6)

    plt.show()


def plot_distributions(original, mean_imputed, knn_imputed, variable, figsize=(15, 3)):
    plt.figure(figsize=figsize)

    # Original data
    plt.subplot(1, 3, 1)
    sns.histplot(original[variable].dropna(), kde=True)
    plt.title(f'Original: "{variable}"')

    # Mean imputed data
    plt.subplot(1, 3, 2)
    sns.histplot(mean_imputed[variable], kde=True)
    plt.title(f'Imputación (mediana): "{variable}"')

    # KNN imputed data
    plt.subplot(1, 3, 3)
    sns.histplot(knn_imputed[variable], kde=True)
    plt.title(f'Imputación (KNN): "{variable}"')

    plt.show()

# SimpleImputer (mediana)


def simple_imputer_mean(df, numerical_vars):
    imputer = SimpleImputer(strategy='mean')
    df[numerical_vars] = imputer.fit_transform(df[numerical_vars])
    return df

# KNN Imputer


def knn_imputer(df, numerical_vars, n_neighbors=5):
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df[numerical_vars] = imputer.fit_transform(df[numerical_vars])
    return df


def outliers_iqr(df, columns):
    outliers = []
    for col in columns:
        # Rango itercuartílico
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        irq_lower_bound = Q1 - 1.5 * IQR
        irq_upper_bound = Q3 + 1.5 * IQR

        # Desviación estándar
        mean = df[col].mean()
        std = df[col].std()
        std_lower_bound = mean - 3 * std
        std_upper_bound = mean + 3 * std

        outliers.append({'Column': col,
                         'IRQ-Percentage': (((df[col] < irq_lower_bound) | (df[col] > irq_upper_bound)).sum() / len(df[col]) * 100).round(2),
                         '3Std-Percentage': (((df[col] < std_lower_bound) | (df[col] > std_upper_bound)).sum() / len(df[col]) * 100).round(2),
                         'IRQ-Count': ((df[col] < irq_lower_bound) | (df[col] > irq_upper_bound)).sum(),
                         '3Std-Count': ((df[col] < std_lower_bound) | (df[col] > std_lower_bound)).sum()
                         })
    return pd.DataFrame(outliers)


def show_unique_types_as_df(df, columns):
    # Crear una lista para almacenar la información
    type_info = []

    for column in columns:
        if column in df.columns:
            types = df[column].apply(lambda x: type(
                x) if not pd.isna(x) else np.nan)
            unique_types = types.unique()
            for t in unique_types:
                type_info.append({
                    'Column': column,
                    'Type': 'NaN' if t is np.nan else str(t)
                })
        else:
            type_info.append({
                'Column': column,
                'Type': 'Not in DataFrame'
            })

    # Convertir la lista a un DataFrame
    type_info_df = pd.DataFrame(type_info)

    return type_info_df

# Definir una función personalizada para concatenar los valores


def concatenate_values(series):
    return ', '.join(series.astype(str))

# Muestra los tipos únicos de datos en las columnas especificadas de un DataFrame, incluyendo NaN.


def show_unique_types(df, columns):
    for column in columns:
        if column in df.columns:
            types = df[column].apply(lambda x: type(
                x) if not pd.isna(x) else np.nan)
            unique_types = types.unique()
            print(f"Columna '{column}' tiene los siguientes tipos únicos:")
            for t in unique_types:
                if t is np.nan:
                    print("  - NaN")
                else:
                    print(f"  - {t}")
        else:
            print(f"La columna '{column}' no está en el DataFrame")

# Imprimir los valores únicos de cada columna


def print_unique_values(unique_values):
    for col, values in unique_values.items():
        print(f"Columna: {col}")
        try:
            print(values.to_numpy())
        except:
            print(values)
        print()  # Imprimir una línea en blanco para separar las columna

# Muestra el porcentaje de valores faltantes.


def print_missing_perc(df, column):
    missing_perc = round((df[column].isna().sum() / df.shape[0]) * 100, 1)
    print(f"""Porcentaje de valores faltantes en la columna {
          column}: {missing_perc}%""")


def plot_locations_over_time(df, color="blue"):
    locations = df["Location"].unique()

    _, ax = plt.subplots(figsize=(12, 7))

    # Plot data for each location
    for location in locations:
        filt = df["Location"] == location
        df_location = df.loc[filt, :]

        # Plot observations present in dataframe
        ax.plot(
            df_location["Date"],
            [location] * len(df_location),
            "o-",
            color=color,
            label=location,
            markersize=0.5,
            linewidth=0.05,
        )

        # Plot null values in target column
        null_indices = df_location.loc[df_location["RainTomorrow"].isnull(
        )].index
        for idx in null_indices:
            ax.plot(df_location.loc[idx, "Date"],
                    location, "ko", markersize=0.15)

    # Customize the plot
    ax.set_yticks(np.arange(len(locations)))
    ax.set_yticklabels(
        locations, fontsize="x-small"
    )  # Increase fontsize for y-axis labels
    ax.set_ylim(-0.5, len(locations) - 0.5)

    xticks = pd.date_range(
        start=df["Date"].min(), end=df["Date"].max(), freq="6MS")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks.strftime("%Y-%m"),
                       fontsize="x-small", rotation=90)

    ax.grid(True, linestyle=":", alpha=0.5)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="lightgreen",
            markersize=5,
            label="Observación presente en el dataframe",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="black",
            markersize=5,
            label="Observación con valor ausente en la columna 'RainTomorrow'",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.075),
        ncol=2,
        fontsize=7,
    )

    plt.suptitle(
        "Observaciones presentes en la serie temporal por centro meteorológico",
        fontsize=10,
    )

    plt.show()


def phi_coefficient(confusion_matrix):
    chi2, p, _, _ = chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum().sum()
    phi = np.sqrt(chi2 / n)
    return phi, p

# Función para graficar un pie plot.


def plot_pie(df, column, ax):
    counts = df[column].value_counts()
    labels = counts.index
    sizes = counts.values
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    ax.set_title(column)

# Función para graficar un gráfico de barras


def plot_bar(df, column, ax):
    sns.countplot(data=df, x=column, ax=ax)
    ax.set_title(column)
    # Nota: Elimino las etiquetas en el eje de las x porque quedan muy apretadas
    ax.set_xlabel('')  # Eliminar la etiqueta del eje x
    ax.set_xticklabels([])  # Eliminar las etiquetas en el eje x

# Función para graficar un histograma.


def plot_hist(df, column, ax):
    sns.histplot(df[column], kde=True, ax=ax)
    ax.set_title(column)

# Función para graficar qq-plot


def plot_qq_plots(df, column, ax):
    sm.qqplot(df[column].dropna(), line='45', fit=True, ax=ax)
    ax.set_title(f'QQ-plot for {column}')

# Funcion para graficar boxplot


def plot_boxplot(df, column, ax):
    sns.boxplot(df[column], ax=ax)
    ax.set_title(column)


def plot_heatmap(correlation_matrix, figsize=(8, 8)):
    plt.figure(figsize=figsize)
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, annot_kws={"size": 8}, cbar=False)
    plt.title("Heatmap de Correlación")

    plt.xticks(fontsize=8)  # Adjust x-axis font size
    plt.yticks(fontsize=8)  # Adjust y-axis font size

    plt.show()


def plot_graph_on_grid(df, columns, graph_type, num_cols=3, figsize=(10, 5)):
    # cálculo del número de filas
    num_rows = (len(columns) + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()

    for i, col in enumerate(columns):
        if graph_type == 'pie':
            plot_pie(df, col, axes[i])
        elif graph_type == 'hist':
            plot_hist(df, col, axes[i])
        elif graph_type == 'bar':
            plot_bar(df, col, axes[i])
        elif graph_type == 'qq-plot':
            plot_qq_plots(df, col, axes[i])
        elif graph_type == 'box-plot':
            plot_boxplot(df, col, axes[i])
        else:
            raise Exception('Tipo de gráfico no soportado.')

    # Elimina cualquier gráfico extra en la grilla
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def evaluate_predictions(y_true, y_pred, algorithm, figsize=(4, 4)):
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Mapping of labels
    labels = ["No", "Yes"]

    # Plot confusion matrix
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
    )
    plt.xlabel("Predicted Values")
    plt.ylabel("Real Values")
    plt.title("Confusion Matrix")
    plt.show()

    row = {
        "Algorithm": algorithm,
        "Sensibility": round(tp / (tp + fn), 3),
        "Specificity": round(tn / (tn + fp), 3),
        "Accuracy": round(accuracy_score(y_true, y_pred), 3),
        "Balanced-Accuracy": round(balanced_accuracy_score(y_true, y_pred), 3),
        "Precision": round(precision_score(y_true, y_pred), 3),  # type: ignore
        "Recall": round(recall_score(y_true, y_pred), 3),  # type: ignore
        "F1 Score": round(f1_score(y_true, y_pred), 3)  # type: ignore
    }

    return row
