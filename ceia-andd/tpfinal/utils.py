# Importación de librerias
import sys # Interactuar con el sistema
import numpy as np # Albegra lineal
import pandas as pd # Procesamiento de datos
import matplotlib.pyplot as plt # Visualización de datos
import seaborn as sns # Visualización de datos estadísticos
from ydata_profiling import ProfileReport # Reporte (profiling)
from scipy.stats import chi2_contingency # Test de chi-cuadrado
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score) # Metricas para evaluación
import geopandas as gpd # Georeferenciacion
from geopandas.datasets import get_path # Ruta de los datos geográficos
from shapely.geometry import Point # Geometría espacial
import re # Expresiones regulares
from itertools import chain, combinations # Iteradores
import osmnx as ox # OpenStreetMap
import statsmodels.api as sm # Regresión lineal
import json

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
            types = df[column].apply(lambda x: type(x) if not pd.isna(x) else np.nan)
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
            types = df[column].apply(lambda x: type(x) if not pd.isna(x) else np.nan)
            unique_types = types.unique()
            print(f"Columna '{column}' tiene los siguientes tipos únicos:")
            for t in unique_types:
                if t is np.nan:
                    print(f"  - NaN")
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
    print(f"Porcentaje de valores faltantes en la columna {column}: {missing_perc}%")

# Evaluar las predicciones en una matriz de confuncion.
def evaluate_predictions(y_true, y_pred, figsize=(4, 4)):
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

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

    # Calculate evaluation metrics
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    # Print evaluation metrics
    print()
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

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
    null_indices = df_location.loc[df_location["RainTomorrow"].isnull()].index
    for idx in null_indices:
        ax.plot(df_location.loc[idx, "Date"], location, "ko", markersize=0.15)

    # Customize the plot
    ax.set_yticks(np.arange(len(locations)))
    ax.set_yticklabels(
        locations, fontsize="x-small"
    )  # Increase fontsize for y-axis labels
    ax.set_ylim(-0.5, len(locations) - 0.5)

    xticks = pd.date_range(start=df["Date"].min(), end=df["Date"].max(), freq="6MS")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks.strftime("%Y-%m"), fontsize="x-small", rotation=90)

    ax.grid(True, linestyle=":", alpha=0.5)

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="lightgreen",
            markersize=5,
            label="Observación presente en el dataframe",
        ),
        plt.Line2D(
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
    sm.qqplot(df[column].dropna(), line ='45', fit=True, ax=ax)
    ax.set_title(f'QQ-plot for {column}')

# Funcion para graficar boxplot
def plot_boxplot(df, column, ax):
    sns.boxplot(df[column], ax=ax)
    ax.set_title(column)

def plot_heatmap(correlation_matrix):
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Heatmap de Correlación")
    plt.show()

def plot_graph_on_grid(df, columns, graph_type, num_cols=3, figsize=(10, 5)):
    num_rows = (len(columns) + num_cols - 1) // num_cols  # cálculo del número de filas

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