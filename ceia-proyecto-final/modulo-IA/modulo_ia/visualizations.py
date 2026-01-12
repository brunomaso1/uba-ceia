import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(plt.rcParamsDefault)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    # "Roboto",
    "DejaVu Sans",
    "Arial",
]  # Prioriza Roboto, si no la encuentra, usa las otras.
plt.rcParams["text.usetex"] = False  # Es crucial para manejar el texto Unicode/acentos

# Constantes de color
TEXT_COLOR = "#026e00"
METRICS_COLOR = "#012c00"
LOSS_COLOR = "#9ac599"
LR_COLOR = "#012c00"
# Para saber la configuración de rcParams: plt.rcParams.keys()
plt.rcParams["figure.dpi"] = 150
plt.rcParams["xtick.color"] = TEXT_COLOR
plt.rcParams["ytick.color"] = TEXT_COLOR
plt.rcParams["text.color"] = TEXT_COLOR
plt.rcParams["grid.color"] = "black"
plt.rcParams["grid.alpha"] = 0.1
plt.rcParams["axes.facecolor"] = "#f9faf3"
plt.rcParams["axes.edgecolor"] = TEXT_COLOR


class TrainingVisualizer:
    def __init__(self, data: str | pd.DataFrame):
        """
        Inicializa el visualizador.
        Args:
            data: Puede ser una ruta a un archivo CSV (str) o un DataFrame de Pandas.
        """
        if isinstance(data, str):
            self.results_df = pd.read_csv(data).set_index("epoch")
        elif isinstance(data, pd.DataFrame):
            if "epoch" in data.columns:
                self.results_df = data.set_index("epoch")
            else:
                self.results_df = data
        else:
            raise ValueError("El parámetro 'data' debe ser una ruta CSV o un DataFrame de Pandas.")

    def plot_losses(self, save_path: str = None):
        """Subplots para las Pérdidas de Entrenamiento y Validación"""
        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        losses = ["box_loss", "cls_loss", "dfl_loss"]
        for ax, loss in zip(axes, losses):
            ax.plot(self.results_df.index, self.results_df[f"train/{loss}"], label=f"train/{loss}", color=LOSS_COLOR)
            ax.plot(
                self.results_df.index,
                self.results_df[f"val/{loss}"],
                label=f"val/{loss}",
                color=LOSS_COLOR,
                linestyle=":",
            )
            ax.set_ylabel("Loss", color=LOSS_COLOR)
            ax.set_title(loss)
            ax.legend()
            ax.grid(True)
            ax.tick_params(axis="y", colors=LOSS_COLOR)
            ax.yaxis.label.set_color(LOSS_COLOR)

        axes[-1].set_xlabel("Epochs", color=TEXT_COLOR)
        fig.suptitle("Pérdidas de entrenamiento y validación", fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()

    def plot_combined_loss_and_metrics(self, save_path: str = None):
        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        losses = ["box_loss", "cls_loss", "dfl_loss"]
        for ax, loss in zip(axes, losses):
            # --- Eje izquierdo: Loss ---
            (line1,) = ax.plot(
                self.results_df.index, self.results_df[f"train/{loss}"], label=f"train/{loss}", color=LOSS_COLOR
            )
            (line2,) = ax.plot(
                self.results_df.index,
                self.results_df[f"val/{loss}"],
                label=f"val/{loss}",
                color=LOSS_COLOR,
                linestyle=":",
                alpha=0.5,
            )
            ax.set_ylabel("Loss", color=LOSS_COLOR)
            ax.set_title(loss)
            ax.grid(True)

            # Cambia color de ticks y label del eje Y izquierdo
            ax.tick_params(axis="y", colors=LOSS_COLOR)
            ax.yaxis.label.set_color(LOSS_COLOR)

            # --- Eje derecho: Métricas ---
            ax2 = ax.twinx()
            (line3,) = ax2.plot(
                self.results_df.index,
                self.results_df["metrics/precision(B)"],
                label="val/precision",
                color=METRICS_COLOR,
            )
            (line4,) = ax2.plot(
                self.results_df.index,
                self.results_df["metrics/recall(B)"],
                label="val/recall",
                color=METRICS_COLOR,
                linestyle=":",
                alpha=0.5,
            )
            ax2.set_ylabel("Precision / Recall", color=METRICS_COLOR)

            # Cambia color de ticks y label del eje Y derecho
            ax2.tick_params(axis="y", colors=METRICS_COLOR)
            ax2.yaxis.label.set_color(METRICS_COLOR)

            # Leyenda combinada
            lines = [line3, line4, line1, line2]
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc="right", frameon=True, title="Curvas")

        axes[-1].set_xlabel("Epochs", color=TEXT_COLOR)
        fig.suptitle("Pérdidas de entrenamiento y validación", fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()

    def plot_metrics(self, save_path: str = None):
        """Subplots para Métricas de Validación"""
        fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

        metrics = [
            ("metrics/precision(B)", "Precisión"),
            ("metrics/recall(B)", "Recall"),
            ("metrics/mAP50(B)", "mAP@50"),
            ("metrics/mAP50-95(B)", "mAP@50-95"),
        ]

        for ax, (col, title) in zip(axes, metrics):
            ax.plot(self.results_df.index, self.results_df[col], label=title, color=METRICS_COLOR)
            ax.set_ylabel("Valor", color=METRICS_COLOR)
            ax.set_title(title)
            ax.legend(loc="lower right")
            ax.grid(True)
            ax.tick_params()

        axes[-1].set_xlabel("Epochs", color=TEXT_COLOR)
        fig.suptitle("Métricas de validación", fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()

    def plot_map(self, save_path: str = None):
        """Grafica combinada de mAP50 y mAP50-95"""
        fig = plt.figure(figsize=(10, 6))
        plt.plot(
            self.results_df.index,
            self.results_df["metrics/mAP50-95(B)"],
            label="mAP@50-95",
            color=METRICS_COLOR,
        )
        plt.plot(
            self.results_df.index,
            self.results_df["metrics/mAP50(B)"],
            label="mAP@50",
            color=METRICS_COLOR,
            linestyle=":",
            alpha=0.5,
        )
        plt.ylabel("Valor", color=TEXT_COLOR)
        plt.tick_params(axis='y', labelcolor=TEXT_COLOR)
        fig.suptitle("mAP@50 y mAP@50-95 durante el entrenamiento", fontsize=14)
        plt.legend(loc="right")
        plt.grid(True)
        plt.xlabel("Epochs", color=TEXT_COLOR)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()

    def plot_lr(self, save_path: str = None):
        """Subplots para learning rate"""
        fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

        lrs = ["lr/pg0", "lr/pg1", "lr/pg2"]
        for ax, lr in zip(axes, lrs):
            ax.plot(self.results_df.index, self.results_df[lr], label=lr, color=LR_COLOR)
            ax.set_ylabel("LR", color=LR_COLOR)
            ax.set_title(lr)
            ax.legend()
            ax.grid(True)
            ax.tick_params()

        axes[-1].set_xlabel("Epochs", color=TEXT_COLOR)
        fig.suptitle("Tasas de Aprendizaje por Grupo de Parámetros", fontsize=14)

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()

    def plot_confusion_matrix(
        self, cm: pd.DataFrame, save_path: str = None, normalize: bool = False, cmap: str = "Greens"
    ):
        """
        Grafica la matriz de confusión dada como DataFrame de Pandas.

        Args:
            cm (pd.DataFrame): Matriz de confusión con índices y columnas como nombres de clases.
            save_path (str, opcional): Ruta para guardar la imagen. Si no se indica, muestra la figura.
            normalize (bool): Si True, normaliza por filas (valores relativos por clase verdadera).
            cmap (str): Colormap de Matplotlib, por defecto 'Greens'.
        """
        matrix = cm.to_numpy().astype(float)
        if normalize:
            matrix = matrix / matrix.sum(axis=1, keepdims=True)
            matrix = np.nan_to_num(matrix)  # Evita NaN si una fila suma 0

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(matrix, interpolation="nearest", cmap=cmap)

        # Ejes y etiquetas
        classes = cm.columns
        ax.set_xticks(np.arange(len(classes)))
        ax.set_yticks(np.arange(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=9, color=TEXT_COLOR)
        ax.set_yticklabels(classes, fontsize=9, color=TEXT_COLOR)

        ax.set_xlabel("Predicted", color=TEXT_COLOR)
        ax.set_ylabel("True", color=TEXT_COLOR)
        ax.set_title(u"Matriz de confusión", fontsize=14, color=TEXT_COLOR, pad=10)

        # Cambiar color de ticks
        ax.tick_params(colors=TEXT_COLOR)

        # Anotar valores
        fmt = ".2f" if normalize else ".0f"
        thresh = matrix.max() / 2.0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                ax.text(
                    j,
                    i,
                    format(value, fmt),
                    ha="center",
                    va="center",
                    color="white" if value > thresh else TEXT_COLOR,
                    fontsize=8,
                )

        # Barra de color
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Proporción" if normalize else "Recuento", rotation=-90, va="bottom", color=TEXT_COLOR)
        cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=TEXT_COLOR)

        plt.grid(False)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()

    def plot_all(self, save_dir: str = None):
        """Genera todas las gráficas"""
        if save_dir:
            self.plot_combined_loss_and_metrics(f"{save_dir}/combined_loss_metrics.png")
            self.plot_map(f"{save_dir}/map.png")
            self.plot_lr(f"{save_dir}/lr.png")
        else:
            self.plot_combined_loss_and_metrics()
            self.plot_map()
            self.plot_lr()
