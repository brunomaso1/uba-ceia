# visualizations.py
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union


class TrainingVisualizer:
    def __init__(self, data: Union[str, pd.DataFrame]):
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
        """Subplots para las pérdidas de entrenamiento y validación"""
        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        losses = ["box_loss", "cls_loss", "dfl_loss"]
        for ax, loss in zip(axes, losses):
            ax.plot(self.results_df.index, self.results_df[f"train/{loss}"], label=f"train/{loss}")
            ax.plot(self.results_df.index, self.results_df[f"val/{loss}"], label=f"val/{loss}")
            ax.set_ylabel("Loss")
            ax.set_title(loss)
            ax.legend()
            ax.grid(True)

        axes[-1].set_xlabel("Epoch")
        fig.suptitle("Entrenamiento y Validación - Pérdidas", fontsize=14)

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        else:
            plt.show()
        plt.close()

    def plot_metrics(self, save_path: str = None):
        """Subplots para métricas de validación"""
        fig, axes = plt.subplots(4, 1, figsize=(10, 16), sharex=True)

        metrics = [
            ("metrics/precision(B)", "Precision"),
            ("metrics/recall(B)", "Recall"),
            ("metrics/mAP50(B)", "mAP@50"),
            ("metrics/mAP50-95(B)", "mAP@50-95"),
        ]

        for ax, (col, title) in zip(axes, metrics):
            ax.plot(self.results_df.index, self.results_df[col], label=title)
            ax.set_ylabel("Valor")
            ax.set_title(title)
            ax.legend()
            ax.grid(True)

        axes[-1].set_xlabel("Epoch")
        fig.suptitle("Métricas de Validación", fontsize=14)

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        else:
            plt.show()
        plt.close()

    def plot_lr(self, save_path: str = None):
        """Subplots para learning rate"""
        fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

        lrs = ["lr/pg0", "lr/pg1", "lr/pg2"]
        for ax, lr in zip(axes, lrs):
            ax.plot(self.results_df.index, self.results_df[lr], label=lr)
            ax.set_ylabel("LR")
            ax.set_title(lr)
            ax.legend()
            ax.grid(True)

        axes[-1].set_xlabel("Epoch")
        fig.suptitle("Tasas de Aprendizaje por Grupo de Parámetros", fontsize=14)

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        else:
            plt.show()
        plt.close()

    def plot_all(self, save_dir: str = None):
        """Genera todas las gráficas"""
        if save_dir:
            self.plot_losses(f"{save_dir}/losses.png")
            self.plot_metrics(f"{save_dir}/metrics.png")
            self.plot_lr(f"{save_dir}/lr.png")
        else:
            self.plot_losses()
            self.plot_metrics()
            self.plot_lr()
