"""Renderizações gráficas (dispersão, fronteira de decisão e boxplots)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np

from .dados import ConjuntoDados
from .experimento import ChaveConfiguracao, ResultadoConfiguracao


class VisualizadorPerceptron:
    """Responsável pelas renderizações de dispersão e de fronteiras lineares."""

    MARGEM_RELATIVA: ClassVar[float] = 0.05
    MARGEM_MINIMA: ClassVar[float] = 0.5
    EPSILON: ClassVar[float] = 1e-12

    def __init__(self, cor_classe_0: str = "red", cor_classe_1: str = "blue") -> None:
        self.cor_classe_0 = cor_classe_0
        self.cor_classe_1 = cor_classe_1

    # ------------------------------------------------------------ API pública
    def plotar_dispersao(
        self,
        dados: ConjuntoDados,
        titulo: str = "Dispersão dos Dados",
        tamanho_ponto: float = 12,
        alpha: float = 0.7,
        figsize: tuple[float, float] = (8, 6),
    ) -> None:
        """Plota a distribuição espacial das classes no plano 2D."""
        fig, ax = plt.subplots(figsize=figsize)
        self._desenhar_amostras(ax, dados, tamanho_ponto, alpha)
        self._configurar_eixo(ax, titulo)
        fig.tight_layout()
        plt.show()

    def plotar_fronteira(
        self,
        dados: ConjuntoDados,
        pesos: np.ndarray,
        titulo: str = "Perceptron — Parte I",
        figsize: tuple[float, float] = (9, 6),
    ) -> None:
        """Plota a reta de decisão w1*x1 + w2*x2 + bias = 0 sobre os dados."""
        fig, ax = plt.subplots(figsize=figsize)
        self._desenhar_fronteira_em_eixo(ax, dados, pesos, titulo, tamanho_ponto=36)
        fig.tight_layout()
        plt.show()

    def plotar_comparacao_treino_teste(
        self,
        treino: ConjuntoDados,
        teste: ConjuntoDados,
        pesos: np.ndarray,
        referencia: ConjuntoDados | None = None,
        figsize: tuple[float, float] = (14, 6),
    ) -> None:
        """Renderiza lado a lado a fronteira sobre os conjuntos de treino e teste."""
        fig, eixos = plt.subplots(1, 2, figsize=figsize)
        self._desenhar_fronteira_em_eixo(
            eixos[0], treino, pesos, f"Treino ({len(treino)} exemplos)", referencia)
        self._desenhar_fronteira_em_eixo(
            eixos[1], teste, pesos, f"Teste ({len(teste)} exemplos)", referencia)
        fig.tight_layout()
        plt.show()

    # ------------------------------------------------------------ internos
    def _desenhar_amostras(self, ax, dados: ConjuntoDados, tamanho_ponto: float, alpha: float) -> None:
        for classe, cor in ((0, self.cor_classe_0), (1, self.cor_classe_1)):
            mascara = dados.y == classe
            ax.scatter(dados.X[mascara, 0], dados.X[mascara, 1],
                       color=cor, s=tamanho_ponto, alpha=alpha, label=f"Classe {classe}")

    def _definir_limites(self, ax, referencia: ConjuntoDados) -> tuple[float, float]:
        """Ajusta os limites do eixo com margem e devolve o intervalo em x."""
        x_min, x_max = referencia.X[:, 0].min(), referencia.X[:, 0].max()
        y_min, y_max = referencia.X[:, 1].min(), referencia.X[:, 1].max()
        margem_x = (x_max - x_min) * self.MARGEM_RELATIVA or self.MARGEM_MINIMA
        margem_y = (y_max - y_min) * self.MARGEM_RELATIVA or self.MARGEM_MINIMA

        limites_x = (x_min - margem_x, x_max + margem_x)
        ax.set_xlim(*limites_x)
        ax.set_ylim(y_min - margem_y, y_max + margem_y)
        return limites_x

    def _desenhar_reta(self, ax, pesos: np.ndarray, limites_x: tuple[float, float]) -> None:
        w1, w2, bias = pesos
        if abs(w2) > self.EPSILON:
            x_reta = np.linspace(*limites_x, 200)
            ax.plot(x_reta, -(w1 * x_reta + bias) / w2, color="black", label="Reta de separação")
        elif abs(w1) > self.EPSILON:
            ax.axvline(x=-bias / w1, color="black", label="Reta de separação")
        else:
            raise ValueError("A reta de separação não pode ser desenhada (pesos nulos).")

    def _desenhar_fronteira_em_eixo(
        self,
        ax,
        dados: ConjuntoDados,
        pesos: np.ndarray,
        titulo: str,
        referencia: ConjuntoDados | None = None,
        tamanho_ponto: float = 12,
    ) -> None:
        self._desenhar_amostras(ax, dados, tamanho_ponto, alpha=0.7)
        limites_x = self._definir_limites(ax, referencia if referencia is not None else dados)
        self._desenhar_reta(ax, pesos, limites_x)
        self._configurar_eixo(ax, titulo)

    @staticmethod
    def _configurar_eixo(ax, titulo: str) -> None:
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_title(titulo)
        ax.legend()
        ax.grid(alpha=0.2)


class VisualizadorExperimento:
    """Gráficos comparativos do experimento de configurações."""

    @staticmethod
    def plotar_boxplots(
        resultados: Mapping[ChaveConfiguracao, ResultadoConfiguracao],
        figsize: tuple[float, float] = (14, 5),
    ) -> None:
        rotulos = [f"η={c.taxa_aprendizado}\n{c.intervalo_pesos}" for c in resultados]
        fig, eixos = plt.subplots(1, 2, figsize=figsize)

        eixos[0].boxplot([r.ajustes for r in resultados.values()], tick_labels=rotulos)
        eixos[0].set_title("Qtde de ajustes por configuração")
        eixos[0].set_ylabel("Ajustes")

        eixos[1].boxplot([r.epocas for r in resultados.values()], tick_labels=rotulos)
        eixos[1].set_title("Épocas até convergência por configuração")
        eixos[1].set_ylabel("Épocas")

        fig.tight_layout()
        plt.show()
