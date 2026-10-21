"""Métricas de desempenho para classificação binária."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

from .dados import ConjuntoDados
from .modelo import Perceptron


@dataclass(frozen=True)
class MetricasClassificacao:
    """Métricas de desempenho para classificação binária (classe positiva = 1)."""

    acuracia: float
    precisao: float
    revocacao: float
    f_score: float

    @classmethod
    def calcular(cls, y_real: np.ndarray, y_pred: np.ndarray) -> MetricasClassificacao:
        return cls(
            acuracia=accuracy_score(y_real, y_pred),
            precisao=precision_score(y_real, y_pred, zero_division=0),
            revocacao=recall_score(y_real, y_pred, zero_division=0),
            f_score=f1_score(y_real, y_pred, zero_division=0),
        )


class AvaliadorBinario:
    """Avalia um classificador treinado sobre conjuntos de dados."""

    ROTULOS: ClassVar[tuple[int, int]] = (0, 1)

    def __init__(self, modelo: Perceptron) -> None:
        self.modelo = modelo

    def matriz_confusao(self, dados: ConjuntoDados) -> np.ndarray:
        y_pred = self.modelo.predict(dados.X)
        return confusion_matrix(dados.y, y_pred, labels=list(self.ROTULOS))

    def metricas(self, dados: ConjuntoDados) -> MetricasClassificacao:
        y_pred = self.modelo.predict(dados.X)
        return MetricasClassificacao.calcular(dados.y, y_pred)
