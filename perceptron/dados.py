"""Representação, carga, validação e partição dos conjuntos de dados."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np


@dataclass(frozen=True)
class ConjuntoDados:
    """Par (atributos, rótulos) imutável de um problema de classificação binária."""

    X: np.ndarray
    y: np.ndarray

    def __post_init__(self) -> None:
        if len(self.X) != len(self.y):
            raise ValueError("X e y devem ter o mesmo número de amostras.")

    def __len__(self) -> int:
        return len(self.y)

    @property
    def classes(self) -> np.ndarray:
        return np.unique(self.y)

    @property
    def contagem_por_classe(self) -> np.ndarray:
        return np.bincount(self.y)

    def normalizar(self, escala: float) -> ConjuntoDados:
        """Retorna um novo conjunto com os atributos divididos por `escala`."""
        if escala <= 0:
            raise ValueError("A escala de normalização deve ser positiva.")
        return ConjuntoDados(self.X / escala, self.y)

    def dividir(
        self,
        proporcao_treino: float = 0.7,
        rng: np.random.RandomState | None = None,
    ) -> tuple[ConjuntoDados, ConjuntoDados]:
        """Particiona aleatoriamente em (treino, teste)."""
        if not 0.0 < proporcao_treino < 1.0:
            raise ValueError("A proporção de treino deve estar entre 0 e 1.")
        rng = rng if rng is not None else np.random.RandomState()

        indices = rng.permutation(len(self))
        n_treino = int(round(proporcao_treino * len(self)))
        idx_treino, idx_teste = indices[:n_treino], indices[n_treino:]

        treino = ConjuntoDados(self.X[idx_treino], self.y[idx_treino])
        teste = ConjuntoDados(self.X[idx_teste], self.y[idx_teste])
        return treino, teste


class CarregadorDados:
    """Responsável pela carga e validação de integridade dos arquivos de dados."""

    N_COLUNAS: ClassVar[int] = 3  # x1, x2, rótulo
    ROTULOS_VALIDOS: ClassVar[tuple[int, ...]] = (0, 1)

    @classmethod
    def carregar(cls, caminho_arquivo: str | Path) -> ConjuntoDados:
        dados = np.asarray(np.load(caminho_arquivo), dtype=float).reshape(-1, cls.N_COLUNAS)

        if not np.all(np.isfinite(dados)):
            raise ValueError("O arquivo contém valores inválidos (NaN ou infinito).")

        rotulos = np.unique(dados[:, 2])
        if not np.all(np.isin(rotulos, cls.ROTULOS_VALIDOS)):
            raise ValueError(f"Rótulos inesperados encontrados: {rotulos}")

        return ConjuntoDados(X=dados[:, :2], y=dados[:, 2].astype(int))
