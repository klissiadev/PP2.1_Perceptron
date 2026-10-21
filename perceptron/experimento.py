"""Benchmark de configurações (taxa de aprendizado x intervalo de pesos)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import product
from typing import NamedTuple

import numpy as np

from .dados import ConjuntoDados
from .modelo import ConfiguracaoPerceptron, Perceptron


class ChaveConfiguracao(NamedTuple):
    """Identifica uma combinação de hiperparâmetros do experimento."""

    taxa_aprendizado: float
    intervalo_pesos: tuple[float, float]


@dataclass(frozen=True)
class ResultadoConfiguracao:
    """Resultados brutos das repetições de uma configuração, com estatísticas derivadas."""

    ajustes: tuple[int, ...]
    epocas: tuple[int, ...]

    @property
    def media_ajustes(self) -> float:
        return float(np.mean(self.ajustes))

    @property
    def desvio_padrao_ajustes(self) -> float:
        return float(np.std(self.ajustes, ddof=1))

    @property
    def min_epocas(self) -> int:
        return min(self.epocas)

    @property
    def max_epocas(self) -> int:
        return max(self.epocas)


class ExperimentoConfiguracoes:
    """Executa o benchmark de configurações com várias repetições."""

    def __init__(
        self,
        taxas_aprendizado: Sequence[float],
        intervalos: Sequence[tuple[float, float]],
        configuracao_base: ConfiguracaoPerceptron | None = None,
        num_repeticoes: int = 10,
        seed: int | None = 13,
    ) -> None:
        self.taxas_aprendizado = tuple(taxas_aprendizado)
        self.intervalos = tuple(intervalos)
        self.configuracao_base = configuracao_base or ConfiguracaoPerceptron()
        self.num_repeticoes = num_repeticoes
        self.seed = seed
        self.resultados: dict[ChaveConfiguracao, ResultadoConfiguracao] = {}

    def executar(self, dados: ConjuntoDados) -> dict[ChaveConfiguracao, ResultadoConfiguracao]:
        """Treina `num_repeticoes` perceptrons para cada configuração."""
        rng = np.random.RandomState(self.seed)  # um único gerador, reprodutível via seed
        self.resultados = {}

        for eta, intervalo in product(self.taxas_aprendizado, self.intervalos):
            configuracao = replace(
                self.configuracao_base, taxa_aprendizado=eta, intervalo_pesos=intervalo)
            self.resultados[ChaveConfiguracao(eta, intervalo)] = self._repetir(
                configuracao, dados, rng)

        return self.resultados

    def _repetir(
        self,
        configuracao: ConfiguracaoPerceptron,
        dados: ConjuntoDados,
        rng: np.random.RandomState,
    ) -> ResultadoConfiguracao:
        ajustes, epocas = [], []
        for _ in range(self.num_repeticoes):
            historico = Perceptron(configuracao, rng=rng).fit(dados.X, dados.y).historico
            ajustes.append(historico.total_ajustes)
            epocas.append(historico.epocas_executadas)
        return ResultadoConfiguracao(ajustes=tuple(ajustes), epocas=tuple(epocas))
