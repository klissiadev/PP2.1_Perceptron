"""Modelo Perceptron de Rosenblatt e seus objetos de configuração/histórico."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConfiguracaoPerceptron:
    """Hiperparâmetros do Perceptron (imutáveis e validados na criação).

    Atributos:
        taxa_aprendizado: Taxa de aprendizado (eta).
        intervalo_pesos: Intervalo uniforme [min, max] dos pesos iniciais.
        max_epocas: Teto máximo de épocas de treinamento.
        embaralhar: Se verdadeiro, permuta as amostras a cada época.
        limiar: Limiar da função de ativação degrau.
    """

    taxa_aprendizado: float = 0.1
    intervalo_pesos: tuple[float, float] = (-0.5, 0.5)
    max_epocas: int = 100_000
    embaralhar: bool = False
    limiar: float = 0.0

    def __post_init__(self) -> None:
        if self.taxa_aprendizado <= 0:
            raise ValueError("A taxa de aprendizado deve ser positiva.")
        if self.intervalo_pesos[0] > self.intervalo_pesos[1]:
            raise ValueError("O intervalo de pesos deve ser (mínimo, máximo).")
        if self.max_epocas < 1:
            raise ValueError("max_epocas deve ser pelo menos 1.")


@dataclass(frozen=True)
class HistoricoTreinamento:
    """Resumo imutável de uma execução de treinamento."""

    pesos_iniciais: np.ndarray
    total_ajustes: int
    epocas_executadas: int
    convergiu: bool
    erros_por_epoca: tuple[int, ...]


class Perceptron:
    """Perceptron de Rosenblatt para classificação binária (classes 0 e 1).

    A aleatoriedade (pesos iniciais e embaralhamento) vem de um gerador injetado
    (`rng`), o que evita estado global e permite reproduzir experimentos.
    """

    def __init__(
        self,
        configuracao: ConfiguracaoPerceptron | None = None,
        rng: np.random.RandomState | None = None,
    ) -> None:
        self.configuracao = configuracao or ConfiguracaoPerceptron()
        self._rng = rng if rng is not None else np.random.RandomState()
        self._pesos: np.ndarray | None = None
        self._historico: HistoricoTreinamento | None = None

    # ------------------------------------------------------------ estado
    @property
    def treinado(self) -> bool:
        return self._pesos is not None

    @property
    def pesos(self) -> np.ndarray:
        """Pesos ajustados [w1, w2, ..., bias] (cópia defensiva)."""
        self._exigir_treinado()
        return self._pesos.copy()

    @property
    def historico(self) -> HistoricoTreinamento:
        self._exigir_treinado()
        return self._historico

    def _exigir_treinado(self) -> None:
        if not self.treinado:
            raise RuntimeError("O modelo deve ser treinado via fit(X, y) antes de ser usado.")

    # ------------------------------------------------------------ cálculo
    @staticmethod
    def _adicionar_bias(X: np.ndarray) -> np.ndarray:
        """Concatena coluna de 1.0 correspondente à entrada do bias."""
        return np.column_stack((X, np.ones(X.shape[0])))

    def _funcao_degrau(self, potencial):
        return np.where(potencial >= self.configuracao.limiar, 1, 0)

    def potencial_ativacao(self, X: np.ndarray) -> np.ndarray:
        """Potencial de ativação u = X_bias @ pesos."""
        self._exigir_treinado()
        return self._adicionar_bias(X) @ self._pesos

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Classes previstas (0 ou 1) para cada amostra de X."""
        return self._funcao_degrau(self.potencial_ativacao(X)).astype(int)

    # ------------------------------------------------------------ treino
    def fit(self, X: np.ndarray, y: np.ndarray) -> Perceptron:
        """Ajusta os pesos pela regra delta e registra o histórico de treinamento."""
        if len(X) != len(y):
            raise ValueError("X e y devem ter o mesmo número de amostras.")

        cfg = self.configuracao
        X_bias = self._adicionar_bias(X)

        pesos = self._rng.uniform(*cfg.intervalo_pesos, size=X_bias.shape[1])
        pesos_iniciais = pesos.copy()
        erros_por_epoca: list[int] = []

        for _ in range(cfg.max_epocas):
            erros = self._executar_epoca(X_bias, y, pesos)
            erros_por_epoca.append(erros)
            if erros == 0:
                break

        self._pesos = pesos
        self._historico = HistoricoTreinamento(
            pesos_iniciais=pesos_iniciais,
            total_ajustes=sum(erros_por_epoca),
            epocas_executadas=len(erros_por_epoca),
            convergiu=erros_por_epoca[-1] == 0,
            erros_por_epoca=tuple(erros_por_epoca),
        )
        return self

    def _executar_epoca(self, X_bias: np.ndarray, y: np.ndarray, pesos: np.ndarray) -> int:
        """Percorre as amostras uma vez, atualizando `pesos` in place.

        Retorna a quantidade de classificações incorretas (= ajustes) da época.
        """
        cfg = self.configuracao
        n_amostras = X_bias.shape[0]
        indices = self._rng.permutation(n_amostras) if cfg.embaralhar else range(n_amostras)

        erros = 0
        for i in indices:
            entrada = X_bias[i]
            saida = self._funcao_degrau(pesos @ entrada)
            erro = y[i] - saida
            if erro != 0:
                pesos += cfg.taxa_aprendizado * erro * entrada
                erros += 1
        return erros
