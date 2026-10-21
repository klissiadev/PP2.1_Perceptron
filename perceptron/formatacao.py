"""Apresentação tabular de resultados (separada da lógica de cálculo)."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from prettytable import PrettyTable

from .avaliacao import MetricasClassificacao
from .experimento import ChaveConfiguracao, ResultadoConfiguracao


class FormatadorExperimento:
    """Tabela dos resultados do experimento de configurações."""

    @staticmethod
    def tabela(resultados: Mapping[ChaveConfiguracao, ResultadoConfiguracao]) -> PrettyTable:
        tabela = PrettyTable()
        tabela.field_names = ["Taxa de Aprendizado", "Intervalo de Pesos",
                              "Qtde Ajustes", "Min Épocas", "Max Épocas"]

        for chave, r in resultados.items():
            minimo, maximo = chave.intervalo_pesos
            tabela.add_row([
                f"η = {chave.taxa_aprendizado}",
                f"({minimo}, +{maximo})",
                f"{r.media_ajustes:.2f} ± {r.desvio_padrao_ajustes:.2f}",
                r.min_epocas,
                r.max_epocas,
            ])
        return tabela


class FormatadorAvaliacao:
    """Tabelas da avaliação de desempenho do classificador."""

    @staticmethod
    def matriz_confusao(matriz: np.ndarray) -> PrettyTable:
        tabela = PrettyTable()
        tabela.field_names = ["Real \\ Previsto", "Previsto 0", "Previsto 1"]
        tabela.add_row(["Real 0", f"{matriz[0, 0]} (VN)", f"{matriz[0, 1]} (FP)"])
        tabela.add_row(["Real 1", f"{matriz[1, 0]} (FN)", f"{matriz[1, 1]} (VP)"])
        return tabela

    @staticmethod
    def metricas(metricas_por_conjunto: Mapping[str, MetricasClassificacao]) -> PrettyTable:
        tabela = PrettyTable()
        tabela.field_names = ["Conjunto", "Acurácia", "Precisão", "Revocação", "F-Score"]
        for nome, m in metricas_por_conjunto.items():
            tabela.add_row([nome, f"{m.acuracia:.4f}", f"{m.precisao:.4f}",
                            f"{m.revocacao:.4f}", f"{m.f_score:.4f}"])
        return tabela
