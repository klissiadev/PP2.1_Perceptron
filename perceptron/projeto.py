"""Identificação do grupo e resolução dos arquivos de dados do trabalho."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar


@dataclass(frozen=True)
class Integrante:
    """Membro do grupo, identificado por nome e matrícula."""

    nome: str
    matricula: str

    @property
    def digito_final(self) -> int:
        return int(self.matricula[-1])


@dataclass(frozen=True)
class Grupo:
    """Equipe do trabalho; deriva o identificador que seleciona o arquivo da Parte II."""

    integrantes: tuple[Integrante, ...]

    MODULO: ClassVar[int] = 4

    @property
    def identificador(self) -> int:
        return sum(i.digito_final for i in self.integrantes) % self.MODULO


@dataclass(frozen=True)
class ArquivosDados:
    """Resolve os caminhos dos arquivos de dados utilizados em cada parte."""

    identificador: int
    diretorio: Path = Path(".")

    SUFIXO: ClassVar[str] = "20262"

    @property
    def parte_1(self) -> Path:
        return self.diretorio / f"dataAll{self.SUFIXO}.txt"

    @property
    def parte_2(self) -> Path:
        return self.diretorio / f"data{self.identificador}{self.SUFIXO}.txt"

    @property
    def parte_3(self) -> Path:
        return self.diretorio / f"dataHoldout{self.SUFIXO}.txt"

    def todos(self) -> tuple[Path, ...]:
        return (self.parte_1, self.parte_2, self.parte_3)

    def verificar_existencia(self) -> dict[Path, bool]:
        return {caminho: caminho.exists() for caminho in self.todos()}
