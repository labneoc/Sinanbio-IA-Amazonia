"""
Geradores de dados sintéticos compartilhados pelos demos do SINANBIO.

Tudo aqui é simulação. As "moléculas" são vetores binários que imitam a
estrutura estatística de fingerprints circulares: moléculas de um mesmo
arcabouço (scaffold) compartilham um núcleo de bits e diferem em substituintes.
Essa é a propriedade mínima necessária para que os fenômenos que queremos
demonstrar (vazamento por divisão aleatória, agrupamento em famílias, lacuna
de cobertura química) apareçam de forma honesta.

O que NÃO é: nenhum bit corresponde a uma subestrutura química real, nenhuma
atividade aqui é medida, nenhuma espécie citada teve composto atribuído.

------------------------------------------------------------------------------
Prof. Bruno Duarte Gomes
Faculdade de Biotecnologia, ICB, UFPA
Laboratório de Neurofisiologia Eduardo Oswaldo Cruz (LNEOC) - ICB - UFPA
Laboratório de Simulação e Biologia Computacional (SimBic) - CCAD-IA, UFPA
Licença MIT.
------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_BITS = 512


@dataclass
class Biblioteca:
    """Biblioteca sintética de moléculas.

    fp        : matriz (n_moleculas, n_bits) binária, análoga a um fingerprint
    scaffold  : índice da família/arcabouço de cada molécula
    rotulo    : rótulo binário de atividade (quando gerado)
    potencia  : valor contínuo latente (quando gerado)
    """

    fp: np.ndarray
    scaffold: np.ndarray
    rotulo: np.ndarray | None = None
    potencia: np.ndarray | None = None

    def __len__(self) -> int:
        return self.fp.shape[0]


def tanimoto(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Similaridade de Tanimoto entre conjuntos de vetores binários.

    a: (n, d), b: (m, d) -> (n, m)
    """
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    inter = a @ b.T
    soma_a = a.sum(axis=1)[:, None]
    soma_b = b.sum(axis=1)[None, :]
    uniao = soma_a + soma_b - inter
    return np.divide(inter, uniao, out=np.zeros_like(inter), where=uniao > 0)


def gerar_scaffolds(n_scaffolds: int, rng: np.random.Generator, n_bits: int = N_BITS,
                    densidade_nucleo: float = 0.09) -> np.ndarray:
    """Cria os núcleos (cores) de cada arcabouço: bits que toda a família carrega."""
    return (rng.random((n_scaffolds, n_bits)) < densidade_nucleo).astype(np.uint8)


def gerar_biblioteca(
    n_moleculas: int,
    nucleos: np.ndarray,
    rng: np.random.Generator,
    scaffolds_permitidos: np.ndarray | None = None,
    p_substituinte: float = 0.05,
    p_perda: float = 0.10,
) -> Biblioteca:
    """Povoa famílias de arcabouços com análogos.

    Cada molécula = núcleo do scaffold - algumas perdas + alguns bits de
    decoração. Isso produz alta similaridade intrafamília e baixa interfamília,
    que é a estrutura que torna a divisão por scaffold um teste mais duro.
    """
    n_bits = nucleos.shape[1]
    if scaffolds_permitidos is None:
        scaffolds_permitidos = np.arange(nucleos.shape[0])
    escolha = rng.choice(scaffolds_permitidos, size=n_moleculas, replace=True)
    fp = nucleos[escolha].copy()
    # decoração (substituintes) e perdas
    decoracao = (rng.random((n_moleculas, n_bits)) < p_substituinte).astype(np.uint8)
    fp = np.maximum(fp, decoracao)
    perda = rng.random((n_moleculas, n_bits)) < p_perda
    fp[perda & (fp == 1)] = 0
    return Biblioteca(fp=fp, scaffold=escolha)


def atribuir_atividade(
    bib: Biblioteca,
    rng: np.random.Generator,
    n_bits_causais: int = 12,
    efeito_scaffold: float = 1.6,
    ruido: float = 0.8,
    limiar_quantil: float = 0.85,
) -> Biblioteca:
    """Gera atividade com DUAS fontes: bits causais (generalizáveis) e um efeito
    de arcabouço (não generalizável para famílias novas).

    É exatamente essa mistura que faz um modelo parecer ótimo sob divisão
    aleatória e piorar sob divisão por scaffold: parte do sinal é memorização
    da família, não química transferível.
    """
    n, d = bib.fp.shape
    bits_causais = rng.choice(d, size=n_bits_causais, replace=False)
    pesos = rng.normal(0, 1.0, size=n_bits_causais)
    sinal_quimico = bib.fp[:, bits_causais].astype(float) @ pesos

    n_scaf = int(bib.scaffold.max()) + 1
    vies_scaffold = rng.normal(0, efeito_scaffold, size=n_scaf)
    sinal_familia = vies_scaffold[bib.scaffold]

    potencia = sinal_quimico + sinal_familia + rng.normal(0, ruido, size=n)
    limiar = np.quantile(potencia, limiar_quantil)
    bib.potencia = potencia
    bib.rotulo = (potencia > limiar).astype(int)
    return bib
