"""
DEMO 01 - A Amazônia como problema de DADOS, não apenas de biodiversidade.

Ideia da palestra
-----------------
Um modelo de IA só enxerga o espaço químico que existe no seu conjunto de
treino. Se a química tropical está sub-representada nas bases que alimentam
esses modelos, então a molécula amazônica cai fora do domínio de aplicabilidade
do modelo: a predição continua saindo, mas com incerteza que o modelo não
declara. A pergunta deixa de ser "qual o melhor algoritmo" e passa a ser "de
quem é o dado".

O que este script simula
------------------------
1. Um "acervo global" de moléculas organizado em famílias de arcabouço.
2. Um "acervo amazônico" pequeno, que ocupa parcialmente famílias ausentes do
   acervo global (novidade de arcabouço).
3. Mede: cobertura por vizinho mais próximo (Tanimoto), fração de moléculas
   amazônicas fora do domínio de aplicabilidade e o efeito do tamanho amostral.

O que este script NÃO é
-----------------------
Não há molécula real, nem base real consultada, nem estimativa de quantos
metabólitos amazônicos estão descritos. Os números do slide sobre bases reais
(NuBBE DB, COCONUT, LOTUS, NPAtlas, GNPS, MIBiG) devem ser verificados na
fonte primária, com data de acesso. Aqui só se demonstra o MECANISMO.

Nível de evidência do que se demonstra: [E5] - ilustração conceitual.

------------------------------------------------------------------------------
Prof. Bruno Duarte Gomes
Faculdade de Biotecnologia, ICB, UFPA
Laboratório de Neurofisiologia Eduardo Oswaldo Cruz (LNEOC) - ICB - UFPA
Laboratório de Simulação e Biologia Computacional (SimBic) - CCAD-IA, UFPA
Licença MIT.
------------------------------------------------------------------------------
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sinanbio.style import (DIR_DADOS, OKABE_ITO, aplicar_estilo,
                            aviso_dados_sinteticos, cabecalho, exportar)
from sinanbio.synth import gerar_biblioteca, gerar_scaffolds, tanimoto

SEMENTE = 20260926
LIMIAR_DOMINIO = 0.40  # Tanimoto abaixo do qual consideramos "fora do domínio"


def main() -> None:
    cabecalho(
        "DEMO 01 · Lacuna de cobertura química",
        "[E5] ilustração conceitual sobre dados sintéticos",
        "O gargalo da IA para a Amazônia é o algoritmo ou o dado?",
    )
    aplicar_estilo()
    rng = np.random.default_rng(SEMENTE)

    # ---- Acervo global e acervo amazônico -------------------------------
    # Famílias 0-89: já amostradas globalmente.
    # Famílias 90-119: exclusivas do acervo amazônico e geradas sob outro
    # regime (núcleos mais densos), caricatura da observação recorrente de que
    # produtos naturais tendem a ser maiores, mais oxigenados e mais ricos em
    # centros sp3 que as coleções sintéticas de triagem. Trate como premissa
    # da simulação, não como medida: o dado empírico precisa ser verificado
    # na literatura antes de ir ao slide.
    nucleos = np.vstack([
        gerar_scaffolds(90, rng, densidade_nucleo=0.09),
        gerar_scaffolds(30, rng, densidade_nucleo=0.16),
    ])

    scaffolds_globais = np.arange(0, 90)
    scaffolds_amazonia = np.arange(70, 120)     # sobreposição parcial + 30 exclusivas

    global_ = gerar_biblioteca(6000, nucleos, rng, scaffolds_globais)
    amazonia = gerar_biblioteca(450, nucleos, rng, scaffolds_amazonia)

    # ---- Cobertura por vizinho mais próximo -----------------------------
    sim = tanimoto(amazonia.fp, global_.fp)
    viz_max = sim.max(axis=1)
    fora_dominio = viz_max < LIMIAR_DOMINIO
    novidade_scaffold = ~np.isin(amazonia.scaffold, scaffolds_globais)

    print(f"Moléculas no acervo global sintético : {len(global_):>6d}")
    print(f"Moléculas no acervo amazônico        : {len(amazonia):>6d} "
          f"({100 * len(amazonia) / (len(global_) + len(amazonia)):.1f}% do total)")
    print(f"Tanimoto ao vizinho mais próximo     : mediana {np.median(viz_max):.3f}")
    print(f"Fora do domínio (Tanimoto < {LIMIAR_DOMINIO})   : "
          f"{fora_dominio.mean() * 100:.1f}% das moléculas amazônicas")
    print(f"Arcabouço inédito no acervo global   : {novidade_scaffold.mean() * 100:.1f}%")
    print()
    print("Leitura: a fração fora do domínio é a fração sobre a qual QUALQUER")
    print("modelo treinado no acervo global extrapola. Extrapolação não é erro")
    print("do modelo; é ausência de dado. Corrige-se com coleta e depósito, não")
    print("com mais parâmetros.")

    # ---- Curva: quanto a cobertura melhora ao depositar dados amazônicos --
    tamanhos = [0, 25, 50, 100, 200, 400, 800, 1600]
    reforco = gerar_biblioteca(1600, nucleos, rng, scaffolds_amazonia)
    consulta = gerar_biblioteca(300, nucleos, rng, scaffolds_amazonia)
    cobertura = []
    for k in tamanhos:
        base = global_.fp if k == 0 else np.vstack([global_.fp, reforco.fp[:k]])
        vmax = tanimoto(consulta.fp, base).max(axis=1)
        cobertura.append(float((vmax >= LIMIAR_DOMINIO).mean() * 100))

    # ---- Figura ----------------------------------------------------------
    # Mapa com eixos interpretáveis em vez de uma projeção 2D genérica.
    # Nota metodológica que vale para o slide: mapas de espaço químico em
    # t-SNE/UMAP/PCA comprimem dezenas de famílias em duas dimensões e
    # rotineiramente escondem justamente a novidade - ela colapsa no centro.
    # Aqui os eixos são quantidades que se pode auditar.
    idx_g = rng.choice(len(global_), size=900, replace=False)
    sim_gg = tanimoto(global_.fp[idx_g], global_.fp)
    sim_gg.sort(axis=1)
    viz_global = sim_gg[:, -2]  # exclui a própria molécula
    complexidade_g = global_.fp[idx_g].sum(axis=1)
    complexidade_a = amazonia.fp.sum(axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), layout="constrained")

    ax = axes[0]
    ax.scatter(complexidade_g, viz_global, s=8, alpha=0.30,
               color=OKABE_ITO["cinza"], marker="o",
               label="Acervo global (amostra n=900)")
    ax.scatter(complexidade_a[~fora_dominio], viz_max[~fora_dominio], s=26,
               color=OKABE_ITO["azul"], marker="^", label="Amazônia · no domínio")
    ax.scatter(complexidade_a[fora_dominio], viz_max[fora_dominio], s=34,
               color=OKABE_ITO["vermelho"], marker="X",
               label="Amazônia · fora do domínio")
    ax.axhline(LIMIAR_DOMINIO, color=OKABE_ITO["vermelho"], linestyle="--",
               linewidth=1.5)
    ax.set(xlabel="Bits ativos (proxy de complexidade estrutural)",
           ylabel="Tanimoto ao vizinho mais próximo conhecido",
           title="Onde o modelo enxerga e onde extrapola")
    ax.legend(loc="lower left")
    aviso_dados_sinteticos(ax)

    ax = axes[1]
    ax.hist(sim.max(axis=1), bins=30, color=OKABE_ITO["azul"], alpha=0.85,
            edgecolor="white")
    ax.axvline(LIMIAR_DOMINIO, color=OKABE_ITO["vermelho"], linestyle="--",
               linewidth=2)
    ax.annotate(f"limiar de domínio = {LIMIAR_DOMINIO}",
                xy=(LIMIAR_DOMINIO, ax.get_ylim()[1] * 0.85),
                xytext=(LIMIAR_DOMINIO + 0.07, ax.get_ylim()[1] * 0.85),
                color=OKABE_ITO["vermelho"], fontsize=10)
    ax.set(xlabel="Tanimoto ao vizinho mais próximo do acervo global",
           ylabel="Nº de moléculas amazônicas",
           title="Distância ao conhecido")
    aviso_dados_sinteticos(ax)

    ax = axes[2]
    ax.plot(tamanhos, cobertura, marker="o", color=OKABE_ITO["verde"])
    ax.set(xlabel="Moléculas amazônicas depositadas na base",
           ylabel="Cobertura de novas amostras (%)",
           title="Depositar dado > ajustar modelo", ylim=(0, 112))
    for x, y in zip(tamanhos, cobertura):
        if x in (0, 100, 400, 1600):
            ax.annotate(f"{y:.0f}%", (x, y), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=10)
    aviso_dados_sinteticos(ax)

    caminho = exportar(
        fig, "01_lacuna_amazonica",
        proveniencia={
            "script": "demos/01_lacuna_amazonica.py",
            "semente": SEMENTE,
            "limiar_dominio_tanimoto": LIMIAR_DOMINIO,
            "transformacoes": ["PCA 2 componentes sobre fingerprint binário sintético"],
        },
    )
    print(f"\nFigura: {caminho}")

    DIR_DADOS.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "id_molecula": np.arange(len(amazonia)),
        "scaffold": amazonia.scaffold,
        "tanimoto_vizinho_global": viz_max,
        "fora_do_dominio": fora_dominio,
        "arcabouco_inedito": novidade_scaffold,
    })
    destino = DIR_DADOS / "01_acervo_amazonico_sintetico.csv"
    df.to_csv(destino, index=False)
    print(f"Tabela: {destino}")


if __name__ == "__main__":
    main()
