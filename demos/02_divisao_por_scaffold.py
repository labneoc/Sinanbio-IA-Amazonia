"""
DEMO 02 - O número que engana: divisão aleatória versus divisão por arcabouço.

Ideia da palestra
-----------------
A maior parte das métricas impressionantes em "IA para produtos naturais" vem
de validação com divisão aleatória. Como análogos de um mesmo arcabouço são
quase idênticos entre si, a divisão aleatória coloca primos da mesma família
nos dois lados da partição: o modelo é avaliado em moléculas que já viu de
outra forma. O AUC sobe, a capacidade real de encontrar uma classe química
NOVA - exatamente o caso amazônico - não sobe junto.

Este é o teste mínimo de honestidade a exigir de qualquer resultado
computacional apresentado como descoberta.

O que este script simula
------------------------
Uma tarefa sintética de classificação de atividade em que o sinal tem duas
componentes: bits causais (transferíveis entre famílias) e um efeito de
arcabouço (memorizável, não transferível). Treina o mesmo modelo sob duas
partições e compara.

Nível de evidência do fenômeno demonstrado: [E4] - o fenômeno de inflação por
divisão aleatória é documentado e reprodutível em benchmarks de
quimioinformática; a demonstração aqui é sobre dado sintético e serve para
tornar o mecanismo visível, não para quantificar o efeito em dado real.

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from sinanbio.style import (DIR_DADOS, OKABE_ITO, aplicar_estilo,
                            aviso_dados_sinteticos, cabecalho, exportar)
from sinanbio.synth import (atribuir_atividade, gerar_biblioteca,
                            gerar_scaffolds, tanimoto)

SEMENTE = 20260926
N_REPETICOES = 15
FRACAO_TESTE = 0.25


def divisao_aleatoria(bib, rng):
    n = len(bib)
    idx = rng.permutation(n)
    corte = int(n * (1 - FRACAO_TESTE))
    return idx[:corte], idx[corte:]


def divisao_por_scaffold(bib, rng):
    """Famílias inteiras vão para treino OU teste, nunca para os dois."""
    familias = np.unique(bib.scaffold)
    rng.shuffle(familias)
    teste_fam, acumulado = [], 0
    alvo = len(bib) * FRACAO_TESTE
    for f in familias:
        if acumulado >= alvo:
            break
        teste_fam.append(f)
        acumulado += int((bib.scaffold == f).sum())
    mascara_teste = np.isin(bib.scaffold, teste_fam)
    return np.where(~mascara_teste)[0], np.where(mascara_teste)[0]


def avaliar(bib, treino, teste, semente):
    modelo = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=semente
    )
    modelo.fit(bib.fp[treino], bib.rotulo[treino])
    p = modelo.predict_proba(bib.fp[teste])[:, 1]
    y = bib.rotulo[teste]
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan, np.nan
    return roc_auc_score(y, p), average_precision_score(y, p)


def main() -> None:
    cabecalho(
        "DEMO 02 · Divisão aleatória vs divisão por arcabouço",
        "[E4] fenômeno documentado em benchmarks; demonstração em dado sintético",
        "Quando um AUC alto significa capacidade de descobrir algo novo?",
    )
    aplicar_estilo()
    rng = np.random.default_rng(SEMENTE)

    nucleos = gerar_scaffolds(80, rng)
    bib = gerar_biblioteca(4000, nucleos, rng)
    bib = atribuir_atividade(bib, rng)

    # Quão parecidas são as moléculas dentro e fora de uma família?
    amostra = rng.choice(len(bib), 600, replace=False)
    sim = tanimoto(bib.fp[amostra], bib.fp[amostra])
    mesma = bib.scaffold[amostra][:, None] == bib.scaffold[amostra][None, :]
    np.fill_diagonal(mesma, False)
    triangular = ~np.eye(len(amostra), dtype=bool)
    sim_intra = sim[mesma]
    sim_inter = sim[triangular & ~mesma]
    print(f"Tanimoto médio intrafamília : {sim_intra.mean():.3f}")
    print(f"Tanimoto médio interfamília : {sim_inter.mean():.3f}")
    print("É essa diferença que a divisão aleatória transforma em vazamento.\n")

    linhas = []
    for r in range(N_REPETICOES):
        rng_r = np.random.default_rng(SEMENTE + r)
        tr, te = divisao_aleatoria(bib, rng_r)
        auc, ap = avaliar(bib, tr, te, SEMENTE + r)
        linhas.append({"repeticao": r, "particao": "aleatória", "auc": auc, "ap": ap})

        rng_r = np.random.default_rng(SEMENTE + r)
        tr, te = divisao_por_scaffold(bib, rng_r)
        auc, ap = avaliar(bib, tr, te, SEMENTE + r)
        linhas.append({"repeticao": r, "particao": "por arcabouço", "auc": auc, "ap": ap})

    df = pd.DataFrame(linhas)
    resumo = df.groupby("particao")[["auc", "ap"]].agg(["mean", "std"])
    print(resumo.round(3).to_string())

    a = df.loc[df.particao == "aleatória", "auc"]
    s = df.loc[df.particao == "por arcabouço", "auc"].dropna()
    print(f"\nQueda média de AUC ao remover o vazamento: "
          f"{a.mean() - s.mean():+.3f} "
          f"({100 * (a.mean() - s.mean()) / a.mean():.1f}% do valor inflado)")
    print("\nRegra prática para o slide: exigir a métrica sob divisão por")
    print("arcabouço (ou por táxon, ou por cluster de rede molecular). A")
    print("diferença entre as duas colunas é o tamanho da ilusão.")

    # ---- Figura ---------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), layout="constrained")

    ax = axes[0]
    ax.hist(sim_inter, bins=40, alpha=0.75, color=OKABE_ITO["cinza"],
            label="pares de famílias diferentes", density=True)
    ax.hist(sim_intra, bins=40, alpha=0.75, color=OKABE_ITO["azul"],
            label="pares da mesma família", density=True, hatch="//",
            edgecolor="white")
    ax.set(xlabel="Similaridade de Tanimoto", ylabel="Densidade",
           title="Análogos são quase cópias")
    ax.legend()
    aviso_dados_sinteticos(ax)

    ax = axes[1]
    posicoes = [0, 1]
    dados = [df.loc[df.particao == p, "auc"].dropna().values
             for p in ["aleatória", "por arcabouço"]]
    cores = [OKABE_ITO["vermelho"], OKABE_ITO["azul"]]
    for pos, valores, cor in zip(posicoes, dados, cores):
        ax.scatter(np.full(len(valores), pos) + rng.normal(0, 0.05, len(valores)),
                   valores, color=cor, alpha=0.75, s=40, zorder=3)
        ax.plot([pos - 0.22, pos + 0.22], [valores.mean()] * 2, color="black",
                linewidth=2.5, zorder=4)
        ax.annotate(f"{valores.mean():.3f}", (pos, valores.mean()),
                    textcoords="offset points", xytext=(30, -4), fontsize=11)
    ax.axhline(0.5, color=OKABE_ITO["cinza"], linestyle=":", linewidth=1.5)
    ax.text(1.35, 0.51, "acaso", color=OKABE_ITO["cinza"], fontsize=10)
    ax.set(xticks=posicoes, xticklabels=["divisão\naleatória", "divisão por\narcabouço"],
           ylabel="AUC-ROC no conjunto de teste", ylim=(0.4, 1.02),
           title=f"O mesmo modelo, duas partições (n={N_REPETICOES})")
    aviso_dados_sinteticos(ax)

    ax = axes[2]
    largura = 0.35
    for i, metrica in enumerate(["auc", "ap"]):
        vals = [df.loc[df.particao == p, metrica].dropna()
                for p in ["aleatória", "por arcabouço"]]
        ax.bar(i - largura / 2, vals[0].mean(), largura, yerr=vals[0].std(),
               color=OKABE_ITO["vermelho"], capsize=4,
               label="aleatória" if i == 0 else None)
        ax.bar(i + largura / 2, vals[1].mean(), largura, yerr=vals[1].std(),
               color=OKABE_ITO["azul"], capsize=4, hatch="//", edgecolor="white",
               label="por arcabouço" if i == 0 else None)
    ax.set(xticks=[0, 1], xticklabels=["AUC-ROC", "precisão média"],
           ylabel="Desempenho (média ± DP)", ylim=(0, 1.05),
           title="A ilusão aparece em qualquer métrica")
    ax.legend(loc="upper right")
    aviso_dados_sinteticos(ax)

    caminho = exportar(
        fig, "02_divisao_por_scaffold",
        proveniencia={
            "script": "demos/02_divisao_por_scaffold.py",
            "semente": SEMENTE,
            "repeticoes": N_REPETICOES,
            "modelo": "RandomForestClassifier(n_estimators=300, min_samples_leaf=2)",
            "incerteza": "desvio-padrão entre repetições (não é IC)",
        },
    )
    print(f"\nFigura: {caminho}")

    DIR_DADOS.mkdir(parents=True, exist_ok=True)
    destino = DIR_DADOS / "02_metricas_por_particao.csv"
    df.to_csv(destino, index=False)
    print(f"Tabela: {destino}")


if __name__ == "__main__":
    main()
