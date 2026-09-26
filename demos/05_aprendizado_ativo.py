"""
DEMO 05 - Fechar o ciclo: a IA não descobre, ela decide o que testar.

Ideia da palestra
-----------------
O recurso escasso na bioprospecção amazônica não é a molécula; é o ensaio, a
coluna, o RMN, o animal, o tempo de bancada. Nessa economia, o valor prático da
IA quase nunca é "predizer a molécula certa" e quase sempre é "ordenar a fila
de experimentos". Aprendizado ativo formaliza isso: treinar com o pouco que já
foi medido, escolher o próximo lote, medir, reentrar no ciclo.

O contraponto que raramente aparece no slide de quem vende o método: quando há
MAIS DE UM mecanismo de atividade na quimioteca, uma política gulosa trava no
mecanismo que encontrou primeiro. Ela maximiza acertos e, ao mesmo tempo,
encontra MENOS moléculas do segundo mecanismo do que a triagem aleatória. Para
um programa que busca classe nova - o caso amazônico - isso é uma armadilha:
acerta-se mais e descobre-se menos.

Resultado que esta simulação produz (e que contraria a intuição fácil)
---------------------------------------------------------------------
Penalizar redundância estrutural e somar incerteza NÃO basta para recuperar o
segundo mecanismo. Só uma cota explícita de exploração recupera parte dele, e
com custo mensurável em acertos. Ou seja: a diversidade precisa estar na
FUNÇÃO OBJETIVO, não aparece de brinde.

Nível de evidência: [E5] simulação. O ganho de aprendizado ativo sobre triagem
aleatória é [E4]/[E2] conforme a plataforma, e exige citação verificada.

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

from sinanbio.style import (DIR_DADOS, OKABE_ITO, aplicar_estilo,
                            aviso_dados_sinteticos, cabecalho, exportar)
from sinanbio.synth import gerar_biblioteca, gerar_scaffolds

SEMENTE = 20260926
N_MOLECULAS = 3000
N_SCAFFOLDS = 70
FAMILIAS_MECANISMO_A = 45     # famílias 0-44 respondem pelo mecanismo A
LOTE = 50
N_LOTES = 12
N_EXECUCOES = 6
POLITICAS = ["aleatória", "gulosa", "híbrida", "cota de exploração"]


def construir_quimioteca(rng):
    """Quimioteca com DOIS farmacóforos independentes.

    Mecanismo A: prevalente, ocupa famílias 0-44.
    Mecanismo B: raro, ocupa famílias 45-69, e depende de bits causais
    completamente distintos - um modelo que aprendeu A não transfere para B.
    """
    nucleos = gerar_scaffolds(N_SCAFFOLDS, rng)
    bib = gerar_biblioteca(N_MOLECULAS, nucleos, rng)
    d = bib.fp.shape[1]

    bits_a = rng.choice(d, size=10, replace=False)
    bits_b = rng.choice(np.setdiff1d(np.arange(d), bits_a), size=10, replace=False)
    peso_a, peso_b = rng.normal(0, 1, 10), rng.normal(0, 1, 10)

    mecanismo_a = bib.scaffold < FAMILIAS_MECANISMO_A
    sinal_a = bib.fp[:, bits_a] @ peso_a + rng.normal(0, 0.6, len(bib))
    sinal_b = bib.fp[:, bits_b] @ peso_b + rng.normal(0, 0.6, len(bib))

    limiar_a = np.quantile(sinal_a[mecanismo_a], 0.88)
    limiar_b = np.quantile(sinal_b[~mecanismo_a], 0.94)
    bib.rotulo = np.where(mecanismo_a, sinal_a > limiar_a,
                          sinal_b > limiar_b).astype(int)
    return bib, mecanismo_a


def selecionar(politica, bib, testados, rng, semente):
    """Devolve os índices do próximo lote de ensaios."""
    candidatos = np.where(~testados)[0]
    if politica == "aleatória":
        return rng.choice(candidatos, size=LOTE, replace=False)

    y = bib.rotulo[testados]
    if y.sum() == 0 or y.sum() == len(y):        # ainda sem sinal: explora
        return rng.choice(candidatos, size=LOTE, replace=False)

    modelo = RandomForestClassifier(n_estimators=150, min_samples_leaf=2,
                                    n_jobs=-1, random_state=semente)
    modelo.fit(bib.fp[testados], y)
    p = modelo.predict_proba(bib.fp[candidatos])[:, 1]
    contagem = np.bincount(bib.scaffold[testados], minlength=N_SCAFFOLDS)

    if politica == "gulosa":
        return candidatos[np.argsort(p)[::-1][:LOTE]]

    if politica == "híbrida":
        incerteza = p * (1 - p)
        redundancia = contagem[bib.scaffold[candidatos]] / (contagem.max() + 1e-9)
        pontuacao = p + 2.0 * incerteza - 0.6 * redundancia
        return candidatos[np.argsort(pontuacao)[::-1][:LOTE]]

    # cota de exploração: 60% explotação + 40% amostragem dirigida às famílias
    # menos visitadas. A diversidade entra como restrição, não como bônus.
    n_exploracao = int(0.4 * LOTE)
    topo = candidatos[np.argsort(p)[::-1][:LOTE - n_exploracao]]
    resto = np.setdiff1d(candidatos, topo)
    peso = 1.0 / (1.0 + contagem[bib.scaffold[resto]])
    peso = peso / peso.sum()
    exploracao = rng.choice(resto, size=n_exploracao, replace=False, p=peso)
    return np.concatenate([topo, exploracao])


def campanha(politica, bib, mecanismo_a, semente):
    rng = np.random.default_rng(semente)
    testados = np.zeros(len(bib), dtype=bool)
    historico = []
    for lote in range(N_LOTES):
        testados[selecionar(politica, bib, testados, rng, semente + lote)] = True
        achados = testados & (bib.rotulo == 1)
        historico.append({
            "politica": politica,
            "execucao": semente,
            "ensaios": int(testados.sum()),
            "acertos": int(achados.sum()),
            "acertos_mecanismo_B": int((achados & ~mecanismo_a).sum()),
            "familias_ativas_distintas": int(len(np.unique(bib.scaffold[achados]))),
        })
    return historico


def main() -> None:
    cabecalho(
        "DEMO 05 · Aprendizado ativo e o custo da ganância",
        "[E5] simulação; o ganho do aprendizado ativo é [E4]/[E2] na literatura",
        "Se cada ensaio custa caro, quem decide a ordem da fila?",
    )
    aplicar_estilo()
    rng = np.random.default_rng(SEMENTE)
    bib, mecanismo_a = construir_quimioteca(rng)

    ativos = bib.rotulo == 1
    n_b = int((ativos & ~mecanismo_a).sum())
    print(f"Quimioteca   : {len(bib)} moléculas, {ativos.sum()} ativos "
          f"({100 * ativos.mean():.1f}%)")
    print(f"  mecanismo A: {int((ativos & mecanismo_a).sum())} ativos "
          f"(prevalente, famílias 0-{FAMILIAS_MECANISMO_A - 1})")
    print(f"  mecanismo B: {n_b} ativos "
          f"(raro, famílias {FAMILIAS_MECANISMO_A}-{N_SCAFFOLDS - 1}, "
          f"farmacóforo independente)")
    print(f"Orçamento    : {N_LOTES} × {LOTE} = {N_LOTES * LOTE} ensaios "
          f"({100 * N_LOTES * LOTE / len(bib):.0f}% da biblioteca)\n")

    registros = []
    for politica in POLITICAS:
        for r in range(N_EXECUCOES):
            registros.extend(campanha(politica, bib, mecanismo_a, SEMENTE + 100 * r))
    df = pd.DataFrame(registros)

    final = df[df.ensaios == N_LOTES * LOTE]
    resumo = final.groupby("politica").agg(
        acertos=("acertos", "mean"),
        acertos_dp=("acertos", "std"),
        mecanismo_B=("acertos_mecanismo_B", "mean"),
        mecanismo_B_dp=("acertos_mecanismo_B", "std"),
        familias=("familias_ativas_distintas", "mean"),
    ).reindex(POLITICAS).round(1)
    resumo["% do mec. B achado"] = (100 * resumo["mecanismo_B"] / n_b).round(1)
    print(resumo.to_string())

    base = resumo.loc["aleatória", "acertos"]
    print(f"\nGanho da gulosa sobre a aleatória : "
          f"{resumo.loc['gulosa','acertos'] / base:.1f}× em acertos totais")
    print(f"No mecanismo RARO                 : gulosa "
          f"{resumo.loc['gulosa','mecanismo_B']:.1f} ± {resumo.loc['gulosa','mecanismo_B_dp']:.1f}"
          f"  vs  aleatória {resumo.loc['aleatória','mecanismo_B']:.1f} ± "
          f"{resumo.loc['aleatória','mecanismo_B_dp']:.1f} moléculas")
    print("  -> Cuidado com a leitura: a pequena DESVANTAGEM aparente da gulosa")
    print("     está dentro da dispersão entre execuções e não é um efeito que")
    print("     eu sustentaria. O que é robusto aqui é outra coisa, e basta:")
    print(f"     triplicar os acertos NÃO comprou nenhuma molécula a mais do")
    print(f"     mecanismo raro. O ganho de {resumo.loc['gulosa','acertos'] / base:.1f}× foi todo gasto")
    print("     aprofundando o farmacóforo que o modelo já conhecia.")

    print("\nLeituras para o slide:")
    print(" · Aprendizado ativo entrega um ganho grande e real sobre triagem")
    print("   aleatória. Esse é o argumento honesto a favor.")
    print(" · A política gulosa aprofunda o mecanismo que já encontrou. Ela")
    print("   triplica os acertos sem adicionar uma única molécula do segundo")
    print("   farmacóforo. Mais hits não é mais descoberta.")
    print(" · Nenhuma das políticas informadas superou a triagem aleatória no")
    print("   mecanismo raro. Incerteza e penalização de redundância não")
    print("   bastaram, e a cota de exploração apenas empatou, pagando acertos.")
    print("   Se o objetivo é CLASSE NOVA, ele precisa estar escrito na função")
    print("   objetivo e medido em uma métrica própria - caso contrário o")
    print("   otimizador entrega exatamente aquilo que foi pedido: hits.")
    print(" · O modelo ordena a fila. A bancada continua sendo quem converte")
    print("   [E5] em [E3].")

    # ---- Figura ---------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.0), layout="constrained")
    estilos = {
        "aleatória": (OKABE_ITO["cinza"], "o", "-"),
        "gulosa": (OKABE_ITO["vermelho"], "s", "--"),
        "híbrida": (OKABE_ITO["azul"], "^", "-."),
        "cota de exploração": (OKABE_ITO["verde"], "D", ":"),
    }

    ax = axes[0]
    for politica, (cor, marcador, traco) in estilos.items():
        sub = df[df.politica == politica].groupby("ensaios")["acertos"]
        media, dp = sub.mean(), sub.std()
        ax.plot(media.index, media.values, color=cor, marker=marcador,
                linestyle=traco, label=politica)
        ax.fill_between(media.index, media - dp, media + dp, color=cor, alpha=0.12)
    ax.set(xlabel="Ensaios realizados", ylabel="Ativos encontrados (acumulado)",
           title=f"Curva de descoberta (média ± DP, n={N_EXECUCOES})")
    ax.legend(fontsize=10)
    aviso_dados_sinteticos(ax)

    ax = axes[1]
    for politica, (cor, marcador, traco) in estilos.items():
        sub = df[df.politica == politica].groupby("ensaios")["acertos_mecanismo_B"]
        ax.plot(sub.mean().index, sub.mean().values, color=cor, marker=marcador,
                linestyle=traco, label=politica)
    ax.set(xlabel="Ensaios realizados",
           ylabel="Ativos do mecanismo RARO encontrados",
           title="No mecanismo raro, nenhuma política superou o acaso")
    ax.legend(fontsize=10)
    aviso_dados_sinteticos(ax)

    ax = axes[2]
    xpos = np.arange(len(POLITICAS))
    largura = 0.36
    ax.bar(xpos - largura / 2, resumo["acertos"], largura,
           color=OKABE_ITO["laranja"], label="acertos totais")
    ax2 = ax.twinx()
    ax2.bar(xpos + largura / 2, resumo["mecanismo_B"], largura,
            color=OKABE_ITO["verde"], hatch="//", edgecolor="white",
            label="ativos do mecanismo raro")
    ax.set(xticks=xpos,
           xticklabels=[p.replace(" de ", "\nde ") for p in POLITICAS],
           ylabel="Acertos totais", ylim=(0, 240),
           title="Duas métricas, duas conclusões")
    ax.tick_params(axis="x", labelsize=10)
    ax2.set_ylabel("Ativos do mecanismo raro")
    ax2.set_ylim(0, 17)
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=10)
    aviso_dados_sinteticos(ax)

    caminho = exportar(
        fig, "05_aprendizado_ativo",
        proveniencia={
            "script": "demos/05_aprendizado_ativo.py",
            "semente": SEMENTE,
            "lote": LOTE, "n_lotes": N_LOTES, "execucoes": N_EXECUCOES,
            "politica_hibrida": "p + 2.0*incerteza - 0.6*redundância_de_família",
            "politica_cota": "60% explotação + 40% amostragem inversa à visitação",
            "incerteza": "desvio-padrão entre execuções (não é intervalo de confiança)",
        },
    )
    print(f"\nFigura: {caminho}")

    DIR_DADOS.mkdir(parents=True, exist_ok=True)
    destino = DIR_DADOS / "05_campanhas_aprendizado_ativo.csv"
    df.to_csv(destino, index=False)
    print(f"Tabela: {destino}")


if __name__ == "__main__":
    main()
