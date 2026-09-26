"""
DEMO 06 - Barreira hematoencefálica: a ponte com a neurociência, e onde ela range.

Ideia da palestra
-----------------
Um composto neuroativo amazônico só interessa se chegar ao encéfalo. Predizer
permeabilidade da barreira hematoencefálica (BHE) é, à primeira vista, o caso
fácil da quimioinformática: existem descritores simples e correlatos
conhecidos (massa, área polar topológica, lipofilia). Três coisas estragam
essa impressão, e todas as três importam para o caso amazônico:

 1. Os conjuntos de treino públicos são dominados por fármacos sintéticos de
    massa moderada. Produtos naturais tendem a ocupar outra região - mais
    massa, mais área polar. O modelo não erra por incompetência; ele erra por
    DESLOCAMENTO DE DOMÍNIO, e erra com confiança alta.
 2. Parte do fenômeno é transporte ativo e efluxo (P-gp e congêneres), que não
    é função dos descritores fisico-químicos usuais. É incerteza irredutível
    para esse conjunto de atributos - nenhuma arquitetura a elimina.
 3. Permeabilidade não é atividade central: atravessar não é agir. Um modelo
    de BHE bem calibrado ainda deixa intocado o problema do alvo.

O que este script simula
------------------------
Um preditor de BHE treinado em espaço "tipo fármaco" e avaliado em dois
regimes: no mesmo domínio e em moléculas "tipo produto natural". Mede acurácia,
CALIBRAÇÃO (diagrama de confiabilidade e erro de calibração esperado, ECE) e
PREDIÇÃO SELETIVA (curva risco-cobertura com abstenção por confiança).

Nível de evidência: [E5] simulação didática. As relações entre TPSA, massa,
lipofilia e permeabilidade da BHE, bem como o papel do efluxo, são [E2]/[E3]
na literatura e devem entrar no slide com citação primária verificada.

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
from sklearn.ensemble import GradientBoostingClassifier

from sinanbio.style import (DIR_DADOS, OKABE_ITO, aplicar_estilo,
                            aviso_dados_sinteticos, cabecalho, exportar)

SEMENTE = 20260926
ATRIBUTOS = ["massa", "tpsa", "logp", "hbd", "rotativas"]


def sigmoide(x):
    return 1.0 / (1.0 + np.exp(-x))


def amostrar(n, regime, rng):
    """Gera descritores em dois regimes de espaço químico."""
    if regime == "tipo_farmaco":
        massa = rng.normal(350, 60, n)
        tpsa = rng.normal(62, 20, n)
        logp = rng.normal(2.5, 1.0, n)
    else:  # tipo_produto_natural
        massa = rng.normal(520, 90, n)
        tpsa = rng.normal(112, 30, n)
        logp = rng.normal(2.0, 1.4, n)
    massa = np.clip(massa, 120, 900)
    tpsa = np.clip(tpsa, 5, 260)
    hbd = np.clip(np.round(tpsa / 26 + rng.normal(0, 0.8, n)), 0, 12)
    rotativas = np.clip(np.round(massa / 65 + rng.normal(0, 1.5, n)), 0, 25)
    return pd.DataFrame({"massa": massa, "tpsa": tpsa, "logp": logp,
                         "hbd": hbd, "rotativas": rotativas})


def rotular(x, rng):
    """Verdade-terreno sintética.

    Difusão passiva favorecida por baixa TPSA, massa moderada e lipofilia
    intermediária (termo quadrático: muito lipofílico também penaliza).
    O termo de EFLUXO é uma variável latente que o modelo nunca observa -
    é a incerteza irredutível para esse conjunto de descritores.
    """
    efluxo = (rng.random(len(x)) < sigmoide(0.010 * (x["massa"] - 420))).astype(float)
    logito = (2.6
              - 0.026 * (x["tpsa"] - 60)
              - 0.0045 * (x["massa"] - 350)
              + 0.55 * x["logp"]
              - 0.13 * x["logp"] ** 2
              - 2.3 * efluxo)
    p = sigmoide(logito)
    return (rng.random(len(x)) < p).astype(int), p.values, efluxo


def ece(y, p, n_bins=10):
    """Erro de calibração esperado, com os bins explícitos (não é métrica mágica)."""
    bordas = np.linspace(0, 1, n_bins + 1)
    total, soma = len(y), 0.0
    pontos = []
    for i in range(n_bins):
        sel = (p > bordas[i]) & (p <= bordas[i + 1]) if i else (p <= bordas[1])
        if sel.sum() == 0:
            pontos.append((np.nan, np.nan, 0))
            continue
        conf, freq = p[sel].mean(), y[sel].mean()
        soma += sel.sum() / total * abs(conf - freq)
        pontos.append((conf, freq, int(sel.sum())))
    return soma, pontos


def curva_risco_cobertura(y, p):
    """Acurácia em função da cobertura, abstendo-se das predições menos confiantes."""
    confianca = np.maximum(p, 1 - p)
    ordem = np.argsort(confianca)[::-1]
    acertou = ((p >= 0.5).astype(int) == y)[ordem]
    cobertura = np.arange(1, len(y) + 1) / len(y)
    acuracia = np.cumsum(acertou) / np.arange(1, len(y) + 1)
    return cobertura, acuracia


def main() -> None:
    cabecalho(
        "DEMO 06 · BHE, calibração e o direito de dizer 'não sei'",
        "[E5] simulação; relações TPSA/massa/efluxo-BHE são [E2]/[E3] na literatura",
        "Um modelo confiante fora do seu domínio é pior que nenhum modelo?",
    )
    aplicar_estilo()
    rng = np.random.default_rng(SEMENTE)

    treino_x = amostrar(4000, "tipo_farmaco", rng)
    treino_y, _, _ = rotular(treino_x, rng)
    teste_dom_x = amostrar(1500, "tipo_farmaco", rng)
    teste_dom_y, p_verdadeiro_dom, _ = rotular(teste_dom_x, rng)
    teste_pn_x = amostrar(1500, "tipo_produto_natural", rng)
    teste_pn_y, p_verdadeiro_pn, efluxo_pn = rotular(teste_pn_x, rng)

    modelo = GradientBoostingClassifier(random_state=SEMENTE)
    modelo.fit(treino_x[ATRIBUTOS], treino_y)

    resultados = {}
    for nome, x, y in [("mesmo domínio", teste_dom_x, teste_dom_y),
                       ("tipo produto natural", teste_pn_x, teste_pn_y)]:
        p = modelo.predict_proba(x[ATRIBUTOS])[:, 1]
        acuracia = ((p >= 0.5).astype(int) == y).mean()
        e, pontos = ece(y, p)
        confianca_media = np.maximum(p, 1 - p).mean()
        resultados[nome] = {"p": p, "y": y, "acuracia": acuracia, "ece": e,
                            "pontos": pontos, "confianca": confianca_media}
        print(f"{nome:<22s} acurácia {acuracia:.3f} | ECE {e:.3f} | "
              f"confiança média declarada {confianca_media:.3f}")

    d = resultados["mesmo domínio"]
    n = resultados["tipo produto natural"]
    print(f"\nQueda de acurácia sob deslocamento de domínio : "
          f"{d['acuracia'] - n['acuracia']:+.3f}")
    print(f"Aumento do erro de calibração (ECE)          : "
          f"{n['ece'] - d['ece']:+.3f}")
    print(f"Lacuna confiança − acurácia (produto natural): "
          f"{n['confianca'] - n['acuracia']:+.3f}")
    print("  -> excesso de confiança: o modelo declara certeza que não tem.")

    # limite superior imposto pela variável latente de efluxo
    bayes_pn = ((p_verdadeiro_pn >= 0.5).astype(int) == teste_pn_y).mean()
    print(f"\nTeto de acurácia com a verdade-terreno probabilística: {bayes_pn:.3f}")
    print("  -> parte do erro é IRREDUTÍVEL para esses descritores: efluxo é")
    print("     uma variável latente que o modelo nunca observa. Nenhuma")
    print("     arquitetura resolve isso; só dado novo (ensaio de transporte).")

    # predição seletiva
    cob_d, acc_d = curva_risco_cobertura(d["y"], d["p"])
    cob_n, acc_n = curva_risco_cobertura(n["y"], n["p"])
    for alvo in (0.5, 0.8):
        i_d = np.searchsorted(cob_d, alvo)
        i_n = np.searchsorted(cob_n, alvo)
        print(f"\nAbstendo-se até cobrir {alvo:.0%} das moléculas:")
        print(f"  mesmo domínio        : acurácia {acc_d[i_d]:.3f}")
        print(f"  tipo produto natural : acurácia {acc_n[i_n]:.3f}")

    print("\nLeituras para o slide:")
    print(" · O erro que importa não é o do conjunto de teste do artigo; é o")
    print("   do conjunto que você tem na mão. Fora do domínio, a acurácia cai")
    print("   E a calibração piora - o modelo erra com convicção.")
    i50_d, i50_n = np.searchsorted(cob_d, 0.5), np.searchsorted(cob_n, 0.5)
    print(" · Abster-se por confiança funciona QUANDO a confiança é calibrada:")
    print(f"   no mesmo domínio, recusar metade dos casos leva a acurácia de")
    print(f"   {d['acuracia']:.3f} para {acc_d[i50_d]:.3f}. Fora do domínio o mesmo")
    print(f"   procedimento leva {n['acuracia']:.3f} para apenas {acc_n[i50_n]:.3f}:")
    print("   a abstenção se apoia justamente na confiança que quebrou. Filtrar")
    print("   por incerteza NÃO é um paraquedas contra deslocamento de domínio.")
    print("   O que serve como paraquedas é detecção explícita de domínio - a")
    print("   distância ao conhecido do DEMO 01 - e não a probabilidade do modelo.")
    print(" · Sempre reportar ECE ao lado de AUC. Um modelo pode ordenar bem")
    print("   e ainda assim mentir sobre a própria probabilidade.")
    print(" · Atravessar a BHE não é agir no SNC. A predição de permeabilidade")
    print("   resolve um gargalo de farmacocinética, não o de alvo.")

    # ---- Figura ---------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.2), layout="constrained")

    ax = axes[0]
    ax.scatter(treino_x["massa"], treino_x["tpsa"], s=6, alpha=0.25,
               color=OKABE_ITO["cinza"], marker="o", label="treino · tipo fármaco")
    ax.scatter(teste_pn_x["massa"], teste_pn_x["tpsa"], s=8, alpha=0.45,
               color=OKABE_ITO["vermelho"], marker="X",
               label="teste · tipo produto natural")
    ax.set(xlabel="Massa molecular (u)", ylabel="Área polar topológica (Å²)",
           title="Deslocamento de domínio, não incompetência")
    ax.legend(loc="upper left", fontsize=10)
    aviso_dados_sinteticos(ax)

    ax = axes[1]
    ax.plot([0, 1], [0, 1], color=OKABE_ITO["cinza"], linestyle=":",
            linewidth=1.5, label="calibração perfeita")
    for nome, cor, marcador in [("mesmo domínio", OKABE_ITO["azul"], "o"),
                                ("tipo produto natural", OKABE_ITO["vermelho"], "X")]:
        pontos = resultados[nome]["pontos"]
        conf = [p_[0] for p_ in pontos]
        freq = [p_[1] for p_ in pontos]
        ax.plot(conf, freq, marker=marcador, color=cor,
                label=f"{nome} (ECE {resultados[nome]['ece']:.3f})")
    ax.set(xlabel="Probabilidade predita", ylabel="Frequência observada",
           xlim=(0, 1), ylim=(0, 1), title="Diagrama de confiabilidade (10 bins)")
    ax.legend(loc="upper left", fontsize=10)
    aviso_dados_sinteticos(ax)

    ax = axes[2]
    ax.plot(cob_d, acc_d, color=OKABE_ITO["azul"], label="mesmo domínio")
    ax.plot(cob_n, acc_n, color=OKABE_ITO["vermelho"], linestyle="--",
            label="tipo produto natural")
    ax.axhline(bayes_pn, color=OKABE_ITO["verde"], linestyle=":", linewidth=2)
    ax.annotate("teto imposto pelo efluxo (variável latente)", (0.30, bayes_pn + 0.012),
                fontsize=9, color=OKABE_ITO["verde"])
    ax.set(xlabel="Cobertura (fração de moléculas em que o modelo responde)",
           ylabel="Acurácia entre as respondidas", xlim=(0.05, 1.0),
           title="Abster-se só ajuda se a confiança for honesta")
    ax.legend(loc="lower left", fontsize=10)
    aviso_dados_sinteticos(ax)

    caminho = exportar(
        fig, "06_bhe_calibracao",
        proveniencia={
            "script": "demos/06_bhe_calibracao.py",
            "semente": SEMENTE,
            "modelo": "GradientBoostingClassifier(padrão sklearn)",
            "atributos": ATRIBUTOS,
            "variavel_latente": "efluxo (não observado pelo modelo)",
            "bins_ece": 10,
        },
    )
    print(f"\nFigura: {caminho}")

    DIR_DADOS.mkdir(parents=True, exist_ok=True)
    df = pd.concat([
        teste_dom_x.assign(regime="mesmo domínio", rotulo=teste_dom_y,
                           p_predito=d["p"]),
        teste_pn_x.assign(regime="tipo produto natural", rotulo=teste_pn_y,
                          p_predito=n["p"], efluxo_latente=efluxo_pn),
    ])
    destino = DIR_DADOS / "06_predicoes_bhe.csv"
    df.to_csv(destino, index=False)
    print(f"Tabela: {destino}")


if __name__ == "__main__":
    main()
