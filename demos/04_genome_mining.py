"""
DEMO 04 - Genome mining: por que regras encontram o que já conhecemos.

Ideia da palestra
-----------------
Detectar cluster biossintético (BGC) por regra especialista funciona muito bem
para as classes modulares e estereotipadas - policetídeo sintase, peptídeo não
ribossomal - porque elas têm gramática: módulos repetidos, domínios canônicos
em ordem previsível. As regras foram escritas a partir dessas classes. O
resultado é uma assimetria que costuma passar em silêncio: a mesma ferramenta
que tem recall alto em PKS/NRPS é quase cega para classes não modulares e
crípticas, que são justamente onde sobra espaço para novidade em microbiomas
pouco amostrados - endófitos amazônicos, solo de várzea, simbiontes de insetos.

Modelos aprendidos a partir de dados reduzem essa cegueira, ao custo de mais
falsos positivos. A decisão entre um e outro não é técnica, é econômica: qual
o custo de perder um cluster verdadeiro contra o de expressar um falso?

O que este script simula
------------------------
Genomas sintéticos como sequências de domínios gênicos, com BGCs de duas
classes (modular e não modular), e dois detectores: um por regra e um por
aprendizado supervisionado em janelas deslizantes.

Nível de evidência: [E5] simulação didática. A assimetria entre classes
modulares e não modulares é [E4]/[E3] na literatura de mineração genômica e
deve ser citada com fonte primária verificada.

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score

from sinanbio.style import (DIR_DADOS, OKABE_ITO, aplicar_estilo,
                            aviso_dados_sinteticos, cabecalho, exportar)

SEMENTE = 20260926
N_GENOMAS = 120
N_GENES = 600          # genes por genoma
N_DOMINIOS = 40        # vocabulário de domínios
JANELA, PASSO = 12, 3

DOM_MODULARES = np.arange(0, 8)    # KS/AT/ACP/C/A/PCP-like: gramática clara
DOM_CRIPTICOS = np.arange(8, 14)   # RiPP/terpeno-like: sinal fraco e ambíguo
DOM_FUNDO = np.arange(14, N_DOMINIOS)


def gerar_genoma(rng):
    """Um genoma: fundo metabólico + BGCs inseridos em posições aleatórias."""
    # fundo: majoritariamente domínios inespecíficos, com vazamento de
    # domínios crípticos - é isso que torna a classe não modular difícil
    genes = rng.choice(DOM_FUNDO, size=N_GENES)
    vazamento = rng.random(N_GENES) < 0.08
    genes[vazamento] = rng.choice(DOM_CRIPTICOS, size=vazamento.sum())

    rotulo = np.zeros(N_GENES, dtype=int)   # 0 = não BGC, 1 = modular, 2 = críptico
    posicoes = rng.choice(np.arange(20, N_GENES - 40), size=6, replace=False)
    posicoes.sort()
    # três clusters de cada classe por genoma, para que a comparação de recall
    # entre classes não dependa de flutuação amostral
    classes_sorteadas = rng.permutation([1, 1, 1, 2, 2, 2])
    ocupado = np.zeros(N_GENES, dtype=bool)

    for p, classe in zip(posicoes, classes_sorteadas):
        classe = int(classe)
        tamanho = int(rng.integers(12, 20)) if classe == 1 else int(rng.integers(5, 9))
        if p + tamanho >= N_GENES or ocupado[p:p + tamanho].any():
            continue
        if classe == 1:
            # módulos repetidos: assinatura forte e estereotipada
            bloco = np.tile(DOM_MODULARES[:3], tamanho // 3 + 1)[:tamanho]
            extra = rng.random(tamanho) < 0.35
            bloco[extra] = rng.choice(DOM_MODULARES[3:], size=extra.sum())
        else:
            # cluster críptico: poucos domínios diagnósticos, muito fundo
            bloco = rng.choice(DOM_FUNDO, size=tamanho)
            n_diag = int(rng.integers(2, 4))
            idx = rng.choice(tamanho, size=n_diag, replace=False)
            bloco[idx] = rng.choice(DOM_CRIPTICOS, size=n_diag)
        genes[p:p + tamanho] = bloco
        rotulo[p:p + tamanho] = classe
        ocupado[p:p + tamanho] = True
    return genes, rotulo


def janelas(genes, rotulo):
    """Vetoriza janelas deslizantes como contagem de domínios (bag-of-domains)."""
    x, y, pos = [], [], []
    for i in range(0, N_GENES - JANELA, PASSO):
        fatia = genes[i:i + JANELA]
        rot = rotulo[i:i + JANELA]
        contagem = np.bincount(fatia, minlength=N_DOMINIOS)
        # a janela é positiva se mais da metade dela pertence a um BGC
        classes, n = np.unique(rot[rot > 0], return_counts=True)
        alvo = int(classes[np.argmax(n)]) if len(classes) and n.max() > JANELA // 2 else 0
        x.append(contagem)
        y.append(alvo)
        pos.append(i)
    return np.array(x), np.array(y), np.array(pos)


def detector_por_regra(x):
    """Proxy da regra especialista: exige domínios modulares canônicos."""
    return (x[:, DOM_MODULARES].sum(axis=1) >= 4).astype(int)


def main() -> None:
    cabecalho(
        "DEMO 04 · Genome mining: regra vs aprendizado",
        "[E5] simulação didática de um fenômeno [E4]/[E3] da literatura",
        "A ferramenta encontra o que existe ou o que ela foi escrita para ver?",
    )
    aplicar_estilo()
    rng = np.random.default_rng(SEMENTE)

    X, Y = [], []
    genoma_exemplo = None
    for gi in range(N_GENOMAS):
        genes, rotulo = gerar_genoma(rng)
        if gi == 0:
            genoma_exemplo = (genes, rotulo)
        xi, yi, _ = janelas(genes, rotulo)
        X.append(xi)
        Y.append(yi)
    X = np.vstack(X)
    Y = np.concatenate(Y)

    # divisão POR GENOMA (nunca janelas do mesmo genoma nos dois lados)
    n_janelas_por_genoma = len(Y) // N_GENOMAS
    genoma_id = np.repeat(np.arange(N_GENOMAS), n_janelas_por_genoma)[: len(Y)]
    treino = genoma_id < int(N_GENOMAS * 0.6)
    teste = ~treino

    modelo = LogisticRegression(max_iter=2000, class_weight="balanced")
    modelo.fit(X[treino], (Y[treino] > 0).astype(int))
    pred_ml = modelo.predict(X[teste])
    pred_regra = detector_por_regra(X[teste])
    y_teste = Y[teste]
    verdadeiro = (y_teste > 0).astype(int)

    resultados = []
    for nome, pred in [("regra especialista", pred_regra), ("modelo aprendido", pred_ml)]:
        linha = {
            "detector": nome,
            "recall_modular": recall_score(y_teste == 1, pred & (y_teste != 2),
                                           zero_division=0),
            "recall_criptico": recall_score(y_teste == 2, pred & (y_teste != 1),
                                            zero_division=0),
            "precisao_global": precision_score(verdadeiro, pred, zero_division=0),
            "recall_global": recall_score(verdadeiro, pred, zero_division=0),
        }
        linha["falsos_positivos"] = int(((pred == 1) & (verdadeiro == 0)).sum())
        resultados.append(linha)
    res = pd.DataFrame(resultados)
    print(res.round(3).to_string(index=False))

    dr = res.set_index("detector")
    print(f"\nRecall em clusters modulares  : regra {dr.loc['regra especialista','recall_modular']:.2f}"
          f"  vs  aprendido {dr.loc['modelo aprendido','recall_modular']:.2f}")
    print(f"Recall em clusters crípticos  : regra {dr.loc['regra especialista','recall_criptico']:.2f}"
          f"  vs  aprendido {dr.loc['modelo aprendido','recall_criptico']:.2f}")
    print(f"Recall GLOBAL (agregado)      : regra {dr.loc['regra especialista','recall_global']:.2f}"
          f"  vs  aprendido {dr.loc['modelo aprendido','recall_global']:.2f}")
    print("  -> o agregado da regra parece bom porque clusters modulares são")
    print("     mais longos e dominam a contagem de janelas. A métrica média")
    print("     apaga exatamente o modo de falha que interessa.")
    print(f"Falsos positivos              : regra {dr.loc['regra especialista','falsos_positivos']}"
          f"  vs  aprendido {dr.loc['modelo aprendido','falsos_positivos']}")
    print()
    print("Leitura para o slide: a regra não está errada; ela é fiel ao que foi")
    print("escrita para ver. O ponto é que o catálogo de BGCs conhecidos e a")
    print("regra que o produziu se retroalimentam. Quebrar esse circuito exige")
    print("dado novo (metagenoma de solo, endófitos, simbiontes) e um detector")
    print("que aceite pagar falso positivo em troca de classe nova - com")
    print("validação heteróloga depois, porque detecção NÃO é produção: um BGC")
    print("predito é [E5] até que o metabólito seja isolado.")

    # ---- Figura ---------------------------------------------------------
    genes, rotulo = genoma_exemplo
    xi, yi, pos = janelas(genes, rotulo)
    p_regra = detector_por_regra(xi)
    p_ml = modelo.predict(xi)

    fig, axes = plt.subplots(2, 2, figsize=(15.0, 8.0), layout="constrained",
                             height_ratios=[1, 1.25])

    ax = axes[0, 0]
    # janela escolhida para conter pelo menos um cluster de cada classe
    def _spans(rot):
        out, i = [], 0
        while i < len(rot):
            if rot[i] > 0:
                j = i
                while j < len(rot) and rot[j] == rot[i]:
                    j += 1
                out.append((i, j, int(rot[i])))
                i = j
            else:
                i += 1
        return out

    spans = _spans(rotulo)
    cripticos = [s_ for s_ in spans if s_[2] == 2]
    centro = cripticos[0][0] if cripticos else spans[0][0]
    inicio = int(np.clip(centro - 100, 0, N_GENES - 220))
    fim = inicio + 220

    for a, b, classe in spans:
        if b < inicio or a > fim:
            continue
        ax.axvspan(a, b, ymin=0.0, ymax=1.0,
                   color=OKABE_ITO["azul"] if classe == 1 else OKABE_ITO["rosa"],
                   alpha=0.9, zorder=1)
        ax.annotate("BGC modular" if classe == 1 else "BGC críptico",
                    ((a + b) / 2, 1.03), ha="center", fontsize=10,
                    annotation_clip=False,
                    color=OKABE_ITO["azul"] if classe == 1 else OKABE_ITO["rosa"])
    for nome, pred, altura, cor, marcador in [
        ("regra", p_regra, 0.70, OKABE_ITO["laranja"], "v"),
        ("modelo", p_ml, 0.30, OKABE_ITO["verde"], "^"),
    ]:
        sel = (pos >= inicio) & (pos < fim) & (pred == 1)
        ax.scatter(pos[sel] + JANELA / 2, np.full(sel.sum(), altura),
                   marker=marcador, s=55, color=cor, zorder=3,
                   label=f"detecção · {nome}")
    ax.set(xlim=(inicio, fim), ylim=(0, 1), yticks=[],
           xlabel="Posição no genoma (gene)",
           title="Trecho de genoma sintético e o que cada detector marca")
    ax.legend(loc="lower right", fontsize=9, ncol=2)

    ax = axes[0, 1]
    ax.axis("off")
    texto = (
        "Assimetria que a regra herda do catálogo\n\n"
        f"recall modular   · regra {dr.loc['regra especialista','recall_modular']:.2f}"
        f"   | aprendido {dr.loc['modelo aprendido','recall_modular']:.2f}\n"
        f"recall críptico  · regra {dr.loc['regra especialista','recall_criptico']:.2f}"
        f"   | aprendido {dr.loc['modelo aprendido','recall_criptico']:.2f}\n"
        f"precisão global  · regra {dr.loc['regra especialista','precisao_global']:.2f}"
        f"   | aprendido {dr.loc['modelo aprendido','precisao_global']:.2f}\n\n"
        f"recall GLOBAL   · regra {dr.loc['regra especialista','recall_global']:.2f}"
        f"   | aprendido {dr.loc['modelo aprendido','recall_global']:.2f}\n\n"
        f"O recall global de {dr.loc['regra especialista','recall_global']:.2f} da regra esconde\n"
        f"um recall de {dr.loc['regra especialista','recall_criptico']:.2f} na classe que importa:\n"
        "métrica agregada apaga o modo de falha.\n\n"
        "A regra compra precisão pagando com\n"
        "cegueira para a classe não modular.\n"
        "O modelo compra recall pagando com\n"
        "falso positivo - que custa bancada."
    )
    ax.text(0.02, 0.95, texto, va="top", ha="left", fontsize=12, family="monospace")

    ax = axes[1, 0]
    largura = 0.35
    categorias = ["BGC modular", "BGC críptico"]
    valores_regra = [dr.loc["regra especialista", "recall_modular"],
                     dr.loc["regra especialista", "recall_criptico"]]
    valores_ml = [dr.loc["modelo aprendido", "recall_modular"],
                  dr.loc["modelo aprendido", "recall_criptico"]]
    xpos = np.arange(2)
    ax.bar(xpos - largura / 2, valores_regra, largura, color=OKABE_ITO["laranja"],
           label="regra especialista")
    ax.bar(xpos + largura / 2, valores_ml, largura, color=OKABE_ITO["verde"],
           hatch="//", edgecolor="white", label="modelo aprendido")
    for x, v in zip(xpos - largura / 2, valores_regra):
        ax.annotate(f"{v:.2f}", (x, v), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=11)
    for x, v in zip(xpos + largura / 2, valores_ml):
        ax.annotate(f"{v:.2f}", (x, v), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=11)
    ax.set(xticks=xpos, xticklabels=categorias, ylabel="Recall", ylim=(0, 1.12),
           title="Onde cada detector enxerga")
    ax.legend()
    aviso_dados_sinteticos(ax)

    ax = axes[1, 1]
    ax.bar(["regra\nespecialista", "modelo\naprendido"],
           [dr.loc["regra especialista", "falsos_positivos"],
            dr.loc["modelo aprendido", "falsos_positivos"]],
           color=[OKABE_ITO["laranja"], OKABE_ITO["verde"]], edgecolor="white")
    ax.set(ylabel="Janelas falsas positivas (conj. teste)",
           title="O preço do recall: bancada gasta")
    aviso_dados_sinteticos(ax)

    caminho = exportar(
        fig, "04_genome_mining",
        proveniencia={
            "script": "demos/04_genome_mining.py",
            "semente": SEMENTE,
            "particao": "por genoma (60% treino / 40% teste)",
            "modelo": "LogisticRegression(class_weight='balanced') sobre bag-of-domains",
        },
    )
    print(f"\nFigura: {caminho}")

    DIR_DADOS.mkdir(parents=True, exist_ok=True)
    destino = DIR_DADOS / "04_desempenho_detectores.csv"
    res.to_csv(destino, index=False)
    print(f"Tabela: {destino}")


if __name__ == "__main__":
    main()
