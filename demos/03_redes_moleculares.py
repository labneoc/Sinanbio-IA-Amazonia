"""
DEMO 03 - Redes moleculares: o gargalo é anotar, não medir.

Ideia da palestra
-----------------
Um extrato bruto de uma planta amazônica rende milhares de features em
LC-MS/MS em poucas horas. A fração desses sinais que recebe identidade
estrutural confiável é pequena - o problema da dereplicação. Redes moleculares
atacam isso agrupando espectros por similaridade de fragmentação: se um nó de
um agrupamento tem identidade conhecida, a vizinhança herda uma hipótese de
classe. Os agrupamentos SEM nenhum nó anotado são, por definição, onde mora a
novidade química - e são o alvo racional de isolamento.

A IA entra aqui em três lugares, com riscos distintos:
  (1) agrupar espectros - barato e robusto;
  (2) propagar classe por vizinhança - útil, mas o erro se propaga junto;
  (3) predizer estrutura de novo a partir do espectro - o passo caro, e onde um
      resultado [E5] é mais frequentemente vendido como se fosse [E3].

Três resultados que este script torna visíveis
----------------------------------------------
1. O viés da biblioteca espectral se transfere para a rede: se a química
   tropical está sub-representada na biblioteca, ela reaparece como
   agrupamento órfão - o que é simultaneamente um problema e uma oportunidade.
2. A propagação de anotação não é identificação. Ela herda o erro do nó de
   origem e o distribui pela vizinhança.
3. O limiar de similaridade é um hiperparâmetro que negocia COBERTURA contra
   ACERTO. Relatar um sem o outro é relatar metade do resultado.

Nível de evidência: [E5] simulação didática. Taxas reais de anotação em
metabolômica não direcionada devem ser citadas de fonte primária verificada,
com a plataforma e o nível de confiança de identificação explícitos.

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
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from sinanbio.style import (DIR_DADOS, OKABE_ITO, aplicar_estilo,
                            aviso_dados_sinteticos, cabecalho, exportar)

SEMENTE = 20260926
N_BINS = 400              # bins de m/z
N_FAMILIAS = 14
N_ESPECTROS = 260
LIMIAR_COSSENO = 0.76
FAMILIAS_BEM_ESTUDADAS = 8   # famílias 0-7 são as representadas na biblioteca


def gerar_espectros(rng):
    """Gera espectros com duas propriedades que existem no dado real.

    (a) Linhagens: famílias biossinteticamente aparentadas compartilham parte
        dos fragmentos. Isso produz agrupamentos quiméricos, que são a causa
        principal do erro de propagação de anotação.
    (b) Análogos de uma família diferem por perdas e adições de fragmentos.
    """
    n_linhagens = 5
    linhagens = [rng.choice(N_BINS, size=5, replace=False) for _ in range(n_linhagens)]

    nucleos = []
    for f in range(N_FAMILIAS):
        lin = f % n_linhagens
        herdados = rng.choice(linhagens[lin], size=5, replace=False)
        proprios = rng.choice(N_BINS, size=rng.integers(2, 5), replace=False)
        picos = np.unique(np.concatenate([herdados, proprios]))
        intens = rng.random(len(picos)) * 0.8 + 0.2
        nucleos.append((picos, intens))

    familia = rng.integers(0, N_FAMILIAS, size=N_ESPECTROS)
    espectros = np.zeros((N_ESPECTROS, N_BINS))
    for i, f in enumerate(familia):
        picos, intens = nucleos[f]
        manter = rng.random(len(picos)) > 0.20
        espectros[i, picos[manter]] = intens[manter] * rng.uniform(0.7, 1.3, manter.sum())
        extras = rng.choice(N_BINS, size=rng.integers(2, 6), replace=False)
        espectros[i, extras] += rng.random(len(extras)) * 0.5
        espectros[i] += rng.random(N_BINS) * 0.01     # linha de base
    espectros /= np.linalg.norm(espectros, axis=1, keepdims=True)
    return espectros, familia


def construir_rede(cos, limiar):
    g = nx.Graph()
    g.add_nodes_from(range(cos.shape[0]))
    ii, jj = np.where(np.triu(cos, 1) >= limiar)
    for i, j in zip(ii, jj):
        g.add_edge(int(i), int(j), weight=float(cos[i, j]))
    return g


def propagar(g, familia, anotado):
    """Cada componente herda a classe majoritária entre seus nós anotados."""
    classe = np.full(len(familia), -1)
    for comp in nx.connected_components(g):
        nos = np.array(sorted(comp))
        conhecidos = nos[anotado[nos]]
        if len(conhecidos):
            classe[nos] = Counter(familia[conhecidos]).most_common(1)[0][0]
    return classe


def metricas(g, familia, anotado, classe):
    alvo = (~anotado) & (classe >= 0)
    acerto = float((classe[alvo] == familia[alvo]).mean()) if alvo.sum() else np.nan
    componentes = list(nx.connected_components(g))
    orfaos = [c for c in componentes if len(c) > 1 and not anotado[sorted(c)].any()]
    return {
        "cobertura": float((anotado | (classe >= 0)).mean()),
        "acerto_propagacao": acerto,
        "n_propagados": int(alvo.sum()),
        "agrupamentos": sum(1 for c in componentes if len(c) > 1),
        "isolados": sum(1 for c in componentes if len(c) == 1),
        "agrupamentos_orfaos": len(orfaos),
        "espectros_orfaos": sum(len(c) for c in orfaos),
        "nos_orfaos": np.array(sorted({n for c in orfaos for n in c}), dtype=int),
    }


def main() -> None:
    cabecalho(
        "DEMO 03 · Redes moleculares e dereplicação",
        "[E5] simulação didática",
        "Onde está a novidade química em um extrato que não sabemos ler?",
    )
    aplicar_estilo()
    rng = np.random.default_rng(SEMENTE)

    espectros, familia = gerar_espectros(rng)
    cos = espectros @ espectros.T
    np.fill_diagonal(cos, 0.0)

    # Dereplicação COM VIÉS: bibliotecas espectrais públicas são construídas a
    # partir do que já foi isolado e publicado, e essa amostra é historicamente
    # enviesada. Famílias 0-7 batem com a biblioteca; 8-13 ("tropicais
    # sub-representadas"), quase nunca. É o DEMO 01 reaparecendo no espectro.
    p_anotacao = np.where(familia < FAMILIAS_BEM_ESTUDADAS, 0.48, 0.02)
    anotado = rng.random(N_ESPECTROS) < p_anotacao

    g = construir_rede(cos, LIMIAR_COSSENO)
    classe = propagar(g, familia, anotado)
    m = metricas(g, familia, anotado, classe)

    sub_representadas = familia >= FAMILIAS_BEM_ESTUDADAS
    orfaos = np.zeros(N_ESPECTROS, dtype=bool)
    orfaos[m["nos_orfaos"]] = True
    frac_trop_orfao = (sub_representadas[orfaos].mean() if orfaos.any() else np.nan)

    print(f"Espectros simulados                 : {N_ESPECTROS}")
    print(f"Arestas (cosseno >= {LIMIAR_COSSENO})          : {g.number_of_edges()}")
    print(f"Agrupamentos (>1 nó) / isolados     : {m['agrupamentos']} / {m['isolados']}")
    print(f"Anotados por biblioteca             : {anotado.sum()} "
          f"({100 * anotado.mean():.1f}%)")
    print(f"  · famílias bem estudadas          : "
          f"{100 * anotado[~sub_representadas].mean():.1f}%")
    print(f"  · famílias sub-representadas      : "
          f"{100 * anotado[sub_representadas].mean():.1f}%")
    print(f"Cobertura após propagação           : {100 * m['cobertura']:.1f}%")
    print(f"Acerto da classe propagada          : "
          f"{100 * m['acerto_propagacao']:.1f}% (n={m['n_propagados']})")
    print(f"Agrupamentos órfãos                 : {m['agrupamentos_orfaos']} "
          f"({m['espectros_orfaos']} espectros)")
    print(f"  · deles, de famílias sub-repres.  : {100 * frac_trop_orfao:.1f}%")

    # ---- Varredura do limiar: cobertura vs acerto ------------------------
    limiares = np.round(np.arange(0.60, 0.93, 0.02), 2)
    varredura = []
    for t in limiares:
        gt = construir_rede(cos, t)
        ct = propagar(gt, familia, anotado)
        mt = metricas(gt, familia, anotado, ct)
        varredura.append({"limiar": t, **{k: v for k, v in mt.items()
                                          if k != "nos_orfaos"}})
    var = pd.DataFrame(varredura)

    print("\nVarredura do limiar de similaridade (trecho):")
    print(var[["limiar", "cobertura", "acerto_propagacao", "agrupamentos_orfaos"]]
          .iloc[::3].round(3).to_string(index=False))
    print("\nLeituras para o slide:")
    print(" 1. O viés da biblioteca vira topologia: as famílias que ninguém")
    print("    estudou viram agrupamentos órfãos. Isso é um mapa de prioridade")
    print("    de isolamento, não um defeito do experimento.")
    print(" 2. Propagar anotação NÃO é identificar. Cobertura alta com acerto")
    print("    baixo é o modo de falha silencioso mais comum aqui.")
    print(" 3. O limiar é uma escolha de política de risco, não um detalhe")
    print("    técnico: ele move cobertura e acerto em direções opostas.")

    # ---- Figura ---------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.6), layout="constrained",
                             width_ratios=[1.35, 1, 1])

    ax = axes[0]
    pos = nx.spring_layout(g, seed=SEMENTE, k=0.30)
    nx.draw_networkx_edges(g, pos, ax=ax, alpha=0.25, width=0.9,
                           edge_color=OKABE_ITO["cinza"])
    propagados = (~anotado) & (classe >= 0) & ~orfaos
    sem_hipotese = (~anotado) & (classe < 0) & ~orfaos
    grupos = [
        (np.where(anotado)[0], OKABE_ITO["azul"], "o", 95, "anotado (biblioteca)"),
        (np.where(propagados)[0], OKABE_ITO["verde"], "^", 70, "hipótese por propagação"),
        (np.where(orfaos)[0], OKABE_ITO["vermelho"], "X", 95,
         "agrupamento órfão · prioridade"),
        (np.where(sem_hipotese)[0], OKABE_ITO["cinza"], "s", 45, "sem hipótese"),
    ]
    for nos, cor, marcador, tam, rotulo in grupos:
        if len(nos):
            nx.draw_networkx_nodes(g, pos, nodelist=nos.tolist(), ax=ax,
                                   node_color=cor, node_shape=marcador,
                                   node_size=tam, label=rotulo)
    ax.set_axis_off()
    ax.set_title(f"Rede molecular sintética (cosseno ≥ {LIMIAR_COSSENO:.2f})")
    ax.legend(loc="upper left", scatterpoints=1, fontsize=10)
    ax.text(0.995, 0.01, "DADOS SINTÉTICOS · ilustração didática",
            transform=ax.transAxes, ha="right", fontsize=8,
            color=OKABE_ITO["cinza"])

    ax = axes[1]
    etapas = ["Espectros\nadquiridos", "Anotados por\nbiblioteca",
              "Cobertura após\npropagação", "Propagação\nCORRETA"]
    valores = [
        100.0,
        100 * anotado.mean(),
        100 * m["cobertura"],
        100 * (anotado.mean() + propagados.mean() * m["acerto_propagacao"]),
    ]
    cores = [OKABE_ITO["cinza"], OKABE_ITO["azul"], OKABE_ITO["verde"],
             OKABE_ITO["laranja"]]
    barras = ax.bar(etapas, valores, color=cores, edgecolor="white")
    for b, v in zip(barras, valores):
        ax.annotate(f"{v:.0f}%", (b.get_x() + b.get_width() / 2, v),
                    textcoords="offset points", xytext=(0, 5), ha="center",
                    fontsize=12)
    ax.set(ylabel="% dos espectros", ylim=(0, 115), title="Funil da dereplicação")
    ax.tick_params(axis="x", labelsize=9)
    aviso_dados_sinteticos(ax)

    ax = axes[2]
    ax.plot(var["limiar"], 100 * var["cobertura"], marker="o",
            color=OKABE_ITO["verde"], label="cobertura de anotação")
    ax.plot(var["limiar"], 100 * var["acerto_propagacao"], marker="s",
            linestyle="--", color=OKABE_ITO["vermelho"],
            label="acerto da propagação")
    ax.axvline(LIMIAR_COSSENO, color=OKABE_ITO["preto"], linestyle=":",
               linewidth=1.5)
    ax.annotate("limiar usado", (LIMIAR_COSSENO, 8),
                xytext=(4, 0), textcoords="offset points", fontsize=10)
    ax.set(xlabel="Limiar de cosseno", ylabel="%", ylim=(0, 105),
           title="Cobertura e acerto puxam para lados opostos")
    ax.legend(loc="center left", fontsize=10)
    aviso_dados_sinteticos(ax)

    caminho = exportar(
        fig, "03_redes_moleculares",
        proveniencia={
            "script": "demos/03_redes_moleculares.py",
            "semente": SEMENTE,
            "limiar_cosseno": LIMIAR_COSSENO,
            "vies_biblioteca": "p=0.48 famílias 0-7; p=0.02 famílias 8-13",
            "layout": f"networkx.spring_layout(seed={SEMENTE})",
        },
    )
    print(f"\nFigura: {caminho}")

    DIR_DADOS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "id_espectro": np.arange(N_ESPECTROS),
        "familia_verdadeira": familia,
        "familia_sub_representada": sub_representadas,
        "anotado_biblioteca": anotado,
        "classe_propagada": classe,
        "em_agrupamento_orfao": orfaos,
    }).to_csv(DIR_DADOS / "03_rede_molecular_sintetica.csv", index=False)
    var.to_csv(DIR_DADOS / "03_varredura_limiar.csv", index=False)
    print(f"Tabelas: {DIR_DADOS / '03_rede_molecular_sintetica.csv'}")
    print(f"         {DIR_DADOS / '03_varredura_limiar.csv'}")


if __name__ == "__main__":
    main()
