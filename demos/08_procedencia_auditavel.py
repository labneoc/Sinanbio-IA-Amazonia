"""
DEMO 08 - Procedência como código: o que a IA herda quando herda o dado.

Ideia da palestra
-----------------
Nos slides sobre Amazônia, ética costuma ser o último bullet. Aqui ela é uma
estrutura de dados. A razão é técnica, não retórica: um modelo é uma função do
conjunto de treino. Se o conjunto de treino carrega acessos sem cadastro, sem
consentimento prévio informado ou sem repartição pactuada, o modelo carrega
isso junto - e ninguém consegue auditar depois o que não foi registrado antes.

O problema novo se chama informação de sequência digital (DSI, na sigla em
inglês): o material físico fica no país de origem, mas a sequência, o espectro
e os descritores circulam livremente e podem ser suficientes para gerar valor.
É a forma contemporânea do velho problema, e o mecanismo multilateral que
deveria endereçá-la esteve em negociação no âmbito da CDB. O ESTADO ATUAL
DESSAS NEGOCIAÇÕES PRECISA SER VERIFICADO antes de qualquer afirmação no
slide - não afirme de memória.

O que este script faz
---------------------
Gera um livro-razão sintético de acessos, aplica cinco regras de auditoria e
mostra a EVAPORAÇÃO DA RASTREABILIDADE ao longo da cadeia: acesso ->
depósito digital -> uso em treino -> produto. Cada elo perde procedência.

Aviso jurídico
--------------
As regras abaixo são uma CARICATURA DIDÁTICA. Não são assessoria jurídica e
não substituem a leitura do texto normativo vigente (Convenção sobre
Diversidade Biológica, Protocolo de Nagoia, Lei nº 13.123/2015 e suas
regulamentações, e as normas do SisGen), cujas versões atuais devem ser
verificadas na fonte oficial. O valor aqui é o PADRÃO: procedência auditável
por código, verificável por terceiros, e não declarada em prosa.

Nível de evidência: [E5] simulação. Casos históricos de apropriação indevida
citados na palestra devem vir com fonte primária verificada, sem exceção.

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

SEMENTE = 20260926
N_ACESSOS = 240

REGRAS = {
    "R1": "acesso ao patrimônio genético sem cadastro prévio",
    "R2": "uso de conhecimento tradicional sem consentimento prévio informado",
    "R3": "dado digital (DSI) publicado a partir de acesso não cadastrado",
    "R4": "modelo treinado para fim tecnológico sem repartição pactuada",
    "R5": "dado digital público sem identificador do acesso de origem",
}
SEVERIDADE = {"R1": 3, "R2": 4, "R3": 4, "R4": 3, "R5": 2}


def gerar_livro_razao(rng):
    """Livro-razão sintético de acessos a patrimônio genético.

    Nenhuma espécie, comunidade ou instituição real é nomeada - e isso é
    deliberado: um exemplo didático não deve produzir uma lista de suspeitos.
    """
    idx = np.arange(1, N_ACESSOS + 1)
    conhecimento_tradicional = rng.random(N_ACESSOS) < 0.35
    cadastro = rng.random(N_ACESSOS) < 0.78
    # o consentimento acompanha o cadastro, mas imperfeitamente
    consentimento = np.where(
        conhecimento_tradicional,
        rng.random(N_ACESSOS) < np.where(cadastro, 0.80, 0.25),
        True,
    )
    finalidade = np.where(rng.random(N_ACESSOS) < 0.30,
                          "desenvolvimento tecnológico", "pesquisa")
    # a cadeia digital: nem todo acesso gera dado, nem todo dado leva identificador
    gerou_dado_digital = rng.random(N_ACESSOS) < 0.62
    identificador_na_base = gerou_dado_digital & (rng.random(N_ACESSOS) < 0.45)
    usado_em_treino = gerou_dado_digital & (rng.random(N_ACESSOS) < 0.55)
    reparticao = (rng.random(N_ACESSOS)
                  < np.where(finalidade == "desenvolvimento tecnológico", 0.40, 0.12))

    return pd.DataFrame({
        "id_acesso": [f"ACS-{i:04d}" for i in idx],
        "taxon_codificado": [f"TX-{rng.integers(1, 90):03d}" for _ in idx],
        "territorio_codificado": [f"TER-{rng.integers(1, 25):02d}" for _ in idx],
        "conhecimento_tradicional_associado": conhecimento_tradicional,
        "cadastro_previo": cadastro,
        "consentimento_previo_informado": consentimento,
        "finalidade": finalidade,
        "dado_digital_publicado": gerou_dado_digital,
        "identificador_de_origem_na_base": identificador_na_base,
        "usado_em_treino_de_modelo": usado_em_treino,
        "reparticao_de_beneficios_pactuada": reparticao,
    })


def auditar(df):
    """Aplica as regras e devolve o livro-razão com as bandeiras e o risco.

    Esta função é o entregável reutilizável deste demo: troque as colunas pelos
    campos reais do seu laboratório e as regras pelo texto normativo vigente.
    """
    f = pd.DataFrame(index=df.index)
    f["R1"] = ~df["cadastro_previo"]
    f["R2"] = df["conhecimento_tradicional_associado"] & ~df["consentimento_previo_informado"]
    f["R3"] = df["dado_digital_publicado"] & ~df["cadastro_previo"]
    f["R4"] = (df["usado_em_treino_de_modelo"]
               & (df["finalidade"] == "desenvolvimento tecnológico")
               & ~df["reparticao_de_beneficios_pactuada"])
    f["R5"] = df["dado_digital_publicado"] & ~df["identificador_de_origem_na_base"]

    risco = sum(f[r].astype(int) * SEVERIDADE[r] for r in REGRAS)
    faixa = pd.cut(risco, bins=[-1, 0, 3, 7, 100],
                   labels=["conforme", "atenção", "alto", "crítico"])
    return pd.concat([df, f.add_prefix("flag_"),
                      pd.Series(risco, name="risco", index=df.index),
                      pd.Series(faixa, name="faixa_de_risco", index=df.index)],
                     axis=1), f


def main() -> None:
    cabecalho(
        "DEMO 08 · Procedência como código",
        "[E5] simulação; regras são caricatura didática, não assessoria jurídica",
        "O que um modelo herda junto com o conjunto de treino?",
    )
    aplicar_estilo()
    rng = np.random.default_rng(SEMENTE)

    livro = gerar_livro_razao(rng)
    auditado, flags = auditar(livro)

    print(f"Acessos no livro-razão sintético: {len(livro)}\n")
    print("Violações por regra:")
    for r, descricao in REGRAS.items():
        n = int(flags[r].sum())
        print(f"  {r} (severidade {SEVERIDADE[r]}) · {n:3d} acessos "
              f"({100 * n / len(livro):4.1f}%) · {descricao}")

    print("\nDistribuição por faixa de risco:")
    print(auditado["faixa_de_risco"].value_counts()
          .reindex(["conforme", "atenção", "alto", "crítico"]).to_string())

    # ---- Evaporação da rastreabilidade -----------------------------------
    etapas = {
        "Acessos registrados": len(livro),
        "Com cadastro prévio": int(livro["cadastro_previo"].sum()),
        "Com consentimento quando há CTA": int(
            (~livro["conhecimento_tradicional_associado"]
             | livro["consentimento_previo_informado"]).sum()),
        "Dado digital publicado": int(livro["dado_digital_publicado"].sum()),
        "...com identificador de origem": int(
            livro["identificador_de_origem_na_base"].sum()),
        "Usado em treino de modelo": int(livro["usado_em_treino_de_modelo"].sum()),
        "...com repartição pactuada": int(
            (livro["usado_em_treino_de_modelo"]
             & livro["reparticao_de_beneficios_pactuada"]).sum()),
    }
    print("\nCadeia de custódia (onde a procedência evapora):")
    for nome, n in etapas.items():
        print(f"  {nome:<34s} {n:4d}  ({100 * n / len(livro):5.1f}%)")

    treinados = livro["usado_em_treino_de_modelo"]
    rastreaveis = treinados & livro["identificador_de_origem_na_base"] & livro["cadastro_previo"]
    print(f"\nDos {int(treinados.sum())} acessos que entraram em treino de modelo, "
          f"{int(rastreaveis.sum())} ({100 * rastreaveis.sum() / max(1, treinados.sum()):.0f}%) "
          f"são plenamente rastreáveis.")
    print("Essa é a fração que um terceiro conseguiria auditar. O resto não é")
    print("necessariamente ilícito - é INVERIFICÁVEL, que para efeito de")
    print("confiança científica dá quase no mesmo.")

    print("\nLeituras para o slide:")
    print(" · A procedência não se perde em um ato de má-fé; ela evapora por")
    print("   atrito, um elo de cada vez, em decisões que ninguém registrou.")
    print(" · Por isso ela precisa ser CAMPO OBRIGATÓRIO no pipeline, não")
    print("   parágrafo no artigo. Se não está na tabela, não existe.")
    print(" · Proposta concreta [E5], testável e barata: todo conjunto de dados")
    print("   do laboratório carrega identificador de acesso por registro; todo")
    print("   modelo publica o manifesto dos identificadores usados no treino;")
    print("   a auditoria roda no CI, como um teste. Procedência vira")
    print("   propriedade verificável do artefato, e não promessa.")
    print(" · Verificar antes de afirmar em público: texto vigente da Lei")
    print("   13.123/2015 e regulamentação, normas do SisGen e o estado atual")
    print("   das negociações sobre DSI no âmbito da CDB.")

    # ---- Figura ----------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.6), layout="constrained")

    ax = axes[0]
    nomes = list(etapas)
    valores = [100 * v / len(livro) for v in etapas.values()]
    cores = [OKABE_ITO["cinza"], OKABE_ITO["azul"], OKABE_ITO["azul"],
             OKABE_ITO["verde"], OKABE_ITO["laranja"], OKABE_ITO["verde"],
             OKABE_ITO["vermelho"]]
    ax.barh(range(len(nomes))[::-1], valores, color=cores, edgecolor="white")
    ax.set(yticks=range(len(nomes))[::-1], yticklabels=nomes,
           xlabel="% dos acessos", xlim=(0, 108),
           title="A procedência evapora por atrito")
    for i, v in zip(range(len(nomes))[::-1], valores):
        ax.annotate(f"{v:.0f}%", (v, i), xytext=(5, -4),
                    textcoords="offset points", fontsize=11)
    ax.tick_params(axis="y", labelsize=10)
    aviso_dados_sinteticos(ax)

    ax = axes[1]
    n_violacoes = [int(flags[r].sum()) for r in REGRAS]
    severidades = [SEVERIDADE[r] for r in REGRAS]
    barras = ax.bar(list(REGRAS), n_violacoes,
                    color=[plt.get_cmap("Oranges")(0.25 + 0.18 * s)
                           for s in severidades], edgecolor="white")
    for b, n, s in zip(barras, n_violacoes, severidades):
        ax.annotate(f"{n}\nsev. {s}", (b.get_x() + b.get_width() / 2, n),
                    xytext=(0, 4), textcoords="offset points", ha="center",
                    fontsize=10)
    ax.set(ylabel="Acessos sinalizados",
           ylim=(0, max(n_violacoes) * 1.28),
           title="Violações por regra (cor = severidade)")
    aviso_dados_sinteticos(ax)

    ax = axes[2]
    tabela = (auditado.groupby(["finalidade", "faixa_de_risco"], observed=False)
              .size().unstack(fill_value=0)
              .reindex(columns=["conforme", "atenção", "alto", "crítico"]))
    n_por_grupo = tabela.sum(axis=1)
    # normalizado dentro de cada finalidade: os grupos têm tamanhos diferentes
    # e comparar contagens brutas seria enganoso
    tabela = 100 * tabela.div(n_por_grupo, axis=0)
    rotulos_x = [f"{i}\n(n={int(n_por_grupo[i])})" for i in tabela.index]
    base = np.zeros(len(tabela))
    cores_faixa = [OKABE_ITO["verde"], OKABE_ITO["amarelo"],
                   OKABE_ITO["laranja"], OKABE_ITO["vermelho"]]
    hachuras = ["", "..", "//", "xx"]
    for coluna, cor, hachura in zip(tabela.columns, cores_faixa, hachuras):
        ax.bar(rotulos_x, tabela[coluna], bottom=base, color=cor,
               hatch=hachura, edgecolor="white", label=coluna)
        base += tabela[coluna].values
    ax.set(ylabel="% dos acessos da finalidade", ylim=(0, 108),
           title="Risco por finalidade declarada")
    ax.legend(title="faixa de risco", fontsize=10)
    ax.tick_params(axis="x", labelsize=10)
    aviso_dados_sinteticos(ax)

    caminho = exportar(
        fig, "08_procedencia_auditavel",
        proveniencia={
            "script": "demos/08_procedencia_auditavel.py",
            "semente": SEMENTE,
            "regras": REGRAS,
            "severidades": SEVERIDADE,
            "aviso": "regras são caricatura didática; não são assessoria jurídica",
        },
    )
    print(f"\nFigura: {caminho}")

    DIR_DADOS.mkdir(parents=True, exist_ok=True)
    destino = DIR_DADOS / "08_livro_razao_auditado.csv"
    auditado.to_csv(destino, index=False)
    print(f"Tabela: {destino}")


if __name__ == "__main__":
    main()
