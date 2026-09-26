"""
DEMO 07 - Descobrir a EQUAÇÃO, não só o padrão: dinâmica de biossíntese.

Ideia da palestra (e o ponto onde o palestrante tem vantagem comparativa)
------------------------------------------------------------------------
Quase toda a IA aplicada a produtos naturais faz reconhecimento de padrão
estático: esta molécula é ativa, este cluster é um BGC, este espectro é desta
classe. O arcabouço de sistemas dinâmicos orientados por dados propõe outra
coisa: recuperar a EQUAÇÃO DE GOVERNO a partir de séries temporais medidas.

Em biossíntese de metabólitos especializados isso é atraente porque o objeto
é literalmente um sistema dinâmico: elicitação, indução de transcrito,
consumo de precursor, acúmulo e degradação do produto. Um modelo esparso e
interpretável dessa dinâmica responde a perguntas que um classificador não
responde - quando colher, quanto elicitar, qual passo limita o rendimento.

AVISO DE HONESTIDADE (importa mais que a demonstração)
------------------------------------------------------
A identificação esparsa de dinâmica é método estabelecido e validado em
mecânica, fluidos e circuitos. Sua aplicação a rotas biossintéticas de
produtos naturais amazônicos é, até onde verificamos, PROPOSTA NOSSA: [E5].
Não apresentar como literatura consolidada. Esta simulação serve para mostrar
o mecanismo e, principalmente, para expor a condição de falha: o método exige
amostragem densa e ruído baixo, e a metabolômica temporal típica - poucos
pontos, amostragem destrutiva, alta variabilidade entre réplicas biológicas -
está justamente no regime em que ele quebra. Dizer isso é parte do argumento,
não uma ressalva de rodapé.

Referência metodológica: Brunton & Kutz, "Data-Driven Science and
Engineering" (no corpus do projeto). A implementação aqui é mínima e própria,
sem dependência externa, para que o código caiba em uma leitura.

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
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter

from sinanbio.style import (DIR_DADOS, OKABE_ITO, aplicar_estilo,
                            aviso_dados_sinteticos, cabecalho, exportar)

SEMENTE = 20260926
ESTADOS = ["P", "E", "M"]     # precursor, enzima induzida, metabólito
T_FINAL = 30.0

# Sistema "verdadeiro" (polinomial, para que a recuperação seja verificável):
#   dP/dt = 0.80 - 0.30 P - 0.90 P E
#   dE/dt = -0.25 E                     (pulso de elicitação na condição inicial)
#   dM/dt = 0.90 P E - 0.15 M
COEF_VERDADEIROS = {
    "P": {"1": 0.80, "P": -0.30, "P*E": -0.90},
    "E": {"E": -0.25},
    "M": {"P*E": 0.90, "M": -0.15},
}


def campo(_t, s):
    p, e, m = s
    return [0.80 - 0.30 * p - 0.90 * p * e,
            -0.25 * e,
            0.90 * p * e - 0.15 * m]


def termos_biblioteca(x):
    """Biblioteca polinomial de grau 2 sobre (P, E, M)."""
    p, e, m = x[:, 0], x[:, 1], x[:, 2]
    colunas = {
        "1": np.ones_like(p), "P": p, "E": e, "M": m,
        "P^2": p * p, "P*E": p * e, "P*M": p * m,
        "E^2": e * e, "E*M": e * m, "M^2": m * m,
    }
    return np.column_stack(list(colunas.values())), list(colunas)


def stlsq(theta, dx, limiar=0.05, n_iter=12):
    """Mínimos quadrados sequencialmente limiarizados (o coração do SINDy).

    Ajusta, zera coeficientes abaixo do limiar, reajusta apenas nos termos que
    sobraram. O limiar é o hiperparâmetro que decide o quão esparso é o modelo
    - e, como todo hiperparâmetro, precisa ser reportado.
    """
    xi = np.linalg.lstsq(theta, dx, rcond=None)[0]
    for _ in range(n_iter):
        pequenos = np.abs(xi) < limiar
        xi[pequenos] = 0.0
        for j in range(dx.shape[1]):
            grandes = ~pequenos[:, j]
            if grandes.sum():
                xi[grandes, j] = np.linalg.lstsq(theta[:, grandes], dx[:, j],
                                                 rcond=None)[0]
    return xi


# Condições iniciais = doses de elicitação distintas. A ordem importa: a
# primeira sozinha representa o experimento "uma dose, uma curva".
CONDICOES = [(0.2, 0.4), (0.2, 1.0), (0.6, 1.6), (1.2, 0.8), (0.1, 2.0)]


def simular(n_pontos, ruido_rel, rng, condicao=(0.2, 1.0)):
    """Integra o sistema para uma condição inicial e observa com ruído."""
    p0, e0 = condicao
    t = np.linspace(0, T_FINAL, n_pontos)
    sol = solve_ivp(campo, (0, T_FINAL), [p0, e0, 0.0], t_eval=t,
                    rtol=1e-10, atol=1e-12)
    limpo = sol.y.T
    escala = limpo.std(axis=0, keepdims=True)
    observado = limpo + rng.normal(0, ruido_rel, limpo.shape) * escala
    return t, limpo, observado


def identificar(series, limiar=0.05):
    """Suaviza, deriva por diferença central e resolve por STLSQ.

    `series` é uma lista de pares (t, observado): UMA entrada por experimento.
    Empilhar experimentos com elicitações diferentes é o que torna a matriz da
    biblioteca bem condicionada - ver a discussão de excitação persistente em
    identificação de sistemas.
    """
    blocos_x, blocos_dx, suaves = [], [], []
    for t, observado in series:
        dt = t[1] - t[0]
        janela = max(5, (len(t) // 20) * 2 + 1)
        if janela >= len(t):
            janela = len(t) - (1 - len(t) % 2)
        if janela < 5:
            raise ValueError("série curta demais para suavizar")
        suave = savgol_filter(observado, window_length=janela, polyorder=3, axis=0)
        derivada = np.gradient(suave, dt, axis=0)
        corte = slice(3, -3)
        blocos_x.append(suave[corte])
        blocos_dx.append(derivada[corte])
        suaves.append(suave)
    theta, nomes = termos_biblioteca(np.vstack(blocos_x))
    xi = stlsq(theta, np.vstack(blocos_dx), limiar=limiar)
    return xi, nomes, suaves


def experimento(n_condicoes, n_pontos, ruido, semente):
    """Roda n_condicoes elicitações e devolve (xi, nomes, t, limpo, observado)."""
    rng = np.random.default_rng(semente)
    series, limpos = [], []
    t = None
    for cond in CONDICOES[:n_condicoes]:
        t, limpo, obs = simular(n_pontos, ruido, rng, cond)
        series.append((t, obs))
        limpos.append((limpo, obs))
    xi, nomes, _ = identificar(series)
    return xi, nomes, t, limpos


def suporte_exato(xi, nomes):
    """O modelo achou exatamente os termos certos (nem a mais, nem a menos)?"""
    return all(
        {n for i, n in enumerate(nomes) if abs(xi[i, j]) > 1e-8}
        == set(COEF_VERDADEIROS[estado])
        for j, estado in enumerate(ESTADOS)
    )


def erro_de_coeficientes(xi, nomes):
    """Erro absoluto médio contra a verdade-terreno, sobre TODOS os termos."""
    erro = 0.0
    for j, estado in enumerate(ESTADOS):
        verdade = np.array([COEF_VERDADEIROS[estado].get(n, 0.0) for n in nomes])
        erro += np.abs(xi[:, j] - verdade).sum()
    return erro / (len(nomes) * len(ESTADOS))


def main() -> None:
    cabecalho(
        "DEMO 07 · Identificação esparsa da dinâmica de biossíntese",
        "[E5] PROPOSTA - método [E1]/[E2] em física, sem validação publicada aqui",
        "Dá para recuperar a equação da rota, e não só classificar o extrato?",
    )
    aplicar_estilo()
    rng = np.random.default_rng(SEMENTE)

    # ---- Uma dose só: o experimento que a maioria faz ---------------------
    xi_1, nomes, t, limpos_1 = experimento(1, 300, 0.02, SEMENTE)
    # ---- Cinco doses de elicitação: excitação persistente -----------------
    xi_5, _, _, limpos_5 = experimento(len(CONDICOES), 300, 0.02, SEMENTE)

    print("Sistema verdadeiro:")
    for estado, termos in COEF_VERDADEIROS.items():
        expr = "  ".join(f"{v:+.2f}·{k}" for k, v in termos.items())
        print(f"  d{estado}/dt = {expr}")

    for rotulo, xi_ in [("UMA curva (uma dose de elicitação)", xi_1),
                        (f"{len(CONDICOES)} curvas (doses distintas)", xi_5)]:
        print(f"\nIdentificado com {rotulo} · 300 pontos, ruído 2%:")
        for j, estado in enumerate(ESTADOS):
            termos = [f"{xi_[i, j]:+.3f}·{n}" for i, n in enumerate(nomes)
                      if abs(xi_[i, j]) > 1e-8]
            print(f"  d{estado}/dt = {'  '.join(termos) if termos else '0'}")
        print(f"  suporte exato: {'SIM' if suporte_exato(xi_, nomes) else 'NÃO'}"
              f" | erro médio de coeficiente: {erro_de_coeficientes(xi_, nomes):.4f}")

    print("\nEste é o resultado central deste demo, e ele NÃO é sobre o algoritmo:")
    print("uma única série temporal deixa a matriz da biblioteca mal condicionada")
    print("- as colunas ficam quase colineares porque a trajetória percorre um")
    print("subespaço estreito. O método falha por falta de EXCITAÇÃO, não por")
    print("falta de dado. Variar a dose de elicitação resolve. Isso é uma")
    print("prescrição de delineamento experimental derivada da matemática.")

    # ---- Fronteira de viabilidade ----------------------------------------
    print("\nVarredura: quando o método deixa de funcionar")
    linhas = []
    for n_cond in (1, 2, 3, 5):
        for n_pontos in (300, 120, 60, 30):
            for ruido in (0.02, 0.05, 0.10, 0.20):
                try:
                    xi_v, nomes_v, _, _ = experimento(
                        n_cond, n_pontos, ruido,
                        SEMENTE + n_cond * 7 + n_pontos + int(1000 * ruido))
                    erro = erro_de_coeficientes(xi_v, nomes_v)
                    ok = suporte_exato(xi_v, nomes_v)
                except Exception:
                    erro, ok = np.nan, False
                linhas.append({"n_condicoes": n_cond, "n_pontos": n_pontos,
                               "ruido": ruido, "erro_coef": erro,
                               "suporte_correto": ok})
    var = pd.DataFrame(linhas)

    print("\nFração de combinações com suporte EXATO, por nº de doses:")
    print((var.groupby("n_condicoes")["suporte_correto"].mean() * 100)
          .round(0).astype(int).to_string())
    print("\nErro médio de coeficiente (5 doses), linha=pontos, coluna=ruído:")
    print(var[var.n_condicoes == 5]
          .pivot(index="n_pontos", columns="ruido", values="erro_coef")
          .round(3).to_string())

    print("\nLeituras para o slide:")
    print(" · O método não ajusta uma curva: devolve uma EQUAÇÃO com termos")
    print("   interpretáveis - qual espécie limita, qual decai, onde está o")
    print("   produto de reação. É o oposto de uma caixa-preta.")
    print(" · O que decide o sucesso não é a quantidade de dado, é o DESENHO.")
    print("   Uma curva longa e limpa falha; várias curtas com elicitações")
    print("   diferentes funcionam.")
    print(" · A fronteira de viabilidade acima cai justamente no regime da")
    print("   metabolômica temporal real: poucos pontos, amostragem destrutiva,")
    print("   variância entre réplicas biológicas. Dizer isso é parte do")
    print("   argumento, não ressalva de rodapé.")
    print(" · Proposta testável [E5]: delinear a coleta para identificação -")
    print("   amostragem densa na fase transiente, elicitação como entrada")
    print("   controlada em pelo menos 3 doses, réplicas suficientes para")
    print("   estimar variância. Depois comparar o modelo identificado com")
    print("   medidas independentes de fluxo. Só então isso vira [E3].")

    # ---- Figura -----------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.2), layout="constrained")

    ax = axes[0]
    cores = [OKABE_ITO["azul"], OKABE_ITO["laranja"], OKABE_ITO["verde"]]
    marcadores = ["o", "s", "^"]
    limpo, observado = limpos_5[1]        # curva de dose intermediária

    def campo_identificado(_t, estado_atual):
        theta_s, _ = termos_biblioteca(np.array(estado_atual)[None, :])
        return (theta_s @ xi_5)[0]

    sol_id = solve_ivp(campo_identificado, (0, T_FINAL),
                       [CONDICOES[1][0], CONDICOES[1][1], 0.0],
                       t_eval=t, rtol=1e-8, atol=1e-10)
    for j, (estado, cor, mk) in enumerate(zip(ESTADOS, cores, marcadores)):
        ax.plot(t, limpo[:, j], color=cor, linewidth=2.2, label=f"{estado} · real")
        ax.plot(t[::14], observado[::14, j], linestyle="none", marker=mk,
                color=cor, alpha=0.45, markersize=5)
        ax.plot(sol_id.t, sol_id.y[j], color=cor, linestyle="--", linewidth=1.6,
                label=f"{estado} · identificado")
    ax.set(xlabel="Tempo (unidades arbitrárias)", ylabel="Concentração (u.a.)",
           title="Modelo recuperado a partir de 5 doses de elicitação")
    ax.legend(fontsize=9, ncol=2)
    aviso_dados_sinteticos(ax)

    ax = axes[1]
    verdade = np.column_stack([
        [COEF_VERDADEIROS[e].get(n, 0.0) for n in nomes] for e in ESTADOS])
    diferenca = np.column_stack([
        np.abs(xi_1 - verdade).sum(axis=1), np.abs(xi_5 - verdade).sum(axis=1)])
    imagem = ax.imshow(diferenca, cmap="Reds", aspect="auto",
                       vmin=0, vmax=max(1e-6, diferenca.max()))
    ax.set(xticks=[0, 1], xticklabels=["1 dose", "5 doses"],
           yticks=range(len(nomes)), yticklabels=nomes,
           title="Erro por termo da biblioteca")
    for i, nome in enumerate(nomes):
        if any(nome in COEF_VERDADEIROS[e] for e in ESTADOS):
            ax.get_yticklabels()[i].set_color(OKABE_ITO["azul"])
            ax.get_yticklabels()[i].set_fontweight("bold")
    fig.colorbar(imagem, ax=ax, label="|erro| somado sobre as 3 equações")
    ax.text(-0.5, len(nomes) + 0.9, "em azul: termos do sistema real",
            fontsize=9, color=OKABE_ITO["azul"])

    ax = axes[2]
    for n_cond, cor, mk in zip([1, 2, 3, 5],
                               [OKABE_ITO["vermelho"], OKABE_ITO["laranja"],
                                OKABE_ITO["azul"], OKABE_ITO["verde"]],
                               ["X", "s", "^", "o"]):
        sub = (var[(var.n_condicoes == n_cond)]
               .groupby("n_pontos")["erro_coef"].mean().sort_index())
        ax.plot(sub.index, sub.values, marker=mk, color=cor,
                label=f"{n_cond} dose{'s' if n_cond > 1 else ''}")
    ax.set(xscale="log", yscale="log", xlabel="Pontos temporais por curva",
           ylabel="Erro médio de coeficiente (média sobre ruídos)",
           title="Desenho experimental > quantidade de dado")
    ax.legend(fontsize=10)
    aviso_dados_sinteticos(ax)

    caminho = exportar(
        fig, "07_dinamica_biossintese",
        proveniencia={
            "script": "demos/07_dinamica_biossintese.py",
            "semente": SEMENTE,
            "sistema_verdadeiro": {k: v for k, v in COEF_VERDADEIROS.items()},
            "biblioteca": "polinomial de grau 2 em (P, E, M)",
            "limiar_stlsq": 0.05,
            "suavizacao": "Savitzky-Golay, polyorder=3, janela ~ n/12",
            "derivada": "diferença central (numpy.gradient)",
        },
    )
    print(f"\nFigura: {caminho}")

    DIR_DADOS.mkdir(parents=True, exist_ok=True)
    destino = DIR_DADOS / "07_fronteira_identificacao.csv"
    var.to_csv(destino, index=False)
    print(f"Tabela: {destino}")


if __name__ == "__main__":
    main()
