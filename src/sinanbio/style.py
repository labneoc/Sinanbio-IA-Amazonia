"""
Estilo e utilitários de figura para o repositório SINANBIO.

Paleta Okabe-Ito (segura para as formas mais comuns de daltonismo). A cor nunca
é o único canal de informação: marcadores, tracejados e rótulos diretos são
usados de forma redundante, como recomendado para figuras acessíveis.

------------------------------------------------------------------------------
Prof. Bruno Duarte Gomes
Faculdade de Biotecnologia, ICB, UFPA
Laboratório de Neurofisiologia Eduardo Oswaldo Cruz (LNEOC) - ICB - UFPA
Laboratório de Simulação e Biologia Computacional (SimBic) - CCAD-IA, UFPA
Licença MIT.
------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# Okabe & Ito (2008), paleta qualitativa acessível.
OKABE_ITO = {
    "preto": "#000000",
    "laranja": "#E69F00",
    "azul_claro": "#56B4E9",
    "verde": "#009E73",
    "amarelo": "#F0E442",
    "azul": "#0072B2",
    "vermelho": "#D55E00",
    "rosa": "#CC79A7",
    "cinza": "#999999",
}

CICLO = [
    OKABE_ITO["azul"],
    OKABE_ITO["vermelho"],
    OKABE_ITO["verde"],
    OKABE_ITO["laranja"],
    OKABE_ITO["rosa"],
    OKABE_ITO["azul_claro"],
]

RAIZ = Path(__file__).resolve().parents[2]
DIR_FIGURAS = RAIZ / "figuras"
DIR_DADOS = RAIZ / "dados_sinteticos"


def aplicar_estilo(escala: str = "apresentacao") -> None:
    """Aplica um estilo global. 'apresentacao' usa fontes maiores (projeção)."""
    base = {
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.prop_cycle": mpl.cycler(color=CICLO),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "image.cmap": "viridis",
    }
    if escala == "apresentacao":
        base.update(
            {
                "font.size": 12,
                "axes.titlesize": 14,
                "axes.labelsize": 12,
                "xtick.labelsize": 11,
                "ytick.labelsize": 11,
                "legend.fontsize": 11,
                "lines.linewidth": 2.0,
                "lines.markersize": 6,
            }
        )
    else:  # 'publicacao'
        base.update(
            {
                "font.size": 8,
                "axes.titlesize": 9,
                "axes.labelsize": 8,
                "xtick.labelsize": 7,
                "ytick.labelsize": 7,
                "legend.fontsize": 7,
                "lines.linewidth": 1.2,
                "lines.markersize": 3.5,
            }
        )
    mpl.rcParams.update(base)


def exportar(fig, nome: str, proveniencia: dict | None = None) -> Path:
    """Salva a figura em PNG e grava um manifesto JSON de procedência ao lado.

    O manifesto existe para que qualquer figura levada ao slide possa ser
    rastreada até o script, a semente aleatória e as transformações que a
    geraram. Nenhuma figura deste repositório representa dado experimental.
    """
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    caminho = DIR_FIGURAS / f"{nome}.png"
    fig.savefig(caminho)
    if proveniencia is not None:
        manifesto = dict(proveniencia)
        manifesto["natureza_dos_dados"] = "SINTÉTICO - gerado por simulação, sem origem experimental"
        manifesto["arquivo"] = caminho.name
        (DIR_FIGURAS / f"{nome}.manifesto.json").write_text(
            json.dumps(manifesto, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    plt.close(fig)
    return caminho


def aviso_dados_sinteticos(ax, texto: str = "DADOS SINTÉTICOS · ilustração didática") -> None:
    """Carimba a figura. Evita que um gráfico didático seja lido como resultado."""
    ax.text(
        0.995,
        -0.16,
        texto,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color=OKABE_ITO["cinza"],
    )


def cabecalho(titulo: str, nivel_evidencia: str, pergunta: str) -> None:
    """Imprime o cabeçalho epistêmico de cada demo no terminal."""
    linha = "=" * 78
    print(linha)
    print(titulo)
    print(linha)
    print(f"Pergunta da palestra : {pergunta}")
    print(f"Nível de evidência   : {nivel_evidencia}")
    print("Dados                : SINTÉTICOS (simulação; nenhum dado experimental real)")
    print(linha)
