"""
Executa todos os demos do SINANBIO em sequência e relata tempo e status.

Uso:
    python run_all.py              # roda tudo
    python run_all.py 02 05        # roda apenas os demos 02 e 05

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
import subprocess
import sys
import time

RAIZ = pathlib.Path(__file__).resolve().parent
DEMOS = sorted((RAIZ / "demos").glob("[0-9][0-9]_*.py"))


def main() -> int:
    filtros = sys.argv[1:]
    selecionados = [d for d in DEMOS
                    if not filtros or any(d.name.startswith(f) for f in filtros)]
    if not selecionados:
        print("Nenhum demo corresponde ao filtro.", file=sys.stderr)
        return 1

    print(f"SINANBIO · executando {len(selecionados)} demo(s)")
    print("Todos os dados gerados são SINTÉTICOS.\n")

    falhas = []
    for demo in selecionados:
        inicio = time.perf_counter()
        processo = subprocess.run([sys.executable, str(demo)],
                                  capture_output=True, text=True)
        duracao = time.perf_counter() - inicio
        estado = "ok " if processo.returncode == 0 else "FALHA"
        print(f"[{estado}] {demo.name:<34s} {duracao:6.1f} s")
        if processo.returncode != 0:
            falhas.append(demo.name)
            print(processo.stderr[-2000:], file=sys.stderr)

    print(f"\nFiguras em : {RAIZ / 'figuras'}")
    print(f"Tabelas em : {RAIZ / 'dados_sinteticos'}")
    if falhas:
        print(f"\nFalharam: {', '.join(falhas)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
