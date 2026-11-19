"""Exportar calibração e casos de erro de uma execução de classificação já guardada."""
import argparse
from pathlib import Path

from .apoio import ErroProjeto
from .metricas import PAPER, agrupar, calibracao_imagem, carregar, csv_gravar, previsao


def analisar(nome, destino, figuras=False):
    manifesto, itens = carregar(nome)
    if manifesto["parametros"]["tipo"] != "imagem":
        raise ErroProjeto("Este comando analisa classificação de imagens, não respostas do assistente.")
    bins, casos = [], []
    for (modo, seed), grupo in agrupar(itens, lambda i: (i["modo"], i["seed"])).items():
        bins.extend({"modo": modo, "seed": seed, **b} for b in calibracao_imagem(grupo))
        for i in grupo:
            if previsao(i) != i["classe_ref"]:
                casos.append({
                    "modo": modo, "seed": seed, "imagem": i["imagem"], "classe_ref": i["classe_ref"],
                    "classe_prevista": previsao(i), "estado": i["estado"],
                    "confianca": (i.get("visual") or {}).get("confianca"), "chave": i["chave"],
                })
    # Erros confiantes primeiro; abstenções sem confiança no fim. Não é uma avaliação humana.
    casos.sort(key=lambda c: (c["confianca"] is None, -(c["confianca"] or 0), c["imagem"]))
    csv_gravar(destino / "calibracao.csv", (
        "modo", "seed", "bin", "limite_inf", "limite_sup", "n", "confianca_media", "acuracia",
    ), bins)
    csv_gravar(destino / "piores_casos.csv", (
        "modo", "seed", "imagem", "classe_ref", "classe_prevista", "estado", "confianca", "chave",
    ), casos)
    if figuras:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as erro:
            raise ErroProjeto("CSV exportados. Para as figuras instalar matplotlib: .\\.venv\\Scripts\\python.exe -m pip install matplotlib") from erro
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot([0, 1], [0, 1], "--", color="grey", label="Correspondência ideal")
        for (modo, seed), grupo in agrupar(bins, lambda b: (b["modo"], b["seed"])).items():
            preenchidos = [b for b in grupo if b["n"]]
            if preenchidos:
                ax.plot([b["confianca_media"] for b in preenchidos], [b["acuracia"] for b in preenchidos],
                        "o-", label=f"{modo}, seed {seed}")
        ax.set(xlabel="Confiança média declarada", ylabel="Proporção de acertos",
               title=f"Calibração · {nome}", xlim=(0, 1), ylim=(0, 1))
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(destino / "calibracao.png", dpi=180)
        plt.close(fig)
    print(f"Análise dos itens guardados em {destino}; {len(casos)} erros ou abstenções. Não afinar no teste.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execucao", required=True)
    parser.add_argument("--destino", type=Path)
    parser.add_argument("--figuras", action="store_true", help="Requer matplotlib, dependência opcional.")
    args = parser.parse_args()
    try:
        analisar(args.execucao, args.destino or PAPER / "analise" / args.execucao, args.figuras)
    except (ErroProjeto, OSError, ValueError) as erro:
        parser.exit(1, f"{erro}\n")


if __name__ == "__main__":
    main()
