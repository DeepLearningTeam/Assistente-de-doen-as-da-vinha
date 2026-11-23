import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from .indice import obter_cliente, pesquisar_query_qdrant, descrever_colecao
from .config import carregar_config
from .apoio import RAIZ, ErroProjeto
from .metricas import posicao_fonte

# Avaliação de recuperação documental integrada, ligada aos mesmos parâmetros da aplicação.


def carregar_perguntas(caminho, split):
    with open(caminho, encoding="utf-8-sig", newline="") as f:
        linhas = list(csv.DictReader(f))
    if split != "todos":
        linhas = [l for l in linhas if l["split"] == split]
    return linhas


PASTA_SCRIPT = Path(__file__).resolve().parent

def resolver_caminho(caminho: str) -> Path:
    """Aceita caminho absoluto, relativo à pasta atual, ou relativo à pasta do script."""
    p = Path(caminho).expanduser()
    if p.is_absolute() and p.exists():
        return p
    for base in (Path.cwd(), PASTA_SCRIPT):
        candidato = (base / p).resolve()
        if candidato.exists():
            return candidato
    raise FileNotFoundError(
        f"Não encontrei '{caminho}'.\n"
        f"  Procurei em: {Path.cwd()} e {PASTA_SCRIPT}"
    )


def main():
    ap = argparse.ArgumentParser(description="Avalia a recuperação do RAG.")
    ap.add_argument("csv", nargs="?", default=str(RAIZ / "data/perguntas/perguntas.csv"))
    ap.add_argument("--modo", choices=["denso", "hibrida"], default="hibrida",
                    help="tipo de pesquisa a avaliar")
    ap.add_argument("--split", default="validacao",
                    choices=["validacao", "teste", "todos"])
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--saida", help="CSV com o resultado por pergunta")
    ap.add_argument("--reranker", action="store_true", help="reordena os candidatos com o cross-encoder")
    ap.add_argument("--candidatos", type=int, default=30)
    ap.add_argument("--limite", type=int, help="Ensaio curto apenas na validação.")
    args = ap.parse_args()

    perguntas = carregar_perguntas(resolver_caminho(args.csv), args.split)
    if args.limite is not None:
        if args.limite < 1 or args.split != "validacao":
            ap.error("--limite exige um número positivo e --split validacao.")
        perguntas = perguntas[:args.limite]
    if not 1 <= args.k <= 10 or args.candidatos < args.k:
        ap.error("Utilizar 1 <= k <= 10 e candidatos >= k.")
    config = carregar_config().model_copy(update={"top_k": args.k, "pesquisa": args.modo,
                                                  "k_candidatos": args.candidatos})
    client = obter_cliente()
    info = descrever_colecao(config)
    print(json.dumps(info, ensure_ascii=False))
    indexados = set(info["documentos"])

    print(f"Split: {args.split} | modo: {args.modo} | reranker: {args.reranker} | "
          f"candidatos: {args.candidatos} | perguntas: {len(perguntas)} | "
          f"documentos no índice: {len(indexados)}")
    print("Estado de validação:",
          dict(Counter(q["estado_validacao"] for q in perguntas)))

    def buscar(pergunta):
        return pesquisar_query_qdrant(
            pergunta, client, modo=config.pesquisa, usar_reranker=args.reranker,
            k_candidatos=config.k_candidatos, limiar=None, config=config,
        )

    resp, scores_acerto = [], []
    scores_recusar, sem_indice, contaminadas, linhas = [], [], [], []

    for q in perguntas:
        res = buscar(q["pergunta"])[:args.k]
        top1 = res[0] if res else None
        doc_id = q["doc_id"]

        if q["expected_action"] == "recusar":
            if doc_id in indexados:                    # fonte reservada indexada por engano
                contaminadas.append(q["id"])
            scores_recusar.append(top1["score"] if top1 else float("-inf"))
            linhas.append({"id": q["id"], "acao": "recusar", "split": q["split"],
                           "top1_doc": top1 and top1["nome_documento"],
                           "top1_pag": top1 and top1["pagina"],
                           "top1_score": top1 and top1["score"]})
            continue

        if doc_id not in indexados:
            sem_indice.append(q["id"])
            # Não retirar estas perguntas do denominador: são falhas de recuperação.
            ranks = {"documento": None, "pág ±0": None, "pág ±1": None}
            resp.append((q, ranks, res))
            linhas.append({"id": q["id"], "acao": "responder", "split": q["split"],
                           **{f"rank_{k}": v for k, v in ranks.items()},
                           "top1_doc": top1 and top1["nome_documento"],
                           "top1_pag": top1 and top1["pagina"],
                           "top1_score": top1 and top1["score"]})
            continue

        paginas = {int(x) for x in re.findall(r"\d+", q["pagina_pdf"])}
        ranks = {
            "documento": posicao_fonte(res, doc_id)[0],
            "pág ±0":    posicao_fonte(res, doc_id, paginas)[0],
            "pág ±1":    posicao_fonte(res, doc_id, paginas, tolerancia=1)[0],
        }
        score_hit = posicao_fonte(res, doc_id, paginas, tolerancia=1)[1]
        if score_hit is not None:
            scores_acerto.append(score_hit)
        resp.append((q, ranks, res))
        linhas.append({"id": q["id"], "acao": "responder", "split": q["split"],
                       **{f"rank_{k}": v for k, v in ranks.items()},
                       "top1_doc": top1 and top1["nome_documento"],
                       "top1_pag": top1 and top1["pagina"],
                       "top1_score": top1 and top1["score"]})

    # ── Recuperação (perguntas 'responder') ──
    n = len(resp)
    print(f"\n'responder' avaliadas: {n} | sem documento no índice: "
          f"{len(sem_indice)} {sem_indice}")
    if contaminadas:
        print(f"ATENÇÃO: 'recusar' cujo documento ESTÁ indexado: {contaminadas}")
    if n:
        for chave in ["documento", "pág ±0", "pág ±1"]:
            rs = [r[chave] for _, r, _ in resp]
            recall = sum(r is not None for r in rs) / n
            mrr = sum(1 / r for r in rs if r) / n
            print(f"  {chave:10s} Recall@{args.k}: {recall:6.1%}   MRR: {mrr:.3f}")

        print("\nFalhas (página exata):")
        for q, r, res in resp:
            if r["pág ±0"] is None:
                achou = ("só a página vizinha" if r["pág ±1"] else
                         "só o documento" if r["documento"] else "nem o documento")
                print(f"  {q['id']} [{q['tipo']}] {q['pergunta'][:70]}…  → {achou}")
                for c in res[:3]:
                    print(f"       {c['nome_documento']} pág {c['pagina']}  "
                          f"score {c.get('score') or 0:.3f}")

    # ── Abstenção: varredura de limiar ──
    if n and scores_recusar and scores_acerto and args.split == "validacao" and (args.modo == "denso" or args.reranker):
        validos = [s for s in scores_recusar if s != float("-inf")] + scores_acerto
        lo, hi = min(validos), max(validos)
        for i in range(16):
            t = lo + (hi - lo) * i / 15
            recall_t = sum(s >= t for s in scores_acerto) / n
            abst_t = sum(s < t for s in scores_recusar) / len(scores_recusar)
            print(f"  limiar {t:7.3f}   recall {recall_t:6.1%}   abstenção {abst_t:6.1%}")

    if args.saida and linhas:
        campos = sorted({k for l in linhas for k in l})
        Path(args.saida).parent.mkdir(parents=True, exist_ok=True)
        with open(args.saida, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=campos)
            w.writeheader()
            w.writerows(linhas)


if __name__ == "__main__":
    try:
        main()
    except Exception as erro:
        raise SystemExit(str(erro) if isinstance(erro, ErroProjeto) else "Falhou a avaliação de recuperação. Verificar src.indice info e os modelos.") from None
