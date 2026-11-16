import argparse
import csv
import inspect
import json
import re
from collections import Counter

from qdrant_client import QdrantClient
from rag import QDRANT_URL, QDRANT_API_KEY, NOME_COLECAO, pesquisar_query_qdrant

def carregar_perguntas(caminho, split):
    with open(caminho, encoding="utf-8-sig", newline="") as f:
        linhas = list(csv.DictReader(f))
    if split != "todos":
        linhas = [l for l in linhas if l["split"] == split]
    return linhas


def documentos_indexados(client):
    nomes, offset = set(), None
    while True:
        pontos, offset = client.scroll(
            NOME_COLECAO, limit=1000, offset=offset,
            with_payload=["nome_documento"], with_vectors=False,
        )
        nomes.update(p.payload["nome_documento"] for p in pontos)
        if offset is None:
            return nomes


def resolver_doc(doc_id, mapa, indexados):
    """nome_documento indexado correspondente ao doc_id, ou None."""
    if doc_id in mapa:
        return mapa[doc_id] if mapa[doc_id] in indexados else None
    padrao = re.compile(rf"{re.escape(doc_id)}(?!\d)", re.IGNORECASE)
    return next((n for n in sorted(indexados) if padrao.match(n)), None)


def procurar(res, nome, paginas, tol=0, so_doc=False):
    """(rank, score) do primeiro resultado correto, ou (None, None)."""
    for i, c in enumerate(res, 1):
        if c["nome_documento"] != nome:
            continue
        if so_doc or any(abs(c["pagina"] - p) <= tol for p in paginas):
            return i, c.get("score")
    return None, None

from pathlib import Path

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
    ap.add_argument("csv", nargs="?", default="perguntas.csv")
    ap.add_argument("--modo", choices=["denso", "hibrida"], default="hibrida",
                    help="tipo de pesquisa a avaliar")
    ap.add_argument("--split", default="validacao",
                    choices=["validacao", "teste", "todos"])
    ap.add_argument("--mapa", help='JSON {"D09": "ficheiro.pdf"}; sem isto, '
                                   "procura PDFs que começam pelo doc_id")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--saida", help="CSV com o resultado por pergunta")
    ap.add_argument("--reranker", action="store_true", help="reordena os candidatos com o cross-encoder")
    ap.add_argument("--candidatos", type=int, default=30)
    args = ap.parse_args()

    mapa = {}
    if args.mapa:
        with open(resolver_caminho(args.mapa), encoding="utf-8") as f:
            mapa = json.load(f)

    perguntas = carregar_perguntas(resolver_caminho(args.csv), args.split)
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    indexados = documentos_indexados(client)

    print(f"Split: {args.split} | modo: {args.modo} | reranker: {args.reranker} | "
          f"candidatos: {args.candidatos} | perguntas: {len(perguntas)} | "
          f"documentos no índice: {len(indexados)}")
    print("Estado de validação:",
          dict(Counter(q["estado_validacao"] for q in perguntas)))

    # Depois do passo 4, a função aceita 'limiar'; aqui desligamos o corte
    # para ver os scores todos e poder varrer limiares.
    params = inspect.signature(pesquisar_query_qdrant).parameters
    if "modo" not in params:
        raise SystemExit("pesquisar_query_qdrant (indice2.py) não tem o parâmetro 'modo'.")

    def buscar(pergunta):
        kwargs = {"modo": args.modo,
                  "usar_reranker": args.reranker,
                  "k_candidatos": args.candidatos}
        if "limiar" in params:                      # só existirá depois do reranker
            kwargs["limiar"] = float("-inf")
        return pesquisar_query_qdrant(pergunta, client, **kwargs)

    resp, scores_acerto = [], []
    scores_recusar, sem_indice, contaminadas, linhas = [], [], [], []

    for q in perguntas:
        res = buscar(q["pergunta"])[:args.k]
        top1 = res[0] if res else None
        nome = resolver_doc(q["doc_id"], mapa, indexados)

        if q["expected_action"] == "recusar":
            if nome:                                   # holdout indexado por engano
                contaminadas.append(q["id"])
            scores_recusar.append(top1["score"] if top1 else float("-inf"))
            linhas.append({"id": q["id"], "acao": "recusar", "split": q["split"],
                           "top1_doc": top1 and top1["nome_documento"],
                           "top1_pag": top1 and top1["pagina"],
                           "top1_score": top1 and top1["score"]})
            continue

        if nome is None:                               # documento não indexado
            sem_indice.append(q["id"])
            continue

        paginas = {int(x) for x in re.findall(r"\d+", q["pagina_pdf"])}
        ranks = {
            "documento": procurar(res, nome, paginas, so_doc=True)[0],
            "pág ±0":    procurar(res, nome, paginas, tol=0)[0],
            "pág ±1":    procurar(res, nome, paginas, tol=1)[0],
        }
        score_hit = procurar(res, nome, paginas, tol=1)[1]
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
    if n and scores_recusar and (args.modo == "denso" or args.reranker):
        validos = [s for s in scores_recusar if s != float("-inf")] + scores_acerto
        lo, hi = min(validos), max(validos)
        for i in range(16):
            t = lo + (hi - lo) * i / 15
            recall_t = sum(s >= t for s in scores_acerto) / n
            abst_t = sum(s < t for s in scores_recusar) / len(scores_recusar)
            print(f"  limiar {t:7.3f}   recall {recall_t:6.1%}   abstenção {abst_t:6.1%}")

    if args.saida and linhas:
        campos = sorted({k for l in linhas for k in l})
        with open(args.saida, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=campos)
            w.writeheader()
            w.writerows(linhas)


if __name__ == "__main__":
    main()