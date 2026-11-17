"""Pesquisa híbrida e reranking do colega, ligados à configuração da aplicação."""
import argparse
import hashlib
import json
import os
import sys
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, FieldCondition, Filter, Fusion, FusionQuery, MatchAny,
    Modifier, Prefetch, SparseVector,
)

from .apoio import ErroProjeto, RAIZ
from .config import carregar_config
from .corpus import HOLDOUT, ler_corpus, selecionadas

NOME_VETOR_DENSO = "vetor_denso"
NOME_VETOR_ESPARSO = "bm25"
DIM_VETOR = 1024


@lru_cache(maxsize=2)
def obter_modelo_bge(modelo="BAAI/bge-m3"):
    # Importação tardia: sem RAG não é necessário carregar Torch ou embeddings.
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(modelo)


@lru_cache(maxsize=1)
def obter_bm25():
    from fastembed import SparseTextEmbedding
    return SparseTextEmbedding("Qdrant/bm25", language="portuguese")


@lru_cache(maxsize=2)
def obter_reranker(modelo="BAAI/bge-reranker-v2-m3"):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(modelo, max_length=512)


@lru_cache(maxsize=4)
def _cliente(url, chave, local):
    if local:
        caminho = Path(local)
        return QdrantClient(path=str(caminho if caminho.is_absolute() else RAIZ / caminho))
    if not url:
        raise ErroProjeto("Falta QDRANT_URL no .env; para uso local define QDRANT_LOCAL_PATH.")
    return QdrantClient(url=url, api_key=chave or None, timeout=30)


def obter_cliente():
    load_dotenv(RAIZ / ".env", override=False)
    return _cliente(os.getenv("QDRANT_URL", ""), os.getenv("QDRANT_API_KEY", ""),
                    os.getenv("QDRANT_LOCAL_PATH", ""))


def validar_colecao(client, config):
    """Confirma o esquema por leitura; não prova a origem dos embeddings."""
    info = client.get_collection(config.qdrant_colecao)
    parametros = info.config.params
    densos = parametros.vectors
    esparsos = parametros.sparse_vectors or {}
    if not isinstance(densos, dict) or NOME_VETOR_DENSO not in densos:
        raise ErroProjeto("Coleção incompatível: falta o vetor nomeado vetor_denso.")
    vetor = densos[NOME_VETOR_DENSO]
    if vetor.size != DIM_VETOR or vetor.distance != Distance.COSINE:
        raise ErroProjeto("Coleção incompatível: esperados 1024 valores e distância Cosine.")
    if NOME_VETOR_ESPARSO not in esparsos or esparsos[NOME_VETOR_ESPARSO].modifier != Modifier.IDF:
        raise ErroProjeto("Coleção incompatível: esperado bm25 com modifier IDF.")
    return info


def filtro_corpus():
    # Impede que fontes reservadas entrem nas respostas numa coleção contaminada.
    return Filter(must=[
        FieldCondition(key="doc_id", match=MatchAny(any=list(selecionadas(ler_corpus())))),
        FieldCondition(key="estado_fonte", match=MatchAny(any=["sim", "condicional"])),
    ])


def recuperar_candidatos(pergunta, client, modo="hibrida", k=30, config=None):
    """Fase 1 (Qdrant): pesquisa densa ou híbrida com fusão RRF."""
    config = config or carregar_config()
    filtro = filtro_corpus()
    denso = obter_modelo_bge(config.embeddings_modelo).encode(
        [pergunta], normalize_embeddings=True,
    )[0].tolist()
    if modo == "denso":
        resposta = client.query_points(
            collection_name=config.qdrant_colecao, query=denso,
            using=NOME_VETOR_DENSO, limit=k, query_filter=filtro, with_payload=True,
        )
    else:
        sp = next(iter(obter_bm25().query_embed(pergunta)))
        resposta = client.query_points(
            collection_name=config.qdrant_colecao,
            prefetch=[
                Prefetch(query=denso, using=NOME_VETOR_DENSO, limit=k, filter=filtro),
                Prefetch(query=SparseVector(indices=sp.indices.tolist(), values=sp.values.tolist()),
                         using=NOME_VETOR_ESPARSO, limit=k, filter=filtro),
            ],
            query=FusionQuery(fusion=Fusion.RRF), limit=k, with_payload=True,
        )
    return [{**(h.payload or {}), "id": str(h.id), "score_pesquisa": h.score}
            for h in resposta.points]


def reranquear(pergunta, candidatos, config=None):
    """Fase 2 (Python): cross-encoder sobre cada par pergunta/passagem."""
    if not candidatos:
        return []
    config = config or carregar_config()
    scores = obter_reranker(config.reranker_modelo).predict(
        [(pergunta, c["texto"]) for c in candidatos], batch_size=16,
    )
    for c, s in zip(candidatos, scores):
        c["score_reranker"] = float(s)
    return sorted(candidatos, key=lambda c: c["score_reranker"], reverse=True)


def pesquisar_query_qdrant(pergunta, client, modo="hibrida", usar_reranker=True,
                           k_candidatos=30, limiar=None, config=None):
    config = config or carregar_config()
    candidatos = recuperar_candidatos(pergunta, client, modo, k_candidatos, config)
    if usar_reranker:
        candidatos = reranquear(pergunta, candidatos, config)
    for c in candidatos:
        c["score"] = c["score_reranker"] if usar_reranker else c["score_pesquisa"]
    # None significa sem corte. Scores RRF/cosseno não são scores do reranker.
    if usar_reranker and limiar is not None:
        candidatos = [c for c in candidatos if c["score"] >= limiar]
    return candidatos[:config.top_k]


def normalizar_passagem(r):
    obrigatorios = ("doc_id", "texto", "pagina", "titulo", "entidade", "data", "url")
    if any(k not in r for k in obrigatorios):
        raise ErroProjeto("Payload incompatível: faltam campos documentais. Executa src.indice info.")
    if not isinstance(r["pagina"], int) or r["pagina"] < 1 or not r["texto"].strip():
        raise ErroProjeto("Payload incompatível: página ou texto inválido.")
    return {**r, "doc": r["doc_id"], "contexto": r.get("cabecalhos", ""),
            "similaridade": r["score"]}


def recuperar(config, consulta):
    """Top-k antes do corte para Recall@k; o pipeline aplica o limiar uma vez."""
    try:
        client = obter_cliente()
        validar_colecao(client, config)
        resultados = pesquisar_query_qdrant(
            consulta, client, modo=config.pesquisa, k_candidatos=config.k_candidatos,
            limiar=None, config=config,
        )
        return [normalizar_passagem(r) for r in resultados]
    except ErroProjeto:
        raise
    except Exception:
        # Exceções HTTP podem conter o URL privado; não as passar para UI/logs.
        raise ErroProjeto("Falhou a recuperação RAG. Confirma modelos, rede e src.indice info.") from None


def descrever_colecao(config):
    """Inspeção integral de payloads, só de leitura, para diagnóstico e avaliação."""
    client = obter_cliente()
    info = validar_colecao(client, config)
    linhas, docs, reservados, invalidos = [], set(), set(), 0
    offset = None
    fontes = ler_corpus()
    while True:
        pontos, offset = client.scroll(config.qdrant_colecao, limit=256, offset=offset,
                                       with_payload=True, with_vectors=False)
        for p in pontos:
            payload = p.payload or {}
            doc = payload.get("doc_id", "")
            docs.add(doc)
            if doc in HOLDOUT:
                reservados.add(doc)
            try:
                normalizar_passagem({**payload, "id": str(p.id), "score": 0})
                if doc not in fontes or payload["pagina"] > int(fontes[doc]["paginas"]):
                    raise ValueError()
                if payload.get("estado_fonte") not in {"sim", "condicional"}:
                    raise ValueError()
            except (ErroProjeto, ValueError):
                invalidos += 1
            linhas.append(json.dumps({"id": str(p.id), "payload": payload}, sort_keys=True, ensure_ascii=False))
        if offset is None:
            break
    if not linhas or invalidos or reservados:
        raise ErroProjeto(f"Coleção vazia/incompatível: payloads inválidos={invalidos}, reservados={sorted(reservados)}.")
    return {
        "colecao": config.qdrant_colecao, "pontos": info.points_count, "documentos": sorted(docs),
        "documentos_em_falta": sorted(set(selecionadas(fontes)) - docs),
        "payloads_sha256": hashlib.sha256("\n".join(sorted(linhas)).encode()).hexdigest(),
        "proveniencia": "Confirmar com o autor pesos BGE-M3, revisão, BM25 português e extração; o esquema não prova a proveniência.",
    }


def main():
    parser = argparse.ArgumentParser(description="Qdrant: diagnóstico e pesquisa só de leitura.")
    sub = parser.add_subparsers(dest="acao", required=True)
    sub.add_parser("info")
    pesquisar = sub.add_parser("pesquisar")
    pesquisar.add_argument("pergunta")
    args = parser.parse_args()
    config = carregar_config()
    try:
        resultado = descrever_colecao(config) if args.acao == "info" else recuperar(config, args.pergunta)
        print(json.dumps(resultado, ensure_ascii=False, indent=2))
    except Exception as erro:
        mensagem = str(erro) if isinstance(erro, ErroProjeto) else "Ligação/coleção indisponível. Confirma o .env e a rede; credenciais omitidas."
        sys.exit(mensagem)


if __name__ == "__main__":
    main()
