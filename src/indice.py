import os

from dotenv import load_dotenv
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector, Prefetch, FusionQuery, Fusion

from sentence_transformers import SentenceTransformer, CrossEncoder

from functools import lru_cache
from .apoio import ErroProjeto

load_dotenv()

# ── Configuração ───────────────────────────────────────────────────────────

MODELO_EMBEDDINGS = "BAAI/bge-m3"
MODELO_RERANKER = "BAAI/bge-reranker-v2-m3"
NOME_COLECAO = "videira_fitossanidade"

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

TOP_K = 5
K_CANDIDATOS=30
LIMIAR_RERANKER = 0.60

NOME_VETOR_DENSO = "vetor_denso"
NOME_VETOR_ESPARSO = "bm25"

_reranker = None
_bm25 = None
_modelo_bge = None

# ── Modelo de embeddings ─────────────────────────
def obter_modelo_bge() -> SentenceTransformer:
    """Carrega o bge-m3 só na primeira utilização (evita ~2 GB em cada import)."""
    global _modelo_bge
    if _modelo_bge is None:
        _modelo_bge = SentenceTransformer(MODELO_EMBEDDINGS)
    return _modelo_bge

def obter_bm25() -> SparseTextEmbedding:
    global _bm25
    if _bm25 is None:
        _bm25 = SparseTextEmbedding("Qdrant/bm25", language="portuguese")
    return _bm25

def obter_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(MODELO_RERANKER, max_length=512)
    return _reranker

@lru_cache(maxsize=1)
def obter_cliente() -> QdrantClient:
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise ErroProjeto("Faltam QDRANT_URL e/ou QDRANT_API_KEY no .env")
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# ── Recuperação ────────────────────────────────────────────────────────────
def recuperar_candidatos(pergunta: str, client: QdrantClient,
                         modo: str = "hibrida", k: int = K_CANDIDATOS) -> list[dict]:
    """Fase 1 (Qdrant): devolve k candidatos, sem reranking."""
    denso = obter_modelo_bge().encode([pergunta], normalize_embeddings=True)[0].tolist()

    if modo == "denso":
        resposta = client.query_points(
            collection_name=NOME_COLECAO, query=denso,
            using=NOME_VETOR_DENSO, limit=k,
        )
    else:
        sp = next(iter(obter_bm25().query_embed(pergunta)))
        resposta = client.query_points(
            collection_name=NOME_COLECAO,
            prefetch=[
                Prefetch(query=denso, using=NOME_VETOR_DENSO, limit=k),
                Prefetch(query=SparseVector(indices=sp.indices.tolist(),
                                            values=sp.values.tolist()),
                         using=NOME_VETOR_ESPARSO, limit=k),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=k,
        )
    return [{**h.payload, "id": h.id, "score_pesquisa": h.score} for h in resposta.points]

def reranquear(pergunta: str, candidatos: list[dict]) -> list[dict]:
    """Fase 2 (Python): o cross-encoder lê (pergunta, chunk) em conjunto e reordena."""
    if not candidatos:
        return []
    scores = obter_reranker().predict(
        [(pergunta, c["texto"]) for c in candidatos], batch_size=16
    )
    for c, s in zip(candidatos, scores):
        c["score_reranker"] = float(s)
    return sorted(candidatos, key=lambda c: c["score_reranker"], reverse=True)

def pesquisar_query_qdrant(pergunta: str, client: QdrantClient,
                           modo: str = "hibrida", usar_reranker: bool = True,
                           k_candidatos: int = K_CANDIDATOS,
                           limiar: float | None = None) -> list[dict]:
    candidatos = recuperar_candidatos(pergunta, client, modo, k_candidatos)

    if usar_reranker:
        candidatos = reranquear(pergunta, candidatos)
        for c in candidatos:
            c["score"] = c["score_reranker"]
        corte = LIMIAR_RERANKER if limiar is None else limiar
        if corte is not None:                      # o corte só vale para o score do reranker
            candidatos = [c for c in candidatos if c["score"] >= corte]
    else:
        for c in candidatos:
            c["score"] = c["score_pesquisa"]

    return candidatos[:TOP_K]

def recuperar(config, consulta: str) -> list[dict]:
    """Interface esperada pela conversa: devolve passagens já filtradas e ordenadas."""
    resultados = pesquisar_query_qdrant(consulta, obter_cliente())
    return [
        {
            **r,
            "doc": r["doc_id"],
            "contexto": r.get("cabecalhos", ""),
            "similaridade": r["score"],
        }
        for r in resultados
    ]
