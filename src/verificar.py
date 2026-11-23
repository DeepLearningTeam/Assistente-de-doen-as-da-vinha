"""Ensaios curtos com modelos reais. Não substituem a avaliação humana das respostas."""
import argparse
import os
import sys
import tempfile
from contextlib import closing
from unittest.mock import patch

from . import indice
from .apoio import RAIZ, ErroProjeto, gravar_json, nova_verificacao
from .config import carregar_config
from .rag import responder

PERGUNTA = "Por onde pode entrar na videira o fungo associado à eutipiose?"
SEM_SUPORTE = "Segundo o corpus, quantos minutos dura a fase azul da doença fictícia zeta da videira?"


def verificar_sem_rag(destino):
    # O modelo é real. Só o acesso ao Qdrant é proibido para demonstrar o isolamento.
    with patch.dict(os.environ, {"QDRANT_URL": "", "QDRANT_API_KEY": "", "QDRANT_LOCAL_PATH": ""}), \
         patch.object(indice, "obter_cliente", side_effect=AssertionError("Modo sem RAG tentou usar Qdrant")):
        resultado = responder(PERGUNTA, "llm_cnn", historico=(), registar=True)
    if resultado.estado != "resposta" or resultado.fontes:
        raise ErroProjeto("O ensaio sem RAG não produziu uma resposta livre. Consultar o JSON/modelo.")
    gravar_json(destino / "sem_rag.json", resultado.model_dump())
    print(f"Sem RAG e sem credenciais: {resultado.estado} — {resultado.texto}")


def verificar_rag(destino):
    config = carregar_config("llm_cnn_rag")
    info = indice.descrever_colecao(config)
    gravar_json(destino / "colecao.json", info)
    passagens = indice.recuperar(config, PERGUNTA)
    gravar_json(destino / "recuperacao.json", passagens)
    if not any(p["doc"] == "D02" and p["pagina"] == 6 for p in passagens):
        raise ErroProjeto("A passagem esperada D02 p. 6 não entrou no top-k; consultar recuperacao.json.")
    resultado = responder(PERGUNTA, "llm_cnn_rag", historico=(), config=config, registar=True)
    gravar_json(destino / "suportada.json", resultado.model_dump())
    if resultado.estado != "resposta" or not any(
        f["doc"] == "D02" and f["pagina"] == 6 for f in resultado.fontes
    ):
        raise ErroProjeto("A geração não citou D02 p. 6. Consultar suportada.json; não basta recuperar a passagem.")
    print(f"Suportada, D02 p. 6: {resultado.texto}")
    sem_suporte = responder(SEM_SUPORTE, "llm_cnn_rag", historico=(), config=config, registar=True)
    gravar_json(destino / "sem_suporte.json", sem_suporte.model_dump())
    if sem_suporte.estado != "recusa":
        raise ErroProjeto("O modelo não recusou a pergunta sem suporte; consultar sem_suporte.json.")
    print(f"Sem suporte: {sem_suporte.estado} ({sem_suporte.motivo}) — {sem_suporte.texto}")


def verificar_indice_local(destino):
    """PDF real e embeddings reais numa coleção temporária, sem escrever no grupo."""
    from qdrant_client import QdrantClient
    from .corpus import ler_corpus
    from .ingestao import processar_corpus, construir_base_qdrant
    config = carregar_config().model_copy(update={"qdrant_colecao": "teste_temporario"})
    chunks = processar_corpus([ler_corpus()["D02"]], config=config, ocr="off")
    amostra = [c for c in chunks if c["pagina"] == 6]
    if not amostra:
        raise ErroProjeto("A extração de D02 não produziu passagens na página 6.")
    (RAIZ / "runs").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=RAIZ / "runs") as tmp:
        with closing(QdrantClient(path=tmp)) as client:
            construir_base_qdrant(amostra, client, config)
            resultados = indice.pesquisar_query_qdrant(PERGUNTA, client, config=config)
            if not resultados or resultados[0]["pagina"] != 6:
                raise ErroProjeto("O índice local não recuperou a amostra da página 6.")
    gravar_json(destino / "indice_local.json", {
        "documento": "D02", "chunks_extraidos": len(chunks), "chunks_indexados": len(amostra),
        "ocr": "off", "config": config.model_dump(), "resultados": resultados,
    })
    print(f"Índice local real: {len(chunks)} passagens extraídas, {len(amostra)} da página 6 indexadas e pesquisadas.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sem-rag", action="store_true")
    parser.add_argument("--rag", action="store_true")
    parser.add_argument("--indice-local", action="store_true")
    args = parser.parse_args()
    if not args.sem_rag and not args.rag and not args.indice_local:
        parser.error("Selecionar --sem-rag, --rag e/ou --indice-local.")
    destino = nova_verificacao("assistente")
    print(f"Relatórios em {destino}.", flush=True)
    try:
        if args.sem_rag:
            verificar_sem_rag(destino)
        if args.rag:
            verificar_rag(destino)
        if args.indice_local:
            verificar_indice_local(destino)
    except Exception as erro:
        gravar_json(destino / "erro.json", {"tipo_erro": type(erro).__name__,
                    "mensagem": str(erro) if isinstance(erro, ErroProjeto) else "Falhou a verificação; detalhes privados omitidos."})
        sys.exit(str(erro) if isinstance(erro, ErroProjeto) else "Falhou a verificação de serviços/modelos; detalhes privados omitidos.")


if __name__ == "__main__":
    main()
