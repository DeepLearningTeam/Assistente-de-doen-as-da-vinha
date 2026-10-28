from langchain_core.documents import Document
from src.corpus import dividir_em_chunks
from src.rag import formatar_fontes, formatar_contexto


def test_chunking_preserves_page_metadata():
    docs = [Document(page_content=("informação técnica " * 100), metadata={"doc_id": "D1", "pagina": 4})]
    chunks = dividir_em_chunks(docs, chunk_size=100, chunk_overlap=10)
    assert len(chunks) > 1
    assert all(chunk.metadata["pagina"] == 4 for chunk in chunks)
    assert all(chunk.metadata["doc_id"] == "D1" for chunk in chunks)


def test_sources_include_page_and_title():
    docs = [Document(page_content="texto", metadata={"doc_id": "D1", "titulo": "Manual", "pagina": 3, "url": "https://example.invalid"})]
    fontes = formatar_fontes(docs)
    assert fontes[0]["titulo"] == "Manual"
    assert fontes[0]["pagina"] == 3


def test_context_labels_passages():
    docs = [Document(page_content="Texto de teste", metadata={"titulo": "Manual", "pagina": 2})]
    contexto = formatar_contexto(docs)
    assert "[S1]" in contexto
    assert "página 2" in contexto
