"""Funções comuns para recuperar passagens e preparar citações.

A suficiência da evidência não é garantida apenas por recuperar documentos. A política de
recusa e os limiares precisam de ser definidos e avaliados pelo grupo.
"""
from __future__ import annotations
from typing import Any
from langchain_core.documents import Document


def recuperar_passagens(retriever: Any, pergunta: str, contexto_visual: str | None = None) -> list[Document]:
    """Pesquisa documentos com pergunta e, opcionalmente, contexto visual da CNN/VLM."""
    consulta = pergunta.strip()
    if contexto_visual:
        consulta += f"\nPossível contexto visual (não é evidência documental): {contexto_visual.strip()}"
    return list(retriever.invoke(consulta))


def formatar_fontes(documentos: list[Document]) -> list[dict[str, Any]]:
    """Devolve fontes com metadados para UI/logs. Não afirma que suportam a resposta."""
    fontes: list[dict[str, Any]] = []
    vistos: set[tuple[str, str, str]] = set()
    for doc in documentos:
        m = doc.metadata or {}
        chave = (str(m.get("doc_id", m.get("titulo", "desconhecido"))),
                 str(m.get("pagina", "?")), str(m.get("url", "")))
        if chave in vistos:
            continue
        vistos.add(chave)
        fontes.append({
            "doc_id": m.get("doc_id", "PREENCHER"),
            "titulo": m.get("titulo", "Título por preencher"),
            "entidade": m.get("entidade", "Entidade por preencher"),
            "data": m.get("data", "Data por verificar"),
            "pagina": m.get("pagina", "Pág. por verificar"),
            "url": m.get("url", "URL por verificar"),
        })
    return fontes


def formatar_contexto(documentos: list[Document]) -> str:
    """Formata passagens com IDs [S1], [S2] para facilitar a verificação das citações."""
    blocos = []
    for i, doc in enumerate(documentos, start=1):
        m = doc.metadata or {}
        titulo = m.get("titulo", "Título por verificar")
        entidade = m.get("entidade", "Entidade por verificar")
        data = m.get("data", "Data por verificar")
        pagina = m.get("pagina", "Pág. por verificar")
        url = m.get("url", "URL por verificar")
        blocos.append(
            f"[S{i}] {titulo} | {entidade} | {data} | página {pagina} | {url}\n"
            f"Passagem:\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(blocos)


def instrucoes_resposta() -> str:
    """Instruções preliminares para o LLM; a política final tem de ser validada."""
    return (
        "Responde em português de Portugal. Usa as passagens fornecidas como evidência para "
        "afirmações técnicas. Para cada afirmação factual, indica o ID da fonte, por exemplo "
        "[S1], e não inventes títulos, páginas, datas, URLs, produtos ou doses. Distingue a "
        "hipótese visual da recomendação documental. Se as passagens não sustentarem uma "
        "resposta segura e específica, diz explicitamente que a documentação recuperada é "
        "insuficiente e recusa recomendar uma ação específica. A resposta deve ser verificada "
        "por pessoas; estas instruções não garantem fidelidade automaticamente."
    )
