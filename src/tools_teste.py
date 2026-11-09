"""Ferramentas determinísticas de demonstração, sem depender de F2/F3.

Os resultados são SIMULADOS e servem apenas para testar o ciclo LangChain -> tool -> LLM.
Não usar estas respostas em experiências nem como evidência agronómica.
"""
from langchain.tools import tool
from langchain.agents import create_agent


@tool
def classificar_folha_demo() -> str:
    """Devolve uma classificação fictícia de uma imagem de folha para testar a ferramenta CNN."""
    return (
        'RESULTADO SIMULADO, NÃO É UMA PREVISÃO REAL: '
        '{"classe":"esca","probabilidades":{"saudavel":0.05,"esca":0.75,'
        '"leaf_blight":0.10,"black_rot":0.10}}'
    )


@tool
def pesquisar_documentacao_demo(consulta: str) -> str:
    """Devolve uma passagem fictícia/curta de teste para verificar a chamada à pesquisa documental."""
    return (
        'PASSAGEM DE DEMONSTRAÇÃO — não utilizar como fonte real. '
        'D02, p. 8: a descrição da esca refere necrose das margens das folhas que avança '
        'para o centro e manchas entre nervuras.'
    )


def criar_agente_demo(llm):
    """Cria um agente LangChain que dispõe das duas ferramentas simuladas."""
    return create_agent(
        model=llm,
        tools=[classificar_folha_demo, pesquisar_documentacao_demo],
        system_prompt=(
            'És um agente de demonstração de chamadas a ferramentas. Antes de responder, '
            'chama obrigatoriamente classificar_folha_demo e pesquisar_documentacao_demo. '
            'Assinala sempre que os resultados são simulados e não constituem aconselhamento.'
        ),
    )
