# Contratos entre componentes — proposta para validação pelo Grupo 2

Este documento define formatos de entrada/saída para evitar integrações incompatíveis. Não substitui a validação dos responsáveis F2/F3/F1.

## 1. Classes visuais canónicas

Usar exatamente estas etiquetas em CSVs, outputs e métricas:

- `saudavel`
- `esca`
- `leaf_blight`
- `black_rot`
- `inconclusivo` (saída de segurança; não é uma classe do FieldVitis)

O FieldVitis tem quatro classes reais; `inconclusivo` serve para recusar uma classificação visual quando a imagem não é folha de videira ou não há evidência visual suficiente.

## 2. CNN — F2 fornece; F4 integra como ferramenta

**Entrada:** imagem RGB (`PIL.Image.Image`).

**Saída mínima:**

```json
{
  "classe": "esca",
  "probabilidades": {
    "saudavel": 0.05,
    "esca": 0.75,
    "leaf_blight": 0.10,
    "black_rot": 0.10
  }
}
```

Os valores são meramente ilustrativos. O classificador real tem de devolver as probabilidades reais do modelo, na mesma ordem/nomenclatura das quatro classes. A calibração da confiança é avaliada separadamente pelo F5.

## 3. Retriever RAG — F3 fornece; F4 integra como ferramenta

**Entrada:** string de consulta em português.

**Saída:** lista de `langchain_core.documents.Document` com:

- `page_content`: texto da passagem;
- `metadata.doc_id`: D01, D02, ...;
- `metadata.titulo`;
- `metadata.entidade`;
- `metadata.data`;
- `metadata.pagina`: número físico da página PDF, começando em 1;
- `metadata.url`;
- `metadata.score`: opcional, se o retriever devolver score de relevância.

Exemplo de uma passagem:

```python
Document(
    page_content="...texto da passagem...",
    metadata={"doc_id":"D02", "titulo":"...", "entidade":"ADVID",
              "data":"2007", "pagina":8, "url":"https://..."}
)
```

**Importante:** a lista de documentos recuperados não prova, por si só, que a resposta é suportada. O agente tem de citar as passagens e a verificação final é humana.

## 4. VLM — execução visual

**Entrada:** caminho para imagem + prompt de classificação zero-shot.

**Saída mínima do módulo `src/vlm.py`:**

```json
{
  "classe": "esca",
  "descricao_visual": "...",
  "limitacoes": "...",
  "texto_bruto": "...",
  "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
  "latencia_s": 2.3,
  "confidence": null
}
```

A confiança fica `null` porque uma frase gerada pelo VLM não é uma probabilidade calibrada. O F2 avalia Macro-F1 por classe.

## 5. Resposta final do sistema

Saída a adotar na integração com a app:

```json
{
  "resposta": "...",
  "fontes": [{"doc_id":"D02", "titulo":"...", "pagina":8, "data":"2007", "url":"..."}],
  "recusou": false,
  "motivo_recusa": null,
  "ferramentas_invocadas": ["classificar_folha", "pesquisar_documentacao"],
  "model_id": "qwen3:1.7b"
}
```

## 6. Fluxos de referência

### A — LLM + tools (CNN + RAG)

1. Receber pergunta e imagem opcional.
2. Se existir imagem, disponibilizar/chamar `classificar_folha` (CNN F2).
3. Chamar `pesquisar_documentacao` (retriever F3), com a pergunta e a hipótese visual.
4. LLM textual gera resposta com citações `[S1]`, `[S2]` e datas das fontes.
5. Se não houver evidência, recusar a recomendação específica.

### B — VLM + RAG

1. Receber pergunta e imagem.
2. VLM interpreta a imagem e devolve classe/descrição.
3. Construir consulta documental com a pergunta + descrição/classe.
4. Retriever F3 devolve passagens.
5. O mesmo LLM textual produz a resposta técnica com citações e recusa.

Este desenho mantém o VLM como componente de visão e o LLM como gerador textual ancorado no RAG, de acordo com o enquadramento do Guia Tema F. Confirmar com o professor se a instrução oral mais recente exige que o VLM seja também o gerador da resposta final; essa variante altera a comparação.

## 7. Comparabilidade dos prompts (B0/B1/B2)

Para a comparação textual formal, usar `construir_prompt_sistema(contexto)` de `src/prompts.py` nas três condições: B0 (sem RAG) recebe o bloco `[SEM DOCUMENTOS RECUPERADOS]`; B1/B2 recebem as passagens recuperadas. Assim o template de sistema e as instruções gerais são iguais e só muda a disponibilidade de evidência. O agente LLM+tools tem instruções adicionais de orquestração e deve ser tratado como uma via de arquitetura distinta, não como substituto silencioso da baseline B0.
