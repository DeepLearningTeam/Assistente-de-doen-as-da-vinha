# Perguntas F4 — conjunto revisto com base nos PDFs do ZIP

**Estado: rascunho para validação humana.** As 40 perguntas foram reconstruídas a partir dos PDFs efetivamente incluídos no `PDFs(1).zip`. Cada linha indica a resposta de referência, o documento, a página física do PDF e a ação esperada.

## 1. Composição do conjunto

- **40 perguntas no total**.
- **30 perguntas com resposta de referência nos documentos**: 20 em documentos marcados `indexar=sim` no `corpus.csv` e 10 em documentos cuja indexação continua `condicional`.
- **10 perguntas de holdout**: respostas localizadas em D10–D12, que permanecem fora do índice e devem provocar recusa quando o RAG não dispõe de suporte.

| Split | Responder | Recusar | Total |
|---|---:|---:|---:|
| Validação (Q001–Q010, Q021–Q025, Q031–Q035) | 15 | 5 | 20 |
| Teste (Q011–Q020, Q026–Q030, Q036–Q040) | 15 | 5 | 20 |
| **Total** | **30** | **10** | **40** |

## 2. Significado dos campos

- `expected_action`: `responder` quando a resposta de referência existe no documento indicado; `recusar` quando a resposta está num documento holdout excluído do índice.
- `estado_fonte`: `candidato_indice` (o corpus marca `indexar=sim`), `condicional` (a indexação depende da confirmação dos direitos/condições de utilização) ou `holdout` (nunca indexar).
- `pagina_pdf`: número físico da página no visualizador PDF, começando em 1; não usar o número impresso no rodapé se for diferente.
- `passagem_ref`: indicação do conteúdo que sustenta a resposta e notas de contexto.
- `estado_validacao`: todas as perguntas permanecem `pendente_validacao_humana` até revisão por elementos do grupo.

**Atenção às 10 perguntas condicionais (Q021–Q030):** a resposta existe nos PDFs D13–D19, mas estes documentos estão marcados `indexar=condicional` no inventário devido a licenças ou autorizações por confirmar. Não executar a avaliação como “respondível pelo índice” até o grupo/docente decidir quais podem ser incorporados. Se algum documento não puder ser indexado, substituir a pergunta por outra com suporte em fonte aprovada ou alterar formalmente o protocolo.

## 3. Cobertura e cautelas

O conjunto abrange eutipiose, esca, black rot, podridão cinzenta, escoriose, míldio, oídio, mancha-das-folhas, doenças do lenho e proteção integrada. As perguntas sobre mancha-das-folhas em D13–D15 **não afirmam que a classe `leaf_blight` do FieldVitis corresponde a Pseudocercospora vitis**. Essa equivalência exige confirmação na documentação original do dataset.

D14 e D15 são documentos brasileiros; D13 também é Embrapa. As perguntas incidem sobre a descrição do documento ou sobre a metodologia do estudo, não sobre a validade de tratamentos brasileiros em Portugal. Não converter produtos/doses de fontes brasileiras em recomendações portuguesas.

As circulares D05, D06 e D09 são avisos regionais datados. As respostas têm de preservar a data e a região; não apresentar avisos de 2026 como regras permanentes ou recomendações atuais fora do seu contexto.

D06 tem nome local `Avisos_Agricolas-Circular_05-2026.pdf`, mas o cabeçalho interno é **Circular n.º 08/2026, de 21/05/2026**. Usar Circular n.º 08/2026 nas respostas e citações.

D07 é apenas um excerto de 9 páginas e não contém a secção necessária sobre doenças fúngicas/mancha-das-folhas. D08 é um folheto de 2 páginas e tem licença CC BY-NC-ND 4.0; não foi usado como fonte das 40 perguntas.

## 4. Separação entre validação e teste

As 20 perguntas de validação podem ser usadas para testar e ajustar prompts, recuperação e política de recusa, dentro do protocolo aprovado. As 20 perguntas de teste devem permanecer fechadas até `PROTOCOLO.md` estar aprovado e commitado. Não usar respostas do teste para escolher modelos, parâmetros, prompts, `top_k` ou limiares de recusa.

O ficheiro `splits.csv` é o split das **imagens** (path/label/grupo/split/seed/hash), não o split destas perguntas. Como o ZIP recebido não contém as 555 imagens nem os respetivos identificadores, `splits.csv` permanece apenas com o cabeçalho; não foram inventadas linhas de imagens.

## 5. Auditoria e revisão obrigatória

A auditoria preliminar das 10 perguntas de recusa está em `holdout_audit.md`. Antes de aprovar o benchmark:

1. Confirmar cada resposta de referência e a página no PDF original.
2. Rever com F3 se cada pergunta holdout deixa de ser respondível a partir do corpus final autorizado, incluindo pesquisa semântica no índice.
3. Rever a situação de licença de todos os documentos condicionais.
4. Registar os revisores e só então mudar `estado_validacao` para um estado aprovado pelo grupo.
5. Manter o conjunto de teste fechado até ao commit do protocolo.
