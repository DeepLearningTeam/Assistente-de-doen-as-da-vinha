# Auditoria preliminar das 10 perguntas de holdout (Q031–Q040)

**Estado: auditoria documental preliminar; requer validação humana e confirmação do protocolo.**

As perguntas Q031–Q040 têm respostas de referência em D10–D12. Estes três documentos têm `indexar=nao` e não podem ser disponibilizados ao índice FAISS nem ao retriever. O objetivo é verificar se a pergunta deixa de ser respondível com o corpus que permanece acessível ao sistema.

| ID | Documento/página | Informação necessária para responder | Verificação preliminar no restante ZIP |
|---|---|---|---|
| Q031 | D10, p. 2 | Limiar superior a 20% de videiras contaminadas e prazo até 31 de março para arranque/destruição de toda a parcela | D01–D09 e D13–D19 não apresentam esta regra específica do Edital 2/2025. D09 tem conteúdo sobre flavescência e tratamentos do vetor, mas não este critério de arranque integral da parcela. |
| Q032 | D10, p. 1 | Métodos explicitamente indicados: estilhaçamento ou fogo e, sempre que possível, no próprio local | D09 recomenda proteção contra a flavescência e apresenta tratamentos do vetor, mas não esta instrução específica sobre métodos de destruição de videiras contaminadas. |
| Q033 | D10, p. 6 | Lista de freguesias de Chaves incluídas na zona demarcada em contenção: Anelhe, Oura e Vidago (união de quatro freguesias) | D09, pp. 11–13, contém uma tabela de tratamentos por concelho/freguesia, mas não inclui Chaves nem a lista do Anexo I de D10. D11/D12 são também holdouts. |
| Q034 | D11, p. 1 | Definição normativa de videira contaminada: resultado laboratorial oficial positivo e videiras com sintomas semelhantes no mesmo ano ou em anos subsequentes | D09 descreve sintomas suspeitos e medidas de proteção, mas não apresenta esta definição normativa completa. |
| Q035 | D11, p. 5 | Freguesias de Boticas na coluna “1.º e 2.º tratamentos” | D09, Quadro 5 (pp. 11–13), não inclui Boticas. A lista específica está no quadro de D11, que permanece excluído. |
| Q036 | D12, p. 2 | Os anos de prospeção que fundamentaram as cinco zonas estabelecidas pelo Despacho 39/G/2024 | D01–D09 e D13–D19 não identificam esta sequência histórica (prospeção de 2022 e 2023 → cinco zonas do Despacho 39/G/2024). |
| Q037 | D12, p. 1 | Periodicidade anual e âmbito nacional da prospeção oficial de flavescência dourada e Scaphoideus titanus | D09 inclui monitorização e avisos regionais de 2026, mas não contém a disposição do Despacho 69/G/2026 sobre prospeção oficial anual em território nacional para a praga e o vetor. |
| Q038 | D12, p. 2 | Contagem e origem das zonas mantidas em erradicação: cinco do Despacho 39/G/2024 + uma do Despacho 72/G/2025 | O restante corpus não apresenta esta contagem histórica e a repartição por despachos; D09 trata de medidas/avisos e tratamentos do vetor em 2026. |
| Q039 | D12, p. 2 | Revogação expressa do Despacho 72/G/2025, de 2 de julho | A revogação é um ato jurídico expresso do Despacho 69/G/2026 e não é indicada nos restantes PDFs candidatos/condicionais. |
| Q040 | D12, p. 1 | Base legal nacional para publicar a lista atualizada de freguesias com presença do vetor e classificação de risco: n.º 3 do artigo 8.º da Portaria 267/2023 | D10/D11 citam o artigo 13.º da mesma Portaria para notificações/medidas, mas os documentos candidatos ao índice não indicam a base legal específica do artigo 8.º, n.º 3, para a atualização da lista e classificação de risco. |

## Sobreposição com a Circular D09

A Circular n.º 11/2026 (D09) inclui, nas páginas PDF 11–13, o Quadro 5 relativo ao número de tratamentos obrigatórios contra a cigarrinha da flavescência dourada, com concelhos e freguesias. Por esse motivo, **não** foram usadas como holdout perguntas sobre “máximo de três tratamentos” nem perguntas sobre freguesias/concelhos presentes nesse quadro. Q035 usa Boticas, que não aparece no Quadro 5 de D09; Q033 usa Chaves e a delimitação da zona demarcada do Anexo I de D10, informação que não aparece em D09.

## Revisão obrigatória antes de aprovar

1. O F3 deve executar a pesquisa das dez perguntas no índice final aprovado, verificando não apenas correspondência literal mas também passagens semanticamente equivalentes.
2. A equipa deve confirmar que D10–D12 nunca são carregados para o índice, retriever, prompt ou contexto de inferência.
3. Dois elementos devem verificar cada resposta de referência e a página física do PDF.
4. Se o corpus indexável mudar (por exemplo, se D07/D08 ou D13–D19 forem autorizados), repetir a auditoria. As fontes D13–D19 são sobretudo de doenças da videira e não parecem cobrir estes detalhes legais, mas a verificação tem de ser repetida no índice final.
5. Não marcar qualquer pergunta como validada sem revisão humana e sem identificar os revisores.

A auditoria é preliminar: uma pesquisa textual e a leitura dos PDFs ajudam a identificar sobreposições, mas não provam automaticamente que o retriever nunca recuperará contexto suficiente para responder.
