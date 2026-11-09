# PDFs locais — NÃO versionar

O ficheiro `PDFs(1).zip` recebido contém 19 PDFs, identificados como D01–D19. Extrai os PDFs para esta pasta (`data/docs/`) sem lhes alterar o conteúdo. O `.gitignore` deve excluir os PDFs; o repositório deve manter os metadados, URL, data, licença, SHA-256 e o mapa local.

Consulta `mapa_ficheiros_locais.csv` para os nomes exatos dos 19 ficheiros. A coluna `indexacao` acompanha `corpus.csv`: `sim` = candidato atual ao índice; `condicional` = aguarda confirmação de direitos/condições; `nao` = fora do índice (holdout ou documento inadequado/incompleto). A indicação `sim` não substitui a confirmação dos termos de utilização identificados no `corpus.csv`.

## Notas importantes sobre os PDFs

- **D06:** o ficheiro chama-se `Avisos_Agricolas-Circular_05-2026.pdf`, mas o cabeçalho interno identifica a **Circular n.º 08/2026, de 21/05/2026**. Nas citações, usar a identificação interna correta (D06 / Circular 08/2026), não “Circular 05/2026”.
- **D07:** o ZIP contém apenas um excerto inicial de 9 páginas de “Frutas do Brasil 35”. Não inclui a secção necessária de doenças fúngicas/mancha-das-folhas. Não usar este excerto para sustentar perguntas sobre `leaf_blight` ou Pseudocercospora.
- **D08:** folheto da Embrapa de 2 páginas; o repositório indica CC BY-NC-ND 4.0. Confirmar com o docente a adequação da indexação RAG.
- **D10–D12:** documentos holdout sobre flavescência dourada. Nunca copiar para `data/index/`, disponibilizar ao retriever ou incluir no contexto de resposta da aplicação. São usados apenas para as respostas de referência das perguntas Q031–Q040.
- **D13 e D15:** documentos Embrapa; a situação CC BY-NC-ND 4.0 deve ser confirmada antes da indexação.
- **D14:** o próprio PDF declara que a reprodução não autorizada constitui violação dos direitos de autor. Obter autorização antes de indexar.
- **D16–D19:** documentos ADVID/OILB para os quais não foi identificada licença aberta explícita nos ficheiros consultados. Confirmar condições de reutilização e indexação com os titulares.
- **Fontes brasileiras (D13–D15):** podem apoiar perguntas de descrição, biologia e metodologia, mas não devem ser usadas para inferir produtos ou doses autorizados em Portugal.
- **Avisos agrícolas D05, D06 e D09:** são documentos regionais e datados. As respostas devem mencionar a circular, a data e o contexto regional; não são recomendações permanentes.

## Integridade dos ficheiros

Os valores SHA-256 do `corpus.csv` foram comparados com os 19 PDFs incluídos no ZIP e coincidem. Depois de extrair os ficheiros para `data/docs/`, executa:

```powershell
python scripts/verificar_corpus.py
```

A validação usa o SHA-256 do ficheiro completo e os nomes locais definidos em `mapa_ficheiros_locais.csv`. Não renomeies os PDFs sem atualizar também o mapa.
