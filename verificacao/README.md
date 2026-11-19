# Evidências dos ensaios da integração

Estes ficheiros são amostras de funcionamento de 16/11/2026. Não representam os resultados finais do projeto nem substituem a avaliação humana.

- `colecao.json`: inspeção de leitura da coleção do grupo; sem URL privado/chave.
- `recuperacao.json`, `suportada.json`, `sem_suporte.json`: passagem esperada, resposta citada e recusa com serviços reais.
- `sem_rag.json`: Qwen3 real com acesso ao Qdrant proibido; contém incorreções observadas na baseline.
- `indice_local.json`: extração de D02 e indexação/pesquisa de duas passagens numa coleção temporária local com modelos reais.
- `integracao-par/`: manifesto e dois itens do ensaio textual emparelhado, com histórico vazio, sem imagem e referência apenas no registo.
- `integracao-foto/`: manifesto e dois itens VLM com a mesma pergunta e fotografia documental; o hash é igual. O VLM classificou incorretamente a folha como saudável; o modo RAG recusou por evidência insuficiente. A fotografia pode ser extraída de D02 com o comando em `INTEGRACAO.md`.
- `recuperacao-ensaio.csv`: ensaio do avaliador de recuperação integrado com uma pergunta.
- `testes.txt`, `ambiente.txt`: resultado dos 49 testes e versões instaladas.

As mensagens registadas usam `[fotografia]` no lugar dos bytes da imagem. Os manifestos têm os hashes do código/protocolo no momento da respetiva execução. As alterações finais de documentação e do ensaio de índice local são posteriores a algumas dessas corridas; os registos originais foram conservados.
