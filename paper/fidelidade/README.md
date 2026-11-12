# Folhas de fidelidade

`python -m src.metricas anotar --execucao NOME` cria três CSV em `paper/fidelidade/NOME/`. Faz a exportação após concluir a execução. Os dois anotadores trabalham nas folhas numeradas. Guardam a decisão conjunta em `consenso.csv`.

- Cada linha corresponde a uma afirmação sugerida pela geração. Lê a resposta completa em `runs/NOME/itens/<chave_resposta>.json`. `trechos_no_json` identifica os excertos exatos no campo `recuperadas`. Abre a página do PDF se o trecho não for suficiente para interpretar o contexto.
- Em `suportada`, usa `S` para suporte integral, `N` para ausência de suporte ou contradição e `parcial` quando só parte da afirmação é sustentada. Uma afirmação sem fonte citada recebe `N` na métrica de fidelidade às citações, mesmo quando é correta.
- Em `correta_face_referencia`, regista `S` ou `N` para a afirmação quando for comparável à referência. Na primeira linha de cada resposta, preenche `resposta_correta` com `S` se o pedido ficou corretamente respondido e completo face à referência e `N` caso contrário. Esta métrica incide nas respostas anotadas. As recusas são medidas em separado.
- Em `tipo_erro`, regista, quando aplicável, `inventado`, `fonte_errada`, `desatualizado`, `sem_citacao` ou `outro`. Preenche `anotador` com a identificação combinada pelo grupo. Usa `paper/tabelas/tabela3.csv` para o catálogo de casos e correções.
- Preserva os IDs entre folhas. Se uma linha contiver duas afirmações independentes, divide-a nas três folhas com os mesmos sufixos de ID. Depois do consenso, usa `python -m src.metricas resumir --execucoes NOME`.
