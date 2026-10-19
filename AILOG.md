- **Data:** 2026-10-19
- **Ação:** Implementação do script de extração em lote `F3/text_Extract.py`.
- **Implementação:** Uso de `os.listdir()` para iterar a pasta `Documents` e `PyMuPDF` para extração de texto e contagem de páginas/caracteres.
- **Decisão Arquitetural:** 
  1. Agrupamento dos metadados extraídos (documento, páginas, caracteres, número de chunks) numa única lista de dicionários. Rejeitada a abordagem de listas paralelas (anti-pattern) para garantir a integridade referencial de cada ficheiro e facilitar a futura conversão para `corpus.csv`.
  2. Uso de caminhos dinâmicos (`os.path.abspath` e `..`) para evitar falhas de "File Not Found" independentemente da diretoria onde o terminal é executado.
  3. Divisão do texto com uma sobreposição (overlap) de 200 caracteres para preservar a semântica entre chunks, essencial para o sistema RAG.