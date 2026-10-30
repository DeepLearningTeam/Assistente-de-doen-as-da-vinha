- **Data:** 2026-10-19
- **Ação:** Implementação do script de extração em lote `F3/text_Extract.py`.
- **Implementação:** Uso de `os.listdir()` para iterar a pasta `Documents` e `PyMuPDF` para extração de texto e contagem de páginas/caracteres.
- **Decisão Arquitetural:** 
  1. Agrupamento dos metadados extraídos (documento, páginas, caracteres, número de chunks) numa única lista de dicionários. Rejeitada a abordagem de listas paralelas (anti-pattern) para garantir a integridade referencial de cada ficheiro e facilitar a futura conversão para `corpus.csv`.
  2. Uso de caminhos dinâmicos (`os.path.abspath` e `..`) para evitar falhas de "File Not Found" independentemente da diretoria onde o terminal é executado.
  3. Divisão do texto com uma sobreposição (overlap) de 200 caracteres para preservar a semântica entre chunks, essencial para o sistema RAG.

---

- **Data:** 2026-10-23
- **Ação:** Criação da estrutura de dados auxiliar para mapeamento FAISS em `src/indice.py`.
- **Implementação:** Lista global `lista_chunks` que armazena dicionários com as chaves `texto`, `nome_documento` e `pagina`. Duas funções auxiliares: `adicionar_chunk()` para inserir chunks e devolver o ID (índice da lista), e `obter_chunk()` para recuperar metadados a partir do ID devolvido pelo FAISS.
- **Decisão Arquitetural:**
  1. Uso do índice nativo da lista Python como ID do FAISS, eliminando a necessidade de estruturas de mapeamento externas (dicionários, bases de dados) e garantindo correspondência direta entre o ID do vetor no FAISS e a posição na lista de metadados.
  2. Separação da lógica de indexação/metadados num módulo próprio (`src/indice.py`), desacoplando-a da extração de texto (`F3/text_Extract.py`) para facilitar a manutenção e reutilização.