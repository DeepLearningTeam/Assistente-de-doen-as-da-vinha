# PROTOCOLO.md — versão de trabalho

**Estado: RASCUNHO — não usar ainda como protocolo experimental fechado.** O Guia Tema F pede que o protocolo fique fechado antes de consultar o conjunto de teste.

## 1. Problema e objetivo

Assistente de apoio a viticultores/técnicos para identificar a doença provável numa folha e obter recomendações apoiadas em documentos técnicos públicos em português, com citações verificáveis e recusa quando a documentação acessível não chega.

## 2. Dados (a verificar nos ficheiros reais)

- Imagens: FieldVitis, 555 imagens, quatro classes indicadas no guia: saudável (158), esca (200), leaf blight (158), black rot (39). Confirmar a distribuição real e a licença CC BY 4.0 no dataset original.
- Texto: sete documentos de partida listados em `data/corpus.csv`; o grupo poderá alargar o corpus. Registar URL, entidade, data, licença e SHA-256; não redistribuir os PDFs.
- Limitação documentada no guia: imagens provenientes de datasets públicos e não especificamente de vinhas portuguesas; não há identificador conhecido de planta/parcela.

## 3. Vias de sistema — proposta a validar com o professor

### A. LLM + ferramentas

LLM de texto com possibilidade de invocar (i) uma ferramenta CNN para a imagem fornecida e (ii) uma ferramenta de pesquisa documental. A resposta final usa o contexto recuperado, apresenta as fontes e recusa sem suporte suficiente.

### B. VLM + recuperação documental

VLM aberto pequeno recebe a imagem e a pergunta, produz a interpretação visual e utiliza o contexto de documentos recuperados para responder com citações. Definir se o próprio VLM gera a resposta final ou se a geração final é feita por um LLM comum.

### Baselines formais do Guia Tema F

O guia também identifica: (i) CNN pré-treinada com fine-tuning, 3 seeds, para imagem; (ii) o mesmo LLM de texto sem recuperação, a responder às mesmas perguntas, para texto. Confirmar com o docente se estes baselines continuam obrigatórios em paralelo com a comparação mais recente LLM+tools versus VLM. Não os eliminar silenciosamente.

## 4. Recuperação documental

- Embeddings: escolher entre BGE-M3 e multilingual-e5 após teste de recursos/qualidade.
- Índice: FAISS, segundo o guia.
- `top_k`: começar com 5 para a avaliação Recall@5; parâmetros de chunking e qualquer limiar de relevância são provisórios e devem ser escolhidos sem usar o teste final.
- Metadados mínimos por passagem: `doc_id`, `titulo`, `entidade`, `data`, `pagina`, `url`.
- A página deve ser de contagem humana (a primeira página do PDF é 1) e verificada no documento original.

## 5. Perguntas e recusas

- Preparar cerca de 40 perguntas com resposta de referência e passagem-fonte.
- Proposta indicada no guia: 30 respondíveis com os documentos indexados e 10 que requerem informação presente em 2–3 documentos deliberadamente excluídos do índice.
- Categorizar as perguntas e escrever em linguagem natural, sem copiar literalmente o texto das fontes.
- Definir previamente o que constitui suporte suficiente, recusa correta e recusa indevida.

## 6. Avaliação

- Fidelidade: percentagem de afirmações da resposta suportadas pelas fontes citadas; verificação manual por dois anotadores. Quem cria uma pergunta não deve ser o único a validar a fidelidade da resposta à mesma pergunta.
- Imagem: Macro-F1 por classe; atenção especial à classe minoritária black rot.
- Pesquisa: Recall@5 da passagem-fonte.
- Recusa: taxa de recusa em perguntas sem resposta e taxa de recusa indevida em perguntas com resposta.
- Outras: consistência em três repetições, latência por resposta e catálogo de alucinações.
- Reproduzir as experiências através de YAMLs em `configs/` e outputs JSON em `runs/`.

## 7. Parâmetros experimentais por fechar

- [ ] Model ID e versão/licença do LLM de texto.
- [ ] Model ID, versão/licença e requisitos computacionais do VLM.
- [ ] Modelo da CNN e interface de inferência confirmados com F2.
- [ ] Embedding model e parâmetros do chunking confirmados com F3.
- [ ] Modelo de recusa e limiar de relevância calibrados apenas em dados de desenvolvimento/validação.
- [ ] Regras para fontes antigas, em particular documentos de 2003/2007/2009, e para recomendações fitossanitárias.
- [ ] Divisões finais e três seeds confirmadas; teste reservado até ao protocolo estar fechado.
- [ ] Confirmada com o professor a comparação principal após as indicações mais recentes.
