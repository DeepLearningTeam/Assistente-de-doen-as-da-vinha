# PROTOCOLO.md — proposta v0.9 para fecho na PL09

**Grupo:** TP2-G2 · Tema F  
**Data desta proposta:** 04/11/2026  
**Estado:** proposta técnica consolidada, pendente de validação pelo grupo/docente. Não executar o conjunto de teste até existir versão aprovada e commitada.  
**Commit de fecho:** preencher depois do commit v1.0.

## 1. Problema e objetivo

Assistente de apoio a viticultores/técnicos para analisar uma fotografia opcional de folha de videira e responder a uma pergunta técnica através de documentação em português. O sistema deve distinguir hipótese visual de diagnóstico confirmado, apresentar fontes com documento, página e data, e recusar recomendações quando a evidência recuperada é insuficiente.

## 2. Dados e corpus

### 2.1 Imagens

FieldVitis: 555 imagens, quatro classes (saudável 158, esca 200, leaf blight 158, black rot 39), licença CC BY 4.0 conforme Guia Tema F. Não existe identificador conhecido de planta/parcela; a unidade de divisão disponível será a imagem individual, salvo confirmação de metadados adicionais. Declarar que as imagens provêm de datasets públicos e não representam especificamente vinhas portuguesas.

Divisão planeada: aproximadamente 70/15/15, estratificada por classe, com três seeds e pelo menos seis imagens black rot no teste. `data/splits.json`/`data/splits.csv` têm de registar IDs, classe, split, seed e hash. A PL08 pede um campo `grupo` e verificação de interseção entre grupos; como não há ID de planta/parcela, usar o ID da imagem como unidade e não inventar agrupamentos biológicos. Confirmar com F1/docente como preencher `grupo`.

### 2.2 Corpus textual

`data/corpus.csv` tem 12 registos: D01–D09 candidatos a índice; D10–D12 holdout. O índice proposto fica D01–D09 apenas depois de resolvidas as pendências de D07 e D08. Em qualquer caso, D10–D12 nunca entram no índice experimental. PDFs originais ficam fora do Git; o repositório guarda URL, data, licença, hash e metadados.

**Pendências documentais:**
- D07: o ficheiro fornecido é um excerto de 9 páginas de uma obra de 131 páginas e não inclui a secção de doenças fúngicas/leaf blight. Não o utilizar como fonte das perguntas leaf blight; procurar versão adequada/licenciada ou declarar a lacuna.
- D08: indica CC BY-NC-ND 4.0. Confirmar com docente/entidade se a indexação para RAG é permitida antes de o incluir.
- D01, D03–D06 e D09: não foi identificada licença aberta explícita em todos os casos; confirmar termos de utilização. Os PDFs são indexados/citados, não redistribuídos.
- D06: o ficheiro local enviado tem nome `Avisos_Agricolas-Circular_05-2026(1).pdf`, mas o conteúdo é Circular n.º 08/2026 de 21/05/2026; o SHA-256 corresponde ao registo D06. Corrigir nome local/citações para não atribuir Circular 05.

### 2.3 Documentos holdout

D10 — Edital 2/2025/FD_ST; D11 — Edital 1/2026/FD_ST; D12 — Despacho n.º 69/G/2026. São utilizados exclusivamente para criar respostas de referência de 10 perguntas cujo comportamento esperado é recusa, porque estes documentos ficam fora do índice. Antes de aprovar cada pergunta holdout, confirmar que os documentos indexados não contêm informação equivalente.

### 2.4 Lacuna de cobertura da classe leaf blight

O conjunto de 40 perguntas desta versão não inclui perguntas respondíveis específicas de `leaf_blight`/mancha-das-folhas. O D07 recebido é apenas um excerto de 9 páginas e não contém a secção pretendida da obra; o D08 descreve mancha-das-folhas, mas o repositório indica licença CC BY-NC-ND 4.0, cuja adequação à indexação tem de ser confirmada pelo docente. Não criar respostas de referência sem uma passagem utilizável e aprovada. Se D08 for autorizado para o índice, considerar acrescentar perguntas e rever a distribuição mantendo os 40 itens e o split antes da afinação.

## 3. Arquitetura e condições a comparar

### 3.1 Condições textuais

- **B0 — LLM sem RAG (baseline):** Qwen3-1.7B local (proposta), sem ferramentas e sem recuperação documental. Responde com conhecimento paramétrico, não deve inventar citações e declara incerteza.
- **B1 — LLM + RAG:** o mesmo LLM textual recebe passagens recuperadas e tem de citar cada afirmação factual.
- **B2 — LLM + RAG + recusa:** B1 mais regra de recusa baseada na ausência de evidência suficiente/limiar validado.

Nas condições textuais B0/B1/B2, usar o mesmo template de sistema de `src/prompts.py`; B0 recebe o bloco de evidência vazio e B1/B2 recebem as passagens recuperadas. Assim a diferença principal é a disponibilidade de evidência (e a política de recusa em B2), não uma reescrita arbitrária do prompt. O agente LLM+tools tem instruções adicionais de orquestração e é uma via de arquitetura separada.

### 3.2 Via de visão A — CNN como ferramenta do LLM

O agente LangChain dispõe de (i) ferramenta CNN do F2, apenas quando existe fotografia, e (ii) ferramenta de recuperação documental do F3. A CNN devolve uma hipótese visual e probabilidades; o LLM deve tratar a classe como hipótese, consultar documentação e responder com citações.

### 3.3 Via de visão B — VLM + RAG

O VLM aberto (proposta: Qwen2.5-VL-3B-Instruct) recebe a imagem e produz classe/descrição visual zero-shot. A pergunta + descrição/classe são usadas para recuperar passagens. Para manter a resposta documental comparável, o mesmo LLM textual gera a resposta final apoiada no RAG, conforme o desenho do Guia Tema F. O F2 avalia a classificação visual CNN vs VLM (e LoRA opcional).

**Pendente de confirmação do professor:** se a indicação oral “LLM + tools (CNN) contra VLMs” implica que o VLM também deve gerar a resposta final, alterar esta secção e registar a decisão. O Guia Tema F descreve o VLM como classificador/descritor visual e o LLM textual como gerador com RAG/citações.

## 4. Perguntas de referência e política de recusa (F4)

Conjunto proposto em `data/perguntas/perguntas.csv`: 40 perguntas para português natural, parafraseadas e com resposta de referência, documento, página, tipo e doença.

- 30 respondíveis com documentos candidatos a índice (D01–D09; nesta versão as perguntas usam D02–D06 e D09).
- 10 respondíveis apenas nos documentos holdout D10–D12; a ação esperada na aplicação é recusar.
- 20 perguntas de validação e 20 de teste; cada split tem 15 respondíveis + 5 holdout.
- O split é feito antes de afinar prompts, `top_k`, tamanho de chunks ou limiar de recusa.
- Não copiar literalmente frases das fontes. Registar a resposta de referência em linguagem concisa e identificar a página física do PDF (página 1-based).
- Quem cria uma pergunta não deve ser o único responsável por verificar a fidelidade da resposta a essa pergunta.

### 4.1 Critério de recusa

A aplicação deve recusar uma recomendação específica quando não recupera suporte documental suficiente. A recusa não deve depender apenas da confiança verbal do LLM. F3/F4/F5 definem e calibram a regra usando validação (por exemplo, limiar de relevância do retriever + instrução explícita), nunca o teste. Meta H2 proposta: recusar pelo menos 80% das perguntas sem suporte no índice e recusar indevidamente no máximo 10% das perguntas respondíveis; avaliar os valores efetivos e ICs, sem os tratar como resultado garantido.

## 5. Hipóteses

- **H1:** RAG aumenta a percentagem de afirmações suportadas face ao LLM sem RAG; estimar diferença emparelhada por pergunta e IC 95% bootstrap.
- **H2:** política de recusa atinge ≥80% de recusa em perguntas sem suporte e ≤10% de recusa indevida em perguntas com suporte; confirmar metas na validação.
- **H3:** VLM zero-shot tem Macro-F1 inferior à CNN afinada; LoRA (se aprovada) poderá reduzir a diferença.
- **H4 (exploratória):** RAG melhora a consistência entre repetições face ao LLM sem RAG, avaliando condição determinística e amostragem.

## 6. Métricas e anotação

- Texto — métrica principal: fidelidade, percentagem de afirmações da resposta que são suportadas pela fonte citada. Folha de verificação: uma linha por afirmação, com `S`, `N` ou `parcial`; fórmula do guia: `S / (S + N + parcial)`.
- Correção face à resposta de referência é medida separadamente da fidelidade.
- Dois anotadores humanos; reportar concordância (kappa de Cohen) e resolver desacordos em reunião.
- Imagem — Macro-F1 e recall por classe; atenção à classe minoritária black rot.
- Recuperação — Recall@5 da passagem-fonte.
- Recusa — taxa de recusa em perguntas sem suporte e taxa de recusa indevida em perguntas respondíveis.
- Outras — consistência entre repetições, latência, ECE quando a confiança estiver disponível/calibrada e catálogo de alucinações.

## 7. Repetições, seeds e registo

- CNN: três seeds de treino, conforme F2.
- VLM zero-shot: prompt fixo e mesmas imagens/splits da comparação visual.
- Texto: baseline sem RAG e condições com RAG nas mesmas perguntas; três repetições por condição, com temperatura/configuração registada. A configuração exata de temperatura e a contagem total devem ser fechadas pelo grupo na PL09.
- Cada execução guarda JSON com pergunta, resposta, prompt/configuração, modelo, passagens recuperadas (quando aplicável), ferramentas chamadas, timestamp, latência, repetição, seed e commit.
- Resultados resumidos em `paper/resultados.csv`; folhas humanas em `paper/fidelidade/`.

## 8. Protocolo de validação/teste e prevenção de fuga

- Validação: 20 perguntas (15 respondíveis + 5 holdout). Pode ser usada para afinar prompts, chunking, `top_k` e limiar de recusa.
- Teste: 20 perguntas (15 respondíveis + 5 holdout). Não usar para afinação, escolha de modelo, prompt, limiar ou debug iterativo.
- Antes de qualquer corrida no teste: aprovar este ficheiro, fazer commit e registar o hash do commit. O teste é executado uma única vez por condição/protocolo fechado, guardando todas as respostas.

**Questão a colocar ao docente na PL09:** o Guia Tema F, fase 2, diz que F4 executa o LLM sem RAG nas 40 perguntas × 3 repetições na semana 9, enquanto o protocolo do mesmo guia e o P2-D1 dividem as 40 em 20 validação + 20 teste antes da afinação, e a PL09 diz que PROTOCOLO tem de estar commitado antes de avaliar no teste. Até obter resposta, adota-se a opção conservadora: executar apenas validação antes do fecho; não abrir o teste.

## 9. Configuração proposta

- LLM textual: `qwen3:1.7b` via Ollama local; verificar capacidade de tool calling antes de escolher definitivamente. Alternativa do P2-D1: Qwen3-4B-Instruct-2507, se recursos permitirem.
- VLM: `Qwen/Qwen2.5-VL-3B-Instruct`; alternativa SmolVLM se houver limitações computacionais.
- Embeddings: BGE-M3; alternativa multilingual-e5.
- Índice: FAISS.
- `top_k`: 5 como ponto de partida, não valor final até validação.
- Chunk size/overlap e limiar de recusa: por decidir na validação.
- Hardware: registar dispositivo, precisão/quantização, memória e tempo. A inferência VLM 3B em GPU T4 foi prevista como viável na proposta, mas precisa de ser medida.

## 10. Decisões obrigatórias antes de marcar v1.0

- [ ] Professor confirma o que significa VLM vs LLM+tools, em particular quem gera a resposta final.
- [ ] F1/F3/docente confirmam se D07 e D08 entram no índice; o índice atual bloqueia-os por omissão.
- [ ] As 40 perguntas e as 10 perguntas holdout são revistas contra os PDFs e contra o corpus indexado.
- [ ] F2 confirma labels/contrato da CNN; F3 confirma contrato do retriever.
- [ ] Temperatura, repetições e custo computacional são fixados.
- [ ] Divisões e IDs de imagens são confirmados; a limitação de não existir ID de planta/parcela fica registada.
- [ ] A equipa confirma a regra de recusa e o procedimento de anotação.
- [ ] Versão final aprovada, commitada e hash registado antes de abrir o conjunto de teste.

## 11. Versão e commit

Esta versão é `v0.9-proposta`, gerada para revisão na PL08/PL09. Depois de fechar todos os pontos, mudar o cabeçalho para `v1.0-fechado`, registar a data e o hash Git do commit. Não declarar o protocolo fechado enquanto existirem caixas por resolver.


## 12. Estado dos objetivos de preparação para a PL09 (04/11/2026)

| Objetivo | Preparado no repositório | Demonstração ainda em falta |
|---|---|---|
| 40 perguntas | `data/perguntas/perguntas.csv` (40 linhas; distribuição 30/10 e 20/20) | Revisão humana de cada resposta/página e aprovação da auditoria de holdouts |
| LLM sem RAG | `src/llm_baseline.py`, `scripts/testar_llm_baseline.py` e `scripts/run_baseline_sem_rag.py` | Execução real com Ollama/modelo local; guardar JSON e JSONL |
| LLM + ferramentas | `src/llm_tools.py` com interfaces CNN/RAG e `src/tools_teste.py` com tools simuladas | Executar tool calling com o modelo local; integrar CNN F2 e retriever F3 |
| VLM | `src/vlm.py` e `scripts/testar_vlm.py` para Qwen2.5-VL-3B-Instruct | Inferência real numa imagem FieldVitis com dependências e hardware disponíveis |
| Contratos | `docs/CONTRATOS_COMPONENTES.md` | Aceitação/ajuste por F2, F3 e F1 |
| Protocolo fechado e Git | Esta versão v0.9; scripts bloqueiam o split de teste por omissão | Aprovação docente/grupo, fechar v1.0, commit local no repositório verdadeiro e registar hash |

As verificações automáticas de estrutura, páginas referenciadas e hashes são descritas no README. Não substituem a revisão humana das respostas nem demonstram que os modelos foram executados.
