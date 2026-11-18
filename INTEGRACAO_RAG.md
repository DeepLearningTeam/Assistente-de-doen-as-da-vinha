> Relatório histórico da integração RAG de 16/11. Para o estado atual da CNN, consulta [INTEGRACAO_CNN.md](INTEGRACAO_CNN.md).

# Integração RAG — revisão e verificação

Data: 16/11/2026. Base: versão atual do Rui. Resultado local em `integrado/`; entrega em `integrado_final.zip`.

## Versões efetivamente comparadas

Foram fornecidos três ZIPs, sem diretórios `.git`. Os originais ficaram preservados em `originais/`, fora da entrega. Não foi possível fazer um merge com histórico, identificar um antepassado comum ou atribuir com segurança a autoria/data de cada alteração.

| ZIP | Raiz interna | SHA-256 do ZIP |
|---|---|---|
| rui_atual.zip | Assistente-de-doen-as-da-vinha-RuiTeixeira | d5ad1227f062ddce0d62e5b8b968acc457b0d7479b844e49db55344269f291bc |
| colega_atual.zip | Assistente-de-doen-as-da-vinha-SilvaLukas | 3ca42d4aa3fb8a5c2425732f3d83eec818a39e3017a083282e2cfc7b0de899fc |
| colega_anterior.zip | Assistente-de-doen-as-da-vinha-bbb4fcbe61360922a95c3b066558ef7c098cf747 | 390bee8f3ae72d210eba154727a69f85c844b12639ec7cfec79ebba40dc57b04 |

O nome do ZIP anterior contém um identificador de commit; isso não demonstra que seja a base comum. Esse snapshot já contém o RAG híbrido monolítico, não uma versão comprovadamente anterior ao desenvolvimento do RAG. Serviu para esclarecer a separação posterior entre `indice.py` e `ingestao.py` e a intenção do avaliador de recuperação.

Entre os dois ZIPs atuais, diferem `.gitignore`, `AI_LOG.md`, `README.md`, `requirements.txt`, `src/avaliar.py` e `src/indice.py`. Só o do colega acrescenta `.vscode/settings.json`, `src/faiss.py`, `src/ingestao.py`, `src/perguntas.csv` e `test.py`. Os prompts, configurações, Gradio, CNN e restantes módulos de conversa são iguais nestes dois snapshots. A cópia integrada foi construída a partir de `rui_atual`, mantendo a sua documentação como base e o seu AI_LOG integral.

Não foram importados o FAISS auxiliar, o teste solto nem a cópia duplicada das perguntas. `src/perguntas.csv` do colega difere do CSV central em Q023. Conservou-se `data/perguntas/perguntas.csv` do Rui; não foi alterada a referência do teste. O AI_LOG do colega contém outros registos históricos, sem identificação da ferramenta; não foi usado para substituir o histórico oficial do Rui nem para inventar atribuições.

## Percurso do RAG reaproveitado

1. `src.corpus importar` identifica os PDF pelo SHA-256 do `data/corpus.csv` e coloca-os em `data/docs/Dxx.pdf`. O ZIP do corpus fica na raiz. Os 19 PDF fornecidos foram importados e os hashes confirmados.
2. `src.ingestao` seleciona `sim` e `condicional`, excluindo sempre D10–D12. Reutiliza os PDF importados; deixou de criar um segundo `src/pdfs_cache` com downloads e hashes apenas avisados.
3. A extração do colega usa PyMuPDF4LLM, Markdown por página, limpeza de marcadores, números isolados e linhas curtas repetidas em muitos locais. Divide por cabeçalhos Markdown e depois por caracteres: 1200, sobreposição 200 e mínimo de 50. A página é física e começa em 1; um chunk não atravessa páginas. O título só é prefixado quando explicitamente solicitado pela função, como no original.
4. BGE-M3 (`BAAI/bge-m3`) gera vetores densos normalizados de 1024 dimensões através de SentenceTransformers. BM25 (`Qdrant/bm25`) usa português. Não se substituiu a implementação por embeddings Ollama, apesar de o modelo `bge-m3:latest` estar instalado no Ollama.
5. O Qdrant usa `vetor_denso` com Cosine e `bm25` com modificador IDF. Os IDs UUID5 continuam derivados de documento, página e texto. Conservam-se payloads: `texto`, `pagina`, `cabecalhos`, `doc_id`, `nome_documento`, `titulo`, `entidade`, `data`, `url`, `temas` e `estado_fonte`.
6. A consulta vem da etapa de âmbito/contextualização do Rui. Quando apropriado, incorpora a hipótese e descrição visual. As pesquisas densa/BM25 devolvem até 30 candidatos e fundem por RRF; BGE-reranker-v2-m3 reordena pares pergunta/passagem, com comprimento máximo de 512 tokens. Guardam-se os scores de pesquisa e reranker.
7. O pipeline recebe cinco passagens antes do corte. Nos modos reforçados só gera com scores do reranker >= 0,60. Não há corte na ablação `llm_rag_sem_limiar`. Sem candidatos suficientes, recusa antes da geração.
8. Mantiveram-se a seleção de passagens pelo modelo, a validação dos identificadores, a verificação de números/atos, a revisão automática do suporte e a recusa/reformulação do Rui. O Gradio constrói a citação com documento, página física, data e link `#page=N`; mostra um excerto curto. A verificação automática não substitui anotação humana.

A construção escreve apenas numa instância Qdrant local. Recusa uma coleção já existente e não oferece `--recriar`/apagamento. O fecho do armazenamento local foi testado, incluindo uma segunda tentativa que preserva os dados existentes.

## Problemas concretos corrigidos

| Problema encontrado | Alteração e efeito |
|---|---|
| A avaliação do Rui importava funções FAISS que o `indice.py` Qdrant já não tinha | A avaliação mantém o fluxo do Rui e usa o diagnóstico/recuperação Qdrant integrados. |
| O avaliador do colega importava `from rag` e constantes que já tinham mudado de módulo | Reaproveitado como `src.avaliar_recuperacao`, com imports relativos e CSV central. |
| Parâmetros fixos no recuperador ignoravam os YAML | Modelos, coleção, pesquisa, candidatos, k, chunks e corte são efetivamente configurados pelos YAML; o `.env` configura os serviços. |
| Corte implícito 0,60 no recuperador e segundo filtro 0,43 no pipeline | O recuperador guarda top-k antes do corte e o pipeline aplica 0,60 uma única vez; o controlo sem limiar passa a desligá-lo realmente. |
| Recall calculado depois de eliminar passagens pelo limiar | A avaliação guarda as cinco passagens anteriores ao corte. Documentos ausentes não são retirados do denominador no avaliador do colega. |
| Não havia filtros documentais na pesquisa do colega | Aplicados filtros por doc_id selecionado e estado_fonte às duas pesquisas; diagnóstico/avaliação detetam fontes reservadas. |
| Cache de PDF e regras de validação duplicadas | Ingestão ligada a `data/docs/` e ao corpus/hash já existente. |
| O fallback de OCR podia usar `repetidas` antes de a inicializar | Inicialização/recomputação após extração e fallback; `--ocr por` falha explicitamente se o OCR exigido falhar. |
| Dependências do colega não incluíam a interface completa e fixavam um extrator inutilizável neste ambiente | Mantidas dependências diretas da aplicação e acrescentadas as do RAG. `pymupdf4llm==2.0` instalou aqui apenas metadados; fixadas versões funcionais 1.28.0 do extrator/PyMuPDF. |
| Ensaio de imagens existente só classificava | Acrescentado `avaliar respostas-imagem`, com a mesma pergunta, imagem e hash, sem histórico. Os JSON destinam-se a revisão humana; `metricas` explica isso se forem usados indevidamente como classificação. |

Os prompts do Rui, a lógica da CNN e o ficheiro da interface foram preservados. Em `rag.py`, a alteração funcional é o filtro passar a usar `limiar_reranker`. O comando principal da avaliação compara por predefinição `llm_cnn` com `llm_cnn_rag`; a condição adicional continua disponível.

## Compatibilidade da coleção real

A leitura real encontrou `videira_fitossanidade` com 989 pontos, schema compatível e 14 documentos: D01–D06, D08, D09, D13, D15–D19. D10–D12 não estavam presentes. D07 e D14 estavam ausentes face aos 16 selecionados no CSV.

Hash dos payloads lidos: `508335824f06281c6192dcab88afcb9ef301769b878ae8c8a282b62aec0f89c0`. A inspeção seguinte manteve esse hash. Só foram feitas operações remotas de leitura: esquema, scroll e consultas. A coleção do grupo não foi reconstruída nem alterada.

O schema e os payloads não comprovam a revisão dos pesos, o motor/OCR ou o processo que criou os vetores. Isso precisa de confirmação do colega. A reconstrução local usa a configuração documentada; pode produzir outros chunks devido à versão do extrator e ao OCR. Não deve ser confundida com a coleção existente.

A passagem esperada de Q001 foi recuperada em primeiro lugar: D02 p. 6, sobre a entrada de Eutypa lata por feridas recentes/de poda, score do reranker 0,99677354. A resposta final citou D02 p. 6 e também p. 7. A presença da passagem/citação foi verificada; a correção integral de todas as frases continua a exigir avaliação humana.

## Condições efetivas do F4

- `llm_cnn` sem imagem é a baseline textual: não chama CNN, embeddings ou Qdrant. `llm_cnn_rag` responde às mesmas perguntas com recuperação integrada. Cada item começa com histórico vazio.
- A linha completa do CSV fica em `referencia`, no JSON; só `pergunta` é passada a `responder`. Os testes e o registo real verificam esta separação.
- O prompt geral e os modelos foram preservados: Qwen3 `qwen3:1.7b-q4_K_M` e Qwen2.5-VL `qwen2.5vl:3b-q4_K_M`, contexto 8192. Sem RAG: até 1000 tokens e JSON de resposta livre. RAG: até 420 tokens e JSON com duas posições de passagens, 1–3 frases. A condição reforçada acrescenta verificações/possível reformulação. Estas diferenças afetam a interpretação causal dos resultados.
- Trocar de modo no Gradio conserva a conversa. Um seguimento sem RAG pode aproveitar texto de respostas RAG anteriores. Usa Nova conversa por condição; volta a selecionar o modo, porque Nova conversa repõe LLM + CNN + RAG.
- `avaliar imagem` mede classificação. `avaliar respostas-imagem` mede o percurso das respostas para posterior revisão humana. No VLM a fotografia chega à análise e à geração; no VLM + RAG chega à análise e contribui para consulta/hipótese, mas a geração final só recebe texto documental e hipótese.

## Verificações realizadas

As evidências pequenas estão em `verificacao/`. Os JSON completos das execuções locais permanecem também em `runs/`, fora do ZIP. Não constituem a avaliação oficial do grupo.

| Verificação | Tipo de execução | Resultado |
|---|---|---|
| Testes existentes e novos | 49 testes unittest; modelos de ensaio nos testes da conversa/índice | Passaram; incluem Qdrant real em memória, filtros, cortes, citações, recusa, histórico, persistência e Gradio HTTP. |
| Extração de PDF em duas páginas | PDF gerado localmente, PyMuPDF4LLM real, sem OCR | Páginas e metadados preservados. |
| Índice temporário local | PDF D02, BGE-M3/BM25/reranker reais | 33 chunks extraídos; dois da p. 6 indexados e pesquisados; armazenamento temporário fechado e removido. |
| Pergunta suportada | Qdrant partilhado e Qwen3 reais | Recuperou D02 p. 6, gerou resposta e citou a página esperada. |
| Pergunta fictícia sem suporte | Qdrant partilhado e Qwen3 reais | Recusa por score insuficiente, sem geração documental. |
| Baseline sem credenciais | Qwen3 real, variáveis Qdrant vazias e acesso ao cliente proibido pelo ensaio | Executou; produziu uma resposta, mas incluiu informação incorreta sobre a entrada da eutipiose. Isso não foi corrigido nos prompts para favorecer a baseline. |
| Avaliação de texto | Comando normal, Q001, duas condições, uma repetição, temperatura 0 | Gerou dois itens e manifesto; referência preservada no registo e ausente dos pedidos ao modelo. |
| Avaliador de recuperação do colega | Comando normal, híbrida + reranker, uma pergunta de validação | Recall@5 de documento/página e MRR iguais a 1 nesta única pergunta; não é uma estimativa do conjunto completo. |
| Respostas com fotografia | Qwen2.5-VL e Qdrant reais, mesma pergunta/foto de D02 p. 8 (imagem interna 36) | VLM respondeu; VLM + RAG recuperou cinco passagens, tentou geração/revisão e recusou por evidência insuficiente. |
| Anotação e resumo | `src.metricas`, execução curta | Criou duas afirmações para revisão; exportou resultados sem inventar anotações humanas. |
| Arranque Gradio | `python -m app.app --porta 7869`, servidor real | HTTP 200 em `/`, `/config` e `/gradio_api/runs`. Os testes da interface verificaram envio, gravação, reabertura e modos. |
| Proteção do .env | Git temporário com o .gitignore final | `.env` ignorado e `.env.example` incluível. Valores de credenciais não entram na documentação/configurações/ZIP. |

Na fotografia documental, o VLM classificou como saudável uma folha visivelmente danificada. O modo documental recusou a resposta após a verificação. O ensaio confirma o percurso da fotografia e da recuperação, não a correção da classificação. A fotografia não é FieldVitis nem um novo resultado de benchmark. A evidência inclui o hash idêntico da imagem nos dois itens e os prompts com marcadores de fotografia. A versão integrada conserva os prompts existentes; melhorar a qualidade visual exige trabalho de validação separado.

O ensaio local de reconstrução foi limitado a uma amostra real. Não foi construído o índice completo dos 16 documentos nesta sessão. O modo OCR português explícito depende da instalação Tesseract/por do teu Windows e não foi executado. A CNN treinada, o FieldVitis real e a avaliação humana permanecem pendentes. Não foi feita revisão visual/cópia no navegador nesta sessão; os testes de interface foram HTTP/eventos.

## Comandos executados e repetição

Executa na raiz de `integrado`, em PowerShell. A primeira instalação usou Python 3.14.0. O launcher `py -3.14` desta máquina não reconheceu o registo do Python; `python -m venv .venv` criou o ambiente com o Python 3.14 disponível. A instalação de dependências exigiu acesso de rede; o estado final passou `pip check`.

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m compileall -q src app tests
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m src.corpus importar ".\docs do corpus.zip"
.\.venv\Scripts\python.exe -m src.indice info
.\.venv\Scripts\python.exe -m src.indice pesquisar "Por onde pode entrar na videira o fungo associado à eutipiose?"
.\.venv\Scripts\python.exe -m src.verificar --sem-rag
.\.venv\Scripts\python.exe -m src.verificar --rag
.\.venv\Scripts\python.exe -m src.verificar --indice-local
.\.venv\Scripts\python.exe -m src.avaliar texto --execucao integracao-par --condicoes llm_cnn llm_cnn_rag --limite 1 --repeticoes 1 --temperaturas 0
.\.venv\Scripts\python.exe -m src.avaliar_recuperacao --split validacao --modo hibrida --reranker --limite 1 --saida runs/recuperacao-ensaio.csv
.\.venv\Scripts\python.exe -m src.avaliar respostas-imagem --execucao integracao-foto --foto ".\runs\verificacao\foto_documental.png" --pergunta "Que sinais se veem nesta folha de videira?"
.\.venv\Scripts\python.exe -m src.metricas anotar --execucao integracao-par
.\.venv\Scripts\python.exe -m src.metricas resumir --execucoes integracao-par --destino runs/metricas-integracao --bootstrap 10
.\.venv\Scripts\python.exe -m app.app --porta 7869
```

A fotografia usada no ensaio foi extraída do PDF fornecido, não gerada. Para repetir exatamente esse ensaio após importar o corpus, cria o ficheiro com:

```powershell
New-Item -ItemType Directory -Force .\runs\verificacao | Out-Null
.\.venv\Scripts\python.exe -c "import pymupdf; d=pymupdf.open('data/docs/D02.pdf'); p=pymupdf.Pixmap(d,36); p.save('runs/verificacao/foto_documental.png')"
```

Na sessão, o corpus foi importado a partir do ZIP existente dentro do snapshot original; o comando acima usa a localização normal descrita no README. A instalação final reproduz-se com `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`. Para repetir avaliações após alterações de código/protocolo, usa outro nome em `--execucao`; o manifesto impede misturar configurações.

O README contém os passos completos para credenciais, índice existente/local, OCR, FieldVitis e avaliação. O ZIP final exclui `.env`, ambientes virtuais, caches, PDF, imagens do dataset, índices e pesos. A cópia de trabalho local conserva o `.env` configurado e os PDF importados para te permitir rever o resultado.
