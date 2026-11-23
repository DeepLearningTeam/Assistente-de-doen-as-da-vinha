# P2 · Tema F — Assistente da Vinha

**Aprendizagem Profunda Aplicada · TP2 · Grupo 2**

## 1. Objetivo

A aplicação ajuda a interpretar uma fotografia de uma folha de videira e a esclarecer perguntas sobre viticultura em português. Nos modos com RAG, procura passagens no corpus, explica a informação e liga as afirmações à página e data da fonte. Quando a evidência é insuficiente, indica-o na conversa.

## 2. Estrutura do repositório

```text
DL_P2_TP2-G2/
├── README.md, PROTOCOLO.md, AI_LOG.md
├── configs/                  Modos, modelos, prompts e parâmetros
├── data/
│   ├── corpus.csv            Metadados, licenças, URLs e SHA-256
│   ├── splits.json           Divisão adaptada da CNN seed 1 e holdout separado
│   ├── originais/            Divisões e manifestos recebidos, preservados
│   ├── perguntas/perguntas.csv
│   ├── docs/                 Pasta local dos PDF
│   └── imagens/              Pasta local do FieldVitis
├── src/                      Visão, corpus, índice, conversa e avaliação
├── notebooks/                Exploração
├── runs/                     Execuções, conversas e checkpoints locais
├── app/app.py                Interface Gradio
├── paper/                    Resultados, fidelidade, leituras, figuras e tabelas
└── requirements.txt
```

## 3. Instruções de setup

A configuração habitual utiliza a coleção Qdrant partilhada e não exige PDF locais nem reconstrução do índice. A preparação documental opcional está na secção 7.

### 3.1. Preparar o ambiente

Clonar o repositório ou descompactar o projeto e abrir um terminal na pasta raiz. Instalar o [Python 3.14](https://www.python.org/downloads/) e o [Ollama](https://ollama.com/download).

Os ficheiros `.env` e `resnet50_seed1.keras` estão no **grupo do WhatsApp** e não acompanham o clone. Utilizar o conteúdo do `.env` para preencher a configuração descrita na secção 3.2.

Colocar o modelo de cinco classes `resnet50_seed1.keras` em `runs/checkpoints/`, dentro da pasta clonada do projeto. O caminho final deve ser `Assistente-de-doen-as-da-vinha\runs\checkpoints\resnet50_seed1.keras`. O modelo é ignorado pelo Git e tem de ser copiado para cada instalação local.

Se o ficheiro partilhado aparecer como `resnet50_seed1.keras.zip`, **renomear para `resnet50_seed1.keras`, removendo apenas o `.zip` final**. Não descompactar: o ficheiro já é um modelo Keras completo. Após a renomeação, colocar em `runs/checkpoints/`. Se já tiver o nome `resnet50_seed1.keras`, copiar diretamente. `configs/cnn.yaml` centraliza o caminho, a seed de treino e a ordem das classes.

**Windows PowerShell**

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Se o launcher `py -3.14` não reconhecer o Python instalado, verificar `python --version`. Se indicar Python 3.14, o ambiente pode ser criado com `python -m venv .venv`.

Os comandos Python para PowerShell usam diretamente o executável do ambiente virtual. Executar os comandos a partir da pasta raiz do projeto.

**Ativação opcional no PowerShell**

Para ativar o ambiente virtual, caso o PowerShell bloqueie o script `Activate.ps1`, executar os dois comandos seguintes na mesma janela.

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

O âmbito `Process` limita o bypass à sessão atual. A alteração termina quando a sessão é fechada.

**macOS ou Linux**

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Com o Ollama aberto, descarregar os modelos. Se o serviço estiver parado, executar `ollama serve` noutro terminal e mantê-lo aberto.

```bash
ollama pull qwen3:1.7b-q4_K_M
ollama pull qwen2.5vl:3b-q4_K_M
ollama list
```

### 3.2. Ligar à coleção partilhada

A coleção do grupo chama-se **`videira_fitossanidade`** e está num **servidor Qdrant remoto**, acessível pela Internet. Guarda as passagens dos documentos, os vetores usados na pesquisa e os metadados das fontes. Cada pessoa liga a sua aplicação à mesma coleção. O código Python e os modelos executam-se no computador dessa pessoa.

**O acesso a esta coleção dispensa a instalação de um servidor Qdrant, a descarga do índice e a reindexação dos documentos.** A instalação de `requirements.txt` já inclui o cliente Python que comunica com o servidor. O grupo tem de manter o servidor acessível e disponibilizar o endereço e a chave de acesso por um meio privado.

**Passo 1 — Criar o ficheiro de configuração**

Se ainda não existir um `.env`, copiar o exemplo na raiz do projeto e abrir o ficheiro local:

```powershell
Copy-Item .env.example .env
notepad .env
```

Se já existir um `.env` preenchido, abrir diretamente com `notepad .env`, preservando os valores existentes. O `.env.example` só contém valores fictícios: clonar ou descompactar o repositório não concede automaticamente acesso ao Qdrant do grupo.

**Passo 2 — Preencher os dados fornecidos pelo grupo**

No `.env`, inserir o conteúdo partilhado no **grupo do WhatsApp** e guardar o ficheiro. O exemplo seguinte mostra apenas as variáveis de acesso ao Qdrant, com valores fictícios:

```dotenv
QDRANT_URL=https://exemplo.invalid:6333
QDRANT_API_KEY=chave_ficticia
QDRANT_COLECAO=videira_fitossanidade
```

`QDRANT_URL` é o endereço do servidor, `QDRANT_API_KEY` permite o acesso e `QDRANT_COLECAO` escolhe a coleção dentro desse servidor. Para usar a coleção partilhada, manter `QDRANT_LOCAL_PATH` comentado ou ausente, pois um caminho local preenchido tem prioridade sobre o servidor remoto.

Manter o `.env` apenas no computador local. É ignorado pelo Git; partilha-se o `.env.example` com valores fictícios. A aplicação carrega o `.env` da raiz do projeto. Variáveis já definidas na sessão PowerShell têm prioridade sobre esse ficheiro. Reiniciar a aplicação após alterar a configuração.

**Passo 3 — Confirmar o acesso e experimentar uma pesquisa**

```powershell
.\.venv\Scripts\python.exe -m src.indice info
.\.venv\Scripts\python.exe -m src.indice pesquisar "Por onde pode entrar na videira o fungo associado à eutipiose?"
```

O primeiro comando deve mostrar a coleção, o número de pontos, os documentos e eventuais documentos em falta. O segundo deve devolver passagens com documento e página. Ambos só leem a coleção. Se a ligação falhar, verificar os valores do `.env`, a ligação à Internet e a disponibilidade do servidor. Após confirmar o acesso, iniciar o Gradio conforme a secção 3.3.

Os modos sem RAG funcionam sem credenciais Qdrant. Sem fotografia, a CNN não é chamada. Para utilizar um índice próprio no computador local, seguir a alternativa da secção 7.2.

A pesquisa documental integrada utiliza `BAAI/bge-m3` por SentenceTransformers, vetores densos normalizados com 1024 dimensões, `Qdrant/bm25` em português, fusão RRF de 30 candidatos, `BAAI/bge-reranker-v2-m3` e cinco passagens finais. Estes modelos são descarregados do Hugging Face na primeira consulta e executados localmente. A primeira execução exige Internet e pode demorar e ocupar vários GB.

A coleção deve ter os vetores nomeados `vetor_denso` (1024, Cosine) e `bm25` (IDF), e os payloads documentais da coleção original. `info` verifica o esquema, páginas, campos e documentos reservados; mostra também documentos em falta. Isso não prova quais os pesos/revisões usados originalmente: a proveniência deve ser confirmada nos registos de construção da coleção. Consultar [INTEGRACAO_RAG.md](INTEGRACAO_RAG.md).

### 3.3. Abrir a interface

**Windows PowerShell**

```powershell
.\.venv\Scripts\python.exe -m app.app
```

**macOS ou Linux**

```bash
python -m app.app
```

Abrir o endereço local indicado no terminal. Introduzir uma pergunta, anexar uma fotografia JPG ou PNG, ou combinar as duas entradas.

| Modo | Quem analisa a fotografia | Quem responde | Consulta ao corpus |
|---|---|---|---|
| LLM + CNN + RAG | CNN | Qwen3 | Sim |
| LLM + CNN | CNN | Qwen3 | Não |
| VLM | Qwen2.5-VL | Qwen2.5-VL | Não |
| VLM + RAG | Qwen2.5-VL | Qwen2.5-VL | Sim |

Sem fotografia, os quatro modos aceitam texto e não carregam a CNN. Para voltar a analisar a imagem anterior, selecionar **Voltar a usar a última fotografia**. **Mudar de modo mantém o histórico. Para comparar condições, iniciar uma Nova conversa por condição e voltar a selecionar o modo pretendido** (Nova conversa repõe LLM + CNN + RAG). O modelo recebe até seis turnos relevantes de histórico. Por baixo do seletor de modo aparecem o nome e a variante do modelo de resposta, o modelo de embeddings quando há RAG e o checkpoint da CNN quando aplicável.

As conversas são guardadas automaticamente após cada resposta em `runs/chats/`.

Nos modos RAG, as citações junto à resposta apontam para o URL da fonte com a indicação da página (`#page=N`). O painel lateral mostra o título, a entidade, a data e um excerto curto. A hipótese visual aparece em separado e não confirma um diagnóstico. Nos modos CNN, imagens classificadas como `Outro` ou com baixa confiança são rejeitadas; os detalhes estão no [protocolo](PROTOCOLO.md#9-configuração).

## 4. Avaliação

### 4.1. Avaliar respostas de texto

A comparação textual principal inclui `llm_cnn`, `llm_rag_sem_limiar` e `llm_cnn_rag`, com as mesmas perguntas, sem fotografia e com histórico vazio em cada item. As três condições usam o mesmo LLM, prompt de sistema, instruções gerais, esquema JSON (`primeira`, `segunda`, `resposta`) e limite `max_resposta_tokens` de `configs/base.yaml` (atualmente 1000 tokens). Sem RAG, os dois campos de seleção de passagens são 0; com RAG, identificam as fontes da resposta. As condições diferem na recuperação, nas citações e na recusa. A CNN não intervém. O CSV de referência fica no registo e não entra no prompt. Para um ensaio curto e para a execução de referência, utilizar nomes diferentes em `--execucao`.

**Windows PowerShell**

```powershell
.\.venv\Scripts\python.exe -m src.avaliar dados
.\.venv\Scripts\python.exe -m src.avaliar texto --execucao ensaio --condicoes llm_cnn llm_rag_sem_limiar llm_cnn_rag --limite 1 --repeticoes 1 --temperaturas 0
.\.venv\Scripts\python.exe -m src.avaliar texto --execucao texto-validacao --condicoes llm_cnn llm_rag_sem_limiar llm_cnn_rag
.\.venv\Scripts\python.exe -m src.metricas anotar --execucao texto-validacao
.\.venv\Scripts\python.exe -m src.metricas resumir --execucoes texto-validacao
```

**macOS ou Linux**

```bash
python -m src.avaliar dados
python -m src.avaliar texto --execucao ensaio --condicoes llm_cnn llm_rag_sem_limiar llm_cnn_rag --limite 1 --repeticoes 1 --temperaturas 0
python -m src.avaliar texto --execucao texto-validacao --condicoes llm_cnn llm_rag_sem_limiar llm_cnn_rag
python -m src.metricas anotar --execucao texto-validacao
python -m src.metricas resumir --execucoes texto-validacao
```

O mesmo comando com o mesmo nome retoma uma execução interrompida compatível. Alterações ao código, parâmetros, dados ou protocolo exigem um nome novo. Os resumos ficam em `runs/<execucao>/metricas/`; para exportar resultados selecionados para o relatório, acrescentar `--destino .\paper` ao comando `resumir`. As folhas de anotação permanecem em `paper/fidelidade/<execucao>/`. Depois de concluída a geração, os dois anotadores preenchem as folhas separadamente e registam o consenso. O resumo calcula as métricas humanas a partir dessas folhas. Consultar [PROTOCOLO.md](PROTOCOLO.md) e [paper/fidelidade/README.md](paper/fidelidade/README.md).

A condição intermédia `llm_rag_sem_limiar`, incluída também por predefinição, conserva a recuperação e as citações, mas desliga o corte e a verificação de suporte. A avaliação de recuperação documental integrada utiliza o CSV central:

```powershell
.\.venv\Scripts\python.exe -m src.avaliar_recuperacao --split validacao --modo hibrida --reranker --saida runs/recuperacao-validacao.csv
```

Os testes e os ensaios curtos com serviços reais repetem-se com:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m src.verificar --sem-rag
.\.venv\Scripts\python.exe -m src.verificar --rag
.\.venv\Scripts\python.exe -m src.verificar --indice-local
```

`--sem-rag` usa o Qwen3 real e proíbe qualquer acesso ao Qdrant. `--rag` verifica a recuperação de D02 p. 6, a citação final e a recusa de uma pergunta fictícia sem suporte. `--indice-local` exige os PDF importados e testa uma amostra de D02 numa coleção temporária local. Cada execução cria uma pasta datada em `runs/verificacao/`, com JSON e hashes do código, configuração e protocolo. Estes são ensaios de funcionamento, não resultados científicos.

### 4.2. Preparar o FieldVitis e avaliar imagens

O FieldVitis e as imagens `Outro` não acompanham o clone. Para preparar os dados completos, colocar `fieldvitis.zip` e `outro.zip` na raiz e extrair para `data/imagens/`. O segundo arquivo contém as 160 imagens adicionais; a aplicação pode usar os pesos sem estes dados, mas a preparação completa exige ambos os conjuntos. Executar os comandos a partir da raiz:

**Windows PowerShell**

```powershell
Expand-Archive -LiteralPath ".\fieldvitis.zip" -DestinationPath ".\data\imagens"
Expand-Archive -LiteralPath ".\outro.zip" -DestinationPath ".\data\imagens"
.\.venv\Scripts\python.exe -m src.preparar_cnn
```

**macOS ou Linux**

```bash
unzip ./fieldvitis.zip -d ./data/imagens
unzip ./outro.zip -d ./data/imagens
python -m src.preparar_cnn
```

As cinco pastas das classes ficam diretamente dentro de `data/imagens/`, sem uma pasta `fieldvitis/` adicional. Preservar os originais; se outro arquivo tiver uma pasta exterior, copiar as classes para estes destinos. A preparação confere MD5 com o manifesto original, calcula SHA-256 e duplicados exatos de píxeis, verifica as partições e adapta os IDs sem voltar a sortear imagens.

```text
data/imagens/Black_rot/
data/imagens/Esca/
data/imagens/Healthy/
data/imagens/Leaf_blight/
data/imagens/Outro/
```

Por exemplo, `Healthy_10.jpg` fica em `data/imagens/Healthy/Healthy_10.jpg`.

`data/splits.json` adapta a **seed de treino 1**, com 715 registos: 454 de treino, 98 de validação, 99 de teste interno e 64 de `holdout`. As 555 imagens FieldVitis mantêm as divisões anteriores. A comparação `avaliar imagem` continua limitada às quatro classes de folhas: 76 de validação, 77 de teste e 50 de holdout; não mede a rejeição de `Outro`. Os registos adicionais conservam o MD5 fornecido e têm SHA-256 por preencher até à preparação com as imagens reais. Os manifestos recebidos ficam em `data/originais/cnn/`.

Para verificar apenas o carregamento, a arquitetura e as classes, sem fotografias, executar `.\.venv\Scripts\python.exe -m src.verificar_cnn --apenas-modelo`. A paridade e os percursos com fotografia exigem as imagens de validação.

Ensaiar a implementação e comparar CNN/VLM na validação:

**Windows PowerShell**

```powershell
.\.venv\Scripts\python.exe -m src.verificar_cnn
.\.venv\Scripts\python.exe -m src.verificar_cnn --servicos
.\.venv\Scripts\python.exe -m src.avaliar imagem --execucao cnn-ensaio --modelos cnn --seeds 1 --split validacao --limite 4
.\.venv\Scripts\python.exe -m src.avaliar imagem --execucao visao-validacao --modelos cnn vlm --seeds 1 --split validacao
.\.venv\Scripts\python.exe -m src.metricas resumir --execucoes visao-validacao
```

**macOS ou Linux**

```bash
python -m src.avaliar imagem --execucao visao-validacao --modelos cnn vlm --seeds 1 --split validacao
python -m src.metricas resumir --execucoes visao-validacao
```

O avaliador verifica o hash do checkpoint, a correspondência das partições com o treino e os hashes das imagens executadas. A seed `1` identifica o treino CNN e a sua divisão; a seed de geração do VLM continua a ser `42`. As seeds `42/43/44` das repetições de texto não representam novos treinos CNN. Apenas estão disponíveis pesos da seed 1: não é possível reproduzir três modelos. `src.preparar_cnn` adapta as divisões originais da CNN, sem gerar novas amostragens.

Após fixar as condições, comparar **classificação** no teste interno de 77 imagens:

```powershell
.\.venv\Scripts\python.exe -m src.avaliar imagem --execucao visao-teste-seed1 --modelos cnn vlm --seeds 1 --split teste
```

As 50 imagens finais ficaram reservadas nesta integração. Só quando o grupo fechar o protocolo, executar uma vez por condição, guardar os resultados e reservá-los para a avaliação final:

```powershell
.\.venv\Scripts\python.exe -m src.avaliar imagem --execucao visao-final-50 --modelos cnn vlm --seeds 1 --split holdout
.\.venv\Scripts\python.exe -m src.metricas resumir --execucoes visao-final-50
```

`--limite` só é aceite na validação. O teste interno já consta das análises originais da CNN; não substitui as 50 imagens reservadas. O teste interno tem seis casos de black rot; o holdout tem quatro, abaixo dos seis previstos no D1. As métricas de classificação não medem correção ou fidelidade das respostas. A auditoria de hashes não deteta quase duplicados nem demonstra separação por videira/local de recolha.

### 4.3. Comparar respostas dos quatro modos

O comando `imagem` anterior mede **classificação**, não a qualidade da resposta à pergunta. Para comparar respostas, fornecer a mesma fotografia e a mesma pergunta aos quatro modos:

```powershell
.\.venv\Scripts\python.exe -m src.avaliar respostas-imagem --execucao respostas-foto-validacao --foto ".\data\imagens\Esca\Esca_0.jpg" --pergunta "Que sinais se veem nesta folha de videira?"
```

O ficheiro tem de existir. Cada condição começa sem histórico e guarda o hash da mesma imagem, os prompts e a resposta em `runs/`. Todos os modos partilham o prompt de sistema, as instruções gerais, o esquema JSON e o limite `max_resposta_tokens` descritos na secção 4.1, incluindo reparações e revisões. Nos dois modos VLM, a mesma fotografia chega à análise visual e à geração final, com a mesma observação; o modo RAG acrescenta as passagens documentais. Nos modos LLM, a CNN fornece a mesma hipótese e probabilidades com e sem RAG, sem fotografia nem descrição de sinais observados ao gerador. A classificação CNN e a observação inicial do VLM mantêm os seus contratos próprios. Para avaliar apenas o par VLM, utilizar `--condicoes vlm vlm_rag`, sem exigir os pesos CNN.

Exportar as folhas e resumir as anotações:

```powershell
.\.venv\Scripts\python.exe -m src.metricas anotar --execucao respostas-foto-validacao
# Preencher as duas folhas e o consenso antes do resumo humano.
.\.venv\Scripts\python.exe -m src.metricas resumir --execucoes respostas-foto-validacao
```

A tabela `runs/respostas-foto-validacao/metricas/tabelas/respostas_imagem.csv` fica separada da classificação. Sem anotação, só se calculam latência e proporção de recusas. A execução corresponde a uma pergunta e uma fotografia: repetir com nomes novos para as outras unidades, escolhidas do split adequado. Uma fotografia arbitrária não é automaticamente um teste independente.

## 5. Dados e referências

- **FieldVitis v2** · 555 fotografias, licença CC BY 4.0. A classe podridão-negra tem 39 imagens e o conjunto não foi recolhido especificamente em vinhas portuguesas. [Dataset e citação](https://doi.org/10.5281/zenodo.17307846)
- **Outro** · 160 imagens de folhas de feijão, flores e animais. A proveniência está em `data/originais/cnn/outro_fontes.csv`.
- **Corpus** · `data/corpus.csv` conserva os 19 registos, incluindo as licenças e os URLs. A configuração inclui 14 documentos, contando com os sete condicionais mantidos. D07 e D14 ficam excluídos da indexação; D10–D12 ficam reservados para a avaliação da recusa. Os PDF e as imagens são ficheiros locais; o índice pode ser a coleção partilhada ou uma coleção local nova.
- **Perguntas** · `data/perguntas/perguntas.csv` contém 40 perguntas, com referência, página e indicação de validação humana. Q023 foi substituída por uma pergunta sobre as consequências da desfolha precoce, suportada por D15, página 1, mantendo 15 perguntas respondíveis e cinco de recusa em cada split. As referências bibliográficas estão em [paper/referencias.bib](paper/referencias.bib).
- **Documentação técnica** · [ChatOllama](https://docs.langchain.com/oss/python/integrations/chat/ollama) · [Gradio](https://www.gradio.app/docs/gradio/blocks)

**Apoio técnico experimental.** Confirmar produtos e doses na DGAV ou com um técnico. [Consulta SIFITO](https://www.dgav.pt/medicamentos/conteudo/produtos-fitofarmaceuticos/divulgacao/)

O registo histórico da integração CNN está em [INTEGRACAO_CNN.md](INTEGRACAO_CNN.md). A comparação e verificação anterior do RAG permanece em [INTEGRACAO_RAG.md](INTEGRACAO_RAG.md).

## 6. Análise dos resultados

Após obter uma execução de classificação, exportar calibração e erros/abstenções ordenados por confiança:

```powershell
.\.venv\Scripts\python.exe -m src.analise --execucao visao-validacao
# Opcional, para desenhar a curva de calibração:
.\.venv\Scripts\python.exe -m pip install matplotlib
.\.venv\Scripts\python.exe -m src.analise --execucao visao-validacao --figuras
```

Os CSV e a figura ficam em `runs/<execucao>/analise/`. O ECE mantém dez intervalos e só usa previsões aceites com confiança; as abstenções aparecem nos casos de erro. A confiança do VLM é declarada pelo modelo, não uma probabilidade calibrada. Este comando não ajusta o modelo. Não utilizar resultados de teste/holdout para afinação.

## 7. Preparação documental opcional

Estas operações permitem importar os PDF e construir um índice local. Para utilizar apenas a coleção partilhada, seguir as instruções da secção 3.

### 7.1. Importar os PDF

Colocar `docs do corpus.zip` na raiz do projeto, ao lado de `README.md` e `requirements.txt`. Manter o ficheiro em formato ZIP. O comando de importação extrai os PDF para `data/docs/` e identifica-os pelo SHA-256. Se o ZIP estiver noutro local, indicar o caminho completo entre aspas.

**Windows PowerShell**

```powershell
.\.venv\Scripts\python.exe -m src.corpus importar ".\docs do corpus.zip"
```

**macOS ou Linux**

```bash
python -m src.corpus importar "./docs do corpus.zip"
```

Para tentar obter os PDF pelos URLs registados, utilizar `.\.venv\Scripts\python.exe -m src.corpus descarregar` no PowerShell ou `python -m src.corpus descarregar` em macOS ou Linux.

### 7.2. Construir uma coleção local nova

Este passo é necessário apenas para reconstruir ou testar o índice. A construção reutiliza a extração Markdown por página, limpeza, divisão por cabeçalhos e chunks de 1200 carateres com sobreposição de 200 da implementação original. Utiliza os PDF importados em `data/docs/`, confirmados por SHA-256, e exclui D07, D10–D12 e D14. Não há escrita no serviço partilhado neste comando.

```powershell
.\.venv\Scripts\python.exe -m src.ingestao --local .\data\qdrant_local --colecao videira_teste --ocr por
```

O comando recusa reutilizar uma coleção já existente e não apaga dados. Para outra construção, escolher um caminho e uma coleção novos. Guardar a configuração da ingestão junto do índice local. Fechar a aplicação antes de abrir o mesmo armazenamento local noutro processo.

Para consultar essa coleção, acrescentar ao `.env`:

```dotenv
QDRANT_LOCAL_PATH=data/qdrant_local
QDRANT_COLECAO=videira_teste
```

Em seguida, executar `src.indice info` e iniciar a aplicação. Para voltar à coleção do grupo, remover ou comentar `QDRANT_LOCAL_PATH` e repor `QDRANT_COLECAO=videira_fitossanidade`. Reiniciar a aplicação após alterar o `.env`.

### 7.3. OCR e Tesseract no Windows

O procedimento principal é `--ocr por`, com Tesseract em português. Configurar o OCR antes de executar a construção local da secção 7.2. Como alternativas, `--ocr auto` mantém a seleção automática do PyMuPDF4LLM/RapidOCR e `--ocr off` usa só o texto extraível dos PDF. Instalar o Tesseract e `por.traineddata` e verificar os caminhos:

```powershell
$tess = 'C:\Program Files\Tesseract-OCR'
Test-Path "$tess\tesseract.exe"
Test-Path "$tess\tessdata\por.traineddata"
$env:Path = "$tess;$env:Path"
$env:TESSDATA_PREFIX = "$tess\tessdata"
tesseract --list-langs
.\.venv\Scripts\python.exe -m src.ingestao --local .\data\qdrant_por --colecao videira_por --ocr por
```

Se `por.traineddata` estiver em `data/tessdata`, utilizar `$env:TESSDATA_PREFIX = (Resolve-Path '.\data\tessdata').Path`. Repetir a configuração em cada nova janela de PowerShell. O modo `por` falha se o OCR português não estiver disponível, sem mudar silenciosamente de motor. Mudar a extração/OCR cria outras passagens e pode mudar os resultados; registar essa alteração antes de comparar ensaios.
