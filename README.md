# P2 · Tema F — Assistente da Vinha

**Aprendizagem Profunda Aplicada · TP2 · Grupo 2**

## 1. Objetivo

A aplicação ajuda a interpretar uma fotografia de uma folha de videira e a esclarecer perguntas sobre viticultura em português de Portugal. Nos modos com RAG, procura passagens no corpus, explica a informação e liga as afirmações à página e data da fonte. Quando a evidência é insuficiente, indica-o na conversa.

## 2. Estrutura do repositório

```text
DL_P2_TP2-G2/
├── README.md, PROTOCOLO.md, AI_LOG.md
├── configs/                  Modos, modelos, prompts e parâmetros
├── data/
│   ├── corpus.csv            Metadados, licenças, URLs e SHA-256
│   ├── splits.json           Divisões de referência do FieldVitis v2
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

## 3. Preparar o ambiente

Descompacta o projeto e abre um terminal na pasta raiz do projeto. Instala o [Python 3.14](https://www.python.org/downloads/) e o [Ollama](https://ollama.com/download).

**Windows PowerShell**

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Se o launcher `py -3.14` não reconhecer o Python instalado, confirma `python --version`. Se indicar Python 3.14, podes criar o ambiente com `python -m venv .venv`.

Os comandos Python para PowerShell usam diretamente o executável do ambiente virtual. Executa-os a partir da pasta raiz do projeto.

**Ativação opcional no PowerShell**

Se quiseres ativar o ambiente virtual e o PowerShell bloquear o script `Activate.ps1`, executa estes dois comandos na mesma janela.

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

O âmbito `Process` limita o bypass à sessão atual. A alteração termina quando fechas essa sessão.

**macOS ou Linux**

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Com o Ollama aberto, descarrega os modelos. Se usares o Ollama pela linha de comandos e o serviço estiver parado, deixa `ollama serve` aberto noutro terminal.

```bash
ollama pull qwen3:1.7b-q4_K_M
ollama pull qwen2.5vl:3b-q4_K_M
ollama list
```

## 4. Preparar o corpus e o índice

**Num computador novo, o setup normal usa a coleção Qdrant já existente do grupo.** Depois de preparares o ambiente na secção 3, segue a secção **4.2** e arranca a aplicação na secção **5**. Para conversar ou avaliar com essa coleção, basta configurar o acesso ao servidor; os PDF locais e a reconstrução do índice não são necessários.

As secções 4.1, 4.3 e 4.4 explicam a preparação dos PDF, a construção de um índice local e o OCR, caso precises de reconstruir ou testar o índice.

### 4.1. Importar os PDF

Coloca `docs do corpus.zip` na raiz do projeto, ao lado de `README.md` e `requirements.txt`. Mantém o ficheiro em formato ZIP. O comando de importação extrai os PDF para `data/docs/` e identifica-os pelo SHA-256. Se o ZIP estiver noutro local, podes indicar o caminho completo entre aspas.

**Windows PowerShell**

```powershell
.\.venv\Scripts\python.exe -m src.corpus importar ".\docs do corpus.zip"
```

**macOS ou Linux**

```bash
python -m src.corpus importar "./docs do corpus.zip"
```

Para tentar obter os PDF pelos URLs registados, usa `.\.venv\Scripts\python.exe -m src.corpus descarregar` no PowerShell ou `python -m src.corpus descarregar` em macOS ou Linux.

### 4.2. Ligar um novo computador à coleção partilhada

A coleção do grupo chama-se **`videira_fitossanidade`** e está num **servidor Qdrant remoto**, acessível pela Internet. Guarda as passagens dos documentos, os vetores usados na pesquisa e os metadados das fontes. Cada pessoa liga a sua aplicação à mesma coleção. O código Python e os modelos executam-se no computador dessa pessoa.

**Para usar esta coleção, não precisas de instalar um servidor Qdrant, descarregar o índice ou voltar a indexar os documentos.** A instalação de `requirements.txt` já inclui o cliente Python que comunica com o servidor. O grupo tem de manter o servidor acessível e fornecer-te o endereço e a chave de acesso por um meio privado.

**Passo 1 — Criar o teu ficheiro de configuração**

Na raiz do projeto, copia o exemplo e abre o ficheiro local:

```powershell
Copy-Item .env.example .env
notepad .env
```

Se já tens um `.env` preenchido, abre-o diretamente com `notepad .env`, preservando os teus valores. O `.env.example` só contém valores fictícios: clonar ou descompactar o repositório não te dá automaticamente acesso ao Qdrant do grupo.

**Passo 2 — Preencher os dados fornecidos pelo grupo**

No `.env`, substitui o endereço e a chave dos exemplos pelos valores reais e mantém o nome da coleção:

```dotenv
QDRANT_URL=https://exemplo.invalid:6333
QDRANT_API_KEY=chave_ficticia
QDRANT_COLECAO=videira_fitossanidade
```

`QDRANT_URL` é o endereço do servidor, `QDRANT_API_KEY` permite o acesso e `QDRANT_COLECAO` escolhe a coleção dentro desse servidor. Para usar a coleção partilhada, mantém `QDRANT_LOCAL_PATH` comentado ou ausente, pois um caminho local preenchido tem prioridade sobre o servidor remoto.

Guarda o `.env` apenas no teu computador. É ignorado pelo Git e está excluído do ZIP de entrega; partilha-se o `.env.example` com valores fictícios. A aplicação carrega o `.env` da raiz do projeto. Variáveis já definidas na sessão PowerShell têm prioridade sobre esse ficheiro. Reinicia a aplicação depois de mudares a configuração.

**Passo 3 — Confirmar o acesso e experimentar uma pesquisa**

```powershell
.\.venv\Scripts\python.exe -m src.indice info
.\.venv\Scripts\python.exe -m src.indice pesquisar "Por onde pode entrar na videira o fungo associado à eutipiose?"
```

O primeiro comando deve mostrar a coleção, o número de pontos, os documentos e eventuais documentos em falta. O segundo deve devolver passagens com documento e página. Ambos só leem a coleção. Se a ligação falhar, confirma os valores do `.env`, a ligação à Internet e a disponibilidade do servidor com o grupo. Depois de confirmares o acesso, arranca o Gradio conforme a secção 5.

| Modo | Precisa de acesso ao Qdrant? |
|---|---|
| LLM + CNN | Não |
| VLM | Não |
| LLM + CNN + RAG | Sim |
| VLM + RAG | Sim |

Os modos sem RAG funcionam sem credenciais Qdrant. Sem fotografia, a CNN não é chamada. Se quiseres usar um índice próprio no teu computador, segue a alternativa local na secção 4.3.

O código conserva a pesquisa do colega: `BAAI/bge-m3` por SentenceTransformers, vetores densos normalizados com 1024 dimensões, `Qdrant/bm25` em português, fusão RRF de 30 candidatos, `BAAI/bge-reranker-v2-m3` e cinco passagens finais. Estes modelos são descarregados do Hugging Face na primeira consulta e executados localmente; não usam o `bge-m3` do Ollama. A primeira execução exige Internet e pode demorar e ocupar vários GB.

A coleção deve ter os vetores nomeados `vetor_denso` (1024, Cosine) e `bm25` (IDF), e os payloads documentais do colega. `info` verifica o esquema, páginas, campos e documentos reservados; mostra também documentos em falta. Isso não prova quais os pesos/revisões usados originalmente: confirma essa proveniência com o colega. Na verificação de 16/11, a coleção tinha 989 passagens e 14 documentos; faltavam D07 e D14 face aos 16 previstos. Não foi reconstruída nem alterada. Consulta [INTEGRACAO.md](INTEGRACAO.md).

### 4.3. Construir uma coleção local nova

Só precisas deste passo para reconstruir/testar o índice. A construção reutiliza a extração Markdown por página, limpeza, divisão por cabeçalhos e chunks de 1200 carateres com sobreposição de 200 do colega. Usa os PDF importados em `data/docs/`, confirmados por SHA-256, e exclui D10–D12. Não há escrita no serviço partilhado neste comando.

```powershell
.\.venv\Scripts\python.exe -m src.ingestao --local .\data\qdrant_local --colecao videira_teste --ocr auto
```

O comando recusa reutilizar uma coleção já existente e não apaga dados. Para outra construção, escolhe um caminho/coleção novos. Guarda a configuração da ingestão junto do índice local. Fecha a aplicação antes de abrir o mesmo armazenamento local noutro processo.

Para consultar essa coleção, acrescenta ao `.env`:

```dotenv
QDRANT_LOCAL_PATH=data/qdrant_local
QDRANT_COLECAO=videira_teste
```

Depois executa `src.indice info` e arranca a aplicação. Para voltar à coleção do grupo, remove/comenta `QDRANT_LOCAL_PATH` e repõe `QDRANT_COLECAO=videira_fitossanidade`. Reinicia a aplicação depois de mudares o `.env`.

### 4.4. OCR e Tesseract no Windows

`--ocr auto` mantém a seleção automática do PyMuPDF4LLM/RapidOCR. `--ocr off` usa só o texto extraível dos PDF. Para exigir Tesseract em português, instala o Tesseract e `por.traineddata`, e confirma os caminhos:

```powershell
$tess = 'C:\Program Files\Tesseract-OCR'
Test-Path "$tess\tesseract.exe"
Test-Path "$tess\tessdata\por.traineddata"
$env:Path = "$tess;$env:Path"
$env:TESSDATA_PREFIX = "$tess\tessdata"
tesseract --list-langs
.\.venv\Scripts\python.exe -m src.ingestao --local .\data\qdrant_por --colecao videira_por --ocr por
```

Se guardaste `por.traineddata` em `data/tessdata`, usa `$env:TESSDATA_PREFIX = (Resolve-Path '.\data\tessdata').Path`. Repete a configuração numa nova janela de PowerShell. O modo `por` falha se o OCR português não estiver disponível, sem mudar silenciosamente de motor. Mudar a extração/OCR cria outras passagens e pode mudar os resultados; regista-o antes de comparar ensaios.

## 5. Abrir a interface

**Windows PowerShell**

```powershell
.\.venv\Scripts\python.exe -m app.app
```

**macOS ou Linux**

```bash
python -m app.app
```

Abre o endereço local indicado no terminal. Escreve uma pergunta, anexa uma fotografia JPG ou PNG, ou combina as duas entradas.

| Modo | Quem analisa a fotografia | Quem responde | Consulta ao corpus |
|---|---|---|---|
| LLM + CNN + RAG | CNN | Qwen3 | Sim |
| LLM + CNN | CNN | Qwen3 | Não |
| VLM | Qwen2.5-VL | Qwen2.5-VL | Não |
| VLM + RAG | Qwen2.5-VL | Qwen2.5-VL | Sim |

Sem fotografia, os quatro modos aceitam texto. Até ligares a CNN, escolhe **VLM** ou **VLM + RAG** para analisar fotografias. Para voltar a analisar a imagem anterior, seleciona **Voltar a usar a última fotografia**. **Mudar de modo mantém o histórico. Para comparar condições, escolhe Nova conversa por condição e volta a selecionar o modo pretendido** (Nova conversa repõe LLM + CNN + RAG). O modelo recebe até seis turnos relevantes de histórico. Por baixo do seletor de modo aparecem o nome e a variante do modelo de resposta, o modelo de embeddings quando há RAG e o checkpoint da CNN quando aplicável.

As conversas são guardadas automaticamente após cada resposta em `runs/chats/`.

Nos modos RAG, as citações junto à resposta abrem a página do PDF. O painel lateral mostra o título, a entidade, a data e um excerto curto. A hipótese visual aparece em separado.

## 6. Ligar a CNN do grupo

Se receberes um modelo ONNX, coloca-o em `runs/checkpoints/cnn.onnx` e instala a dependência. No PowerShell, usa `.\.venv\Scripts\python.exe -m pip install onnxruntime==1.32.0`. Em macOS ou Linux, usa `python -m pip install onnxruntime==1.32.0`.

Confirma com o teu colega o pré-processamento e a ordem das classes em `configs/cnn.yaml`. Para outros formatos, usa `src/cnn_colega.py`. O contrato e a configuração estão em [runs/checkpoints/README.md](runs/checkpoints/README.md).

## 7. Avaliação

### 7.1. Avaliar respostas de texto

A comparação principal do F4 é `llm_cnn` contra `llm_cnn_rag`, com as mesmas perguntas, sem fotografia e com histórico vazio em cada item. A CNN não intervém. O CSV de referência fica no registo e não entra no prompt. Para um ensaio curto e para a execução de referência, usa nomes diferentes em `--execucao`.

**Windows PowerShell**

```powershell
.\.venv\Scripts\python.exe -m src.avaliar dados
.\.venv\Scripts\python.exe -m src.avaliar texto --execucao ensaio --condicoes llm_cnn llm_cnn_rag --limite 1 --repeticoes 1 --temperaturas 0
.\.venv\Scripts\python.exe -m src.avaliar texto --execucao texto-validacao --condicoes llm_cnn llm_cnn_rag
.\.venv\Scripts\python.exe -m src.metricas anotar --execucao texto-validacao
.\.venv\Scripts\python.exe -m src.metricas resumir --execucoes texto-validacao
```

**macOS ou Linux**

```bash
python -m src.avaliar dados
python -m src.avaliar texto --execucao ensaio --condicoes llm_cnn llm_cnn_rag --limite 1 --repeticoes 1 --temperaturas 0
python -m src.avaliar texto --execucao texto-validacao --condicoes llm_cnn llm_cnn_rag
python -m src.metricas anotar --execucao texto-validacao
python -m src.metricas resumir --execucoes texto-validacao
```

O mesmo comando com o mesmo nome retoma uma execução interrompida. Depois de concluída a geração, os dois anotadores preenchem as folhas separadamente e registam o consenso. O resumo calcula as métricas humanas a partir dessas folhas. Consulta [PROTOCOLO.md](PROTOCOLO.md) e [paper/fidelidade/README.md](paper/fidelidade/README.md).

A condição adicional `llm_rag_sem_limiar` conserva a recuperação e as citações, mas desliga o corte e a verificação de suporte. A avaliação de recuperação do colega também está disponível, com o import corrigido e o CSV central:

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

`--sem-rag` usa o Qwen3 real e proíbe qualquer acesso ao Qdrant. `--rag` verifica a recuperação de D02 p. 6, a citação final e a recusa de uma pergunta fictícia sem suporte. `--indice-local` exige os PDF importados e testa uma amostra de D02 numa coleção temporária local. Guarda JSON em `runs/verificacao/`; estes são ensaios de funcionamento, não resultados científicos.

### 7.2. Preparar o FieldVitis e avaliar imagens

Para seguir os comandos abaixo, guarda `fieldvitis.zip` na raiz de `DL_P2_TP2-G2`, ao lado de `README.md`. Extrai-o para `data/imagens/`.

**Windows PowerShell**

```powershell
Expand-Archive -LiteralPath ".\fieldvitis.zip" -DestinationPath ".\data\imagens"
```

**macOS ou Linux**

```bash
unzip ./fieldvitis.zip -d ./data/imagens
```

As quatro pastas das classes ficam diretamente dentro de `data/imagens/`. Se a extração criar uma pasta exterior chamada `fieldvitis`, abre-a e move as quatro pastas para `data/imagens/`. Mantém estes nomes, que correspondem aos caminhos do `data/splits.json` incluído.

```text
data/imagens/Black_rot/
data/imagens/Esca/
data/imagens/Healthy/
data/imagens/Leaf_blight/
```

Por exemplo, `Healthy_10.jpg` fica em `data/imagens/Healthy/Healthy_10.jpg`.

Usa as divisões com que o teu colega treinou a CNN. O manifesto incluído é uma referência reproduzível para o FieldVitis v2. Se os pesos do grupo seguirem outro manifesto, indica-o com `--splits`, no formato de `data/splits.json`.

**Windows PowerShell**

```powershell
.\.venv\Scripts\python.exe -m src.avaliar imagem --execucao visao-validacao --modelos cnn vlm --cnn-checkpoint "runs/checkpoints/cnn_{seed}.onnx"
.\.venv\Scripts\python.exe -m src.metricas resumir --execucoes texto-validacao visao-validacao
```

**macOS ou Linux**

```bash
python -m src.avaliar imagem --execucao visao-validacao --modelos cnn vlm --cnn-checkpoint "runs/checkpoints/cnn_{seed}.onnx"
python -m src.metricas resumir --execucoes texto-validacao visao-validacao
```

### 7.3. Comparar respostas VLM com e sem RAG

O comando `imagem` anterior mede **classificação**, não a qualidade da resposta à pergunta. Para comparar respostas, fornece a mesma fotografia e a mesma pergunta aos dois modos:

```powershell
.\.venv\Scripts\python.exe -m src.avaliar respostas-imagem --execucao respostas-foto-validacao --foto ".\data\imagens\Healthy\Healthy_10.jpg" --pergunta "Que sinais se veem nesta folha de videira?"
```

O ficheiro tem de existir. Cada condição começa sem histórico e guarda o hash da mesma imagem, os prompts e a resposta em `runs/`. No VLM, a foto chega à análise visual e à geração final. No VLM + RAG, a foto chega à análise visual; a geração final recebe a hipótese visual e os excertos documentais, sem voltar a receber os bytes da imagem. Este percurso preserva o código do Rui. Avalia os JSON por revisão humana; este ensaio não entra nas métricas automáticas de classificação de `src.metricas`.

## 8. Dados e referências

- **FieldVitis v2** · 555 fotografias, licença CC BY 4.0. A classe podridão-negra tem 39 imagens e o conjunto não foi recolhido especificamente em vinhas portuguesas. [Dataset e citação](https://doi.org/10.5281/zenodo.17307846)
- **Corpus** · `data/corpus.csv` conserva os 19 registos, incluindo as licenças e os URLs. A configuração inclui os 16 candidatos, contando com os condicionais, e reserva D10–D12 para a avaliação da recusa. Os PDF e as imagens são ficheiros locais; o índice pode ser a coleção partilhada ou uma coleção local nova.
- **Perguntas** · `data/perguntas/perguntas.csv` conserva as 40 perguntas fornecidas, com referência, página e indicação de validação humana. As referências bibliográficas estão em [paper/referencias.bib](paper/referencias.bib).
- **Documentação técnica** · [ChatOllama](https://docs.langchain.com/oss/python/integrations/chat/ollama) · [OllamaEmbeddings](https://docs.langchain.com/oss/python/integrations/embeddings/ollama) · [Gradio](https://www.gradio.app/docs/gradio/blocks)

**Apoio técnico experimental.** Confirma produtos e doses na DGAV ou com um técnico. [Consulta SIFITO](https://www.dgav.pt/medicamentos/conteudo/produtos-fitofarmaceuticos/divulgacao/)

O detalhe da comparação de versões, das alterações e das verificações está em [INTEGRACAO.md](INTEGRACAO.md).
