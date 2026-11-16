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
└── .env                      Chave de coleção e link de coleção onde está o indice Qdrant (Vou enviar o ficheiro para o whatsapp no grupo)
```

## 3. Preparar o ambiente

Descompacta o projeto e abre um terminal na pasta raiz do projeto. Instala o [Python 3.14](https://www.python.org/downloads/) e o [Ollama](https://ollama.com/download).

**Windows PowerShell**

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

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
ollama pull bge-m3
ollama list
```

## 4. Preparar o corpus e o índice

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

### 4.2. Configurar o Tesseract

O OCR reconhece texto em páginas dos PDF com pouco texto extraível. O projeto usa o executável do Tesseract e o ficheiro de idioma português `por.traineddata`.

**Windows PowerShell**

Instala o Tesseract com o idioma português (`por`). Se já o tens instalado, começa por verificar os caminhos. Ajusta `$tess` se o instalaste noutra pasta.

```powershell
$tess = 'C:\Program Files\Tesseract-OCR'
Test-Path "$tess\tesseract.exe"
Test-Path "$tess\tessdata\por.traineddata"
```

Se ambos devolverem `True`, acrescenta a pasta do executável ao `PATH` e indica a pasta dos idiomas. Usa a mesma janela de PowerShell para configurar o Tesseract e construir o índice.

```powershell
$env:Path = "$tess;$env:Path"
$env:TESSDATA_PREFIX = "$tess\tessdata"
tesseract --list-langs
```

Se guardaste `por.traineddata` em `data/tessdata` na raiz do projeto, usa `$env:TESSDATA_PREFIX = (Resolve-Path '.\data\tessdata').Path` na segunda linha. Confirma que `por` aparece na lista. Se abrires outra janela do PowerShell, repete a configuração nessa sessão.

**macOS ou Linux**

Instala o Tesseract e o idioma português com o gestor de pacotes do teu sistema. Confirma que `tesseract --list-langs` inclui `por`.

### 4.3. Construir o índice com OCR em português

VERSÂO NOVA

Se precisares de criar o índice novamente, apenas precisas correr o comando: python ingestao.py --recriar --indexar

Se não quiseres recriar mas apenas acrescentar novos documentos, apenas usa: python ingestao.py --indexar

Verifica se tens o venv com todos os requirements para poderes correr o ficheiro
-----------------------------------------------------------------------------------------------------------------------------------------
Com o Ollama a correr e o `bge-m3` instalado, usa `--ocr por` para exigir OCR em português. A construção pode demorar vários minutos.

**Windows PowerShell**

```powershell
.\.venv\Scripts\python.exe -m src.indice construir --ocr por
.\.venv\Scripts\python.exe -m src.indice info
```

**macOS ou Linux**

```bash
python -m src.indice construir --ocr por
python -m src.indice info
```
-----------------------------------------------------------------------------------------------------------------------------------------

**Alternativas**

| Opção | Comportamento | Windows PowerShell | macOS ou Linux |
|---|---|---|---|
| `--ocr off` | Usa apenas o texto já presente nos PDF. | `.\.venv\Scripts\python.exe -m src.indice construir --ocr off` | `python -m src.indice construir --ocr off` |
| `--ocr auto` | Usa OCR em português se encontrar o Tesseract com `por`. Caso contrário, usa apenas o texto dos PDF. É o modo aplicado quando omites `--ocr`. | `.\.venv\Scripts\python.exe -m src.indice construir --ocr auto` | `python -m src.indice construir --ocr auto` |

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

Sem fotografia, os quatro modos aceitam texto. Até ligares a CNN, escolhe **VLM** ou **VLM + RAG** para analisar fotografias. Para voltar a analisar a imagem anterior, seleciona **Voltar a usar a última fotografia**. O modelo recebe até seis turnos relevantes de histórico. Por baixo do seletor de modo aparecem o nome e a variante do modelo de resposta, o modelo de embeddings quando há RAG e o checkpoint da CNN quando aplicável.

As conversas são guardadas automaticamente após cada resposta em `runs/chats/`.

Nos modos RAG, as citações junto à resposta abrem a página do PDF. O painel lateral mostra o título, a entidade, a data e um excerto curto. A hipótese visual aparece em separado.

## 6. Ligar a CNN do grupo

Se receberes um modelo ONNX, coloca-o em `runs/checkpoints/cnn.onnx` e instala a dependência. No PowerShell, usa `.\.venv\Scripts\python.exe -m pip install onnxruntime==1.31.1`. Em macOS ou Linux, usa `python -m pip install onnxruntime==1.31.1`.

Confirma com o teu colega o pré-processamento e a ordem das classes em `configs/cnn.yaml`. Para outros formatos, usa `src/cnn_colega.py`. O contrato e a configuração estão em [runs/checkpoints/README.md](runs/checkpoints/README.md).

## 7. Avaliação

### 7.1. Avaliar respostas de texto

Para um ensaio curto de validação e para a execução de texto de referência, usa nomes diferentes em `--execucao`.

**Windows PowerShell**

```powershell
.\.venv\Scripts\python.exe -m src.avaliar dados
.\.venv\Scripts\python.exe -m src.avaliar texto --execucao ensaio --condicoes llm_cnn_rag --limite 1 --repeticoes 1 --temperaturas 0
.\.venv\Scripts\python.exe -m src.avaliar texto --execucao texto-validacao
.\.venv\Scripts\python.exe -m src.metricas anotar --execucao texto-validacao
.\.venv\Scripts\python.exe -m src.metricas resumir --execucoes texto-validacao
```

**macOS ou Linux**

```bash
python -m src.avaliar dados
python -m src.avaliar texto --execucao ensaio --condicoes llm_cnn_rag --limite 1 --repeticoes 1 --temperaturas 0
python -m src.avaliar texto --execucao texto-validacao
python -m src.metricas anotar --execucao texto-validacao
python -m src.metricas resumir --execucoes texto-validacao
```

O mesmo comando com o mesmo nome retoma uma execução interrompida. Depois de concluída a geração, os dois anotadores preenchem as folhas separadamente e registam o consenso. O resumo calcula as métricas humanas a partir dessas folhas. Consulta [PROTOCOLO.md](PROTOCOLO.md) e [paper/fidelidade/README.md](paper/fidelidade/README.md).

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

## 8. Dados e referências

- **FieldVitis v2** · 555 fotografias, licença CC BY 4.0. A classe podridão-negra tem 39 imagens e o conjunto não foi recolhido especificamente em vinhas portuguesas. [Dataset e citação](https://doi.org/10.5281/zenodo.17307846)
- **Corpus** · `data/corpus.csv` conserva os 19 registos, incluindo as licenças e os URLs. A configuração inclui os 16 candidatos, contando com os condicionais, e reserva D10–D12 para a avaliação da recusa. Os PDF, as imagens e o índice são ficheiros locais.
- **Perguntas** · `data/perguntas/perguntas.csv` conserva as 40 perguntas fornecidas, com referência, página e indicação de validação humana. As referências bibliográficas estão em [paper/referencias.bib](paper/referencias.bib).
- **Documentação técnica** · [ChatOllama](https://docs.langchain.com/oss/python/integrations/chat/ollama) · [OllamaEmbeddings](https://docs.langchain.com/oss/python/integrations/embeddings/ollama) · [Gradio](https://www.gradio.app/docs/gradio/blocks)

**Apoio técnico experimental.** Confirma produtos e doses na DGAV ou com um técnico. [Consulta SIFITO](https://www.dgav.pt/medicamentos/conteudo/produtos-fitofarmaceuticos/divulgacao/)
