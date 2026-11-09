# P2 — Tema F: Assistente de doenças da vinha

> **Estado:** esqueleto inicial de trabalho — ainda não é uma aplicação completa nem reproduz resultados experimentais.  
> **Grupo:** substituir `G2` pelo número real do grupo no nome da pasta/repositório.  
> **Objetivo deste README:** orientar a preparação do ambiente local, a verificação do código inicial e os primeiros passos de desenvolvimento.

## 1. Objetivo do projeto

Construir e avaliar um assistente que recebe uma pergunta e, opcionalmente, uma fotografia de folha de videira. O sistema deve apoiar a identificação provável da doença, recuperar informação de documentos técnicos em português e apresentar respostas com fontes verificáveis; quando não houver evidência suficiente, deve recusar responder.

Explorar duas vias em paralelo:

1. **LLM + ferramentas**, em que o LLM pode chamar uma CNN para interpretar a fotografia e uma ferramenta de pesquisa documental.
2. **VLM + recuperação documental**, em que um modelo visão-linguagem recebe a pergunta e a fotografia e utiliza evidência documental para responder.

Esta indicação deve ser conciliada pelo grupo com a baseline formal do Guia Tema F (CNN pré-treinada com fine-tuning para a imagem e LLM sem RAG para texto). A arquitetura experimental final tem de ficar registada em `PROTOCOLO.md` antes dos testes finais.

## 2. Estrutura do repositório

```text
DL_P2_TP2-G2/
├── README.md
├── AI_LOG.md
├── PROTOCOLO.md
├── requirements.txt
├── .env.example
├── .gitignore
├── configs/
│   ├── baseline_llm_sem_rag.yaml
│   ├── llm_tools.yaml
│   └── vlm_rag.yaml
├── data/
│   ├── README.md
│   ├── corpus.csv
│   ├── splits.json
│   ├── splits.csv              # template do pipeline de imagens; preencher com F1/F2
│   ├── docs/                   # PDFs: fora do Git; mapa_ficheiros_locais.csv no Git
│   ├── images/                 # FieldVitis: fora do Git
│   ├── index/                  # índice gerado: fora do Git
│   └── perguntas/
│       ├── perguntas.csv
│       ├── README.md
│       └── holdout_audit.md
├── src/
│   ├── __init__.py
│   ├── imagem.py               # interface para a CNN do F2
│   ├── vlm.py                  # interface para o VLM
│   ├── corpus.py               # extração PDF por página e chunking
│   ├── indice.py               # embeddings + FAISS
│   ├── rag.py                  # recuperação e formatação de passagens/fontes
│   ├── llm_tools.py            # agente LangChain + CNN/retriever do grupo
│   ├── llm_baseline.py         # execução do LLM sem RAG/ferramentas
│   ├── tools_teste.py          # ferramentas simuladas para testar tool calling
│   ├── prompts.py              # prompts comuns e prompt de visão
│   └── avaliacao.py            # registo de execuções
├── scripts/                    # validação CSV/PDF, baseline, agente demo, VLM e índice
├── docs/                       # contratos de componentes e runbook PL09
├── notebooks/                  # exploração; não guardar aqui os resultados finais
├── runs/                       # JSON/JSONL leves de experiências; checkpoints volumosos ignorados
├── scripts/                    # execução e validação para a PL09
├── docs/                       # contratos de componentes e runbook
├── app/
│   └── app.py                  # estrutura Gradio ainda sem modelos ligados
├── tests/
│   ├── test_corpus.py
│   ├── test_questions.py
│   └── test_vlm_and_prompts.py
└── paper/
    ├── resultados.csv
    ├── fidelidade/
    │   └── folha_verificacao.csv
    ├── leituras/
    ├── figuras/
    └── tabelas/
```

A estrutura preserva os elementos da secção **2.4 do Guia Tema F**. `src/llm_tools.py` foi acrescentado para explicitar a responsabilidade de integração atribuída pelo professor.

## 3. Preparar Python e o ambiente virtual

O Python 3.14 pode ser experimentado primeiro. Como algumas bibliotecas científicas dependem da disponibilidade de versões pré-compiladas para cada combinação de Python e sistema operativo, se a instalação de dependências falhar, usa Python 3.11 ou 3.12 num ambiente virtual novo.

### 3.1 Windows (PowerShell)

Confirma a versão instalada:

```powershell
py -3.14 --version
```

Deverás ver algo semelhante a `Python 3.14.x`. Se o comando não existir, experimenta `python --version`. Se não tiveres Python instalado, instala-o a partir de [python.org](https://www.python.org/downloads/).

Cria o ambiente virtual na pasta principal do projeto:

```powershell
py -3.14 -m venv .venv
```

Ativa-o:

```powershell
.\.venv\Scripts\Activate.ps1
```

Se o PowerShell bloquear a ativação por política de execução, autoriza scripts apenas para esta sessão do terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Depois volta a executar o comando de ativação. Quando estiver ativo, o início da linha do terminal deverá mostrar `(.venv)`.

**Se o Python 3.14 causar incompatibilidades ao instalar os requisitos:**

1. Fecha o ambiente virtual, se estiver ativo, executando `deactivate`.
2. Remove a pasta `.venv` do projeto (confirma primeiro que estás na pasta correta):

```powershell
Remove-Item -Recurse -Force .venv
```

3. Instala Python 3.11 ou 3.12 a partir de [python.org](https://www.python.org/downloads/).
4. Cria e ativa novamente o ambiente, por exemplo com Python 3.11:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Se `py -3.11` não for reconhecido, confirma que essa versão está instalada. Não alteres versões de bibliotecas ao acaso para contornar erros.

### 3.2 macOS / Linux

Confirma a versão:

```bash
python3 --version
```

Cria e ativa o ambiente virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Se a versão instalada não for compatível com os requisitos, instala Python 3.11 ou 3.12 e substitui `python3` pelo comando correspondente à versão instalada.

## 4. Instalar dependências

Confirma que o ambiente virtual está ativo antes de instalar as bibliotecas. Atualiza o instalador de pacotes:

```bash
python -m pip install --upgrade pip
```

Instala as dependências listadas no repositório:

```bash
python -m pip install -r requirements.txt
```

O processo pode demorar e descarregar vários pacotes. Os pacotes Python listados são gratuitos/open-source; os pesos dos modelos são descarregados separadamente e a execução depende dos recursos do computador ou do ambiente de computação.

Se a instalação falhar:

1. Lê as últimas linhas do erro e identifica qual pacote falhou.
2. Confirma que o ambiente `.venv` está ativo e que o VS Code utiliza esse mesmo interpretador.
3. Se o erro estiver relacionado com a disponibilidade de uma versão pré-compilada (por exemplo, uma biblioteca científica no Python 3.14), considera recriar `.venv` com Python 3.11 ou 3.12.
4. Não instales pacotes extra nem alteres `requirements.txt` sem registar e compreender a razão.

## 5. Verificar o esqueleto inicial

Executa os comandos seguintes a partir da pasta principal do projeto e com o ambiente virtual ativo.

### 5.1 Verificar a sintaxe dos ficheiros Python

```powershell
python -m compileall -q src app tests
```

Se o comando terminar sem mensagens de erro, a sintaxe dos ficheiros selecionados foi compilada com sucesso. Isto **não** demonstra que os modelos ou a aplicação estejam funcionalmente integrados.

### 5.2 Executar os testes iniciais

```powershell
python -m pytest -q
```

Os testes iniciais verificam aspetos como a divisão do texto em fragmentos (*chunks*) e a preservação de metadados das fontes. Se aparecer um erro, guarda a mensagem completa, em especial as últimas linhas, antes de alterar o código.

## 6. Abrir a interface Gradio

O Gradio permite criar uma interface web com código Python. O ficheiro `app/app.py` contém uma interface inicial, ainda sem o backend integrado.

Executa:

```powershell
python app/app.py
```

O terminal deverá mostrar um endereço local, geralmente semelhante a `http://127.0.0.1:7860`. Abre esse endereço num navegador.

Nesta fase, a interface apresenta campos para pergunta e imagem e permite selecionar uma das abordagens experimentais. **É esperado que ainda indique que o backend não está ligado.** Não apresentes a interface como protótipo funcional até ligares e testares a CNN, a pesquisa documental e o LLM/VLM.

Para parar a aplicação, volta ao terminal e prime `Ctrl+C`.

## 7. Configurar um LLM local com Ollama (opcional, recomendado para evitar APIs pagas)

A via LLM + ferramentas está preparada para utilizar um modelo servido localmente pelo [Ollama](https://ollama.com/download). A biblioteca Python LangChain e o Ollama não exigem uma API comercial para esta configuração local, mas a execução continua a depender do hardware disponível.

### 7.1 Instalar e verificar Ollama

1. Instala o Ollama através de [ollama.com/download](https://ollama.com/download).
2. Abre um novo terminal e verifica a instalação:

```powershell
ollama --version
```

3. Escolhe um modelo depois de consultar a licença, o tamanho, a capacidade de responder em português e o suporte a **tool calling** (chamadas de ferramentas). Não assumas que qualquer modelo consegue usar ferramentas de forma fiável.
4. Descarrega o modelo escolhido, substituindo `<nome-do-modelo>` pelo identificador real:

```powershell
ollama pull <nome-do-modelo>
```

5. Confirma que aparece na lista local:

```powershell
ollama list
```

O nome tem de corresponder ao identificador apresentado pelo Ollama.

### 7.2 Criar e preencher o `.env`

Cria o ficheiro local a partir do exemplo:

```powershell
Copy-Item .env.example .env
```

Abre `.env` no VS Code e substitui o valor provisório de `OLLAMA_MODEL` pelo nome exato do modelo descarregado. Confirma também o endereço local:

```text
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=nome-exato-do-modelo
```

Os outros valores de `.env.example` são configurações provisórias para os embeddings, o VLM, o índice e os PDFs; deverão ser confirmados quando o grupo escolher os modelos e as localizações.

**Não envies `.env` para o Git nem coloques segredos nesse ficheiro.** O `.gitignore` já exclui `.env`. Os JSON/JSONL leves em `runs/` são versionáveis para permitir reproduzir e auditar as experiências; checkpoints pesados e índices FAISS ficam fora do Git.

### 7.3 Testar apenas a geração de texto

Com o Ollama disponível, o modelo descarregado e `.env` preenchido, executa na pasta principal do projeto:

```powershell
python -c "from dotenv import load_dotenv; load_dotenv(); from src.llm_tools import criar_llm_local; llm = criar_llm_local(); print(llm.invoke('Responde apenas com a palavra OK.').content)"
```

Este teste verifica apenas se o LLM local responde a uma mensagem. **Não** verifica chamadas às ferramentas, recuperação documental, citações ou recusa. Estes comportamentos exigem testes adicionais antes de executar as experiências.

## 8. Orientação sobre os ficheiros

| Ficheiro/pasta | Para que serve |
|---|---|
| `README.md` | Instalação, execução, estrutura e estado do projeto. |
| `AI_LOG.md` | Registo obrigatório da utilização de IA por todos os elementos: pedido, o que foi aproveitado e como foi verificado. |
| `PROTOCOLO.md` | Protocolo experimental. As decisões ainda em aberto devem ser confirmadas com o grupo e fechadas antes de avaliar o conjunto de teste. |
| `configs/` | Configurações YAML por experiência: modelos, prompts, parâmetros e sementes. Os valores atuais são provisórios. |
| `src/llm_tools.py` | Estrutura LangChain para ligar um LLM a ferramentas. |
| `src/imagem.py` | Interface que deverá ser ligada à CNN do F2. |
| `src/vlm.py` | Ponto de entrada para escolher e executar o VLM. |
| `src/corpus.py` | Extração de texto dos PDFs por página e divisão em fragmentos. |
| `src/indice.py` | Embeddings e índice FAISS. |
| `src/rag.py` | Recuperação, preparação do contexto e formatação de fontes. |
| `src/avaliacao.py` | Estrutura inicial para registar execuções. |
| `data/perguntas/perguntas.csv` | Perguntas de teste, resposta de referência e localização da evidência. |
| `paper/fidelidade/folha_verificacao.csv` | Registo da avaliação humana das afirmações das respostas. |
| `runs/` | Outputs experimentais leves (JSON/JSONL) por pergunta/repetição; os resultados finais a reportar também devem ser organizados em `paper/`. Checkpoints grandes ficam fora do Git. |

## 9. Dados e documentação

- Colocar os PDFs localmente em `data/docs/` e as imagens FieldVitis em `data/images/`; ambos ficam ignorados pelo Git.
- Manter `data/corpus.csv` no Git com URL, entidade, data, licença e SHA-256 de cada PDF.
- **Confirmar URL, data e licença na fonte original.** “Disponível publicamente” não significa automaticamente que um documento tem licença aberta.
- Não incluir PDFs de terceiros no repositório. O Guia Tema F indica que os documentos são indexados e citados, não redistribuídos.
- Guardar as perguntas de avaliação em `data/perguntas/perguntas.csv`, com resposta de referência e passagem/página que a suporta.
- Guardar os outputs experimentais em `runs/`, não apenas em notebooks.

## 10. Próximos passos recomendados

1. Confirmar com o grupo a arquitetura experimental decorrente das indicações mais recentes do professor.
2. Preencher e validar `data/corpus.csv` com URLs, licenças, datas e hashes reais.
3. Alinhar com F2 a interface da CNN e com F3 a interface do retriever.
4. Escolher/testar o LLM local que suporte chamadas a ferramentas; não assumir que qualquer modelo as suporta bem.
5. Implementar primeiro um caminho mínimo e controlado antes de adotar um agente com decisões autónomas.
6. Escolher e testar o VLM de forma independente, antes de o integrar no fluxo RAG.
7. Atualizar `PROTOCOLO.md` e os YAMLs antes de executar experiências finais.
8. Registar a utilização de IA em `AI_LOG.md` e verificar cada bloco de código antes de o usar.

## Fontes técnicas

- Guia Tema F, secção 2.4 (estrutura do repositório), secção 3 (papéis) e secções 4–6 (métodos e avaliação).
- Documentação oficial LangChain: https://docs.langchain.com/oss/python/langchain/agents e https://docs.langchain.com/oss/python/langchain/tools
- Documentação oficial ChatOllama: https://docs.langchain.com/oss/python/integrations/chat/ollama
- Visual Studio Code — ambientes Python: https://code.visualstudio.com/docs/python/environments
- Python — transferências: https://www.python.org/downloads/
- Ollama — transferências: https://ollama.com/download
- Git: https://git-scm.com/


## Estado dos objetivos para a PL09

| Objetivo | O que fica preparado nesta versão | O que ainda falta demonstrar no teu computador/grupo |
|---|---|---|
| 40 perguntas de referência | CSV com 40 propostas, respostas, páginas e split 20/20 | Revisão humana das respostas e auditoria de exclusividade dos 10 holdouts |
| LLM sem RAG | Código para smoke test e execução das 20 perguntas de validação × 3 repetições | Configurar/iniciar Ollama, descarregar o modelo e executar os scripts |
| LLM + ferramentas | Agente LangChain real com interfaces para CNN/RAG e agente de demonstração com duas ferramentas simuladas | Executar o agente com um modelo que faça tool calling; depois substituir as tools demo pelas funções reais F2/F3 |
| VLM | Interface Qwen2.5-VL e script zero-shot com registo JSON | Instalar PyTorch/Transformers e executar com uma imagem real FieldVitis, idealmente em GPU T4/Colab |
| Comunicação entre componentes | Documento com contratos de entrada/saída CNN, retriever, VLM e resposta | Validar os contratos com F2, F3 e F1 |
| Protocolo fechado e commitado | `PROTOCOLO.md` v0.9, com decisões em aberto e salvaguarda para não correr o teste | Aprovação do grupo/docente, resolver pendências, fechar v1.0 e fazer commit no repositório real |

Esta distinção é intencional: os artefactos e os scripts ficam preparados, mas nenhuma inferência com Ollama/VLM nem qualquer commit remoto é declarado como realizado nesta sessão.

## 11. Preparação para a PL09 — tarefas F4 e integração técnica

Esta versão acrescenta uma implementação inicial dos primeiros objetivos definidos para a PL09. Distingue sempre **código preparado** de **experiências efetivamente executadas**: a inferência com Ollama e VLM só fica demonstrada depois de correr os comandos no teu ambiente e guardar os respetivos JSON/JSONL.

### 11.1. Validar o conjunto de 40 perguntas

O ficheiro `data/perguntas/perguntas.csv` contém 40 perguntas propostas, com resposta de referência, documento, página PDF, tipo, doença, ação esperada e split. A distribuição é:

- 30 perguntas respondíveis por documentos que são candidatos ao índice;
- 10 perguntas cuja resposta de referência está nos holdouts D10–D12 e cuja ação esperada no sistema RAG é recusar;
- 20 perguntas de validação (15 respondíveis + 5 holdout) e 20 de teste (15 respondíveis + 5 holdout).

Executa:

```powershell
.\.venv\Scripts\python.exe scripts/validar_perguntas.py
.\.venv\Scripts\python.exe scripts/verificar_referencias_perguntas.py --docs data/docs
```

O segundo comando confirma que os PDFs e as páginas referenciadas existem e contêm texto extraível. **Nenhum dos dois comandos valida cientificamente, por si só, a resposta de referência nem prova exclusividade dos holdouts.** A equipa deve ler as passagens e aprovar o conteúdo. A auditoria preliminar está em `data/perguntas/holdout_audit.md`.

A Circular n.º 11/2026 (D09) contém, nas páginas PDF 11–12, uma tabela de tratamentos para Arouca e Barcelos e as categorias de 1, 2 e 3 tratamentos. Por isso, as perguntas holdout preliminares sobre estes detalhes foram substituídas por perguntas sobre inclusão total/parcial e freguesias no Anexo I de D10.

**Lacuna de cobertura:** o conjunto atual não inclui perguntas respondíveis específicas de `leaf_blight`/mancha-das-folhas. O D07 fornecido é um excerto incompleto e não contém a secção pretendida; o D08 descreve a mancha-das-folhas, mas a licença CC BY-NC-ND 4.0 carece de confirmação do docente antes da indexação. Não inventar perguntas para preencher a classe enquanto não houver uma fonte utilizável e aprovada.

### 11.2. Confirmar o corpus e as licenças

Coloca localmente os PDFs em `data/docs/`, seguindo `data/docs/mapa_ficheiros_locais.csv`. Depois executa:

```powershell
.\.venv\Scripts\python.exe scripts/verificar_corpus.py
```

A verificação com os PDFs que acompanham esta entrega confirmou os 12 SHA-256. D07 e D08 ficam marcados como `condicional`: D07 é um excerto incompleto e não contém a secção pretendida sobre leaf blight; D08 indica CC BY-NC-ND 4.0 e requer confirmação do docente antes da indexação. D10–D12 são holdouts e nunca entram no índice.

O ficheiro local que se chama `Avisos_Agricolas-Circular_05-2026(1).pdf` contém a Circular n.º 08/2026, de 21/05/2026 (D06). As citações devem usar a identificação interna correta D06 / Circular 08/2026.

### 11.3. Testar o LLM sem RAG (smoke test)

Com Ollama em execução e `OLLAMA_MODEL=qwen3:1.7b` (ou o modelo realmente escolhido) no `.env`:

```powershell
ollama list
.\.venv\Scripts\python.exe scripts/testar_llm_baseline.py
```

Este teste faz uma pergunta única ao LLM sem acesso a ferramentas ou documentos e guarda um JSON em `runs/llm_baseline_sem_rag/`. Só depois de o smoke test funcionar, executa as 20 perguntas de validação três vezes:

```powershell
.\.venv\Scripts\python.exe scripts/run_baseline_sem_rag.py --split validacao --repeticoes 3 --temperatura 0
```

Para uma condição de amostragem, podes executar separadamente `--temperatura 0.7`, desde que esta condição seja aprovada e registada no protocolo. O script guarda o prompt de sistema, modelo, temperatura, pergunta, resposta, latência, repetição e commit em JSONL. Não usa CNN, retriever ou ferramentas.

**Não abrir o split `teste` antes da aprovação e commit do protocolo.** O script bloqueia `--split teste` e `--split todos` sem a opção explícita `--confirmar-protocolo-fechado`. Há uma tensão entre a execução de 40 perguntas × 3 repetições descrita na fase 2 do Guia Tema F e a divisão 20 validação + 20 teste, com teste fechado até ao protocolo commitado. Perguntar ao docente na PL09 e manter a opção conservadora até resposta.

### 11.4. Testar LLM + ferramentas LangChain

```powershell
.\.venv\Scripts\python.exe scripts/testar_agente_demo.py
```

Este teste usa `classificar_folha_demo` e `pesquisar_documentacao_demo`, ferramentas simuladas, para verificar o ciclo LLM → ferramenta → resultado → LLM. Guarda um JSON em `runs/llm_tools_demo/`. Os resultados são fictícios, não constituem evidência agronómica e não podem ser usados nos resultados do projeto. Se o modelo não executar as duas ferramentas, repete ou testa outro modelo com tool calling.

A integração real está em `src/llm_tools.py`: receberá a CNN do F2 como função Python e o retriever do F3 como objeto com `.invoke(consulta)`. Os contratos estão em `docs/CONTRATOS_COMPONENTES.md`.

### 11.5. Testar o VLM zero-shot numa imagem

O VLM inicial é `Qwen/Qwen2.5-VL-3B-Instruct`. Para a inferência, instala `requirements-vlm.txt` e PyTorch apropriado ao hardware (o Colab com GPU T4 é a opção prevista no Guia Tema F). Consulta o seletor oficial de instalação do PyTorch: https://pytorch.org/get-started/locally/.

Coloca uma fotografia real do FieldVitis ou outra imagem autorizada em `data/images/` e executa:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-vlm.txt
.\.venv\Scripts\python.exe -m scripts.testar_vlm --imagem data/images/EXEMPLO.jpg
```

O script guarda um JSON em `runs/vlm_zero_shot/`, com classe, descrição, limitações, modelo, dispositivo, precisão e latência. `confidence` fica nulo porque texto livre não é uma probabilidade calibrada. Não usar o VLM para recomendar tratamentos; a resposta técnica deve ser fundamentada no RAG e no LLM textual, salvo decisão diferente do professor.

### 11.6. Contratos, protocolo e commit

Consulta `docs/CONTRATOS_COMPONENTES.md` para alinhar entradas/saídas da CNN (F2), retriever (F3), VLM e resposta final; e `docs/PL09_RUNBOOK.md` para a sequência prática.

O `PROTOCOLO.md` está em **v0.9 — proposta**, não em versão fechada. Não é correto marcar como fechado antes de a equipa/docente resolver, pelo menos:

- se o VLM só classifica/descreve ou também gera a resposta final;
- se D07/D08 podem entrar no índice e com que condições;
- como compatibilizar 40 perguntas × 3 repetições com o teste reservado;
- parâmetros de geração, recuperação, recusa e condições de custo;
- contrato de saída da CNN e do retriever;
- divisão efetiva das imagens, sem inventar IDs de planta/parcela.

Depois de o grupo aprovar, atualiza `PROTOCOLO.md` para v1.0, regista a data, faz o commit e guarda o hash:

```powershell
git status
git add PROTOCOLO.md data/perguntas/perguntas.csv data/perguntas/holdout_audit.md configs docs AI_LOG.md
git commit -m "Fecha protocolo experimental P2 Tema F v1.0"
git rev-parse HEAD
```

**Não executar a avaliação final no teste antes desse commit.** O commit deve ser feito localmente no repositório do grupo; esta entrega não publica nada no Git remoto.
