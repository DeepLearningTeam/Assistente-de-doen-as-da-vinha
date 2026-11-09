# Runbook F4 — preparação para a PL09 (12/11/2026)

## Objetivo demonstrável

Até à PL09, pretende-se ter: (1) 40 perguntas preparadas; (2) LLM sem RAG executável; (3) agente LangChain a chamar ferramentas de teste; (4) VLM a analisar pelo menos uma fotografia; (5) contratos entre módulos acordados; (6) PROTOCOLO aprovado e commitado antes de abrir o teste.

## A. Preparar a instalação

No PowerShell, na pasta do repositório, usa o ambiente virtual que já criaste:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Se o `.env` já existir e tiver configurações locais, não o substituas. Edita-o manualmente e confirma `OLLAMA_MODEL=qwen3:1.7b`.

Verifica:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts app tests
.\.venv\Scripts\python.exe scripts/validar_perguntas.py
.\.venv\Scripts\python.exe scripts/verificar_referencias_perguntas.py --docs data/docs
```

## B. Confirmar o corpus e as perguntas

Coloca localmente os PDFs em `data/docs/`, segundo `data/docs/mapa_ficheiros_locais.csv`. Não os comites no Git.

```powershell
.\.venv\Scripts\python.exe scripts/verificar_corpus.py
.\.venv\Scripts\python.exe scripts/validar_perguntas.py
.\.venv\Scripts\python.exe scripts/verificar_referencias_perguntas.py --docs data/docs
```

Os holdouts D10–D12 não podem ir para o índice. D07 é um excerto incompleto e não sustenta a classe leaf blight; D08 tem licença CC BY-NC-ND 4.0 e requer confirmação do docente antes da indexação.

## C. LLM sem RAG (baseline F4)

1. Instala/inicia Ollama e descarrega o modelo escolhido pelo grupo (por defeito no projeto: `qwen3:1.7b`).
2. Testa a ligação:

```powershell
ollama list
.\.venv\Scripts\python.exe -c "from dotenv import load_dotenv; load_dotenv(); from src.llm_baseline import criar_llm_local; llm=criar_llm_local(); print(llm.invoke('Responde apenas OK.').content)"
```

3. Faz primeiro o smoke test de uma pergunta: `.\.venv\Scripts\python.exe scripts/testar_llm_baseline.py`. Depois corre apenas as 20 perguntas de validação, três vezes:

```powershell
.\.venv\Scripts\python.exe scripts/run_baseline_sem_rag.py --split validacao --repeticoes 3 --temperatura 0
```

O script guarda JSONL em `runs/llm_baseline_sem_rag/`, com pergunta, resposta, modelo, temperatura, latência, repetição e commit. Não usa retriever, CNN ou ferramentas.

**Não executar `--split teste` antes da aprovação e commit do `PROTOCOLO.md`.** O script exige `--confirmar-protocolo-fechado` como salvaguarda. A PL09 e o Guia Tema F têm uma tensão entre a execução de 40 perguntas × 3 repetições e a divisão 20 validação/20 teste; pedir confirmação ao docente antes de qualquer corrida no teste.

## D. Testar o ciclo LangChain → ferramenta → LLM

As ferramentas deste primeiro teste são simuladas; servem apenas para testar tool calling, não são resultados agronómicos.

```powershell
.\.venv\Scripts\python.exe scripts/testar_agente_demo.py
```

O output deve mostrar mensagens de ferramenta para `classificar_folha_demo` e `pesquisar_documentacao_demo`. Se o modelo não chamar as duas ferramentas, repetir ou experimentar o modelo alternativo, registando a alteração.

## E. Testar o VLM zero-shot numa imagem

Instala as dependências extra no ambiente adequado (preferencialmente Colab com GPU T4, conforme a estimativa inicial do grupo):

```bash
pip install -r requirements-vlm.txt
```

O PyTorch deve ser instalado conforme o sistema operativo e CPU/CUDA; em Colab usa normalmente o PyTorch já fornecido. Coloca uma imagem de folha em `data/images/` (não a publiques no Git se tiver restrições) e executa:

```bash
python scripts/testar_vlm.py --imagem data/images/EXEMPLO.jpg
```

O script guarda automaticamente JSON em `runs/vlm_zero_shot/`, com `classe`, `descricao_visual`, `limitacoes`, modelo, hardware, tempo e imagem/ID. `confidence` fica nulo: não interpretar texto livre como confiança calibrada.

## F. Fechar e versionar o protocolo

Antes de consultar o teste, confirmar com o grupo/docente:

- se o VLM só classifica/descreve ou também gera a resposta final;
- se o índice definitivo terá 7 ou 9 documentos (D07/D08 pendentes);
- como compatibilizar a instrução PL09 de 40 perguntas × 3 repetições com a divisão de 20 validação + 20 teste;
- temperatura/repetições finais e critérios de recusa;
- contrato de saída da CNN e do retriever.

Depois de aprovar, atualizar o estado e a versão no `PROTOCOLO.md`, fazer commit e só então autorizar a corrida final no teste:

```powershell
git status
git add PROTOCOLO.md data/perguntas/perguntas.csv data/perguntas/holdout_audit.md configs docs AI_LOG.md
git commit -m "Fecha protocolo experimental P2 Tema F v1.0"
git rev-parse HEAD
```

Registar o hash do commit no protocolo/AI_LOG. Não usar a amostra de teste para escolher prompts, `top_k`, chunks ou limiar de recusa.
