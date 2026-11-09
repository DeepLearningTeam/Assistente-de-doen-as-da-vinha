# AI_LOG.md — registo de utilização de IA

> Atualizar este ficheiro todos os membros. Registar o que foi aproveitado e como foi efetivamente verificado; não declarar como testado algo que ainda não foi executado no ambiente do grupo.

| Data | Ferramenta | Pedido/resumo | O que foi aproveitado | Como foi verificado / pendente |
|---|---|---|---|---|
| 2026-10-24 | ChatGPT | Criar o esqueleto do repositório do P2 Tema F mantendo a estrutura do Guia Tema F e prevendo LangChain, via LLM+tools, VLM, RAG e avaliação. | Estrutura inicial de diretórios; templates de configuração; interfaces de módulos; documentação e lista de tarefas. | O grupo deve rever os requisitos; testar a instalação; validar APIs; testar os componentes com os modelos/dados reais. Nenhum resultado experimental foi gerado nessa fase. |
| 2026-10-27 | OpenAI Codex | Corrigir o script de cálculo SHA-256. | Caminho do PDF atualizado para o ficheiro existente e tornado relativo ao diretório do script. | Executado `python data/docs/calcular_sha256.py` e validada a geração de hash SHA-256. |
| 2026-11-04 | ChatGPT | Desenvolver os primeiros elementos para a PL09 com base no starter, corpus, PDFs, P2-D1 e Guia Tema F. | Rascunho de 40 perguntas/respostas de referência; executor de baseline LLM sem RAG; agente LangChain com ferramentas de demonstração; interface de inferência Qwen2.5-VL; contratos CNN/retriever; auditoria preliminar de holdouts; protocolo v0.9 e runbook. | Verificados: compilação/sintaxe Python; `pytest` com 6 testes aprovados e 1 ignorado (os testes de corpus que exigem `langchain_core`, ausente neste ambiente); distribuição 30/10, 20/20 e 15/5 por split; existência de páginas referenciadas; SHA-256 dos 12 PDFs recebidos. Não executados nesta sessão: Ollama/tool calling, inferência real do VLM, CNN do F2, retriever do F3 e anotação humana das perguntas. Todas as perguntas continuam pendentes de aprovação humana; foi corrigida a contaminação inicial de três holdouts após comparar com D09 e documentada em `holdout_audit.md`. A validação automática confirma esquema/páginas, não a correção semântica. Protocolo ainda não está fechado nem commitado no Git do grupo. |

## Modelo para novas entradas

| Data | Ferramenta | Pedido/resumo | O que foi aproveitado | Como foi verificado / pendente |
|---|---|---|---|---|
| AAAA-MM-DD | ... | ... | ... | ... |
