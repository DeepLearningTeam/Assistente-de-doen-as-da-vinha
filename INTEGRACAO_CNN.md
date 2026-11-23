# Integração CNN

## Substituição por CNN v2 — 23/11/2026

Integrada a versão fornecida em `cnn_v2/`, mantendo o adaptador e os percursos existentes da aplicação.

- **Modelo:** ResNet50 seed 1, com cinco saídas na ordem Black_rot, Esca, Healthy, Leaf_blight, Outro. SHA-256: `385083d14e04fb23a4bfde46737eae4e06d5f4e9ff731ee36ba5779f946c7d8e`.
- **Pré-processamento:** EXIF, RGB, padding preto bilinear para 224 × 224 e `preprocess_input` ResNet50, sem divisão por 255.
- **Rejeição:** `Outro` ou confiança máxima inferior a 0,58, conforme o código e a validação fornecidos. `recognized=False` passa a `Visual.estado="rejeitada"`, sem hipótese de doença, pesquisa RAG ou geração da resposta. A etapa de âmbito continua anterior à CNN.
- **Dados:** 715 registos, com 454/98/99 imagens de treino/validação/teste e 64 de holdout. As 555 imagens originais mantêm as divisões anteriores. Os manifestos e a referência CNN foram substituídos pelos recebidos.
- **Avaliação:** a comparação CNN/VLM continua nas quatro classes de folhas, com 76/77/50 imagens de validação/teste/holdout. Não mede a rejeição de `Outro` e mantém as métricas existentes.
- **Ambiente:** mantidas as dependências da aplicação. O novo modelo, guardado com Keras 3.13.2, carregou no TensorFlow/Keras já instalado.

Passaram os 92 testes automáticos, `pip check`, `compileall` e a validação dos dados. Com os pesos reais, duas entradas sintéticas confirmaram a paridade do pré-processamento, probabilidades e decisão com o código recebido. O ruído foi rejeitado nos dois modos LLM antes de recuperar ou gerar; a etapa de âmbito foi simulada. Uma imagem cinzenta foi aceite como saudável, pelo que a rejeição não garante reconhecer toda a imagem fora do domínio. Estes ensaios não são resultados científicos.

A avaliação em fotografias reais, incluindo a rejeição de `Outro`, não foi executada: faltam as imagens nos caminhos operacionais. Nas 160 entradas adicionais do manifesto, o SHA-256 fica por preencher; conserva-se o MD5 fornecido até executar `src.preparar_cnn` com os ficheiros reais. Não foi feita inferência no teste ou holdout.

Reiniciar a aplicação para utilizar o código e os pesos atualizados. O percurso VLM, os prompts, o RAG e as definições de fidelidade/correção foram preservados.

Os materiais de origem da CNN v2 estão preservados no commit `14c2618a00b3934a112dfce0516646d2764aa7f3`, na antiga pasta `referencia_cnn/`. Essa pasta foi retirada da aplicação. O verificador atual confirma o contrato de entrada, as probabilidades e a regra de rejeição; os testes automáticos verificam o pré-processamento. A paridade com o código recebido, registada acima, pertence à verificação anterior à remoção.

## Integração anterior — 18/11/2026

> Registo histórico: contratos e resultados observados na integração de 18/11/2026. As referências a ficheiros e caminhos descrevem essa versão. Os relatórios antigos de `verificacao/` foram removidos na limpeza do projeto; as versões commitadas continuam acessíveis no histórico Git. Para instalação, configuração e verificações atuais, consultar [README.md](README.md) e [PROTOCOLO.md](PROTOCOLO.md).

### Entrega e proveniência

Pasta de trabalho: `integrado_cnn/`. Base: `integrado_final.zip`, extraída sem modificar o arquivo. Não existia repositório Git na pasta nem no ZIP; por isso não foi criado um ramo sobre um histórico inexistente. Para verificar a exclusão do `.env` usou-se um repositório temporário exterior à entrega e as regras `.gitignore` do projeto, antes da cópia do segredo.

Materiais identificados:

| Ficheiro original | Conteúdo utilizado |
|---|---|
| `integrado_final.zip` | Aplicação atual, quatro modos, integração RAG, testes e documentação |
| `DL_P2_TP2_G2-F1-main.zip` | `inference/predict.py`, README, requirements, dois notebooks, manifestos, holdout e splits |
| `resnet50_seed1.keras.zip` | Arquivo Keras completo: `metadata.json`, `config.json`, `model.weights.h5` |
| `fieldvitis.zip` | 555 imagens nas quatro pastas diretamente na raiz do arquivo |
| `.env` | `QDRANT_URL` e `QDRANT_API_KEY`, carregadas pelo mecanismo normal; valores omitidos |

O modelo foi copiado byte a byte para `runs/checkpoints/resnet50_seed1.keras`. O ZIP da entrega original incluía este ficheiro; os pesos não acompanham o clone atual. Não foram descarregados outros pesos CNN, convertida a arquitetura ou feito novo treino. Os originais permanecem na pasta inicial. `referencia_cnn/` conserva código, documentação e notebooks recebidos; `data/originais/` conserva manifestos e divisões dos dois projetos.

### Contrato confirmado

ResNet50 com 23 595 908 parâmetros, cabeça de quatro classes, augmentation e dropout desativados na inferência. Os metadados do arquivo indicam Keras 3.15.1. A associação à seed 1 vem do nome, README e notebook de treino originais; não se voltou a executar o treino para a demonstrar.

- Entrada `(None, 224, 224, 3)`, float32, NHWC.
- EXIF, RGB, resize bilinear direto 224 × 224, `keras.applications.resnet50.preprocess_input` (BGR e médias ImageNet). Sem divisão por 255 ou normalização PyTorch.
- Saída `(None, 4)` com softmax no próprio modelo. Sem segunda softmax ou renormalização.
- Índices `0 Black_rot → black_rot`, `1 Esca → esca`, `2 Healthy → saudavel`, `3 Leaf_blight → leaf_blight`.
- `Leaf_blight` é apresentado/pesquisado como «Mancha-das-folhas (leaf blight)», conservando o nome do dataset, sem tradução para míldio nem atribuição de agente causal não documentado.
- `compile=False` dispensa o estado do otimizador. Cache por caminho/alteração do ficheiro, com exclusão mútua no primeiro carregamento. Foram observados um miss e cinco hits na verificação real do modelo.
- Quatro classes fechadas: sem limiar inventado, sem rejeição validada de imagens fora do domínio, sem afirmar que softmax é certeza de diagnóstico. O estado `ok` indica contrato técnico válido. A nota de hipótese acompanha a saída.

O pré-processamento foi adaptado diretamente da implementação original para receber a imagem PIL usada pela aplicação. No teste de paridade executou-se também o pré-processamento original e `model.predict` com os pesos recebidos: arrays de entrada exatamente iguais e saídas iguais dentro de `atol=1e-6` nas quatro imagens indicadas em `verificacao/cnn/modelo_real.json`.

### Alterações de código

| Ficheiros | Motivo |
|---|---|
| `src/cnn_resnet.py` | Implementar o adaptador existente, carregar Keras tardiamente, reutilizar o modelo, manter o pré-processamento e mapear classes |
| `configs/cnn.yaml`, `src/imagem.py` | Ativar o backend Python/Keras, seed 1, probabilidades já normalizadas e ausência de limiar; manter o contrato Visual |
| `src/rag.py` | Entregar hipótese e probabilidades ao LLM, acrescentar a classe CNN atual à consulta e impedir reinjeção automática da CNN anterior sem foto |
| `app/app.py` | CNN recebe a nova imagem antes do JPEG usado no histórico; legenda «Valor softmax da CNN» |
| `src/preparar_cnn.py`, `data/splits.json` | Conferir dados, duplicados e sobreposições; adaptar as partições do treino sem nova amostragem |
| `src/avaliar.py`, `src/splits.py` | Validar correspondência entre pesos/divisões, distinguir seeds, permitir holdout explícito e proteger o manifesto contra nova amostragem |
| `requirements.txt` | Todas as dependências, incluindo CNN, compatíveis com o Windows/Python existente |
| `src/verificar_cnn.py`, `tests/test_cnn.py` | Verificações repetíveis do contrato, cache, classes, percursos e proteção da avaliação |
| `README.md`, `PROTOCOLO.md`, `AI_LOG.md`, `runs/checkpoints/README.md`, `INTEGRACAO_CNN.md` | Instruções atuais e ligação ao relatório histórico de RAG |

Os prompts gerais e específicos do RAG, YAML experimentais, `src/indice.py`, `src/vlm.py` e `src/metricas.py` foram comparados com o ZIP base e preservados. As definições de fidelidade/correção da secção 6 do protocolo também foram preservadas. A pesquisa, limiares documentais, limites de geração, fotografia na geração VLM e condições textuais não foram redesenhados. O `.env.example` existente é compatível e continua apenas com valores fictícios.

Nos dois modos LLM uma nova foto chama a CNN; sem foto não a chama. A classe atual acrescenta contexto à consulta RAG mesmo que a etapa de âmbito não assinale `usa_imagem`. O gerador recebe a hipótese, o valor softmax e a distribuição. Esses números não alteram scores de pesquisa, reranking ou corte documental. O histórico textual continua a existir, mas a CNN anterior não é reinjetada como análise visual corrente. A opção explícita de voltar a usar a fotografia mantém-se; nesse caso é analisada a versão JPEG conservada no histórico. Nos modos VLM mantém-se o percurso visual anterior.

### Dados e avaliação

555 IDs do manifesto limpo correspondem às imagens fornecidas, incluindo MD5. Confirmados nas seeds originais 0/1/2: 352 treino, 76 validação, 77 teste interno e 50 reservadas, sem interseções de IDs ou de conteúdo exato. Não há duplicados SHA-256 nem duplicados exatos após descodificação RGB/EXIF. Esta auditoria não identifica quase duplicados nem prova independência por videira/local de recolha.

Confirmada a fuga da avaliação antiga: **53/84** imagens do teste da seed 42 pertencem ao treino da CNN seed 1. O manifesto antigo foi removido da pasta de trabalho por não ser utilizado; a versão permanece no histórico Git. O avaliador rejeita essas divisões para estes pesos. O manifesto ativo representa apenas a seed 1 e preserva cada ID. As seeds 0 e 2 ficam arquivadas, mas faltam os respetivos pesos.

Seed de treino/divisão CNN: **1**. Seed de amostragem das 50 reservadas: **2026**. Seed de geração VLM: **42**, conforme configuração existente. As repetições textuais mantêm 42/43/44. Nenhuma destas últimas representa treinos CNN adicionais.

Não foi feita inferência nas 50 reservadas nem nos 77 testes internos nesta integração. A inspeção de ficheiros/hashes do holdout foi apenas uma auditoria de integridade. Os comandos de avaliação oficial estão no README. Os ensaios de funcionamento usam validação e não servem para conclusões sobre Macro-F1 final, fidelidade ou correção das respostas.

### Verificações realizadas

| Verificação | Resultado e evidência |
|---|---|
| Testes existentes e novos | 62 testes passaram, sem falhas/erros/ignorados; `verificacao/cnn/testes.txt` e `testes_resumo.json`. Serviços simulados nos testes de regressão; TensorFlow real nos testes de pré-processamento |
| Sintaxe e dependências | `compileall` e `pip check` passaram; ambiente em `ambiente.txt`, sem venv no ZIP |
| CNN real | Uma imagem de validação de cada classe: entrada correta, probabilidades finitas, soma unitária, classes e paridade original confirmadas; `modelo_real.json` |
| CLI CNN | `cnn-validacao-final`: quatro imagens de validação, quatro saídas válidas; JSON por imagem e manifesto em `runs/`. Amostra inicial ordenada de black_rot, não estimativa de desempenho global |
| CLI CNN/VLM | `comparacao-validacao-cnn-vlm`: os dois modelos reais avaliaram a mesma imagem de validação, com IDs/hash iguais e seed de treino/divisão 1; um par, sem inferência sobre teste ou holdout |
| Métricas | Exportação ensaiada em `verificacao/cnn/metricas-ensaio/`, fora das tabelas académicas. A amostra pequena não permite conclusões científicas |
| Proteção de avaliação | CLI rejeitou manifesto antigo e `--split holdout --limite 1`; aplicação importa sem carregar TensorFlow/Keras; `guardas_cli.json` |
| Ollama | Os dois modelos configurados estavam disponíveis; digests em `servicos.json`; geração real sem substituição de modelos |
| Qdrant | Ligação, esquema, payloads e recuperação reais só por leitura: 989 pontos/14 documentos; D07 e D14 continuam ausentes. Hash de payloads e contagem iguais antes/depois; `qdrant_so_leitura.json` |
| CNN → consulta documental | Esca prevista em `Esca/Esca_0.jpg` acrescentada à consulta. Cinco passagens, incluindo D02 p. 9 e D09 p. 4 sobre esca, misturadas com passagens gerais/outras doenças; não se afirma que todo o top-k é relevante |
| LLM + CNN | Resposta real com foto e resposta textual sem foto; `llm_cnn_foto.json`, `llm_cnn_texto.json` |
| LLM + CNN + RAG | Foto: recuperação e geração executadas, resposta recusada por `evidencia_insuficiente`; texto sobre esca: resposta com citação D02 p. 5. `llm_cnn_rag_*.json` |
| VLM | Respostas reais com/sem foto. Na imagem rotulada esca o VLM sugeriu saudável, erro de classificação que não foi corrigido por prompts; `vlm_*.json` |
| VLM + RAG | Primeira foto falhou no formato JSON. Repetição isolada, sem alterações, completou análise/recuperação/geração/verificação com recusa por suporte insuficiente. Texto também recusado. `percursos_reais.json` e `vlm_rag_foto_repeticao.json` |
| Gradio | Arranque por `-m app.app --porta 7865`; HTTP 200 em `/`, `/config` e `/gradio_api/runs`; upload real por `/enviar_turno`, CNN/Ollama reais, persistência e pergunta seguinte sem foto/sem CNN. `gradio_real.json` |
| Preservação experimental | Comparação com o arquivo base: prompts, VLM, índice, métricas, configurações e definições humanas preservados; `preservacao.json` |

A verificação completa de 76 imagens foi iniciada e interrompida após seis itens, para limitar o ensaio de funcionamento. A pasta exploratória parcial foi preservada na cópia de trabalho, mas não incluída no ZIP. Houve ainda um ensaio inicial de quatro imagens durante a preparação do ambiente; o ZIP usa a execução identificada como final e as verificações do ambiente definitivo. Não se apresenta nenhuma destas corridas como avaliação final.

### Ambiente e limitações

Windows 11 x64, Python **3.14.0**, TensorFlow **2.22.0rc1**, módulo Keras **3.16.0.dev2026111703** da distribuição `keras-nightly`. A RC declara dependência de `keras-nightly`; instalar também `keras` cria sobreposição de ficheiros. O ambiente foi corrigido para manter só `keras-nightly` fixado. O modelo guardado com Keras 3.15.1 carregou e manteve paridade numérica. O `tensorflow[and-cuda]` da configuração original foi adaptado à execução CPU em Windows, sem o extra CUDA. Não foi necessário mudar de Python; GPU não foi testada. [Distribuição oficial](https://pypi.org/project/tensorflow/2.22.0rc1/) e [instruções TensorFlow](https://www.tensorflow.org/install/pip).

Os ensaios BGE-M3/reranker usaram pesos já existentes na cache, com `HF_HUB_OFFLINE=1`; os modelos não foram substituídos. Num computador novo, o percurso normal precisa de descarregar os modelos RAG já configurados. As versões diretas restantes foram mantidas. Nas duas pequenas corridas finais do avaliador e no servidor Gradio de teste usaram-se `TF_NUM_INTRAOP_THREADS=2` e `TF_NUM_INTEROP_THREADS=1` apenas no processo; não foram alterados os valores de produção. Os tempos dependem dessa configuração, do aquecimento e da execução concorrente, e não são benchmarks.

O código transporta e apresenta a hipótese corretamente, mas **a geração não garante prudência nem exatidão**. O LLM sem RAG produziu descrições de sinais que a CNN não fornece e linguagem excessiva sobre diagnóstico; o VLM classificou incorretamente uma folha; o RAG recuperou passagens pertinentes mas também outras e recusou respostas. A nota determinística de hipótese e a descrição do classificador continuam visíveis. Estes comportamentos foram registados, sem reformular os prompts experimentais ou afinar com o holdout. Exigem avaliação humana e eventual trabalho posterior explicitamente separado.

Não se executaram OCR/Tesseract ou reconstrução documental nesta tarefa: os PDF do corpus não foram fornecidos e a coleção partilhada bastou aos ensaios. As instruções Windows foram preservadas com `--ocr por` como principal e `auto`/`off` como alternativas. Confirmar a apresentação no navegador, as respostas e as fontes com revisão humana; obter os pesos das outras seeds se forem necessários três treinos.

### Utilização rápida no PowerShell

Na pasta do projeto, verificar `python --version` (3.14). No Windows desta sessão `python` estava disponível, mas o launcher `py -3.14` não enumerava instalações:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m app.app
```

Os ficheiros `.env` e `resnet50_seed1.keras` estão no grupo do WhatsApp. Preencher a configuração local com o conteúdo do `.env` partilhado e colocar o modelo em `runs/checkpoints/`, conforme o README. Se o modelo aparecer como `.keras.zip`, remover apenas o `.zip` final por renomeação. Colocar `fieldvitis.zip` na raiz do projeto, extrair para `data/imagens/` e executar `src.preparar_cnn`.

```powershell
.\.venv\Scripts\python.exe -m src.avaliar imagem --execucao cnn-ensaio --modelos cnn --seeds 1 --split validacao --limite 4
.\.venv\Scripts\python.exe -m src.avaliar imagem --execucao visao-teste-seed1 --modelos cnn vlm --seeds 1 --split teste
```

O segundo comando é uma avaliação do teste interno, para executar após fechar as condições. O README tem o comando separado das 50 imagens finais. O ZIP da entrega original excluía `.env`, ambientes virtuais, caches, conversas de teste com bytes de imagens e o FieldVitis; incluía código, modelo, divisões, documentação e evidência selecionada de verificação.
