# Assistente de Doenças da Vinha — Deep Learning Aplicado

**Unidade curricular:** Deep Learning Aplicado
**Ano letivo:** 2026/2027
**Projeto:** P2 — Tema F: LLMs/VLMs e RAG multimodal
**Grupo:** DL_P2_TP2-G2
**Estado:** Em desenvolvimento

## 1. Descrição do projeto

Este projeto tem como objetivo desenvolver um assistente multimodal de apoio à identificação de doenças da vinha, combinando análise de imagens de folhas com recuperação de informação em documentos técnicos (*Retrieval-Augmented Generation*, RAG).

O utilizador poderá fornecer uma fotografia de uma folha de videira e colocar uma pergunta em português. O sistema deverá identificar a doença provável, recuperar informação relevante de documentação técnica e produzir uma resposta fundamentada em fontes citadas.

Quando a documentação disponível não for suficiente para responder, o assistente deverá recusar responder em vez de apresentar recomendações sem suporte documental.

O sistema tem fins académicos e experimentais, não substituindo o aconselhamento de técnicos especializados nem a consulta de documentação fitossanitária oficial e atualizada.

## 2. Objetivos

Os principais objetivos são:

- Comparar uma CNN com fine-tuning e um modelo de visão-linguagem (VLM) na classificação de doenças da vinha.
- Desenvolver um pipeline RAG sobre documentos técnicos públicos em português.
- Produzir respostas com citações que identifiquem o documento e a página de origem.
- Avaliar a fidelidade das respostas às fontes, a correção das respostas e a capacidade de recusa.
- Analisar a consistência das respostas entre repetições e identificar casos de alucinação.
- Disponibilizar uma aplicação de demonstração com envio de imagens, perguntas em texto e apresentação das fontes.

## 3. Equipa e responsabilidades

| Elemento                  | Papel                                         | Responsabilidades principais                                                                                       |
| ------------------------- | --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| Francisco Veiga (al82066) | F1 — Dados, protocolo e integração            | Organização dos dados, corpus documental, protocolo experimental, integração da aplicação e registo dos resultados |
| João Pires (al81351)      | F2 — Classificação da folha                   | CNN com fine-tuning, avaliação por classe, Grad-CAM e comparação com VLM                                           |
| Lucas Silva (al81362)     | F3 — Recuperação e geração com citações       | Chunking, embeddings, índice vetorial, recuperação, geração com citações e política de recusa                      |
| Rui Teixeira (al77411)    | F4 — Perguntas, baseline sem RAG e fidelidade | Conjunto de perguntas de referência, baseline textual sem RAG e coordenação da avaliação da fidelidade             |
| Rui Esteves (al78779)     | F5 — Consistência, recusa e incerteza         | Análise de consistência, avaliação das recusas, calibração da confiança e análise de casos difíceis                |

Todos os elementos participam na implementação, na escrita do relatório e na verificação da fidelidade das respostas, de acordo com o protocolo do projeto.

## 4. Arquitetura prevista

O sistema será desenvolvido através dos seguintes componentes:

1. **Entrada:** fotografia de uma folha de videira e/ou pergunta em texto.
2. **Classificação da imagem:** CNN e VLM para identificação da classe provável da doença.
3. **Recuperação documental:** pesquisa das passagens mais relevantes em documentos técnicos indexados.
4. **Geração da resposta:** modelo de linguagem condicionado pelas passagens recuperadas.
5. **Citações e recusa:** apresentação das fontes utilizadas e recusa quando não existe suporte documental suficiente.
6. **Interface:** aplicação Gradio para interação com o assistente.

### Modelos e tecnologias previstos

- **Classificação:** CNN pré-treinada com fine-tuning.
- **Visão-linguagem:** VLM aberto de pequena dimensão, a selecionar e validar.
- **Embeddings:** modelo multilingue, com opções como BGE-M3 ou multilingual-e5.
- **Recuperação vetorial:** FAISS.
- **Processamento documental:** PyMuPDF e OCR quando necessário.
- **Interface:** Gradio.
- **Ambiente experimental:** Python e Google Colab, conforme a disponibilidade computacional.

A seleção final dos modelos, parâmetros e configurações será documentada durante o desenvolvimento.

## 5. Dados e fontes documentais

### 5.1. Dataset de imagens

O projeto utiliza o FieldVitis, constituído por 555 fotografias de folhas de videira, distribuídas por quatro classes:

| Classe      | Número de imagens |
| ----------- | ----------------: |
| Saudável    |               158 |
| Esca        |               200 |
| Leaf blight |               158 |
| Black rot   |                39 |
| **Total**   |           **555** |

A licença indicada para o dataset é CC BY 4.0.

O conjunto de dados é pequeno e desequilibrado, particularmente na classe black rot. Além disso, as imagens provêm de datasets públicos e não representam necessariamente as condições de uma vinha portuguesa. Estas limitações serão consideradas na análise dos resultados.

### 5.2. Corpus documental

O RAG utilizará documentos técnicos públicos em português, incluindo materiais de entidades como DGADR, ADVID, DRAP-Norte, ISA, CCDR-N e Embrapa.

Os documentos serão identificados num inventário (`data/corpus.csv`), incluindo, sempre que disponível:

- Título e entidade responsável;
- URL de origem;
- Data de publicação e data de consulta;
- Licença ou condições de utilização;
- Hash do ficheiro;
- Informação necessária para identificar páginas e passagens.

Os documentos serão utilizados para indexação e recuperação, respeitando as respetivas condições de utilização. Os PDFs não serão incluídos no repositório Git.

A atualidade e a cobertura documental serão consideradas na interpretação das respostas, especialmente quando estiverem em causa recomendações fitossanitárias.

## 6. Metodologia e avaliação

A avaliação será realizada de acordo com um protocolo experimental definido e versionado antes da avaliação final.

### Classificação de imagens

Serão comparados:

- CNN com fine-tuning;
- VLM em zero-shot;
- VLM com LoRA, caso seja viável e aprovado pelo grupo.

As métricas incluirão Macro-F1 e recall por classe, com análise das limitações associadas ao desequilíbrio dos dados.

### Geração de respostas

Serão comparadas as seguintes abordagens:

- LLM sem recuperação documental (baseline);
- LLM com RAG;
- RAG com política de recusa.

A avaliação incluirá:

- Percentagem de afirmações suportadas pelas fontes citadas;
- Correção face às respostas de referência;
- Recall\@5 da passagem-fonte;
- Taxa de recusa correta e de recusa indevida;
- Consistência entre repetições;
- Latência das respostas.

A fidelidade às fontes e a correção das respostas serão avaliadas separadamente. A verificação da fidelidade será realizada por avaliadores humanos, de acordo com as regras estabelecidas no protocolo.

Os resultados, as configurações e as limitações serão documentados para permitir a reprodução das experiências.

## 7. Estrutura do repositório

```text
DL_P2_TP2-G<n>/
├── README.md
├── AI_LOG.md
├── PROTOCOLO.md
├── requirements.txt
├── configs/
├── data/
│   ├── corpus.csv
│   ├── splits.json
│   └── perguntas/
├── src/
│   ├── imagem.py
│   ├── vlm.py
│   ├── corpus.py
│   ├── indice.py
│   └── rag.py
├── notebooks/
├── runs/
├── app/
│   └── app.py
└── paper/
    ├── resultados.csv
    ├── fidelidade/
    ├── leituras/
    ├── figuras/
    └── tabelas/
```

A estrutura poderá ser ajustada durante o desenvolvimento, mantendo a separação entre dados, código, experiências, aplicação e documentação científica.

Os dados, documentos e checkpoints de modelos não serão adicionados ao Git sem uma verificação prévia das licenças, do tamanho e das necessidades de armazenamento.

## 8. Preparação e execução

O ambiente de execução e as dependências ainda serão definidos e validados pelo grupo.

Quando estiver disponível uma primeira versão funcional, esta secção será atualizada com:

1. Pré-requisitos e versão de Python;
2. Criação e ativação do ambiente virtual;
3. Instalação das dependências;
4. Preparação dos dados e configuração dos modelos;
5. Comandos para executar os módulos e a aplicação Gradio.

Os passos de execução serão testados num ambiente limpo antes da entrega final.

## 9. Reprodutibilidade e utilização responsável

Para garantir a rastreabilidade do trabalho:

- O `PROTOCOLO.md` documentará as hipóteses, as divisões de dados, as métricas e as condições experimentais.
- As seeds, os splits e as configurações serão guardados e versionados.
- As experiências terão configurações identificáveis e resultados guardados.
- As respostas geradas conservarão, sempre que aplicável, os parâmetros, as passagens recuperadas e as citações.
- O `paper/resultados.csv` servirá de registo estruturado dos resultados experimentais.
- A utilização de ferramentas de IA no desenvolvimento e na escrita será registada no `AI_LOG.md`, de acordo com as regras da unidade curricular.
- As afirmações do relatório sobre os resultados serão sustentadas por evidência experimental.

O assistente será apresentado como uma ferramenta experimental de apoio à decisão. Não deverá inventar fontes nem apresentar produtos, doses ou recomendações fitossanitárias como oficialmente validados sem suporte documental adequado.

## 10. Estado atual e próximos marcos

**Estado atual:** organização inicial do projeto e preparação do protocolo experimental.

| Marco                    | Data prevista      | Objetivo                                   |
| ------------------------ | ------------------ | ------------------------------------------ |
| P2-D1                    | 29/10/2026         | Proposta do projeto e apresentação inicial |
| P2-D2                    | 24/11/2026         | Seminário sobre o método e avaliação       |
| P2-D3                    | 10/12/2026         | Demonstração do protótipo                  |
| P2-D4                    | 17/12/2026         | Resultados e discussão                     |
| Relatório e documentação | 04/01/2027         | Entrega do relatório e repositório final   |
| P2-D5                    | 05/01 e 07/01/2027 | Apresentação e defesa individual           |

As datas e os entregáveis serão acompanhados pelo grupo e atualizados caso existam alterações oficiais.

## 11. Referências e documentação

As referências científicas, os artigos utilizados e os documentos técnicos serão organizados ao longo do projeto.

A bibliografia final será mantida em `paper/referencias.bib`, incluindo as referências aos modelos, métodos, dataset e documentos utilizados.

---

*Projeto académico desenvolvido no âmbito da unidade curricular Deep Learning Aplicado, ano letivo 2026/2027.*
