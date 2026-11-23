# Referência do componente CNN

Esta pasta conserva os materiais de origem necessários à rastreabilidade da ResNet50 integrada:

- `inference/predict.py`: implementação usada na comparação de paridade do pré-processamento.
- `notebooks/preparacao_dados_fieldvitis.ipynb`: preparação das imagens e divisões.
- `notebooks/treino_cnn_4modelos.ipynb`: treino e seleção dos modelos.
- `requirements.txt`: dependências do ambiente de origem, conservadas como referência.

A aplicação executa `src/cnn_resnet.py`. A instalação da versão integrada utiliza exclusivamente o `requirements.txt` da raiz do projeto.

Os manifestos e as divisões de origem estão em `data/originais/cnn/`. O manifesto utilizado na avaliação é `data/splits.json`. O modelo esperado é `runs/checkpoints/resnet50_seed1.keras`.

O comando de paridade é descrito no [README principal](../README.md#52-preparar-o-fieldvitis-e-avaliar-imagens). Os resultados existentes nos notebooks pertencem às respetivas execuções de treino e não representam novas avaliações da aplicação.


