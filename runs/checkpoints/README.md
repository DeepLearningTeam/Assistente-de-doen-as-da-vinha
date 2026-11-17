# Ligar a CNN do grupo

Pede ao teu colega o ficheiro do modelo, a ordem das classes, o pré-processamento, o tipo de saída e o manifesto usado para a divisão das imagens.

## Modelo ONNX

1. Coloca o ficheiro em `runs/checkpoints/cnn.onnx` ou muda `checkpoint` em `configs/cnn.yaml`.
2. Instala `python -m pip install onnxruntime==1.32.0` no ambiente virtual.
3. Ajusta os campos de imagem ao treino. `largura` e `altura` definem a resolução, `escala`, `media` e `desvio` definem a normalização RGB, e `layout` define a ordem dos canais. `direto` redimensiona a fotografia inteira. `centro` recorta ao centro para o rácio do modelo. Para outra transformação, usa o adaptador Python.
4. Em `classes`, escreve a ordem real das saídas. Usa `saudavel`, `esca`, `leaf_blight` e `black_rot`. Se existir uma saída treinada para outras imagens, usa `outra` na posição correspondente.
5. Escolhe `saida: logits` se o modelo devolver valores antes de softmax, ou `saida: probabilidades` se já devolver probabilidades. Se o modelo tiver várias saídas, indica `nome_saida`.

## Pesos com inferência Python

Coloca os pesos nesta pasta, muda `backend` para `python` e configura `checkpoint`. A função `prever(imagem, checkpoint)` em `src/cnn_colega.py` recebe uma imagem PIL RGB e o caminho para os pesos. Podes apontar `adaptador` para a função do colega com o formato `modulo:funcao`.

A função devolve um dicionário com `probabilidades`, com as quatro classes e valores cuja soma é 1. Pode também devolver `e_folha` para uma decisão explícita de rejeição, `descricao` e `mapa_atencao` como imagem PIL. O mapa só aparece quando o adaptador o disponibiliza. Uma CNN fechada nas quatro classes pode atribuir confiança alta a outras imagens. O limiar de confiança é uma regra de abstenção e a deteção de imagens fora da vinha depende de um mecanismo validado pelo grupo.

Para pesos diferentes por seed, mantém a mesma convenção de pré-processamento e usa nomes como `cnn_42.onnx`, `cnn_43.onnx` e `cnn_44.onnx` na avaliação. Mantém os pesos e os dados de imagem no ambiente local.
