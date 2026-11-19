# Modelo CNN recebido

Esta pasta inclui `resnet50_seed1.keras`: cópia byte a byte de `resnet50_seed1.keras.zip`, sem descompactar o seu conteúdo. É um arquivo Keras completo (arquitetura, pesos e estado de treino). O SHA-256 é registado em `data/splits.json` e `verificacao/cnn/modelo_real.json`. Não há descarga de outra rede nem conversão ONNX.

O caminho e a seed estão em `configs/cnn.yaml`. A aplicação usa `src/cnn_resnet.py`, adaptado de `inference/predict.py`, arquivado em `referencia_cnn/`. A ordem de classes vem de `data/originais/cnn/splits.json`. Não alterar a ordem nem aplicar softmax ou normalização RGB adicional. Consultar a secção 6 do README.

Se o modelo faltar, copiar o original, a partir da raiz desta entrega:

```powershell
Copy-Item -LiteralPath "..\resnet50_seed1.keras.zip" -Destination ".\runs\checkpoints\resnet50_seed1.keras"
```

A ausência do ficheiro ou de TensorFlow/Keras só bloqueia a CNN quando é pedida uma análise com fotografia nos modos LLM. Texto e VLM não a carregam. Só foram fornecidos pesos da seed de treino 1. O avaliador rejeita outras seeds e checkpoints que não correspondam ao hash recebido; integrar novos pesos exige atualizar explicitamente configuração, proveniência e verificações.
