# Modelo CNN

O modelo necessário é a versão de cinco classes de `resnet50_seed1.keras` (CNN v2), disponível no grupo do WhatsApp. Colocar o ficheiro nesta pasta; os pesos antigos de quatro classes não são compatíveis:

```text
runs/checkpoints/resnet50_seed1.keras
```

Se aparecer como `resnet50_seed1.keras.zip`, remover apenas o `.zip` final por renomeação, sem descompactar. O ficheiro é um modelo Keras completo e é ignorado pelo Git.

O caminho, a seed de treino e as classes estão em `configs/cnn.yaml`. A inferência é implementada em `src/cnn_resnet.py`, com pré-processamento e ordem de classes descritos no [protocolo](../../PROTOCOLO.md).

A CNN só é carregada quando há fotografia num modo LLM. O avaliador verifica o hash dos pesos e a correspondência com as partições. Os pesos disponíveis correspondem à seed de treino 1. Reiniciar a aplicação após substituir o modelo.
