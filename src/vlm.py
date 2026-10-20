import torch
import time
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

"""
Este script é instanciado no google Colab devido à grande quantidade de parâmetros que o modelo utiliza, é necessário uma GPU com memória Virtual de 16GB RAM para correr sem problemas
"""

# Inicialização de VLM 
def load_vlm():
    vram_inicial = torch.cuda.memory_allocated() / (1024 ** 3)
    print(f"VRAM antes de carregar o modelo: {vram_inicial:.2f} GB")

    # 2. Carregamento otimizado em meia precisão (fp16)
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"

    print("A carregar modelo para a GPU...")
    tempo_inicio_load = time.time()

    # O device_map="auto" encarrega-se de enviar os pesos para a GPU (cuda)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)

    tempo_fim_load = time.time()

    # 3. Medição após carregamento
    vram_pos_load = torch.cuda.memory_allocated() / (1024 ** 3)
    vram_gasta_modelo = vram_pos_load - vram_inicial

    print(f"Tempo de carregamento: {tempo_fim_load - tempo_inicio_load:.2f} segundos")
    print(f"VRAM ocupada pelo modelo: {vram_gasta_modelo:.2f} GB")
    print(f"VRAM total alocada: {vram_pos_load:.2f} GB")

    return model, processor

def describe_leaf(image_path, model, processor):

    mensagens = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image", 
                    "image": image_path
                },
                {
                    "type": "text", 
                    "text": "Descreve detalhadamente o estado desta folha de videira. Indica se apresenta algum sintoma visível de doença."
                },
            ],
        }
    ]

    text_prompt = processor.apply_chat_template(
    mensagens, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(mensagens)

    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda") # Enviar explicitamente para a GPU

    # 4. Medição de tempo e memória ANTES da inferência
    torch.cuda.synchronize() # Força a GPU a terminar tarefas pendentes antes de medir
    vram_antes_inferencia = torch.cuda.memory_allocated() / (1024 ** 3)
    tempo_inicio = time.time()

    # 5. Geração da resposta (Inferência)
    with torch.no_grad(): # Desativa o cálculo de gradientes (não estamos a treinar, poupa muita VRAM)
        generated_ids = model.generate(**inputs, max_new_tokens=150)

    # 6. Medição de tempo e memória DEPOIS da inferência
    torch.cuda.synchronize()
    tempo_fim = time.time()
    vram_pico = torch.cuda.max_memory_allocated() / (1024 ** 3)

    # O modelo devolve o prompt de entrada junto com a resposta; precisamos de cortar a entrada
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    resposta_texto = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    print(f"--- RESULTADOS DA INFERÊNCIA ---")
    print(f"Resposta do Modelo:\n{resposta_texto[0]}\n")
    print(f"Tempo de inferência: {tempo_fim - tempo_inicio:.2f} segundos")
    print(f"VRAM alocada antes da inferência: {vram_antes_inferencia:.2f} GB")
    print(f"Pico de VRAM durante a inferência: {vram_pico:.2f} GB")
    print(f"Custo extra de memória da inferência: {vram_pico - vram_antes_inferencia:.2f} GB\n")

    # 7. Limpeza e Estabilização da VRAM
    del inputs, generated_ids, generated_ids_trimmed, image_inputs
    torch.cuda.empty_cache()
    print("Cache da GPU limpo para a próxima imagem.")

    return resposta_texto[0]

if __name__ == "__main__":
    # Carrega o modelo apenas uma vez
    modelo_carregado, processador_carregado = load_vlm()
    
    # Testa a imagem
    caminho_teste = "data/fieldvitis/Healthy/Healthy_1.jpg"
    resposta = describe_leaf(caminho_teste, modelo_carregado, processador_carregado)
