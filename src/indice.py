# ---------------------------------------------------------------------------
# indice.py — Extração de texto de PDFs, chunking recursivo e estrutura
#              de dados auxiliar para mapeamento FAISS -> metadados.
#
# O índice da lista (0, 1, 2...) corresponde ao ID interno do FAISS,
# permitindo recuperar os metadados diretamente após uma pesquisa.
# ---------------------------------------------------------------------------

import os
import pickle
import pymupdf as pydf

# ── Constantes de chunking ─────────────────────────────────────────────────
TOKENS_ALVO = 300          # Tamanho-alvo de cada chunk em tokens
TOKENS_SOBREPOSICAO = 50   # Sobreposição entre chunks consecutivos
CHARS_POR_TOKEN = 4        # Estimativa: 1 token ≈ 4 caracteres

CHARS_ALVO = TOKENS_ALVO * CHARS_POR_TOKEN              # 1200 caracteres
CHARS_SOBREPOSICAO = TOKENS_SOBREPOSICAO * CHARS_POR_TOKEN  # 200 caracteres

RECALL = 5

# ── Estrutura de dados auxiliar ────────────────────────────────────────────
lista_chunks: list[dict] = []


# ── Funções de acesso à lista ──────────────────────────────────────────────
def adicionar_chunk(texto: str, nome_documento: str, pagina: int) -> int:
    """Adiciona um chunk à lista e devolve o ID (índice) atribuído.

    Args:
        texto: Conteúdo textual do chunk.
        nome_documento: Nome do ficheiro/documento de origem.
        pagina: Número da página de onde o chunk foi extraído.

    Returns:
        O índice do chunk na lista (== ID no FAISS).
    """
    chunk = {
        "texto": texto,
        "nome_documento": nome_documento,
        "pagina": pagina,
    }
    lista_chunks.append(chunk)
    return len(lista_chunks) - 1


def obter_chunk(indice: int) -> dict:
    """Devolve o dicionário de metadados do chunk dado o seu índice/ID FAISS."""
    return lista_chunks[indice]

import re

# ── Divisão recursiva de texto ─────────────────────────────────────────────
def _dividir_recursivo(texto: str, max_chars: int) -> list[str]:
    """Divide *texto* em fragmentos ≤ *max_chars* usando uma estratégia
    recursiva: primeiro tenta separar por parágrafos (``\\n\\n``), depois por
    pontos finais (``'. '`` / ``'. '`` após ``!`` ou ``?``), e em último
    recurso corta por caracteres.
    """
    texto = texto.strip()
    if not texto:
        return []
    if len(texto) <= max_chars:
        return [texto]

    texto_limpo = re.sub(r'\s+', ' ', texto).strip()

    # 1.ª tentativa — separar por parágrafos
    paragrafos = texto_limpo.split("\n\n")
    if len(paragrafos) > 1:
        return _juntar_segmentos(paragrafos, max_chars)

    # 2.ª tentativa — separar por frases (ponto final seguido de espaço)
    frases = _separar_por_frases(texto_limpo)
    if len(frases) > 1:
        return _juntar_segmentos(frases, max_chars)

    # Último recurso — corte duro por caracteres
    fragmentos = []
    for i in range(0, len(texto_limpo), max_chars):
        fragmentos.append(texto_limpo[i:i + max_chars])
    return fragmentos

import pymupdf4llm
from tqdm import tqdm
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

def processar_pdf_markdown(lista_pdfs):

    lista_chunks_formatada = []

    for caminho_pdf in lista_pdfs:
        nome_ficheiro = os.path.basename(caminho_pdf)
        try:
        # O pymupdf4llm é chamado aqui
            paginas_md = pymupdf4llm.to_markdown(caminho_pdf, page_chunks=True, write_images=False)
        except Exception as e:
            print(f"Falha ao ler {nome_ficheiro}: {e}")
            continue

        # 2. Configura o divisor para agrupar o texto sob os seus respetivos títulos
        headers_to_split_on = [
            ("#", "Secao"),
            ("##", "Subsecao"),
            ("###", "Topico"),
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)

        for pagina in tqdm(paginas_md, desc=f"A processar páginas de {nome_ficheiro}"):
            # Extraímos apenas a string (texto markdown) para alimentar o primeiro divisor
            texto_da_pagina = pagina["text"] 
            num_pagina = pagina["metadata"].get("page", 0) + 1 # O pymupdf começa no 0
            
            # Corte 1: Isola os cabeçalhos (recebe 'str', devolve 'List[Document]')
            chunks_logicos = markdown_splitter.split_text(texto_da_pagina)
            
            # Corte 2: Garante o limite de 300 tokens (recebe 'List[Document]', devolve 'List[Document]')
            chunks_finais = text_splitter.split_documents(chunks_logicos)
            
            # 3. Conversão para o TEU formato original (para não estragar a Fase 2)
            for doc in chunks_finais:
                # Recupera a hierarquia visual (ex: "Secao > Subsecao")
                cabecalhos = " > ".join([v for k, v in doc.metadata.items()])
                
                # Injeta o título nativamente dentro do texto para o FAISS
                texto_enriquecido = f"[{cabecalhos}]\n{doc.page_content}" if cabecalhos else doc.page_content
                
                # Recria o dicionário exato que a tua variável 'lista_chunks' exige
                lista_chunks_formatada.append({
                    "texto": texto_enriquecido,
                    "nome_documento": nome_ficheiro,
                    "pagina": num_pagina
                })

    return lista_chunks_formatada

def _separar_por_frases(texto: str) -> list[str]:
    """Separa texto em frases usando delimitadores '. ', '! ', '? '.

    Devolve cada frase **com** o delimitador incluído no final.
    """
    frases: list[str] = []
    actual = ""
    i = 0
    while i < len(texto):
        actual += texto[i]
        # Verifica se o caractere é um fim de frase seguido de espaço
        if texto[i] in ".!?" and i + 1 < len(texto) and texto[i + 1] == " ":
            frases.append(actual)
            actual = ""
        i += 1
    if actual:  # resto final
        frases.append(actual)
    return frases

def _juntar_segmentos(segmentos: list[str], max_chars: int) -> list[str]:
    """Agrupa *segmentos* pequenos num único bloco até atingir *max_chars*.

    Se um segmento individual ultrapassar *max_chars*, é dividido
    recursivamente com ``_dividir_recursivo``.
    """
    blocos: list[str] = []
    bloco_atual = ""

    for seg in segmentos:
        seg = seg.strip()
        if not seg:
            continue

        # Segmento único demasiado grande → dividir recursivamente
        if len(seg) > max_chars:
            if bloco_atual:
                blocos.append(bloco_atual)
                bloco_atual = ""
            blocos.extend(_dividir_recursivo(seg, max_chars))
            continue

        # Tenta juntar ao bloco atual
        candidato = (bloco_atual + "\n\n" + seg).strip() if bloco_atual else seg
        if len(candidato) <= max_chars:
            bloco_atual = candidato
        else:
            blocos.append(bloco_atual)
            bloco_atual = seg

    if bloco_atual:
        blocos.append(bloco_atual)

    return blocos

def _aplicar_sobreposicao(blocos: list[str], sobreposicao: int) -> list[str]:
    """Reintroduz *sobreposicao* caracteres do bloco anterior no início de
    cada bloco seguinte, para preservar contexto entre chunks consecutivos.
    """
    if sobreposicao <= 0 or len(blocos) <= 1:
        return blocos

    resultado = [blocos[0]]
    for i in range(1, len(blocos)):
        prefixo = blocos[i - 1][-sobreposicao:]
        resultado.append(prefixo + blocos[i])
    return resultado

# ── Extração + chunking por página ─────────────────────────────────────────
def processar_pdf(path_pdf: str) -> int:
    """Lê um PDF, divide o texto de cada página em chunks recursivos com
    sobreposição e anexa-os a ``lista_chunks``.

    Args:
        path_pdf: Caminho absoluto (ou relativo) para o ficheiro PDF.

    Returns:
        O número total de chunks criados para este documento.
    """
    nome_doc = os.path.basename(path_pdf)
    total_chunks = 0

    with pydf.open(path_pdf) as doc:
        for num_pagina, pagina in enumerate(doc, start=1):
            texto_pagina = pagina.get_text("text")
            if not texto_pagina or not texto_pagina.strip():
                continue

            # Divide recursivamente e aplica sobreposição
            blocos = _dividir_recursivo(texto_pagina, CHARS_ALVO)
            blocos = _aplicar_sobreposicao(blocos, CHARS_SOBREPOSICAO)

        for bloco in blocos:
            adicionar_chunk(
                texto=bloco,
                nome_documento=nome_doc,
                pagina=num_pagina,
            )
            total_chunks += 1

    return total_chunks

# Mudar tipo de processamento documental para corpus.csv e não pasta de documentos
def processar_pasta(directory: str) -> list[dict]:
    """Percorre todos os PDFs numa pasta, processa-os e devolve um resumo.

    Args:
        directory: Caminho para a pasta com os PDFs.

    Returns:
        Lista de dicionários com ``documento`` e ``num_chunks`` por ficheiro.
    """
    resultados = []

    if not os.path.exists(directory):
        print(f"ERRO: A pasta '{directory}' não existe.")
        return resultados

    for ficheiro in sorted(os.listdir(directory)):
        if ficheiro.lower().endswith(".pdf"):
            full_path = os.path.join(directory, ficheiro)
            resultados.append(full_path)

    return resultados

from sentence_transformers import SentenceTransformer
import faiss

modelo_bge = SentenceTransformer("BAAI/bge-m3")

def construir_indice_faiss(lista_chunks):

    textos_a_vetorizar = [c["texto"] for c in lista_chunks] # Conversão de dict para lista simples
    print(textos_a_vetorizar[0])

    if not textos_a_vetorizar:
        raise ValueError("Não há texto para vetorizar: a lista de chunks está vazia.")

    vetores = modelo_bge.encode(textos_a_vetorizar, normalize_embeddings=True) # Normalização dos vetores para indexação no faiss
    dimensao = vetores.shape[1]
    indice_faiss = faiss.IndexFlatIP(dimensao)
    indice_faiss.add(vetores)
    return indice_faiss

def pesquisar_query_base_de_dados(query: str, indice_faiss):

    prompt = [query]

    prompt_normalizada = modelo_bge.encode(prompt, normalize_embeddings=True)

    distancias, ids = indice_faiss.search(prompt_normalizada, RECALL)

    resposta=[]

    for i in ids[0]:

        resposta.append(lista_chunks[i])

    return resposta

import ollama

def gerar_resposta_llm(pergunta_utilizador, indice_faiss):
    # Aqui irás formatar a variável 'contexto_extraido' (os teus 5 chunks)
    # numa única string legível para injetar no prompt
    resultados = pesquisar_query_base_de_dados(pergunta_utilizador, indice_faiss)

    print("\n--- DEBUG FAISS: O que foi extraído do disco? ---")
    for i, chunk in enumerate(resultados):
        print(f"Resultado {i+1} (Pág {chunk['pagina']}): {chunk['texto'][:100]}...")
    print("---------------------------------------------------\n")

    if not resultados:
        contexto_formatado = "Nenhum contexto encontrado."
    else:
        partes=[]
        for chunk in resultados:
            partes.append(f"Documento: {chunk['nome_documento']} (Pág. {chunk['pagina']})\nInformação: {chunk['texto']}")
        contexto_formatado = "\n\n---\n\n".join(partes)

    # 3. Definir as restrições arquiteturais
    instrucao_sistema = (
        "És um assistente técnico agrícola especializado em viticultura.\n"
        "A tua tarefa é extrair a resposta à pergunta do utilizador utilizando APENAS os documentos fornecidos no contexto.\n\n"
        "Diretrizes:\n"
        "- Sintetiza os factos presentes no contexto de forma clara e direta.\n"
        "- Inclui o 'nome_documento' de onde retiraste a resposta no final.\n"
        "- Caso a informação fornecida no contexto não tenha absolutamente nenhuma relação com a pergunta, e apenas nesse caso, deves responder com: 'Não há informação disponível na base de dados para este problema.'"
    )

    prompt_utilizador = f"Contexto extraído da base de dados:\n{contexto_formatado}\n\nPergunta do viticultor: {pergunta_utilizador}"

    # 4. Invocar o LLM localmente
    print(f"\nA consultar o Phi-3 para a pergunta: '{pergunta_utilizador}'...")
    resposta = ollama.chat(model='phi3', messages=[
        {'role': 'system', 'content': instrucao_sistema},
        {'role': 'user', 'content': prompt_utilizador}
    ], stream=True)
    for chunk in resposta:
        yield chunk['message']['content']

# ── Execução direta (teste) ────────────────────────────────────────────────
if __name__ == "__main__":
    diretoria_script = os.path.dirname(os.path.abspath(__file__))
    path_documentos = os.path.abspath(
        os.path.join(diretoria_script, "..", "Documents")
    )

    # print(f"A processar PDFs em: {path_documentos}\n")
    # resumo = processar_pasta(path_documentos)

    # if not resumo:
    #     print("Não foram encontrados ficheiros PDF.")
    #     exit()

    # lista_chunks = processar_pdf_markdown(resumo)

    # for doc in resumo:
    #     print(f"  {doc['documento']}  ->  {doc['num_chunks']} chunks")

    # print(f"\nTotal de chunks na lista_chunks: {len(lista_chunks)}")

    # indice_faiss = construir_indice_faiss(lista_chunks)
    
    # print("\nA guardar a base de dados no disco...")
    # faiss.write_index(indice_faiss, "base_dados_videira.faiss")

    indice_faiss = faiss.read_index("base_dados_videira.faiss")

    with open("metadados_chunks.pkl", "rb") as f:
        lista_chunks = pickle.load(f)

    # # Guarda os metadados (a tua lista de dicionários)
    # with open("metadados_chunks.pkl", "wb") as f:
    #     pickle.dump(lista_chunks, f)

    # Teste de Produção (RAG Completo)
    pergunta_teste = "O que é uma uva?"
    resposta_final = gerar_resposta_llm(pergunta_teste, indice_faiss)

    print("\n=== RESPOSTA DO ASSISTENTE ===")
    for palavra in gerar_resposta_llm(pergunta_teste, indice_faiss):
        print(palavra, end='', flush=True)
    print("==============================\n")

    # testar_recall_at_5(indice_faiss,dados_de_teste)

    # # Mostra os primeiros 3 chunks como exemplo
    # for i, c in enumerate(lista_chunks[:3]):
    #     print(f"\n--- Chunk {i} (pág. {c['pagina']}, {c['nome_documento']}) ---")
    #     print(c["texto"][:200] + "..." if len(c["texto"]) > 200 else c["texto"])

