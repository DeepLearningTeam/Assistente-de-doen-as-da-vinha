import pymupdf as pydf
import os

def extrair_processar(path_pdf, size_chunk=1000, sobreposicao=200): # 4 Caracteres equivalem a 1 token
    """
    Extração de texto, conta as páginas/caracteres e divide em chunks com sobreposição
    """
    try:
        doc = pydf.open(path_pdf)
        texto_completo=""

        for pagina in doc:
            texto_completo+=pagina.get_text("text")+ "\n"

        num_paginas = len(doc)
        num_caracteres = len(texto_completo)

        chunks=[]
        step=size_chunk-sobreposicao    # Step indica onde começa o próximo chunk

        for i in range(0, num_caracteres, step):
            chunks.append(texto_completo[i:i+size_chunk])

        return num_paginas, num_caracteres, chunks

    except Exception as e:
        print(f"Erro ao ler {path_pdf}: {e}")
        return 0,0,[]

def analisar_pasta(directory):
    resultados=[]

    if not os.path.exists(directory):
        print(f"ERRO: A pasta {directory} não existe.")
        return resultados

    # Percorre todos os ficheiros na pasta
    for ficheiro in os.listdir(directory):
        if ficheiro.lower().endswith(".pdf"):
            full_path=os.path.join(directory,ficheiro)

            # Extrai os dados do PDF
            paginas, caracteres, chunks = extrair_processar(full_path)

            # Agrupa os dados num dicionário e adiciona à lista
            resultados.append({
                "documento": ficheiro,
                "paginas": paginas,
                "caracteres": caracteres,
                "num_chunks": len(chunks)
            })
    return resultados

if __name__ == "__main__":
    directory=os.path.dirname(os.path.abspath(__file__))
    path_documentos=os.path.abspath(os.path.join(directory,"..","Documents"))

    lista_documentos = analisar_pasta(path_documentos)

    # Imprime os resultados organizados
    for doc in lista_documentos:
        print(f"Ficheiro: {doc['documento']} | Páginas: {doc['paginas']} | Caracteres: {doc['caracteres']} | Chunks: {doc['num_chunks']}")
