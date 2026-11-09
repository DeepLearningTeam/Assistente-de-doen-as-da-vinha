from __future__ import annotations
import argparse
import csv
from pathlib import Path
import fitz


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Confirma que todos os PDFs referidos nas perguntas existem e que as páginas existem/têm texto.'
    )
    parser.add_argument('--perguntas', default='data/perguntas/perguntas.csv')
    parser.add_argument('--mapa', default='data/docs/mapa_ficheiros_locais.csv')
    parser.add_argument('--docs', default='data/docs')
    args = parser.parse_args()

    with Path(args.perguntas).open(encoding='utf-8-sig', newline='') as f:
        perguntas = list(csv.DictReader(f))
    with Path(args.mapa).open(encoding='utf-8-sig', newline='') as f:
        mapa = {r['doc_id']: r['nome_ficheiro_local'] for r in csv.DictReader(f)}

    erros: list[str] = []
    avisos: list[str] = []
    documentos: dict[str, fitz.Document] = {}
    try:
        for q in perguntas:
            qid, doc_id = q['id'], q['doc_id']
            if doc_id not in mapa:
                erros.append(f'{qid}: {doc_id} não existe em mapa_ficheiros_locais.csv')
                continue
            if doc_id not in documentos:
                caminho = Path(args.docs) / mapa[doc_id]
                if not caminho.exists():
                    erros.append(f'{qid}: PDF em falta: {caminho}')
                    continue
                documentos[doc_id] = fitz.open(caminho)
            if doc_id not in documentos:
                continue
            pagina = int(q['pagina_pdf'])
            pdf = documentos[doc_id]
            if pagina < 1 or pagina > len(pdf):
                erros.append(f'{qid}: página {pagina} fora do intervalo 1..{len(pdf)} de {doc_id}')
                continue
            texto = pdf[pagina - 1].get_text('text').strip()
            if not texto:
                avisos.append(f'{qid}: página {pagina} de {doc_id} não tem texto extraível; conferir visualmente/OCR')
        print(f'Perguntas verificadas: {len(perguntas)}')
        print(f'PDFs abertos: {len(documentos)}')
        if avisos:
            print('\nAVISOS:')
            for aviso in avisos:
                print(' -', aviso)
        if erros:
            print('\nERROS:')
            for erro in erros:
                print(' -', erro)
            raise SystemExit(1)
        print('\nTodas as referências apontam para páginas existentes com texto extraível.')
        print('Isto não valida a correção da resposta de referência; é necessária revisão humana do conteúdo e da página.')
    finally:
        for pdf in documentos.values():
            pdf.close()


if __name__ == '__main__':
    main()
