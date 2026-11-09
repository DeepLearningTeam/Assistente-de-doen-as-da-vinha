from __future__ import annotations
import argparse, csv, hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for bloco in iter(lambda:f.read(1024*1024),b''): h.update(bloco)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description='Compara os PDFs locais com os SHA-256 registados em data/corpus.csv.')
    p.add_argument('--docs',default='data/docs')
    p.add_argument('--corpus',default='data/corpus.csv')
    p.add_argument('--mapa',default='data/docs/mapa_ficheiros_locais.csv')
    args=p.parse_args()
    root=Path(args.docs)
    with open(args.corpus,encoding='utf-8-sig',newline='') as f: corpus={r['doc_id']:r for r in csv.DictReader(f)}
    with open(args.mapa,encoding='utf-8-sig',newline='') as f: mapa=list(csv.DictReader(f))
    falhas=0
    for row in mapa:
        doc=row['doc_id']; file=root/row['nome_ficheiro_local']
        if row['indexacao']=='nao': state='HOLDOUT — não indexar'
        elif row['indexacao']=='condicional': state='CONDICIONAL — confirmar conteúdo/licença'
        else: state='candidato a indexação'
        if not file.exists():
            print(f'{doc}: FICHEIRO EM FALTA: {file.name} [{state}]'); falhas+=1; continue
        actual=sha256(file); expected=corpus[doc].get('sha256','').strip()
        ok=actual.lower()==expected.lower()
        print(f'{doc}: {"SHA OK" if ok else "SHA DIFERENTE"} | {file.name} | {state}')
        if not ok:
            print(f'    esperado={expected}\n    atual={actual}')
            falhas+=1
    if falhas: raise SystemExit(f'\n{falhas} ficheiro(s) em falta ou com SHA diferente.')
    print('\nVerificação de ficheiros concluída.')

if __name__=='__main__': main()
