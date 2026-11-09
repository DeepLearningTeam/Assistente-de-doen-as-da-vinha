from __future__ import annotations
import argparse
import csv
from collections import Counter
from pathlib import Path

REQUIRED = {
    'id', 'pergunta', 'tipo', 'doenca', 'resposta_ref', 'doc_id',
    'pagina_pdf', 'expected_action', 'split', 'estado_validacao'
}


def main() -> None:
    parser = argparse.ArgumentParser(description='Valida esquema e distribuição das 40 perguntas F4.')
    parser.add_argument('--csv', default='data/perguntas/perguntas.csv')
    args = parser.parse_args()
    path = Path(args.csv)
    if not path.exists():
        parser.error(f'CSV não encontrado: {path}. Executa a partir da pasta principal do repositório.')
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = set(reader.fieldnames or [])

    errors: list[str] = []
    warnings: list[str] = []
    missing = REQUIRED - fields
    if missing:
        errors.append(f'Colunas em falta: {sorted(missing)}')
    ids = [r.get('id', '') for r in rows]
    if len(rows) != 40:
        errors.append(f'Esperadas 40 perguntas; encontradas {len(rows)}.')
    if len(set(ids)) != len(ids):
        errors.append('Há IDs duplicados.')

    counts = Counter(r.get('expected_action') for r in rows)
    if counts != Counter({'responder': 30, 'recusar': 10}):
        errors.append(f'Distribuição esperada 30 responder/10 recusar; encontrada {dict(counts)}')
    split_counts = Counter(r.get('split') for r in rows)
    if split_counts != Counter({'validacao': 20, 'teste': 20}):
        errors.append(f'Esperadas 20 validação/20 teste; encontrado {dict(split_counts)}')
    for split in ('validacao', 'teste'):
        subset = [r for r in rows if r.get('split') == split]
        count = Counter(r.get('expected_action') for r in subset)
        if count != Counter({'responder': 15, 'recusar': 5}):
            errors.append(f'{split}: esperado 15 responder/5 recusar; encontrado {dict(count)}')

    for r in rows:
        qid = r.get('id', '?')
        for col in ('pergunta', 'resposta_ref', 'doc_id', 'pagina_pdf'):
            if not r.get(col, '').strip():
                errors.append(f'{qid}: campo {col} vazio')
        try:
            if int(r.get('pagina_pdf', '0')) < 1:
                errors.append(f'{qid}: pagina_pdf tem de ser >= 1')
        except ValueError:
            errors.append(f'{qid}: pagina_pdf não é inteiro')
        if r.get('expected_action') == 'recusar' and r.get('doc_id') not in {'D10', 'D11', 'D12'}:
            errors.append(f'{qid}: holdout não aponta para D10-D12')
        if r.get('expected_action') == 'responder' and r.get('doc_id') in {'D10', 'D11', 'D12'}:
            errors.append(f'{qid}: pergunta respondível usa documento holdout')
        if r.get('estado_validacao') != 'validada_humanamente':
            warnings.append(f'{qid}: ainda não está validada humanamente')

    print(f'Perguntas: {len(rows)}')
    print(f'Esperado: {dict(counts)} | splits: {dict(split_counts)}')
    for split in ('validacao', 'teste'):
        count = Counter(r.get('expected_action') for r in rows if r.get('split') == split)
        print(f'  {split}: {dict(count)}')
    if warnings:
        print(f'\nAviso: {len(warnings)} pergunta(s) aguardam validação humana. Exemplo: {warnings[0]}')
    if errors:
        print('\nERROS:')
        for error in errors:
            print(' -', error)
        raise SystemExit(1)
    print('\nEstrutura e distribuição OK.')
    print('Isto não comprova a correção científica das respostas nem a exclusividade dos holdouts; consultar data/perguntas/holdout_audit.md e validar manualmente.')


if __name__ == '__main__':
    main()
