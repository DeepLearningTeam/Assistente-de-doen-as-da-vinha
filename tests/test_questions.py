import csv
from pathlib import Path
from collections import Counter


def test_question_set_distribution():
    path=Path('data/perguntas/perguntas.csv')
    with path.open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==40
    assert len({r['id'] for r in rows})==40
    assert Counter(r['expected_action'] for r in rows)=={'responder':30,'recusar':10}
    assert Counter(r['split'] for r in rows)=={'validacao':20,'teste':20}
    for split in ('validacao','teste'):
        c=Counter(r['expected_action'] for r in rows if r['split']==split)
        assert c=={'responder':15,'recusar':5}


def test_holdout_questions_use_only_d10_d12():
    with Path('data/perguntas/perguntas.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    holdouts=[r for r in rows if r['expected_action']=='recusar']
    assert len(holdouts)==10
    assert all(r['doc_id'] in {'D10','D11','D12'} for r in holdouts)


def test_question_fields_are_not_empty():
    with Path('data/perguntas/perguntas.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    for r in rows:
        for field in ('id','pergunta','tipo','doenca','resposta_ref','doc_id','pagina_pdf','split'):
            assert r[field].strip(), (r['id'],field)
