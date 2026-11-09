from __future__ import annotations
import argparse, csv, json, os, subprocess, time
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv
from src.llm_baseline import criar_llm_local, responder_sem_rag
from src.prompts import BASELINE_SYSTEM_PROMPT


def git_commit() -> str | None:
    try:
        return subprocess.check_output(['git','rev-parse','HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def main():
    parser=argparse.ArgumentParser(description='Executa a baseline LLM sem RAG sobre perguntas CSV.')
    parser.add_argument('--csv', default='data/perguntas/perguntas.csv')
    parser.add_argument('--split', choices=['validacao','teste','todos'], default='validacao')
    parser.add_argument('--repeticoes', type=int, default=3)
    parser.add_argument('--temperatura', type=float, default=0.0, help='Temperatura da geração; registar cada condição em separado.')
    parser.add_argument('--modelo', default=None)
    parser.add_argument('--output', default='runs/llm_baseline_sem_rag')
    parser.add_argument('--confirmar-protocolo-fechado', action='store_true', help='Obrigatório para executar perguntas do split teste.')
    args=parser.parse_args()
    if args.repeticoes < 1:
        parser.error('--repeticoes tem de ser >= 1')
    if args.split in {'teste','todos'} and not args.confirmar_protocolo_fechado:
        parser.error('O split de teste está reservado. Só continuar depois de aprovação e commit do PROTOCOLO.md; usa --confirmar-protocolo-fechado para confirmar.')
    load_dotenv()
    caminho=Path(args.csv)
    if not caminho.exists():
        parser.error(f'CSV não encontrado: {caminho}')
    with caminho.open(encoding='utf-8-sig',newline='') as f:
        perguntas=list(csv.DictReader(f))
    if args.split != 'todos':
        perguntas=[p for p in perguntas if p['split']==args.split]
    if not perguntas:
        parser.error('Não há perguntas para o split selecionado.')
    llm=criar_llm_local(args.modelo,temperature=args.temperatura)
    model_id=args.modelo or os.getenv('OLLAMA_MODEL','qwen3:1.7b')
    out_dir=Path(args.output); out_dir.mkdir(parents=True,exist_ok=True)
    timestamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out_file=out_dir/f'baseline_{args.split}_{timestamp}.jsonl'
    commit=git_commit()
    count=0
    with out_file.open('w',encoding='utf-8') as fo:
        for p in perguntas:
            for rep in range(1,args.repeticoes+1):
                t0=time.perf_counter()
                result=responder_sem_rag(llm,p['pergunta'],temperature=args.temperatura)
                row={
                    'experimento':'baseline_llm_sem_rag', 'model_id':model_id,
                    'config':{'temperature':args.temperatura,'tools':False,'rag':False},
                    'prompt_sistema':BASELINE_SYSTEM_PROMPT,
                    'commit':commit, 'pergunta_id':p['id'], 'split':p['split'],
                    'repeticao':rep, 'pergunta':p['pergunta'],
                    'expected_action_rag':p['expected_action'], 'resposta':result['resposta'],
                    'latencia_s':result['latencia_s'], 'timestamp_utc':datetime.now(timezone.utc).isoformat(),
                    'fonte_recuperada':False,
                }
                fo.write(json.dumps(row,ensure_ascii=False)+'\n'); fo.flush(); count+=1
                print(f"[{count}/{len(perguntas)*args.repeticoes}] {p['id']} repetição {rep} — {result['latencia_s']} s")
    print(f'\nConcluído. Registos: {count}. Ficheiro: {out_file}')
    print('Não foram consultados documentos nem ferramentas nesta baseline.')

if __name__=='__main__': main()
