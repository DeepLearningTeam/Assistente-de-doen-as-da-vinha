"""Adaptar as divisões originais da CNN, sem gerar novas amostragens."""
import argparse
import csv
import hashlib
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from PIL import Image, ImageOps

from .apoio import ErroProjeto, RAIZ, gravar_json, ler_json, sha256, FICHEIRO_SPLITS, PASTA_IMAGENS, nova_verificacao
from .cnn_resnet import MAPEAMENTO, SPLITS_ORIGINAIS, carregar_classes
from .imagem import config_cnn

PARTICOES = {"train": "treino", "val": "validacao", "test": "teste"}
SEED_MODELO = 1


def divisoes_originais(seed):
    original = ler_json(SPLITS_ORIGINAIS)
    chave = str(seed)
    if chave not in original["seeds"]:
        raise ErroProjeto(f"Não existe divisão de treino para a seed {seed}.")
    conjuntos = {PARTICOES[k]: list(v) for k, v in original["seeds"][chave].items()}
    conjuntos["holdout"] = list(original["holdout"])
    for nome, ids in conjuntos.items():
        if len(ids) != len(set(ids)):
            raise ErroProjeto(f"IDs repetidos em {nome}, seed {seed}.")
    for a, b in combinations(conjuntos, 2):
        if set(conjuntos[a]) & set(conjuntos[b]):
            raise ErroProjeto(f"Sobreposição de IDs entre {a} e {b}, seed {seed}.")
    return conjuntos


def validar_manifesto(manifesto, seeds):
    """Confirma a correspondência entre o manifesto e as divisões dos pesos."""
    imagens = manifesto["imagens"]
    if len({i["id"] for i in imagens}) != len(imagens):
        raise ErroProjeto("O manifesto de avaliação contém IDs duplicados.")
    for seed in seeds:
        if seed != SEED_MODELO:
            raise ErroProjeto("O checkpoint configurado corresponde à seed de treino 1.")
        conjuntos = divisoes_originais(seed)
        for split, ids in conjuntos.items():
            encontrados = {i["id"] for i in imagens if i["splits"].get(str(seed)) == split}
            if encontrados != set(ids):
                raise ErroProjeto(f"O split {split} não corresponde ao treino da ResNet50 seed {seed}.")
        if len(imagens) != sum(map(len, conjuntos.values())):
            raise ErroProjeto("O manifesto contém imagens alheias às divisões originais.")
    for item in imagens:
        classe = MAPEAMENTO.get(Path(item["id"]).parts[0])
        if item["classe"] != classe:
            raise ErroProjeto(f"Classe incompatível com o caminho: {item['id']}.")


def preparar(pasta):
    carregar_classes()
    original = ler_json(SPLITS_ORIGINAIS)
    with (SPLITS_ORIGINAIS.parent / "manifest_clean.csv").open(encoding="utf-8-sig", newline="") as f:
        limpos = list(csv.DictReader(f))
    with (SPLITS_ORIGINAIS.parent / "holdout.csv").open(encoding="utf-8-sig", newline="") as f:
        reservados = list(csv.DictReader(f))
    if {i["path"] for i in reservados} != set(original["holdout"]) or len(reservados) != 50:
        raise ErroProjeto("O holdout.csv não corresponde às 50 imagens reservadas em splits.json.")
    if len(limpos) != len({i["path"] for i in limpos}):
        raise ErroProjeto("Há IDs duplicados no manifesto limpo.")
    por_bytes, por_pixels, hashes = defaultdict(list), defaultdict(list), {}
    raiz = pasta.resolve()
    for item in limpos:
        identificador = item["path"]
        foto = (raiz / identificador).resolve()
        if not foto.is_relative_to(raiz) or not foto.is_file():
            raise ErroProjeto(f"Imagem em falta: {identificador}. As classes devem ficar diretamente em data/imagens/.")
        conteudo = foto.read_bytes()
        if hashlib.md5(conteudo).hexdigest() != item["md5"]:
            raise ErroProjeto(f"A imagem difere do manifesto original da CNN: {identificador}.")
        digest = hashlib.sha256(conteudo).hexdigest()
        hashes[identificador] = digest
        por_bytes[digest].append(identificador)
        # Auditoria de conteúdo apenas: não há inferência nem afinação no holdout.
        with Image.open(foto) as im:
            rgb = ImageOps.exif_transpose(im).convert("RGB")
            pixels = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            por_pixels[pixels].append(identificador)
    auditoria = {}
    for seed in map(int, original["seeds"]):
        conjuntos = divisoes_originais(seed)
        if set().union(*map(set, conjuntos.values())) != set(hashes):
            raise ErroProjeto(f"As divisões da seed {seed} não cobrem o manifesto limpo.")
        intersecoes = {}
        for a, b in combinations(conjuntos, 2):
            grupos_a = {hashes[i] for i in conjuntos[a]}
            grupos_b = {hashes[i] for i in conjuntos[b]}
            intersecoes[f"{a}/{b}"] = len(grupos_a & grupos_b)
            if grupos_a & grupos_b:
                raise ErroProjeto(f"Duplicados byte a byte entre {a}/{b}, seed {seed}.")
            for grupo in por_pixels.values():
                if set(grupo) & set(conjuntos[a]) and set(grupo) & set(conjuntos[b]):
                    raise ErroProjeto(f"Píxeis iguais entre {a}/{b}, seed {seed}.")
        auditoria[str(seed)] = {"contagens": {k: len(v) for k, v in conjuntos.items()},
                                 "sobreposicoes_sha256": intersecoes}
    conjuntos = divisoes_originais(SEED_MODELO)
    particao = {i: nome for nome, ids in conjuntos.items() for i in ids}
    imagens = [{"id": i, "classe": MAPEAMENTO[Path(i).parts[0]], "sha256": hashes[i],
                "grupo": i, "splits": {str(SEED_MODELO): particao[i]}} for i in sorted(hashes)]
    cfg = config_cnn()
    if cfg.seed_treino != SEED_MODELO:
        raise ErroProjeto("A preparação suporta a seed de treino 1 do checkpoint configurado.")
    modelo = RAIZ / cfg.checkpoint
    manifesto = {
        "dataset": "FieldVitis v2", "seeds": [SEED_MODELO],
        "classes_originais": original["classes"], "imagens": imagens,
        "proveniencia": {"splits_originais_sha256": sha256(SPLITS_ORIGINAIS),
                         "seed_treino": SEED_MODELO, "seed_reserva": original["politica"]["holdout_seed"],
                         "checkpoint_sha256": sha256(modelo)},
    }
    validar_manifesto(manifesto, [SEED_MODELO])
    relatorio = {
        "imagens": len(imagens), "classes": dict(Counter(i["classe"] for i in imagens)),
        "seeds_originais": auditoria,
        "duplicados_bytes": [g for g in por_bytes.values() if len(g) > 1],
        "duplicados_pixels_rgb_exif": [g for g in por_pixels.values() if len(g) > 1],
        "holdout_inferencia_executada": False,
        "limite_auditoria": "Hashes exatos; não deteta quase duplicados, a mesma folha ou a mesma videira.",
    }
    return manifesto, relatorio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imagens", type=Path, default=PASTA_IMAGENS)
    args = parser.parse_args()
    try:
        manifesto, relatorio = preparar(args.imagens)
        gravar_json(FICHEIRO_SPLITS, manifesto)
        pasta = nova_verificacao("dados-cnn")
        gravar_json(pasta / "divisoes.json", relatorio)
        print(f"{len(manifesto['imagens'])} imagens verificadas; seed de treino 1: 352/76/77 + 50 reservadas.")
        print(f"Relatório de integridade em {pasta}.")
    except (ErroProjeto, OSError, ValueError) as erro:
        parser.exit(1, f"{erro}\n")


if __name__ == "__main__":
    main()
