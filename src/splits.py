"""Divisões estratificadas, reproduzíveis e auditáveis do FieldVitis."""
import argparse
import random
from collections import defaultdict
from pathlib import Path

from .apoio import ErroProjeto, RAIZ, gravar_json, sha256

PASTA_IMAGENS = RAIZ / "data/imagens"
FICHEIRO_SPLITS = RAIZ / "data/splits.json"
SEEDS = (42, 43, 44)
PASTAS = {"healthy": "saudavel", "esca": "esca", "leaf_blight": "leaf_blight", "black_rot": "black_rot"}


def inventariar(pasta):
    """Relaciona caminhos relativos e hashes; agrupa duplicados byte a byte."""
    grupos = defaultdict(list)
    for imagem in sorted(pasta.rglob("*")):
        classe = PASTAS.get(imagem.parent.name.casefold())
        if not imagem.is_file() or not classe or imagem.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        digest = sha256(imagem)
        grupos[digest].append({
            "id": imagem.relative_to(pasta).as_posix(), "classe": classe,
            "sha256": digest, "splits": {},
        })

    classes = defaultdict(list)
    for grupo in grupos.values():
        classe = grupo[0]["classe"]
        if any(x["classe"] != classe for x in grupo):
            raise ErroProjeto("Há imagens com o mesmo SHA-256 e rótulos diferentes.")
        for imagem in grupo:
            imagem["grupo"] = grupo[0]["id"]
        classes[classe].append(grupo)
    if set(classes) != set(PASTAS.values()):
        raise ErroProjeto("Extrai as quatro pastas do FieldVitis para data/imagens/.")
    return classes


def dividir(pasta, seeds):
    classes = inventariar(pasta)
    contagens = {}
    for seed in seeds:
        rng = random.Random(seed)
        contagens[str(seed)] = {}
        for classe, grupos in sorted(classes.items()):
            ordem = list(grupos)
            rng.shuffle(ordem)
            total = sum(map(len, ordem))
            alvo = max(1, round(total * 0.15))
            teste = max(6, alvo) if classe == "black_rot" else alvo
            numeros = {"teste": 0, "validacao": 0, "treino": 0}
            for grupo in ordem:
                if numeros["teste"] < teste:
                    split = "teste"
                elif numeros["validacao"] < alvo:
                    split = "validacao"
                else:
                    split = "treino"
                for imagem in grupo:
                    imagem["splits"][str(seed)] = split
                numeros[split] += len(grupo)
            if min(numeros.values()) == 0 or numeros["teste"] < teste:
                raise ErroProjeto(f"Imagens ou grupos insuficientes na classe {classe}.")
            contagens[str(seed)][classe] = numeros
    imagens = [i for grupos in classes.values() for grupo in grupos for i in grupo]
    return {
        "dataset": "FieldVitis v2", "doi": "10.5281/zenodo.17307846",
        "seeds": list(seeds), "proporcoes_treino_validacao_teste": [0.7, 0.15, 0.15],
        "contagens": contagens, "imagens": sorted(imagens, key=lambda i: i["id"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imagens", type=Path, default=PASTA_IMAGENS)
    parser.add_argument("--saida", type=Path, default=RAIZ / "data/splits_amostragem.json")
    parser.add_argument("--seeds", type=int, nargs=3, default=SEEDS)
    args = parser.parse_args()
    if args.saida.resolve() == FICHEIRO_SPLITS.resolve():
        parser.error("Para a CNN recebida utilizar src.preparar_cnn; não substituir as divisões de treino por nova amostragem.")
    if len(set(args.seeds)) != 3:
        parser.error("Indicar três seeds distintas.")
    try:
        dados = dividir(args.imagens, args.seeds)
        gravar_json(args.saida, dados)
        print(f"{len(dados['imagens'])} imagens divididas. Manifesto em {args.saida}.")
        for seed, classes in dados["contagens"].items():
            print(f"Seed {seed}: {classes}")
    except (ErroProjeto, OSError) as erro:
        parser.exit(1, f"{erro}\n")


if __name__ == "__main__":
    main()
