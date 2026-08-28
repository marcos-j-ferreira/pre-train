"""
Divide um arquivo .bin de tokens em N shards menores, para testes rápidos.

Não precisa (e não deve, pra ser rápido) ler o dataset inteiro: usa mmap e
recorta apenas os tokens necessários pra montar os shards pedidos.

Uso:
    python shards.py --input train.bin --outdir shards/ \
        --num_shards 10 --shard_size 100000 --dtype uint16

    # amostrar de posições aleatórias do arquivo, em vez de pegar do início
    python shards.py --input train.bin --outdir shards/ \
        --num_shards 10 --shard_size 10000000 --dtype uint16 --random --seed 42
"""

import argparse
import os
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Fatiar .bin de tokens em shards de teste")
    p.add_argument("--input", required=True, help="Caminho do .bin de entrada")
    p.add_argument("--outdir", required=True, help="Diretório de saída dos shards")
    p.add_argument("--num_shards", type=int, required=True, help="Quantidade de shards a criar")
    p.add_argument("--shard_size", type=int, required=True, help="Tokens por shard")
    p.add_argument("--dtype", default="uint16", help="dtype dos tokens no .bin (ex: uint16, uint32)")
    p.add_argument("--random", action="store_true", help="Sortear posição inicial de cada shard (default: sequencial a partir do início)")
    p.add_argument("--seed", type=int, default=None, help="Seed pro sorteio (só usado com --random)")
    return p.parse_args()


def main():
    args = parse_args()
    dtype = np.dtype(args.dtype)

    total_bytes = os.path.getsize(args.input)
    total_tokens = total_bytes // dtype.itemsize
    needed = args.num_shards * args.shard_size

    print(f"Arquivo de entrada: {args.input}")
    print(f"Total de tokens disponíveis: {total_tokens:,}")
    print(f"Tokens necessários ({args.num_shards} shards x {args.shard_size}): {needed:,}")

    if args.shard_size > total_tokens:
        raise ValueError(
            f"shard_size ({args.shard_size}) maior que o total de tokens do arquivo ({total_tokens})."
        )
    if not args.random and needed > total_tokens:
        raise ValueError(
            f"Modo sequencial precisa de {needed:,} tokens, mas o arquivo só tem {total_tokens:,}. "
            f"Use --random pra sortear posições, ou reduza --num_shards / --shard_size."
        )

    os.makedirs(args.outdir, exist_ok=True)
    data = np.memmap(args.input, dtype=dtype, mode="r")

    rng = np.random.default_rng(args.seed) if args.random else None
    max_start = total_tokens - args.shard_size  # inclusive

    for i in range(args.num_shards):
        if args.random:
            start = int(rng.integers(0, max_start + 1))
        else:
            start = i * args.shard_size

        shard = np.array(data[start : start + args.shard_size])  # copia pra fora do mmap
        out_path = os.path.join(args.outdir, f"shard_{i:04d}.bin")
        shard.tofile(out_path)

        print(f"shard_{i:04d}.bin -> tokens [{start:,}:{start + args.shard_size:,}] "
              f"({os.path.getsize(out_path):,} bytes)")

    print(f"\n{args.num_shards} shards salvos em: {args.outdir}")


if __name__ == "__main__":
    main()