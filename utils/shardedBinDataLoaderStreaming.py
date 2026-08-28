

"""Loader streaming para shards binarias de tokens.

O loader nunca materializa o dataset (nem uma permutacao com um indice por
token) na RAM.  A ordem de uma epoca e definida por duas permutacoes
pseudoaleatorias, sem repeticao:

* os blocos de cada shard sao visitados em ordem embaralhada;
* os offsets validos dentro do bloco atual tambem sao visitados em ordem
  embaralhada.

As permutacoes usam uma rede de Feistel com cycle-walking. Elas sao bijetivas
no intervalo solicitado e precisam de memoria O(1). Um batch costuma tocar
somente um bloco contiguo do arquivo, o que preserva a aleatoriedade sem
transformar cada token em uma leitura aleatoria de disco.
"""

import glob
import os
import queue
import threading

import numpy as np
import torch


class ShardedBinDataLoaderStreaming:
    """Fornece batches infinitos de ``(x, y)`` a partir de varios ``.bin``.

    Cada arquivo contem somente token ids consecutivos, no ``dtype_dataset``.
    Uma janela nunca atravessa a fronteira entre duas shards: isso evita criar
    contexto artificial entre documentos/arquivos diferentes.

    Args:
        shuffle_block_samples: quantidade maxima de janelas de treinamento em
            um bloco de I/O.  65.536 amostras com tokens uint16 ocupam apenas
            cerca de 128 KiB no arquivo (mais a janela no fim), favorecendo o
            cache de paginas do SO. Aumente para reduzir seeks; diminua para
            misturar shards com mais frequencia.
    """

    # Constantes da funcao de mistura SplitMix64. Operacoes sao feitas com
    # inteiros Python e mascaradas para manter resultados identicos em todas
    # as plataformas.
    _MASK64 = (1 << 64) - 1
    _ROUND_CONSTANT = 0x9E3779B97F4A7C15

    def __init__(
        self,
        shard_dir: str,
        pattern: str,
        block_size: int,
        batch_size: int,
        dtype_dataset,
        device,
        seed: int = 42,
        prefetch_batches: int = 3,
        verbose: bool = True,
        log_every: int = 50,
        shuffle_block_samples: int = 65_536,
        start_fraction: float = 0.0,
        end_fraction: float = 1.0,
    ):
        if block_size <= 0:
            raise ValueError("block_size deve ser positivo.")
        if batch_size <= 0:
            raise ValueError("batch_size deve ser positivo.")
        if prefetch_batches <= 0:
            raise ValueError("prefetch_batches deve ser >= 1.")
        if shuffle_block_samples <= 0:
            raise ValueError("shuffle_block_samples deve ser positivo.")
        if not 0.0 <= start_fraction < end_fraction <= 1.0:
            raise ValueError("start_fraction e end_fraction devem obedecer 0 <= inicio < fim <= 1.")

        self.block_size = int(block_size)
        self.batch_size = int(batch_size)
        self.shuffle_block_samples = int(shuffle_block_samples)
        self.dtype_dataset = np.dtype(dtype_dataset)
        self.device = torch.device(device)
        self.is_cuda = self.device.type == "cuda"
        self.verbose = verbose
        self.log_every = log_every
        self._seed = int(seed)
        self._closed = False
        self._producer_error = None

        shard_paths = sorted(glob.glob(os.path.join(shard_dir, pattern)))
        if not shard_paths:
            raise FileNotFoundError(
                f"Nenhum shard encontrado em '{shard_dir}' com o padrao '{pattern}'"
            )
        self.shard_paths = shard_paths
        self.num_shards = len(shard_paths)

        # Abrir um memmap nao le o conteudo para o heap Python. Shards vazias
        # sao mantidas como None, pois numpy.memmap nao aceita arquivo vazio.
        self._mmaps = []
        token_counts = np.zeros(self.num_shards, dtype=np.int64)
        if self.verbose:
            print(f"[loader] descobrindo shards em '{shard_dir}' (padrao='{pattern}')")

        for i, path in enumerate(shard_paths):
            byte_count = os.path.getsize(path)
            if byte_count % self.dtype_dataset.itemsize:
                raise ValueError(
                    f"Shard '{path}' tem {byte_count} bytes, tamanho incompativel "
                    f"com dtype {self.dtype_dataset}."
                )
            count = byte_count // self.dtype_dataset.itemsize
            arr = np.memmap(path, dtype=self.dtype_dataset, mode="r") if count else None
            self._mmaps.append(arr)
            token_counts[i] = count
            if self.verbose:
                print(f"  [{i:04d}] {os.path.basename(path)} -> {count:,} tokens")

        # Para x=[off:off+B] e y=[off+1:off+B+1], o ultimo offset valido e
        # L-B-1; portanto ha exatamente L-B amostras validas.
        full_samples_per_shard = np.maximum(0, token_counts - self.block_size)
        self._sample_starts = np.floor(full_samples_per_shard * start_fraction).astype(np.int64)
        sample_ends = np.floor(full_samples_per_shard * end_fraction).astype(np.int64)
        self.samples_per_shard = np.maximum(0, sample_ends - self._sample_starts)
        self.total_samples = int(self.samples_per_shard.sum())
        empty_shards = int((self.samples_per_shard == 0).sum())
        if empty_shards and self.verbose:
            print(
                f"[loader][aviso] {empty_shards} shard(s) tem menos tokens que "
                f"block_size+1 e serao ignorados no sampler."
            )
        if self.total_samples == 0:
            self._close_mmaps()
            raise ValueError("Nenhum shard tem tokens suficientes para block_size dado.")

        # Um bloco jamais cruza shards. Estes arrays tem somente um elemento
        # por shard, nao um por token/amostra.
        self.blocks_per_shard = (
            (self.samples_per_shard + self.shuffle_block_samples - 1)
            // self.shuffle_block_samples
        )
        self.block_boundaries = np.concatenate([[0], np.cumsum(self.blocks_per_shard)])
        self.total_blocks = int(self.block_boundaries[-1])

        if self.verbose:
            print(
                f"[loader] {self.num_shards} shards | {token_counts.sum():,} tokens | "
                f"{self.total_samples:,} amostras | {self.total_blocks:,} blocos "
                f"(block_size={self.block_size}, shuffle_block={self.shuffle_block_samples})"
            )

        self._epoch = 0
        self._next_block_position = 0
        self._current_shard = None
        self._current_block_start = 0
        self._current_block_count = 0
        self._current_offset_position = 0
        self._batches_produced = 0
        self._samples_produced = 0
        self._shard_access_count = np.zeros(self.num_shards, dtype=np.int64)

        self._queue = queue.Queue(maxsize=prefetch_batches)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._producer_loop,
            name="sharded-bin-prefetch",
            daemon=True,
        )
        self._thread.start()
        if self.verbose:
            print(
                f"[loader] prefetch iniciado (buffer={prefetch_batches} batches, "
                f"seed={seed}, device={self.device})"
            )

    # ------------------------------------------------------------------ #
    # Permutacao O(1) em memoria
    # ------------------------------------------------------------------ #
    @classmethod
    def _mix64(cls, value: int) -> int:
        value = (value + cls._ROUND_CONSTANT) & cls._MASK64
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & cls._MASK64
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & cls._MASK64
        return (value ^ (value >> 31)) & cls._MASK64

    @classmethod
    def _permute_index(cls, index: int, size: int, key: int) -> int:
        """Retorna uma permutacao pseudoaleatoria de ``[0, size)``.

        A rede de Feistel e bijetiva em um dominio potencia de dois. O
        cycle-walking descarta valores fora de ``size`` sem criar tabelas de
        indices. ``size`` e limitado a 2**62, muito acima de datasets usuais
        de tokens e suficiente para manter a aritmetica simples e portavel.
        """
        if size == 1:
            return 0
        if not 0 <= index < size:
            raise ValueError("index fora do dominio da permutacao")
        if size > (1 << 62):
            raise ValueError("O sampler suporta no maximo 2**62 elementos.")

        half_bits = max(1, (size - 1).bit_length() // 2)
        domain_bits = half_bits * 2
        # Se ceil(log2(size)) for impar, arredondar para o proximo dominio de
        # Feistel com duas metades iguais.
        if (1 << domain_bits) < size:
            half_bits += 1
            domain_bits += 2
        mask = (1 << half_bits) - 1
        value = index
        while True:
            left = value >> half_bits
            right = value & mask
            for round_id in range(6):
                round_key = cls._mix64(key + round_id * cls._ROUND_CONSTANT)
                left, right = right, left ^ (cls._mix64(right ^ round_key) & mask)
            value = (left << half_bits) | right
            if value < size:
                return value

    def _start_epoch(self):
        self._epoch += 1
        self._next_block_position = 0
        self._current_shard = None
        self._current_offset_position = 0
        # Chaves diferentes tornam cada epoca uma ordem independente, mas o
        # resultado continua reprodutivel para seed e numero de batches iguais.
        self._block_key = self._mix64(self._seed ^ (self._epoch * 0xD1B54A32D192ED03))
        self._offset_key = self._mix64(self._seed ^ (self._epoch * 0x94D049BB133111EB))
        if self.verbose:
            print(f"[loader] epoca {self._epoch} iniciada -> {self.total_samples:,} amostras")

    def _select_next_block(self):
        if self._epoch == 0 or self._next_block_position >= self.total_blocks:
            self._start_epoch()

        shuffled_block = self._permute_index(
            self._next_block_position, self.total_blocks, self._block_key
        )
        self._next_block_position += 1
        shard_id = int(np.searchsorted(self.block_boundaries, shuffled_block, side="right") - 1)
        local_block = shuffled_block - int(self.block_boundaries[shard_id])
        local_start = local_block * self.shuffle_block_samples
        remaining = int(self.samples_per_shard[shard_id]) - local_start

        self._current_shard = shard_id
        self._current_block_start = int(self._sample_starts[shard_id]) + local_start
        self._current_block_count = min(self.shuffle_block_samples, remaining)
        self._current_offset_position = 0

    def _next_sample_location(self):
        if self._current_shard is None or self._current_offset_position >= self._current_block_count:
            self._select_next_block()
        offset_in_block = self._permute_index(
            self._current_offset_position,
            self._current_block_count,
            self._offset_key ^ self._mix64(self._current_shard + self._current_block_start),
        )
        self._current_offset_position += 1
        return self._current_shard, self._current_block_start + offset_in_block

    # ------------------------------------------------------------------ #
    # Leitura e prefetch
    # ------------------------------------------------------------------ #
    def _build_batch(self):
        # Uma unica alocacao por tensor; evitamos astype por sample, que criava
        # 2 * batch_size arrays temporarios adicionais.
        x_np = np.empty((self.batch_size, self.block_size), dtype=np.int64)
        y_np = np.empty_like(x_np)
        for row in range(self.batch_size):
            shard_id, offset = self._next_sample_location()
            arr = self._mmaps[shard_id]
            x_np[row] = arr[offset : offset + self.block_size]
            y_np[row] = arr[offset + 1 : offset + self.block_size + 1]
            self._shard_access_count[shard_id] += 1

        x = torch.from_numpy(x_np)
        y = torch.from_numpy(y_np)
        if self.is_cuda:
            x = x.pin_memory()
            y = y.pin_memory()

        self._batches_produced += 1
        self._samples_produced += self.batch_size
        if self.verbose and self._batches_produced % self.log_every == 0:
            top_shards = np.argsort(self._shard_access_count)[::-1][: min(3, self.num_shards)]
            top_str = ", ".join(
                f"shard{s}={self._shard_access_count[s]}" for s in top_shards
            )
            print(
                f"[loader] batch #{self._batches_produced} | epoca {self._epoch} | "
                f"shards mais acessados: {top_str}"
            )
        return x, y

    def _producer_loop(self):
        try:
            while not self._stop_event.is_set():
                batch = self._build_batch()
                while not self._stop_event.is_set():
                    try:
                        self._queue.put(batch, timeout=0.2)
                        break
                    except queue.Full:
                        pass
        except Exception as error:
            self._producer_error = error
            # Entrega o erro depois dos batches ja enfileirados, sem nunca ficar
            # bloqueado indefinidamente se close() for chamado.
            while not self._stop_event.is_set():
                try:
                    self._queue.put(error, timeout=0.2)
                    break
                except queue.Full:
                    pass

    # ------------------------------------------------------------------ #
    # API publica e limpeza
    # ------------------------------------------------------------------ #
    def get_batch(self):
        if self._closed:
            raise RuntimeError("O loader ja foi fechado.")
        if self._producer_error is not None and self._queue.empty():
            raise RuntimeError("A thread de prefetch falhou.") from self._producer_error
        item = self._queue.get()
        if isinstance(item, Exception):
            raise RuntimeError("A thread de prefetch falhou.") from item
        x, y = item
        if self.is_cuda:
            return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
        return x.to(self.device), y.to(self.device)

    def __iter__(self):
        return self

    def __next__(self):
        return self.get_batch()

    def print_shard_stats(self):
        print(f"[loader] acessos por shard (total {self._batches_produced} batches):")
        for i, path in enumerate(self.shard_paths):
            print(f"  {os.path.basename(path)}: {self._shard_access_count[i]:,} amostras lidas")

    def _close_mmaps(self):
        for arr in self._mmaps:
            if arr is not None:
                arr._mmap.close()
        self._mmaps.clear()

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._stop_event.set()
        # Libera um produtor que esteja aguardando espaco na fila.
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            raise RuntimeError("A thread de prefetch nao encerrou; memmaps mantidos abertos por seguranca.")
        self._close_mmaps()
        if self.verbose:
            print("[loader] thread de prefetch e memmaps encerrados")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
