"""Pipeline minimal para BPE usando o binding `rustbpe`.

Organizado como uma classe `BPEPipeline`, com dois modos de uso:

1) Treino (precisa do rustbpe instalado):
    pipe = BPEPipeline()
    pipe.train("dados.txt", vocab_size=5000, special_tokens=["<|endoftext|>"])
    pipe.save("artifacts")
    pipe.tokenize_dataset("dados.txt", "dataset")

2) Inferência (não precisa retreinar nem do rustbpe):
    pipe = BPEPipeline.load_vocab("artifacts")
    ids = pipe.encode("texto qualquer")
    texto = pipe.decode(ids)

3) Adicionar tokens especiais após treino:
    pipe.add_special_tokens(["<|pad|>", "<|unk|>"])
    pipe.save("artifacts")  # persiste os novos tokens

Tokens especiais são tratados FORA do fluxo BPE, com prioridade absoluta
(igual ao tiktoken / GPT-4): nunca são divididos pelo encode normal.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import regex as re

try:
    from rustbpe import Tokenizer
except ImportError:          # rustbpe só é necessário para treinar
    Tokenizer = None         # type: ignore


# ---------------------------------------------------------------------------
# Constante de separação: usada no padrão de split de tokens especiais
# ---------------------------------------------------------------------------
_SPECIAL_TOKEN_RE_FLAGS = re.UNICODE


class BPEPipeline:
    """Treina, salva, carrega e usa um tokenizer BPE byte-level.

    Tokens especiais são gerenciados separadamente do vocabulário BPE:
    - Nunca são divididos pelo encode (prioridade absoluta).
    - Têm IDs próprios, contíguos a partir do vocab_size do BPE.
    - São persistidos em special_tokens.json dentro do diretório de artefatos.
    - Podem ser definidos no treino OU adicionados depois com add_special_tokens().
    """

    def __init__(self) -> None:
        # rustbpe.Tokenizer "vivo" — só existe logo após `train()`
        self.tok: Optional["Tokenizer"] = None

        # Estado serializável do tokenizer BPE
        self.token_bytes: List[bytes] = []            # id -> bytes do token
        self.merges: List[Tuple[int, int, int]] = []  # (left_id, right_id, novo_id)
        self.pattern: Optional[str] = None            # regex de pré-tokenização

        # Tokens especiais: str -> id  (ex.: {"<|endoftext|>": 5000})
        # A ordem de inserção define os IDs — dict preserva ordem no Python 3.7+
        self._special_tokens: Dict[str, int] = {}
        # Inverso: id -> str, reconstruído em _build_lookup_tables()
        self._special_tokens_inv: Dict[int, str] = {}

        # Tabelas auxiliares construídas por _build_lookup_tables()
        self._bytes_to_id: Dict[bytes, int] = {}
        self._max_token_len: int = 1
        self._compiled_pattern = None
        # Regex que divide o texto nos tokens especiais (reconstruído dinamicamente)
        self._special_split_re = None

    # ------------------------------------------------------------------
    # Propriedades
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Tamanho do vocabulário BPE (sem contar tokens especiais)."""
        return len(self.token_bytes)

    @property
    def full_vocab_size(self) -> int:
        """Tamanho total: BPE + tokens especiais."""
        return len(self.token_bytes) + len(self._special_tokens)

    @property
    def vocab(self):

        #  tokens = list(self.pipe.vocab.keys())
        return {i: b.hex() for i, b in enumerate(self.token_bytes) if b}
    
    # ------------------------------------------------------------------
    # Treino
    # ------------------------------------------------------------------

    def train(
        self,
        dataset_path: str,
        vocab_size: int,
        special_tokens: Optional[List[str]] = None,
    ) -> "BPEPipeline":
        """Treina o BPE e, opcionalmente, registra tokens especiais.

        Args:
            dataset_path: Caminho para o arquivo de texto de treino.
            vocab_size:   Tamanho do vocabulário BPE (sem tokens especiais).
            special_tokens: Lista de strings a serem tratadas como tokens
                            especiais (nunca divididas pelo BPE).
        """
        if Tokenizer is None:
            raise RuntimeError("rustbpe não está instalado; treino indisponível.")

        self.tok = Tokenizer()
        with open(dataset_path, "r", encoding="utf8") as f:
            self.tok.train_from_iterator(f, vocab_size)

        self.pattern = self.tok.get_pattern()
        ranks = self.tok.get_mergeable_ranks()   # list[(bytes, id)]
        self._build_from_ranks(ranks)

        if special_tokens:
            self.add_special_tokens(special_tokens)

        return self

    def _build_from_ranks(self, ranks: List[Tuple[bytes, int]]) -> None:
        """Reconstrói token_bytes e merges a partir de get_mergeable_ranks()."""
        ranks_sorted = sorted(ranks, key=lambda x: x[1])

        token_bytes: List[bytes] = [bytes([i]) for i in range(256)]
        bytes_to_id: Dict[bytes, int] = {b: i for i, b in enumerate(token_bytes)}

        merges: List[Tuple[int, int, int]] = []
        for b, idx in ranks_sorted:
            if idx < len(token_bytes):
                token_bytes[idx] = b
                bytes_to_id[b] = idx
                continue

            for split in range(1, len(b)):
                left_b, right_b = b[:split], b[split:]
                if left_b in bytes_to_id and right_b in bytes_to_id:
                    merges.append((bytes_to_id[left_b], bytes_to_id[right_b], idx))
                    break

            if idx >= len(token_bytes):
                token_bytes.extend([b""] * (idx - len(token_bytes) + 1))
            token_bytes[idx] = b
            bytes_to_id[b] = idx

        self.token_bytes = token_bytes
        self.merges = merges
        self._build_lookup_tables()


    def _build_lookup_tables(self) -> None:
        """Reconstrói todas as tabelas de busca rápida."""
        self._bytes_to_id = {b: i for i, b in enumerate(self.token_bytes) if b}
        self._max_token_len = max((len(b) for b in self.token_bytes if b), default=1)
        self._compiled_pattern = re.compile(self.pattern) if self.pattern else None
        self._rebuild_special_tables()

    def _rebuild_special_tables(self) -> None:
        """Reconstrói as tabelas de tokens especiais e o regex de split."""
        self._special_tokens_inv = {v: k for k, v in self._special_tokens.items()}

        if self._special_tokens:
            # Ordena do maior para o menor para que o match seja guloso (longest-first)
            sorted_tokens = sorted(self._special_tokens.keys(), key=len, reverse=True)
            pattern = "(" + "|".join(re.escape(t) for t in sorted_tokens) + ")"
            self._special_split_re = re.compile(pattern, _SPECIAL_TOKEN_RE_FLAGS)
        else:
            self._special_split_re = None

    # ------------------------------------------------------------------
    # Tokens especiais — API pública
    # ------------------------------------------------------------------

    def add_special_tokens(self, tokens: List[str]) -> Dict[str, int]:
        """Adiciona tokens especiais ao tokenizer.

        Pode ser chamado a qualquer momento: durante o treino ou depois
        de carregar um vocabulário existente.  IDs novos são atribuídos
        de forma contígua a partir de `full_vocab_size` atual.

        Args:
            tokens: Lista de strings (ex. ["<|pad|>", "<|unk|>"]).

        Returns:
            Dicionário {token: id} com TODOS os tokens especiais registrados
            após a operação (incluindo os que já existiam).
        """
        for token in tokens:
            if token in self._special_tokens:
                continue   # já existe — preserva o ID original
            new_id = self.vocab_size + len(self._special_tokens)
            self._special_tokens[token] = new_id

        self._rebuild_special_tables()
        return dict(self._special_tokens)

    def get_special_token_id(self, token: str) -> int:
        """Retorna o ID de um token especial ou levanta KeyError."""
        if token not in self._special_tokens:
            raise KeyError(f"Token especial não encontrado: {token!r}")
        return self._special_tokens[token]

    def get_special_tokens(self) -> Dict[str, int]:
        """Retorna cópia do mapeamento {token: id} de tokens especiais."""
        return dict(self._special_tokens)

    # ------------------------------------------------------------------
    # Persistência
    # ------------------------------------------------------------------

    def save(self, outdir: str) -> None:
        """Salva vocab.json, merges.txt, meta.json e special_tokens.json."""
        os.makedirs(outdir, exist_ok=True)

        # --- vocab.json ---------------------------------------------------
        vocab = {}
        for i, b in enumerate(self.token_bytes):
            if not b:
                continue
            entry = {"bytes": b.hex()}
            try:
                entry["text"] = b.decode("utf-8")
            except UnicodeDecodeError:
                entry["text"] = None
            vocab[i] = entry

        with open(os.path.join(outdir, "vocab.json"), "w", encoding="utf8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)

        # --- merges.txt ---------------------------------------------------
        with open(os.path.join(outdir, "merges.txt"), "w", encoding="utf8") as f:
            for left, right, result in self.merges:
                f.write(f"{left} {right} {result}\n")

        # --- meta.json ----------------------------------------------------
        meta = {"pattern": self.pattern, "vocab_size": self.vocab_size}
        with open(os.path.join(outdir, "meta.json"), "w", encoding="utf8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        # --- special_tokens.json ------------------------------------------
        # Persiste apenas se houver tokens definidos; sobrescreve sempre
        # para refletir adições feitas com add_special_tokens() pós-treino.
        with open(os.path.join(outdir, "special_tokens.json"), "w", encoding="utf8") as f:
            json.dump(self._special_tokens, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_vocab(cls, outdir: str) -> "BPEPipeline":
        """Carrega todos os artefatos e devolve um BPEPipeline pronto para uso."""
        pipe = cls()

        # --- vocab.json ---------------------------------------------------
        with open(os.path.join(outdir, "vocab.json"), "r", encoding="utf8") as f:
            vocab = json.load(f)

        max_id = max(int(i) for i in vocab.keys())
        token_bytes: List[bytes] = [b""] * (max_id + 1)
        for i_str, entry in vocab.items():
            token_bytes[int(i_str)] = bytes.fromhex(entry["bytes"])

        for i in range(256):
            if not token_bytes[i]:
                token_bytes[i] = bytes([i])

        # --- merges.txt ---------------------------------------------------
        merges: List[Tuple[int, int, int]] = []
        with open(os.path.join(outdir, "merges.txt"), "r", encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                left, right, result = (int(x) for x in line.split())
                merges.append((left, right, result))

        # --- meta.json ----------------------------------------------------
        with open(os.path.join(outdir, "meta.json"), "r", encoding="utf8") as f:
            meta = json.load(f)

        pipe.token_bytes = token_bytes
        pipe.merges = merges
        pipe.pattern = meta["pattern"]
        pipe._build_lookup_tables()   # inclui _rebuild_special_tables()

        # --- special_tokens.json (opcional) --------------------------------
        special_path = os.path.join(outdir, "special_tokens.json")
        if os.path.exists(special_path):
            with open(special_path, "r", encoding="utf8") as f:
                pipe._special_tokens = json.load(f)
            pipe._rebuild_special_tables()

        return pipe

    # ------------------------------------------------------------------
    # Encode / Decode
    # ------------------------------------------------------------------

    def encode(self, text: str, allowed_special: str = "all") -> List[int]:
        """Codifica `text` em IDs.

        Args:
            text: Texto a codificar.
            allowed_special: Controla como tokens especiais são tratados.
                "all"  — todos os tokens especiais são reconhecidos (padrão).
                "none" — nenhum token especial é reconhecido; o texto que
                         coincide com um token especial é tokenizado pelo BPE
                         normal (como se fossem caracteres comuns).
                set    — conjunto de strings: apenas esses tokens especiais
                         são reconhecidos.
        """
        if self.tok is not None:
            # Sessão de treino ativa: usa o rustbpe diretamente para os
            # tokens BPE, depois aplica tokens especiais sobre o resultado.
            # (rustbpe não conhece nossos tokens especiais extras)
            return self._encode_with_special(text, allowed_special)

        return self._encode_with_special(text, allowed_special)

    def decode(self, ids: List[int]) -> str:
        """Decodifica IDs em texto.

        Tokens especiais são reconstruídos como suas strings originais.
        """
        parts: List[bytes] = []
        for i in ids:
            if i in self._special_tokens_inv:
                # Flush dos bytes acumulados antes do token especial
                parts.append(self._special_tokens_inv[i].encode("utf-8"))
            elif 0 <= i < len(self.token_bytes):
                parts.append(self.token_bytes[i])
            else:
                raise ValueError(f"ID desconhecido: {i}")
        return b"".join(parts).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # Encode interno
    # ------------------------------------------------------------------

    def _encode_with_special(
        self, text: str, allowed_special: str | set
    ) -> List[int]:
        """Divide o texto nos tokens especiais permitidos, depois aplica
        o BPE normal nas partes que sobram.  Estratégia idêntica à do
        tiktoken."""
        # Decide quais tokens especiais participam desta chamada
        if allowed_special == "all":
            active = self._special_tokens
        elif allowed_special == "none" or not self._special_tokens:
            active = {}
        else:
            # allowed_special é um set/list de strings
            active = {t: self._special_tokens[t] for t in allowed_special
                      if t in self._special_tokens}

        if not active:
            # Caminho rápido: nenhum token especial — BPE direto
            return self._encode_bpe(text)

        # Reconstrói o regex apenas para os tokens ativos (cache implícito
        # se for o conjunto completo, reutilizando self._special_split_re)
        if active is self._special_tokens:
            split_re = self._special_split_re
        else:
            sorted_tokens = sorted(active.keys(), key=len, reverse=True)
            pattern = "(" + "|".join(re.escape(t) for t in sorted_tokens) + ")"
            split_re = re.compile(pattern, _SPECIAL_TOKEN_RE_FLAGS)

        ids: List[int] = []
        for part in split_re.split(text):
            if part in active:
                ids.append(active[part])   # ID direto, sem passar pelo BPE
            elif part:
                ids.extend(self._encode_bpe(part))
        return ids

    def _encode_bpe(self, text: str) -> List[int]:
        """Tokeniza `text` com BPE puro (sem tratar tokens especiais)."""
        if self.tok is not None:
            # rustbpe vivo (sessão de treino) — usa encode nativo para velocidade
            return self.tok.encode(text)
        return self._encode_pure_python(text)

    def _encode_pure_python(self, text: str) -> List[int]:
        if self._compiled_pattern is None:
            raise RuntimeError(
                "Tokenizer sem padrão de pré-tokenização (pattern). "
                "Treine ou recarregue via load_vocab()."
            )
        ids: List[int] = []
        for chunk in self._compiled_pattern.findall(text):
            ids.extend(self._maximal_munch(chunk.encode("utf-8")))
        return ids

    def _maximal_munch(self, data: bytes) -> List[int]:
        """Greedy longest-match contra o vocabulário BPE."""
        ids: List[int] = []
        i, n = 0, len(data)
        while i < n:
            for length in range(min(self._max_token_len, n - i), 0, -1):
                piece = data[i : i + length]
                token_id = self._bytes_to_id.get(piece)
                if token_id is not None:
                    ids.append(token_id)
                    i += length
                    break
            else:
                raise ValueError(f"Byte não encontrado no vocabulário: {data[i]!r}")
        return ids
    
    def eos_id(self) -> Optional[int]:
        """Retorna o ID do token especial </s> se estiver registrado."""
        return self._special_tokens.get("<|eos|>")

    # ------------------------------------------------------------------
    # Tokenização de dataset -> train.bin / val.bin
    # ------------------------------------------------------------------

    def tokenize_dataset(
        self,
        dataset_path: str,
        outdir: str,
        split_ratio: float = 0.01,
        eos_token: Optional[str] = None,
    ) -> bool:
        """Tokeniza um dataset linha a linha e salva binários numpy.

        Args:
            dataset_path: Arquivo de texto de entrada.
            outdir:       Diretório de saída para train.bin (e val.bin).
            split_ratio:  Fração das linhas destinada à validação.
            eos_token:    Token especial a ser inserido após cada linha
                          (ex. "<|endoftext|>").  Deve estar em
                          add_special_tokens() previamente.
        """
        import random

        if eos_token is not None and eos_token not in self._special_tokens:
            raise ValueError(
                f"eos_token {eos_token!r} não está registrado. "
                "Chame add_special_tokens([eos_token]) antes."
            )
        eos_id = self._special_tokens.get(eos_token) if eos_token else None

        train_ids: List[int] = []

        with open(dataset_path, "r", encoding="utf8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                ids = self.encode(line)
                if eos_id is not None:
                    ids.append(eos_id)
                # Agora salva todas as linhas em train_ids sem divisão
                train_ids.extend(ids)

        os.makedirs(outdir, exist_ok=True)
        np.array(train_ids, dtype=np.uint16).tofile(
            os.path.join(outdir, "train.bin")
        )

        return True


        # Implementação a ser testado...
    def train_multiple(
        self,
        dataset_paths: List[str],
        vocab_size: int,
        special_tokens: Optional[List[str]] = None,
        num_workers: int = 4,
        max_chunks_per_file: int = 1000,
    ) -> "BPEPipeline":
        """Treina o BPE usando múltiplos arquivos de texto com streaming."""
        if Tokenizer is None:
            raise RuntimeError("rustbpe não está instalado; treino indisponível.")

        if not dataset_paths:
            raise ValueError("Lista de dataset_paths vazia.")

        print(f"[train_multiple] Iniciando treino com {len(dataset_paths)} arquivos")
        print(f"[train_multiple] Vocab size: {vocab_size}")

        # Cria um gerador que itera sobre todos os arquivos
        def text_generator():
            total_processed = 0
            for idx, filepath in enumerate(dataset_paths, 1):
                if not os.path.exists(filepath):
                    print(f"[train_multiple] Aviso: arquivo não encontrado: {filepath}")
                    continue
                    
                try:
                    file_size = os.path.getsize(filepath)
                    print(f"[train_multiple] Lendo arquivo {idx}/{len(dataset_paths)}: "
                        f"{os.path.basename(filepath)} ({file_size / (1024*1024):.2f} MB)")
                    
                    with open(filepath, "r", encoding="utf8") as f:
                        content = f.read()
                        if content.strip():
                            total_processed += 1
                            yield content
                            
                    print(f"  [train_multiple] Arquivo {idx} processado ({total_processed} até agora)")
                    
                except Exception as e:
                    print(f"[train_multiple] Erro ao ler {filepath}: {e}")
                    continue

        # Treina o tokenizador usando o gerador
        print(f"[train_multiple] Treinando tokenizador com streaming...")
        self.tok = Tokenizer()
        self.tok.train_from_iterator(text_generator(), vocab_size)

        print(f"[train_multiple] Treino concluído!")

        # Extrai os dados finais do tokenizer
        self.pattern = self.tok.get_pattern()
        ranks = self.tok.get_mergeable_ranks()
        self._build_from_ranks(ranks)

        # Adiciona tokens especiais se fornecidos
        if special_tokens:
            self.add_special_tokens(special_tokens)

        print(f"[train_multiple] Vocabulário final: {self.vocab_size} tokens BPE "
            f"+ {len(self._special_tokens)} especiais = {self.full_vocab_size} total")

        return self


# ---------------------------------------------------------------------------
# Exemplo de uso
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    texto = "Olá, mundo! Este é um teste do tokenizador BPE."

    pipe = BPEPipeline()


    pipe.train("saida.txt", vocab_size=10000, special_tokens=["<|endoftext|>"])
    pipe.save("artifacts")

    pipe.tokenize_dataset("saida.txt", "dataset", eos_token="<|endoftext|>")

    text_ids = pipe.encode(texto)
    print("Texto:", texto)
    print("IDs de tokens:", text_ids)
    print("Texto reconstruído:", pipe.decode(text_ids))

    print(pipe.eos_id())