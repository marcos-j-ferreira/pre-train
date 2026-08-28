"""
Arquitetura base - DeepSeek

Define os componentes do modelo Transformer, incluindo atenção multi-cabeças,
normalização e uma camada Mixture of Experts (MoE) para substituir o MLP denso.

 Implementação moderna...

"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """LayerNorm com suporte a bias opcional."""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.ndim = ndim

    def forward(self, input):
        return F.layer_norm(input, (self.ndim,), self.weight, self.bias, 1e-5)


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = MultiHead(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        # MLP denso substituído por camada MoE
        self.mlp = MoELayer(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        mlp_out, aux_loss = self.mlp(self.ln_2(x))
        x = x + mlp_out
        return x, aux_loss


class MultiHead(nn.Module):
    """Implementação MultiHead Attention para teste."""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.num_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.num_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class Expert(nn.Module):
    """Um único especialista — mesma estrutura do MLP original."""

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class MoELayer(nn.Module):
    """
    Camada de Mixture of Experts com roteamento top-k (estilo Switch
    Transformer / Mixtral).

    Parâmetros esperados em config:
        num_experts            -> nº total de especialistas
        num_experts_per_tok    -> quantos especialistas cada token usa (top-k)
        moe_aux_loss_coef      -> peso da loss de balanceamento de carga
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.aux_loss_coef = getattr(config, 'moe_aux_loss_coef', 0.01)

        assert 1 <= self.top_k <= self.num_experts, \
            "num_experts_per_tok deve estar entre 1 e num_experts"

        # Rede de roteamento (gate): decide, para cada token, quais experts usar
        self.gate = nn.Linear(config.n_embd, self.num_experts, bias=False)

        # Banco de especialistas
        self.experts = nn.ModuleList([Expert(config) for _ in range(self.num_experts)])

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.view(-1, C)  # (B*T, C)
        num_tokens = x_flat.size(0)

        # Logits e probabilidades de roteamento
        router_logits = self.gate(x_flat)                     # (B*T, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)       # (B*T, num_experts)

        # Seleciona os top-k especialistas por token
        topk_probs, topk_idx = torch.topk(router_probs, self.top_k, dim=-1)  # (B*T, top_k)
        # Renormaliza os pesos dos experts escolhidos para somarem 1
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        out = torch.zeros_like(x_flat)

        # Processa token-a-token por especialista (implementação simples e didática;
        # para escala maior, prefira dispatch via índices/einsum ou kernels dedicados)
        for expert_id in range(self.num_experts):
            # Máscara: quais posições (token, slot-k) escolheram este expert
            mask = (topk_idx == expert_id)          # (B*T, top_k)
            if not mask.any():
                continue

            token_idx, slot_idx = mask.nonzero(as_tuple=True)
            if token_idx.numel() == 0:
                continue

            expert_input = x_flat[token_idx]
            expert_output = self.experts[expert_id](expert_input)

            weights = topk_probs[token_idx, slot_idx].unsqueeze(-1)
            out.index_add_(0, token_idx, expert_output * weights)

        out = out.view(B, T, C)

        # ---- Loss de balanceamento de carga (auxiliary load-balancing loss) ----
        # Incentiva o roteador a distribuir os tokens uniformemente entre os experts.
        # f_i: fração de tokens roteados (no top-1) para o expert i
        # P_i: probabilidade média (softmax) atribuída ao expert i
        with torch.no_grad():
            top1_idx = topk_idx[:, 0]
            expert_mask = F.one_hot(top1_idx, num_classes=self.num_experts).float()
            f = expert_mask.mean(dim=0)  # (num_experts,)

        P = router_probs.mean(dim=0)  # (num_experts,)
        aux_loss = self.aux_loss_coef * self.num_experts * torch.sum(f * P)

        return out, aux_loss


class ModeloCompleto(nn.Module):
    """Versão do modelo com camadas MoE no lugar do MLP denso."""

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # Embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        # Blocks
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layer)])

        # Final layer norm
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)

        # Language model head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.wte.weight = self.lm_head.weight

        # Inicialização
        self.apply(self._init_weights)

        # Inicialização especial para projeções residuais
        for name, param in self.named_parameters():
            if name.endswith('c_proj.weight'):
                torch.nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.num_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.config.block_size

        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)

        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)

        total_aux_loss = 0.0
        for block in self.blocks:
            x, aux_loss = block(x)
            total_aux_loss = total_aux_loss + aux_loss

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            # Soma a loss de balanceamento de carga dos experts à loss principal
            loss = ce_loss + total_aux_loss

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx