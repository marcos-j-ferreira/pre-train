"""
Loop de treinamento - Para a arquitetura do do GPT-2 / Para o MoE será necessario algumas alterações
1 - Carrega o dataset via mmap para não ocupar muito espaço na memória.
2 - Constrói o modelo.
3 - Faz o forward e o backward.
4 - Faz uso inteligente do hardware com dtypes e torch.compile.
5 - Registra logs importantes: tokens vistos, tokens por segundo, loss, learning rate (LR), dt, entre outros.
6 - Salva checkpoints conforme as configurações definidas.
7 - Salva o modelo final em FP16 para ocupar menos espaço.

"""

import math
import torch
import numpy as np
import yaml
from typing import Dict, Any
from pathlib import Path
import time
import gc
import signal
import sys
import psutil

from utils.shardedBinDataLoaderStreaming import ShardedBinDataLoaderStreaming
from artefatos_vocab.tokenizador import  BPEPipeline
from model.transformers import ModeloCompleto



def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml"""
    config_path = Path(__file__).resolve().with_name("config.yaml")
    with config_path.open("r", encoding="utf8") as file:
        return yaml.safe_load(file)


# config.num_experts = 8           # nº de especialistas por camada
# config.num_experts_per_tok = 2   # quantos cada token ativa (top-k)
# config.moe_aux_loss_coef = 0.01  # peso da loss de balanceamento


# Configuração do modelo
class Config:
    def __init__(self, config_dict):
        self.vocab_size = config_dict.get("vocab_size", 10000)
        self.n_embd = config_dict.get("embedding_dim", 128)
        self.num_head = config_dict.get("num_heads", 2)
        self.num_layer = config_dict.get("num_layers", 2)
        self.dropout = config_dict.get("dropout", 0.0)
        self.bias = config_dict.get("bias", False)
        self.block_size = config_dict.get("block_size", config_dict.get("seq_len", 128))
        self.num_experts = config_dict.get("num_experts", 8)
        self.num_experts_per_tok = config_dict.get("num_experts_per_tok", 4)
        self.moe_aux_loss_coef = config_dict.get("moe_aux_loss_coef", 0.01)

        

# Save final
def save_model(model_save_path, model):
    """Salva o modelo em FP16"""
    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
    state_dict = model.state_dict()
    fp16_state_dict = {}
    for key, value in state_dict.items():
        if value.is_floating_point():
            fp16_state_dict[key] = value.half()
        else:
            fp16_state_dict[key] = value
    
    torch.save(fp16_state_dict, model_save_path)
    print(f"Modelo salvo em FP16: {model_save_path}")

def safe_exit():
    """Função para saída segura"""
    print("\nInterrupção detectada. Finalizando treino...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    sys.exit(0)

# Status do hardware
def get_resource_stats(device):
    stats = {}

    # RAM do processo (e do sistema)
    process = psutil.Process()
    stats["ram_used_gb"] = process.memory_info().rss / 1e9
    stats["ram_percent"] = psutil.virtual_memory().percent

    # GPU / VRAM
    if device.type == "cuda":
        stats["vram_alloc_gb"] = torch.cuda.memory_allocated() / 1e9
        stats["vram_reserved_gb"] = torch.cuda.memory_reserved() / 1e9
        stats["vram_max_alloc_gb"] = torch.cuda.max_memory_allocated() / 1e9
        # utilização (%) e VRAM total, via nvidia-smi
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            stats["gpu_util_percent"] = util.gpu
            stats["vram_total_gb"] = mem.total / 1e9
        except ImportError:
            pass

    return stats


# Learning rate dinamico - boa pratica
def get_lr( step: int, lr: float, warmup_steps: int, max_steps: int, min_lr: float) -> float:
    """Warmup linear seguido de cosine decay até min_lr."""
    if warmup_steps > 0 and step < warmup_steps:
        return min(lr * (step + 1) / warmup_steps, lr)

    if step >= max_steps:
        return min_lr

    decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    decay_ratio = min(max(decay_ratio, 0.0), 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    # Protege contra um erro de arredondamento de ponto flutuante na fronteira.
    return min(max(min_lr + coeff * (lr - min_lr), min_lr), lr)



def validate_training_config(max_steps: int, warmup_steps: int, min_lr: float, max_lr: float) -> None:
    """Impede um agendador inválido antes que o treino comece."""
    if max_steps <= 0:
        raise ValueError("treino.max_steps deve ser maior que zero.")
    if not 0 <= warmup_steps <= max_steps:
        raise ValueError("treino.warmup_steps deve estar entre 0 e treino.max_steps.")
    if not 0 < min_lr <= max_lr:
        raise ValueError("É necessário que 0 < treino.min_lr <= treino.learning_rate.")

    checkpoints = {0, max_steps - 1, max_steps}
    if warmup_steps:
        checkpoints.update({warmup_steps - 1, warmup_steps})
    values = [get_lr(step, max_lr, warmup_steps, max_steps, min_lr) for step in checkpoints]
    # Durante o warmup o LR começa abaixo de min_lr; depois dele, min_lr é o piso.
    if any(not 0 < value <= max_lr for value in values):
        raise RuntimeError("O agendador produziu um LR fora dos limites configurados.")
    if any(get_lr(step, max_lr, warmup_steps, max_steps, min_lr) < min_lr
           for step in checkpoints if step >= warmup_steps):
        raise RuntimeError("O agendador produziu um LR fora dos limites configurados.")


@torch.no_grad()
def estimate_loss(model, loader, num_batches: int, device, amp_dtype, use_amp: bool) -> float:
    """Calcula a loss media em batches reservados para validacao."""
    was_training = model.training
    model.eval()
    losses = []
    for _ in range(num_batches):
        x, y = loader.get_batch()
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            _, loss = model(x, y)
        losses.append(loss.item())
    if was_training:
        model.train()
    return sum(losses) / len(losses)


def main():
    # Configurar signal handlers para interrupção segura
    signal.signal(signal.SIGINT, lambda sig, frame: safe_exit())
    signal.signal(signal.SIGTERM, lambda sig, frame: safe_exit())
    
    # Carregar as configurações do arquivo config.yaml
    config = load_config()
    
    config_model = config.get("model", {})
    config_training = config.get("treino", {})
    print("\nConfigurações carregadas")
    print(f"vocab size: {config_model.get('vocab_size')}")
    
    # Environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    
    # Build model
    config_obj = Config(config_model)
    model = ModeloCompleto(config_obj).to(device)
    
    # Parâmetros do modelo
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total de parâmetros do modelo: {total_params:,} | type {next(model.parameters()).dtype}")
    
    # Configuração do treinamento 
    block_size = config_model.get("block_size") # janela de contexto dos tokens
    max_step = config_training.get("max_steps", 100000) # número de atualizações que será feitas
    batch_size = config_training.get("batch_size", 32)  # número de amostras por passos, block_size * batch = 1 passo
    grad_accum_steps = max(1, config_training.get("grad_accum_steps", 1))  # (block_size * batch) * grad_acc = batch global = 1 passo
    print_interval = config_training.get("print_interval", 1)  # 
    warmup_steps = config_training.get("warmup_steps", 1000)  
    min_lr = float(config_training.get("min_lr", 1e-5))
    max_lr = float(config_training.get("learning_rate", 1e-3))
    validate_training_config(max_step, warmup_steps, min_lr, max_lr)
    use_amp = config_training.get("use_amp", True) and device.type == "cuda"  # hardware
    amp_dtype = torch.bfloat16 if config_training.get("use_bfloat16", False) else torch.float16 # hardware
    use_grad_scaler = use_amp and amp_dtype == torch.float16 # hardware
    compile_model = config_training.get("torch_compile", False) # hardware

    # Save 
    eval_interval = int(config_training.get("eval_interval", 200)) 
    eval_batches = int(config_training.get("eval_batches", 20))
    checkpoint_interval = int(config_training.get("checkpoint_interval", eval_interval))
    checkpoint_dir = Path(config_training.get("checkpoint_dir", "checkpoints"))
    early_stopping_patience = int(config_training.get("early_stopping_patience", 10))
    early_stopping_min_delta = float(config_training.get("early_stopping_min_delta", 0.0))
    if eval_interval <= 0 or eval_batches <= 0 or checkpoint_interval <= 0:
        raise ValueError("Os intervalos de avaliacao/checkpoint e eval_batches devem ser maiores que zero.")
    if early_stopping_patience <= 0 or early_stopping_min_delta < 0:
        raise ValueError("early_stopping_patience deve ser positivo e min_delta nao pode ser negativo.")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_grad_scaler)
    print(f"lr_min: {min_lr} | max: {max_lr}")
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=max_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=float(config_training.get("weight_decay", 0.1)),
    )

    config_dataset = config.get("dataset", {})
    validation_fraction = float(config_dataset.get("validation_fraction", 0.05))
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("dataset.validation_fraction deve estar entre 0 e 1.")
    loader_kwargs = {
        "shard_dir": config_dataset.get("shard_dir", "dataset/shards"),
        "pattern": config_dataset.get("shard_pattern", "shard_*.bin"),
        "block_size": block_size,
        "batch_size": batch_size,
        "dtype_dataset": np.uint16,
        "device": device,
        "prefetch_batches": config_training.get("prefetch_factor", 10),
        "verbose": False,
        "log_every": 10,
        "shuffle_block_samples": config_training.get("block_shuffle_size", 65_536),
    }
    seed = int(config_dataset.get("seed", 42))

    loader = ShardedBinDataLoaderStreaming(
        seed=seed, start_fraction=0.0, end_fraction=1.0 - validation_fraction, **loader_kwargs
    )
    val_loader = ShardedBinDataLoaderStreaming(
        seed=seed + 1, start_fraction=1.0 - validation_fraction, end_fraction=1.0, **loader_kwargs
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    
    step = 0
    stop_training = False

    start_time = time.time()
    
    effective_batch_size = batch_size * grad_accum_steps  # NOVO
    print(f"Iniciando treinamento por {max_step:,} steps...")
    print(f"Batch size: {batch_size} | Grad accum: {grad_accum_steps} | "
          f"Batch efetivo: {effective_batch_size} | AMP: {use_amp}, Device: {device}")

    # compile the model
    raw_model = model
    if compile_model:
        print("compiling the model... (takes a ~minute)")
        model = torch.compile(model) # requires PyTorch 2.0

    tokens_total = 0
    best_val_loss = float("inf")
    patience_counter = 0
    last_val_loss = None

    prompts = config_training.get("prompts", "Brasil")
    infer = config_training.get("infer", False)
    inter = config_training.get("inter", 1000)

    if infer:
        pipe = BPEPipeline().load_vocab("artefatos_vocab")

    while step < max_step and not stop_training:
        optimizer.zero_grad(set_to_none=True)

        lr = get_lr(step, lr=max_lr, warmup_steps=warmup_steps, max_steps=max_step, min_lr=min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        tokens_this_step = 0
        step_loss_accum = 0.0  # soma da loss "real" (não escalada) dos micro-steps

        step_start = time.perf_counter()

        # --- Loop de microbatch / gradient accumulation ---
        for micro_step in range(grad_accum_steps):
            x, y = loader.get_batch()
            tokens_this_step += x.numel()

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                _, loss = model(x, y)

            if not torch.isfinite(loss):
                raise FloatingPointError(f"Loss não finita no step {step}: {loss.item()}")

            step_loss_accum += loss.item()

            # Escala a loss pelo número de accumulation steps para manter
            # a magnitude do gradiente equivalente a um batch efetivo único
            loss_scaled = loss / grad_accum_steps
            scaler.scale(loss_scaled).backward()


        loss_value = step_loss_accum / grad_accum_steps

        # --- Clipping / norma do gradiente ---
        grad_clip = config_training.get("grad_clip", 0)
        if use_grad_scaler:
            scaler.unscale_(optimizer)  # necessário antes do clip/leitura da norma real

        if grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        else:
            # sem clipping, mas ainda assim calcula a norma só para diagnóstico/log
            # (max_norm=inf faz o coeficiente de clip ser ~0, ou seja, não mexe nos grads)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"))

        scaler.step(optimizer)
        scaler.update()

        if device.type == "cuda":
            torch.cuda.synchronize()

        step_time_ms = (time.perf_counter() - step_start) * 1000
        tok_per_sec = tokens_this_step / (step_time_ms / 1000)
        tokens_total += tokens_this_step

        #avg_loss = sum(janela) / len(janela)  # média móvel dos últimos 100 steps

        # --- Print de progresso ---
        if step % print_interval == 0 or step == max_step - 1:
            elapsed = time.time() - start_time
            lr_current = optimizer.param_groups[0]['lr']

            res = get_resource_stats(device)
            gpu_str = ""
            if device.type == "cuda":
                gpu_str = (f" | GPU: {res.get('gpu_util_percent', 0):.0f}%"
                        f" | VRAM: {res['vram_alloc_gb']:.2f}/{res.get('vram_total_gb', 0):.1f}GB")

                                                            # | Avg(100): {avg_loss:.4f}
            print(f"Step {step:2d} | Loss: {loss_value:.4f} | "
                f"LR: {lr_current:.10f} | Tempo: {elapsed/60:.1f}min"
                f" | Norm: {grad_norm:.4f} | tokens: {tokens_total:,.0f}"
                f" | tokens/sec {tok_per_sec:,.0f} | dt: {step_time_ms:.1f}ms"
                f" | RAM: {res['ram_used_gb']:.2f}GB ({res['ram_percent']:.0f}%)"
                f"{gpu_str}")

        #--- Validação + Early Stopping ---
        if step % eval_interval == 0 or step == max_step - 1:
            last_val_loss = estimate_loss(model, val_loader, eval_batches, device, amp_dtype, use_amp)
            loss_val = last_val_loss
            print(f"Step {step} | Loss de validação: {loss_val:.4f}")

            if loss_val < best_val_loss - early_stopping_min_delta:
                best_val_loss = loss_val
                patience_counter = 0
                best_model_path = checkpoint_dir / "best_model.pt"
                torch.save({
                    'step': step,
                    'model_state_dict': raw_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict() if use_grad_scaler else None,
                    'loss_val': loss_val,
                }, best_model_path)
                print(f"✓ Novo melhor modelo salvo em: {best_model_path} (loss: {loss_val:.4f})")
            else:
                patience_counter += 1
                print(f"Early Stopping: {patience_counter}/{early_stopping_patience} sem melhora")

            if patience_counter >= early_stopping_patience:
                print(f"Early stopping ativado no step {step}. Melhor loss: {best_val_loss:.4f}")
                stop_training = True

        # --- Checkpoint periódico ---
        if step % checkpoint_interval == 0 or step == max_step - 1 or stop_training:
            checkpoint = {
                'step': step,
                'model_state_dict': raw_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict() if use_grad_scaler else None,
                'loss_val': last_val_loss,
                'best_val_loss': best_val_loss,
                'patience_counter': patience_counter,
                'lr': lr,
            }
            checkpoint_path = checkpoint_dir / f"checkpoint_step_{step}.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint salvo em: {checkpoint_path}")

        # inferencias
        if infer and step % inter  == 0 and step > 5:
            print()
            model.eval()

            for _ in range(0, 3):
                with torch.no_grad():
                    input_ids = pipe.encode(prompts)
                    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)

                    output = model.generate(input_ids, max_new_tokens=100, temperature=1.0)
                    generated_text = pipe.decode(output[0].tolist())
                    print("Generated text:", generated_text)
                    print()

                    model.train()
        
        step += 1
        if step >= max_step:
            stop_training = True

    loader.close()
    val_loader.close()

    # Salvar modelo final
    model_save_path = config_training.get("model_save_path", "model_final.pt")
    save_model(model_save_path, raw_model)
    
    # Limpeza
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"Treinamento concluído! Modelo salvo em {model_save_path}")
    print(f"Total de steps: {step}, Tempo total: {(time.time() - start_time)/60:.1f}min")

if __name__ == "__main__":
    main()
