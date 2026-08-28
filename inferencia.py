"""
interferência no modelo 

As informações são lidas do .yaml

"""


from pyexpat import model

import torch
import numpy as np
import yaml
from typing import Dict , Any
from pathlib import Path

def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml"""
    config_path = Path(__file__).resolve().with_name("config.yaml")
    with config_path.open("r", encoding="utf8") as file:
        return yaml.safe_load(file)

# Antes
#from model.transformers import ModeloCompleto
# novo
from model.transformers import ModeloCompleto

# Configuração do modelo
class Config:
    def __init__(self, config_dict):
        self.vocab_size = config_dict.get("vocab_size", 10000)
        self.n_embd = config_dict.get("embedding_dim", 128)
        self.num_head = config_dict.get("num_heads", 2)
        self.num_layer = config_dict.get("num_layers", 2)
        self.dropout = config_dict.get("dropout", 0.0)
        self.bias = config_dict.get("bias", False)
        self.block_size = config_dict.get("block_size", 128)  # tamanho da sequência de entrada


from artefatos_vocab.tokenizador import  BPEPipeline

def main():

    pipe = BPEPipeline().load_vocab("artefatos_vocab")
    config_model = load_config().get("model", {})


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModeloCompleto(Config(config_model)).to(device)


    model_save_path = "checkpoints/model_final.pth"

    if Path(model_save_path).exists():

        print(f"Carregando modelo de {model_save_path}")

        checkpoint = torch.load(
            model_save_path,
            map_location=device
        )

        # Caso o checkpoint seja diretamente o state_dict
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

    else:
        print(
            f"Arquivo de modelo não encontrado: "
            f"{model_save_path}. Treinando do zero."
        )


    prompts = "One day"
    # output
    # Generated text: texto de entrada, como já relatado no centenário do

    print()

    for _ in range(3):
        model.eval()
        with torch.no_grad():
            input_ids = pipe.encode(prompts)
            input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)



            output = model.generate(input_ids, max_new_tokens=50, temperature=1.0)
            generated_text = pipe.decode(output[0].tolist())
            print("Generated text:", generated_text)
            print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    main()