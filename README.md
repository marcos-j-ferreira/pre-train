# Objetivo e observações

Realizar o pré-treinamento de um modelo de linguagem, usando de forma eficiente o hardware disponível, manipulação dos dados (com janelas deslizantes aleatórias e carregamento eficiente), forward/backward do modelo.

> Treinamento não "coloca conhecimento" diretamente no modelo. Ele modifica os parâmetros para que o modelo represente uma distribuição estatística aprendida a partir dos dados.

Os parâmetros são inicializados de forma aleatória e, à medida que o modelo é exposto a distribuições de dados muito diversas e ricas, ele "aprende" — o que, na prática, se resume a:

> Aprender a prever o próximo token.

O objetivo explícito de treino é a previsão de tokens; as capacidades mais complexas (resumir, traduzir, responder perguntas etc.) surgem como consequência emergente dessa otimização sobre dados suficientemente ricos e diversos.

Ao observar enormes quantidades de sequências, o modelo começa a construir representações internas que capturam:

- relações entre palavras;
- sintaxe;
- semântica;
- padrões linguísticos;
- fatos presentes nos dados;
- estruturas de documentos;
- padrões de código;
- relações matemáticas;
- estilos;
- padrões de raciocínio presentes nos dados;
- associações entre conceitos.

---

*Esse ponto me chama atenção:*

> Assim como o cérebro humano, ao ser exposto a muitos exemplos de conhecimento sobre os mais diversos temas, começa a "generalizar" — o modelo segue uma lógica parecida (sem que isso seja uma comparação direta): ao observar uma variedade enorme e rica de dados, ele aprende a "generalizar", o que é o que torna esse tipo de modelo útil — a capacidade de combinar n pontos de conhecimento em algo novo.

*Generalização emerge de:*

1. **Diversidade dos dados vistos** — diferentes exemplos, contextos e variações.
2. **Riqueza informacional dos dados** — dados que contêm padrões relevantes e relações úteis.
3. **Capacidade do modelo de aprender** — capacidade de extrair e representar esses padrões sem apenas memorizar os exemplos.

*Experimentos realizados*

Não adianta usar o dump da Wikipédia em um modelo pequeno: ele não terá capacidade (número de parâmetros) suficiente para absorver tantos artigos e informações. Da mesma forma, não adianta treinar só com Wikipédia e esperar um modelo com alta capacidade de generalização — o conhecimento enciclopédico é denso em fatos, mas pouco diverso em estilo e estrutura. É necessário um dataset diverso, com conversas, artigos, fóruns, código etc. Assim o modelo aprende conhecimento, sintaxe e língua de forma mais equilibrada.

Da mesma forma, não adianta usar um modelo muito grande com poucos dados: existe uma relação empírica entre tamanho do modelo e volume de dados — as **Chinchilla scaling laws** (Hoffmann et al., 2022), que propõem algo próximo de **20 tokens de treino por parâmetro** como ponto de treino compute-ótimo (ou seja, o melhor uso do orçamento de computação disponível, não necessariamente o teto de qualidade do modelo).

O GPT-2 foi visto como estado da arte porque, mesmo sem qualquer etapa de fine-tuning, já era capaz de responder perguntas, resumir, traduzir e realizar diversas outras tarefas em modo *zero-shot* — algo pouco comum em modelos de linguagem anteriores a ele.

Ao final do pré-treinamento, o que se tem é um **auto-completador sofisticado** — dizendo de forma um pouco crua: o modelo sai como uma "massinha", cheio de conhecimento, conteúdo, sintaxe e língua, mas ainda sem instrução de como usar isso. É só no SFT (*Supervised Fine-Tuning*) que o modelo se torna de fato utilizável para chat, conversas, código, Q&A, entre outros usos.

---

*De forma resumida*

**Após o pré-treinamento você terá:**

> um modelo probabilístico capaz de produzir sequências segundo a distribuição que aprendeu.

# Resumo

Um pré-treinamento de modelo se divide em 4 partes principais. Quais são elas e como foram implementadas:

### Modelo

Foram desenvolvidas duas arquiteturas e suas variações, **MoE** e **GPT-2**, ambas baseadas no paper *"Attention Is All You Need"*, usando somente a parte **decoder-only**. É uma arquitetura que permite contextos longos graças ao mecanismo de atenção, empilhando múltiplas camadas de blocos de atenção + feed-forward.

A arquitetura **MoE** (Mixture of Experts) só se destaca de fato em escala — com poucos parâmetros e pouco dado, o ganho real dessa arquitetura não aparece, apenas "faíscas" do comportamento esperado.

### Dados

Duas abordagens possíveis: preparar os próprios dados ou usar datasets prontos.

**Dados prontos**

O HuggingFace oferece muitos datasets prontos para uso, sendo necessárias apenas pequenas remoções de estrutura e a tokenização.

Alguns que usei para testar:

- [Hugging Face – FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) — bilhões de tokens
- [Hugging Face – FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) — bilhões de tokens
- [Hugging Face – TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) — aproximadamente 1 bilhão de tokens

O número de tokens final depende do tokenizador usado e do seu vocabulário (vocab size) — o mesmo texto gera contagens de tokens diferentes conforme o tokenizador escolhido.

**Preparar os próprios dados**

Essa pipeline envolve: coleta → remoção de estrutura (markup/HTML) → remoção de duplicatas → limpeza → outras etapas conforme o dado.

Realizei esse processo e preparei um dataset da Wikipédia pt-br: baixei um dump, executei toda essa pipeline e salvei o resultado no HuggingFace.

Nesse dataset você encontra:

1. O dataset completo — todos os artigos em pt-br da Wikipédia — em um único arquivo `.txt`.
2. O mesmo dataset em formato `.jsonl` (um artigo por linha).
3. O dataset já tokenizado, salvo em `.bin`, em `float16`/`uint16` para ocupar menos espaço em disco.

Esses dados já foram limpos, normalizados e salvos:

[Hugging Face – dumpWiki](https://huggingface.co/datasets/marcos-j-leemes/wikipedia-pt-clean/tree/main) — ~739 milhões de tokens, com vocabulário de ~10 mil tokens (incluindo tokens especiais).

O tokenizador usado é o **BPE** (Byte Pair Encoding), com vocab size de 10 mil. É possível adaptar essa escolha — por exemplo, usando o tokenizador do GPT-2 disponível na biblioteca `transformers`. Tanto o tokenizador quanto o dataset ficam a critério de quem for reproduzir o treino.

### Loop de treinamento

O loop foi desenvolvido para ser eficiente tanto no carregamento dos dados e na construção do modelo quanto no cálculo do loss e na correção dos pesos (backward + optimizer step). Foram adicionados logs para monitorar o treinamento e checkpoints automáticos, buscando aproveitar melhor o hardware disponível (uso de `torch.compile`, mixed precision, gradient accumulation e scheduler de LR com warmup + cosine decay, etc).

### Ambiente

Foram usados dois ambientes para treinar os modelos: meu ambiente local e o Google Colab.

O Colab, na versão gratuita, oferece GPUs (tipicamente uma **NVIDIA T4**, com cerca de 15–16 GB de VRAM) por tempo limitado. O limite de uso não é fixo nem publicado oficialmente pelo Google — ele é dinâmico, varia conforme demanda e disponibilidade, e pode mudar de um dia para o outro. Na prática, é comum a sessão ser interrompida após algumas horas de uso contínuo por dia. É uma GPU eficiente para o custo (zero), mas não aguenta modelos muito grandes: acima de ~20 milhões de parâmetros o treino já começa a ficar lento, sendo necessário avaliar o trade-off entre aumentar o contexto ou aumentar o modelo.

Meu ambiente local possui uma **GeForce GTX/GT 750** — uma placa bem antiga da NVIDIA, com 4 GB de VRAM — usada apenas para testes rápidos e validações, não para treinos completos.

# Como rodar/usar esse código

O primeiro passo é definir qual dataset será usado para o treinamento — esse é o ponto principal antes de qualquer outra configuração.

Clone o repositório na sua máquina:

```bash
git clone https://github.com/marcos-j-ferreira/pre-train.git
```

> Se você estiver acessando pelo Colab, também é possível abrir o terminal, clonar o repositório e navegar pelos arquivos de forma parecida com o ambiente local.

Em seguida, ajuste o `config.yaml` com:

1. Tamanho do modelo (número de camadas, heads, dimensão do embedding etc.)
2. Configurações de **global batch size**
3. Configurações do treino (passos, learning rate, warmup...)
4. Configurações de avaliação
5. Configurações de checkpoint

Depois de definir as configurações iniciais, é preciso baixar um dataset já tokenizado e colocá-lo na pasta `dataset/`. O nome usado no `config.yaml` deve bater com o nome real dos arquivos:

```yaml
dataset:
  shard_dir: "dataset/shards"
  shard_pattern: "shard_*.bin"
```

Depois disso, basta rodar o script principal do treino:

```bash
python main.py
```

No arquivo [`exemplo_log.txt`](./exemplo_log.txt) você encontra um exemplo do que deve aparecer no terminal durante o treinamento.

**Observações:**

- Verifique se a GPU que você vai usar tem suporte a `torch.compile` e a treino em precisão mista (AMP) — caso contrário, o treinamento pode falhar ou você precisa desativar essas opções no `config.yaml`.
- O tamanho do modelo e do batch precisa respeitar a VRAM disponível; ajuste `gradient accumulation` para simular um batch maior sem estourar a memória.

# Próximas implementações

- [ ] Implementar suporte a paralelismo com múltiplas GPUs (DDP)
- [ ] Permitir retomar o treinamento a partir de um checkpoint salvo (*resume training*)