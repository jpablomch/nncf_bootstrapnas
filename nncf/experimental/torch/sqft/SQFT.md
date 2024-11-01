# SQFT

An end-to-end solution for low-precision sparse parameter-efficient fine-tuning of large pre-trained models (LPMs). 
SQFT enables effective model manipulation in resource-constrained environments. 
For more details, please refer to the [paper](https://arxiv.org/abs/2410.03750).

SQFT mainly includes:

Step 1. Sparsifies and quantizes a pre-trained model.

Step 2. Fine-tunes this model using Neural Low-rank Adapter Search (NLS) with SparsePEFT to generate a trained weight-sharing super-adapter.

Step 3. Searches for optimal sub-adapters via the advanced search algorithm or Returns the heuristic sub-adapter without any searching.

Regarding Step 1, SQFT employs a simple but effective pruning approach [Wanda](https://arxiv.org/abs/2306.11695) to sparsify the model, 
and then applies [GPTQ](https://arxiv.org/abs/2210.17323) or [AWQ](https://arxiv.org/abs/2306.00978) to quantized the sparse model (optional), 
serving as the base model (frozen) for adapter training. 

For fine-tuning, to enable the elastic adapter with Neural Low-rank Adapter Search (NLS) training, we employ some of [BootstrapNAS](https://github.com/openvinotoolkit/nncf/tree/develop/nncf/experimental/torch/nas/bootstrapNAS) features, 
offering a range of compression algorithms tailored for optimizing neural networks. 
In this context, its role is to make the adapters elastic, allowing the extraction of the optimal sub-adapters from the trained super-adapter.

The parameters for generating, training, searching on the super-adapter are defined in a configuration file within two exclusive subsets of parameters for training and searching:

```json5
"sqft": {
        "training": {
            "algorithm": "neural_lora_search",
            "elasticity": {
                "available_elasticity_dims": ["width"],
                "width": {
                    "overwrite_groups": [
                        [
                            ...
                        ]
                    ],
                    "overwrite_groups_widths": [
                        ...
                    ]
                }
            }
        },
        "search": {
            ...
        }
    }
```

`"available_elasticity_dims": ["width"]` means the hidden size of the weight matrix, more precisely, it represents the low-rank size of the LoRA adapter.
In SQFT solution, the design of the low-rank search space is crucial, including the allocation of dependency groups and the design of the search space for each group. 
For example, adopting the grouping `[[Q, K, V], [Up], [Down]]` adapters in 0-th layer, with each group's search space being `[32, 24, 16]`, i.e.,

- `[Q, K, V]`: `[32, 24, 16]`
- `[Up]`: `[32, 24, 16]`
- `[Down]`: `[32, 24, 16]`

```json
"width": {
    "overwrite_groups": [
        [
            "PeftModelForCausalLM/LoraModel[base_model]/LlamaForCausalLM[model]/LlamaModel[model]/ModuleList[layers]/LlamaDecoderLayer[0]/LlamaAttention[self_attn]/Linear[q_proj]/ModuleDict[lora_A]/NNCFLinear[default]/linear_0",
            "PeftModelForCausalLM/LoraModel[base_model]/LlamaForCausalLM[model]/LlamaModel[model]/ModuleList[layers]/LlamaDecoderLayer[0]/LlamaAttention[self_attn]/Linear[k_proj]/ModuleDict[lora_A]/NNCFLinear[default]/linear_0",
            "PeftModelForCausalLM/LoraModel[base_model]/LlamaForCausalLM[model]/LlamaModel[model]/ModuleList[layers]/LlamaDecoderLayer[0]/LlamaAttention[self_attn]/Linear[v_proj]/ModuleDict[lora_A]/NNCFLinear[default]/linear_0"
        ],
        [
            "PeftModelForCausalLM/LoraModel[base_model]/LlamaForCausalLM[model]/LlamaModel[model]/ModuleList[layers]/LlamaDecoderLayer[0]/LlamaMLP[mlp]/Linear[up_proj]/ModuleDict[lora_A]/NNCFLinear[default]/linear_0"
        ],
        [
            "PeftModelForCausalLM/LoraModel[base_model]/LlamaForCausalLM[model]/LlamaModel[model]/ModuleList[layers]/LlamaDecoderLayer[0]/LlamaMLP[mlp]/Linear[down_proj]/ModuleDict[lora_A]/NNCFLinear[default]/linear_0"
        ]
    ],
    "overwrite_groups_widths": [
        [32, 24, 16], [32, 24, 16], [32, 24, 16]
    ]
}
```

Note that the length of groups should be equal to the length of the group widths, and we only set the output hidden 
size space of LoRA-A in the config, as the input hidden size of LoRA-B will be automatically pruned according to LoRA-A.

In the search section, you specify the search algorithm, e.g., `NSGA-II` and its parameters. For example:

```json
"search": {
    "algorithm": "NSGA2",
    "num_evals": 200,
    "population": 10,
}
```

List of parameters about search can refer to [BootstrapNAS.md](../nas/bootstrapNAS/BootstrapNAS.md):

For more information about SQFT and to cite this work, please refer to the following publications:


[SQFT: Low-cost Model Adaptation in Low-precision Sparse Foundation Models](https://arxiv.org/abs/2410.03750)

```bibtex
@article{munoz2024sqft,
  title = {SQFT: Low-cost Model Adaptation in Low-precision Sparse Foundation Models},
  author = {J. Pablo Muñoz and Jinjie Yuan and Nilesh Jain},
  journal = {The 2024 Conference on Empirical Methods in Natural Language Processing (Findings)},
  year = {2024},
  url = {}
}
```

[Shears: Unstructured Sparsity with Neural Low-rank Adapter Search](https://arxiv.org/abs/2404.10934)

```bibtex
@inproceedings{munoz-etal-2024-shears,
    title = "Shears: Unstructured Sparsity with Neural Low-rank Adapter Search",
    author = "Mu{\~n}oz, J. Pablo  and
      Yuan, Jinjie  and
      Jain, Nilesh",
    editor = "Yang, Yi  and
      Davani, Aida  and
      Sil, Avi  and
      Kumar, Anoop",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 6: Industry Track)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.naacl-industry.34",
    doi = "10.18653/v1/2024.naacl-industry.34",
    pages = "395--405",
    abstract = "Recently, several approaches successfully demonstrated that weight-sharing Neural Architecture Search (NAS) can effectively explore a search space of elastic low-rank adapters (LoRA), allowing the parameter-efficient fine-tuning (PEFT) and compression of large language models. In this paper, we introduce a novel approach called Shears, demonstrating how the integration of cost-effective sparsity and a proposed Neural Low-rank adapter Search (NLS) algorithm can further improve the efficiency of PEFT approaches. Results demonstrate the benefits of Shears compared to other methods, reaching high sparsity levels while improving or with little drop in accuracy, utilizing a single GPU for a pair of hours.",
}
```
