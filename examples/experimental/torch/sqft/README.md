# Setup

SQFT (Sparse Quantized Fine-Tuning) is an advanced end-to-end solution designed for the low-precision, sparse, and parameter-efficient fine-tuning of large pre-trained models (LPMs). 
This innovative approach facilitates effective model adaptation and manipulation, particularly in resource-constrained environments where computational efficiency is paramount.
The implementation of SQFT is supported by the Neural Network Compression Framework (NNCF). 
For detailed documentation on the implementation, please refer to the [SQFT Documentation](../../../../nncf/experimental/torch/sqft/SQFT.md).
To help you get started with SQFT, we have prepared a comprehensive quickstart guide. 
This [Quickstart Guide](./README.md) provides a step-by-step example of how to fine-tune a model using SQFT.

## NNCF and PyTorch

Install NNCF and PyTorch using the latest instructions in https://github.com/openvinotoolkit/nncf/blob/develop/docs/Installation.md.

The ```examples``` folder from the NNCF repository ***is not*** included when you install NNCF using a package manager. To run SQFT with Neural LoRA Search (NLS) examples, you will need to obtain this folder from the repository and add it to your path.

## Additional Dependencies

The examples in the NNCF repo have additional requirements, such as transformers, peft, etc., which are not installed with NNCF. You will need to install them using:

```bash
pip install 'numpy<2.0.0' setuptools==69.5.1 wheel
# transformers for Neural Low-rank Adapter Search (NLS)
git clone https://github.com/huggingface/transformers.git && cd transformers && git checkout v4.44.2
git apply --ignore-space-change --ignore-whitespace <path to NNCF>/examples/experimental/torch/sqft/transformers.patch && pip install -e .

pip install peft==0.10.0
pip install datasets accelerate sentencepiece protobuf
```

## SQFT Fine-tuning Example

We use Mistral-7B-v0.3 + GSM8K as an example to show the SQFT fine-tuning example. 
See the original [SQFT repo](https://github.com/IntelLabs/Hardware-Aware-Automated-Machine-Learning/tree/main/SQFT) for other settings and models.

SQFT mainly includes:

1. Sparsifies and quantizes a pre-trained model.

2. Fine-tunes this model using Neural Low-rank Adapter Search (NLS) with SparsePEFT to generate a trained weight-sharing super-adapter.

3. Searches for optimal sub-adapters via the advanced search algorithm or Returns the heuristic sub-adapter without any searching.

#### Sparsification and Quantization

Before fine-tuning, SQFT needs to employ some sparsification and quantization methods to compress the model serving as the base model (frozen) before adapter fine-tuning.
For sparsification, SQFT employs a simple but effective pruning approach [Wanda](https://github.com/locuslab/wanda) to sparsify the language model.
Note that the sparsifying step can use any other weight sparsity algorithm. 
Feel free to try other sparse approaches for the base model before training.
For quantization (optional), SQFT applies [GPTQ](https://arxiv.org/abs/2210.17323) or [AWQ](https://arxiv.org/abs/2306.00978) to quantized the sparse model. 
For more details about Sparsification and Quantization, refer to the original [SQFT repo](https://github.com/IntelLabs/Hardware-Aware-Automated-Machine-Learning/tree/main/SQFT).
Next, we will use the compressed model [IntelLabs/sqft-mistral-7b-v0.3-50-base](https://huggingface.co/IntelLabs/sqft-mistral-7b-v0.3-50-base) as an example.

#### Neural Low-rank Adapter Search (NLS) Finetuning

After obtaining the compressed base model, SQFT applied Neural Low-rank Adapter Search (NLS) training strategy to make the model adapted to a target downstream dataset.

To run super-adapter training, use the ```run_sqft.py``` script located [here](./run_sqft.py) and the sample SQFT NNCF configuration ```config.json``` from [here](../nls_examples/config.json).

The file ```config.json``` contains a sample configuration for generating a super-adapter ([more details](../../../../nncf/experimental/torch/sqft/SQFT.md)). 
The target is to generate a super-adapter on the sparse Mistral-7B-v0.3 trained with [GSM8K](https://huggingface.co/datasets/gsm8k). 
The file should be modified depending on the model and LoRA configs to be used as input for NLS.

Here is an example command to train a super-adapter:

```bash
cd <path to NNCF>/examples/experimental/torch/sqft
python run_sqft.py \
    --model_name_or_path IntelLabs/sqft-mistral-7b-v0.3-50-base \
    --do_train \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 3 \
    --learning_rate 3e-4 \
    --warmup_steps 100 \
    --optim adamw_torch \
    --fp16 \
    --output_dir <path to super-adapter> \
    --logging_steps 20 \
    --save_strategy epoch \
    --save_total_limit 1 \
    --lora \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --target_modules q_proj,k_proj,v_proj \
    --nncf_config sqft_examples/config.json \
    --search_space 8,6,4
```

The output of running ```run_sqft.py``` will be a super-adapter, which includes multiple high-performing sub-adapters.
