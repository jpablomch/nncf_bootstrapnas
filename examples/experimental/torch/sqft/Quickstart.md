# Setup

## NNCF and PyTorch

Install NNCF and PyTorch using the latest instructions in https://github.com/openvinotoolkit/nncf/blob/develop/docs/Installation.md.

The ```examples``` folder from the NNCF repository ***is not*** included when you install NNCF using a package manager. To run SQFT with Neural LoRA Search (NLS) examples, you will need to obtain this folder from the repository and add it to your path.

## Additional Dependencies

The examples in the NNCF repo have additional requirements, such as transformers, peft, etc., which are not installed with NNCF. You will need to install them using:

```bash
# transformers
git clone https://github.com/huggingface/transformers.git && cd transformers && git checkout v4.44.2
git apply --ignore-space-change --ignore-whitespace <path to NNCF>/examples/experimental/torch/text_generation/transformers.patch && pip install -e .

pip install peft==0.10.0
pip install datasets accelerate sentencepiece protobuf
```

## Example

To run an example of super-adapter generation and sub-adapter search, use the ```bootstrap_nas.py``` script located [here](./run_nls.py) and the sample ```config.json``` from [here](../nls_examples/config.json).

The file ```config.json``` contains a sample configuration for generating a super-adapter. The sample file is configured to generate a super-network from [Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) trained with [GSM8K](https://huggingface.co/datasets/gsm8k). The file should be modified depending on the model to be used as input for NLS.

Use the following to test training a super-adapter:

```bash
cd <path to NNCF>/examples/experimental/torch/text_generation
python run_nls.py \
    --model_name_or_path meta-llama/Meta-Llama-3-8B \
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
    --nncf_config <path to this repo>/nls_examples/config.json \
    --search_space 8,6,4
```

The output of running ```run_nls.py``` will be a super-adapter, which includes multiple high-performing sub-adapters.
