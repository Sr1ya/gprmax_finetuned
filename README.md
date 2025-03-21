                                            This is Just Demo Usage.

---

# Fine-Tuning Meta-Llama-3.1-8B-Instruct-bnb-4bit on gprmax_train Dataset

This repository contains the code and instructions for fine-tuning the `Meta-Llama-3.1-8B-Instruct-bnb-4bit` model on the `gprmax_train` dataset using the `unsloth` library. The fine-tuning process was optimized for memory and time efficiency, leveraging an L4 GPU with 32GB of RAM and 8 vCPUs. The fine-tuned model has been uploaded to Hugging Face for public use.

## Model Details
- **Base Model**: `Meta-Llama-3.1-8B-Instruct-bnb-4bit`
- **Fine-Tuning Dataset**: `gprmax_train`
- **Fine-Tuning Framework**: `unsloth` (for memory and time efficiency)
- **Hardware**: L4 GPU (32GB RAM, 8 vCPU)
- **Quantization**: 4-bit (bnb)

## Hugging Face Model
The fine-tuned model is available on Hugging Face:  
[Link to Fine-Tuned Model]([https://huggingface.co/sriyaflows/gprmax_8])

## Requirements
To reproduce the fine-tuning process or use the fine-tuned model, ensure you have the following dependencies installed:

```bash
pip install unsloth
```

## Fine-Tuning Process

### 1. Prepare the Dataset
The `gprmax_train` dataset was preprocessed and formatted for instruction fine-tuning. Ensure the dataset is in the following format:

```json
{
  "instruction": "Your instruction here",
  "input": "Your input data here",
  "output": "Expected output here"
}
```

### 2. Load the Base Model
The base model `Meta-Llama-3.1-8B-Instruct-bnb-4bit` was loaded in 4-bit precision using the `bitsandbytes` library to reduce memory usage.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import bitsandbytes as bnb

model_name = "Meta-Llama-3.1-8B-Instruct-bnb-4bit"
model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

### 3. Fine-Tune with Unsloth
The `unsloth` library was used to optimize the fine-tuning process for memory and time efficiency.

```python
from unsloth import FastLanguageModel

# Load the model with unsloth
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# Prepare the dataset
from datasets import load_dataset
dataset = load_dataset("IraGia/gprMax_Train")

# Fine-tune the model
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank
    lora_alpha=32,  # LoRA alpha
    target_modules=["q_proj", "v_proj"],  # Target modules for LoRA
    lora_dropout=0.1,
    bias="none",
    use_gradient_checkpointing=True,
)

# Training arguments
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    save_steps=500,
    logging_dir="./logs",
    logging_steps=10,
    learning_rate=2e-5,
    fp16=True,
    optim="adamw_torch",
)

# Train the model
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
)

trainer.train()
```

### 4. Save and Upload the Model
After fine-tuning, the model was saved and uploaded to Hugging Face.

```python
model.save_pretrained("./fine-tuned-model")
tokenizer.save_pretrained("./fine-tuned-model")

# Upload to Hugging Face
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="./fine-tuned-model",
    repo_id="your-username/your-model-name",
    repo_type="model",
)
```

## Usage
To use the fine-tuned model, load it from Hugging Face:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "sriyaflows/gprmax_8"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Generate text
input_text = "Your input here"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Results
The fine-tuned model achieves improved performance on the `gprmax_train` dataset, with reduced memory usage and faster training times compared to traditional fine-tuning methods.

## License
This model is licensed under the same terms as the base `Meta-Llama-3.1-8B-Instruct-bnb-4bit` model. Refer to the [Hugging Face model card](https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct) for details.

## Acknowledgments
- Thanks to Hugging Face for the `transformers` library.
- Thanks to the `unsloth` team for their memory-efficient fine-tuning tools.
- Thanks to the creators of the `gprmax_train` dataset.

---

