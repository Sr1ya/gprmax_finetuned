# %% [markdown]
# ### Installation

# %%
!pip install unsloth

# %% [markdown]
# ### Unsloth

# %%
from unsloth import FastLanguageModel
import torch
max_seq_length = 2048
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
  
)

# %% [markdown]
# We now add LoRA adapters so we only need to update 1 to 10% of all parameters!

# %%
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  
    loftq_config = None, 
)

# %%
gprmax_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = gprmax_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

from datasets import load_dataset
dataset = load_dataset("IraGia/gprMax_Train", split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True,)

# %%
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 8,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
    ),
)

# %%
# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# %%
trainer_stats = trainer.train()

# %%
# @title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

# %%
FastLanguageModel.for_inference(model) # Enable native 2x faster inference
inputs = tokenizer(
[
    gprmax_prompt.format(
        "write this input file", # instruction
        "The receiver should be placed at 4.5696, 58.6801, 36.3701 The internal resistance of the voltage source is 831.6273451640435 Ohms The discretisation step should be 1.5901 meters. I want the PML to be 9 cell thick The dimensions of the model are 15.8559 x 75.1634 x 39.5206 meters. The central frequency will be equal with 0.134 GHz Generate random layers The duration of the signal will be 2.2815719340168907e-05 seconds The polarization of the pulse should 'x' The pulse should be gaussianprime Tx is a Hertzian Dipole", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
tokenizer.batch_decode(outputs)

# %%
# gprmax_prompt
FastLanguageModel.for_inference(model) # Enable native 2x faster inference
inputs = tokenizer(
[
    gprmax_prompt.format(
        "Make this model", # instruction
        "Generate random layers The dimensions of the model are 2.3594 x 41.214 x 66.2451 meters. The excitation pulse should be gaussiandotdotnorm Tx is a Hertzian Dipole I want the PML to be 77 cell thick The duration of the signal will be 1.1431824134811208e-05 seconds The central frequency will be equal with 3877.271 MHz The source should be placed at the coordinates 0.2455, 20.9791, 49.2353 meters The internal resistance of the voltage source is 630.1484646516255 Ohms The receiver should be placed at the coordinates (0.2455, 20.9791, 49.2353) meters The polarization of the pulse should 'x'", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)

# %%
model.save_pretrained("lora_model")  # Local saving
tokenizer.save_pretrained("lora_model")

# %%
if False:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "lora_model", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference

# gprmax_prompt = You MUST copy from above!

inputs = tokenizer(
[
    gprmax_prompt.format(
        "What is a famous tall tower in Paris?", # instruction
        "", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)

# %%
model.save_pretrained_gguf("model", tokenizer, quantization_method = "f16")

# %%
!git clone --recursive https://github.com/ggerganov/llama.cpp

# %%
!git clone --recursive https://github.com/ggerganov/llama.cpp

# %%
!make clean -C llama.cpp

# %%
!sudo apt install cmake -y


# %%
!cmake --version


# %%



