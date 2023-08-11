# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Prompt Tuning
# MAGIC This lesson introduces how to apply prompt tuning to your model of choice using [Parameter-Efficient Fine-Tuning (PEFT) library developed by HuggingFace](https://huggingface.co/docs/peft/index). This PEFT library supports multiple methods to reduce the number of parameters for fine-tuning, including prompt tuning and LoRA. For a complete list of methods, refer to their [documentation](https://huggingface.co/docs/peft/main/en/index#supported-methods). Only a subset of models and tasks are supported by this PEFT library for the time being, including GPT-2, LLaMA; for pairs of models and tasks supported, refer to this [page](https://huggingface.co/docs/peft/main/en/index#supported-models).
# MAGIC
# MAGIC
# MAGIC ### ![Dolly](https://files.training.databricks.com/images/llm/dolly_small.png) Learning Objectives
# MAGIC 1. Apply prompt tuning to your model of choice
# MAGIC 1. Fine-tune on your provided dataset
# MAGIC 1. Save and share your model to HuggingFace hub
# MAGIC 1. Conduct inference using the fine-tuned model
# MAGIC 1. Compare outputs from randomly- and text-initialized fine-tuned model vs. foundation model

# COMMAND ----------

# MAGIC %pip install peft==0.4.0

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC [Auto Classes](https://huggingface.co/docs/transformers/main/en/model_doc/auto#auto-classes) helps you automatically retrieve the relevant model and tokenizers, given the pre-trained models you are interested in using. 
# MAGIC
# MAGIC Causal language modeling refers to the decoding process, where the model predicts the next token based on only the tokens on the left. The model cannot see the future tokens, unlike masked language models that have full access to tokens bidirectionally. A canonical example of a causal language model is GPT-2. You also hear causal language models being described as autoregresssive as well. 
# MAGIC
# MAGIC API docs:
# MAGIC * [AutoTokenizer](https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer)
# MAGIC * [AutoModelForCausalLM](https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoModelForCausalLM)
# MAGIC
# MAGIC In this demo, we will be using `bigscience/bloomz-560m` as our **foundation** causal LM to generate text. You can read more about [`bloomz` model here](https://huggingface.co/bigscience/bloomz). It was trained on [multi-lingual dataset](https://huggingface.co/datasets/bigscience/xP3), spanning 46 languages and 13 programming langauges. The dataset covers a wide range of NLP tasks, including Q/A, title generation, text classification.

# COMMAND ----------

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "bigscience/bloomz-560m"

tokenizer = AutoTokenizer.from_pretrained(model_name)
foundation_model = AutoModelForCausalLM.from_pretrained(model_name)

# COMMAND ----------

# MAGIC %md
# MAGIC Before doing any fine-tuning, we will ask the model to generate a new phrase to the following input sentence. 

# COMMAND ----------

input1 = tokenizer("Two things are infinite: ", return_tensors="pt")

foundation_outputs = foundation_model.generate(
    input_ids=input1["input_ids"], 
    attention_mask=input1["attention_mask"], 
    max_new_tokens=7, 
    eos_token_id=tokenizer.eos_token_id
    )
print(tokenizer.batch_decode(foundation_outputs, skip_special_tokens=True))

# COMMAND ----------

# MAGIC %md
# MAGIC The output is not too bad. However, the dataset BLOOMZ is pre-trained on doesn't cover anything about inspirational English quotes. Therefore, we are going to fine-tune `bloomz-560m` on [a dataset called `Abirate/english_quotes`](https://huggingface.co/datasets/Abirate/english_quotes)  containing exclusively inspirational English quotes, with the hopes of using the fine-tuned version to generate more quotes later! 

# COMMAND ----------

from datasets import load_dataset

data = load_dataset("Abirate/english_quotes", cache_dir=DA.paths.datasets+"/datasets")

data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)
train_sample = data["train"].select(range(50))
display(train_sample) 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Onto fine-tuning: define PEFT configurations for random initialization
# MAGIC
# MAGIC Recall that prompt tuning allows both random and initialization of soft prompts or also known as virtual tokens. We will compare the model outputs from both initialization methods later. For now, we will start with random initialization, where all we provide is the length of the virtual prompt. 
# MAGIC
# MAGIC API docs:
# MAGIC * [PromptTuningConfig](https://huggingface.co/docs/peft/main/en/package_reference/tuners#peft.PromptTuningConfig)
# MAGIC * [PEFT model](https://huggingface.co/docs/peft/main/en/package_reference/tuners#peft.PromptTuningConfig)

# COMMAND ----------

from peft import  get_peft_model, PromptTuningConfig, TaskType, PromptTuningInit

peft_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    prompt_tuning_init=PromptTuningInit.RANDOM,
    num_virtual_tokens=4,
    tokenizer_name_or_path=model_name
)
peft_model = get_peft_model(foundation_model, peft_config)
print(peft_model.print_trainable_parameters())

# COMMAND ----------

# MAGIC %md
# MAGIC That's the beauty of PEFT! It allows us to drastically reduce the number of trainable parameters. Now, we can proceed with using [HuggingFace's `Trainer` class](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#trainer) and its [`TrainingArugments` to define our fine-tuning configurations](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments). 
# MAGIC
# MAGIC The `Trainer` class provides user-friendly abstraction to leverage PyTorch under the hood to conduct training. 

# COMMAND ----------

from transformers import TrainingArguments
import os

output_directory = os.path.join(DA.paths.working_dir, "peft_outputs")

if not os.path.exists(DA.paths.working_dir):
    os.mkdir(DA.paths.working_dir)
if not os.path.exists(output_directory):
    os.mkdir(output_directory)

training_args = TrainingArguments(
    output_dir=output_directory, # Where the model predictions and checkpoints will be written
    no_cuda=True, # This is necessary for CPU clusters. 
    auto_find_batch_size=True, # Find a suitable batch size that will fit into memory automatically 
    learning_rate= 3e-2, # Higher learning rate than full fine-tuning
    num_train_epochs=5 # Number of passes to go through the entire fine-tuning dataset 
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train
# MAGIC
# MAGIC We will also use `Data Collator` to help us form batches of inputs to pass in to the model for training. Go [here](https://huggingface.co/docs/transformers/main/en/main_classes/data_collator#data-collator) for documentation.
# MAGIC
# MAGIC Specifically, we will be using `DataCollatorforLanguageModeling` which will additionally pad the inputs to the maximum length of a batch since the inputs can have variable lengths. Refer to [API docs here](https://huggingface.co/docs/transformers/main/en/main_classes/data_collator#transformers.DataCollatorForLanguageModeling).
# MAGIC
# MAGIC Note: This cell might take ~10 mins to train. **Decrease `num_train_epochs` above to speed up the training process.** On another hand, you might notice that this cells triggers a whole new MLflow run. [MLflow](https://mlflow.org/docs/latest/index.html) is an open source tool that helps to manage end-to-end machine learning lifecycle, including experiment tracking, ML code packaging, and model deployment. You can read more about [LLM tracking here](https://mlflow.org/docs/latest/llm-tracking.html).

# COMMAND ----------

from transformers import Trainer, DataCollatorForLanguageModeling

trainer = Trainer(
    model=peft_model, # We pass in the PEFT version of the foundation model, bloomz-560M
    args=training_args,
    train_dataset=train_sample,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False) # mlm=False indicates not to use masked language modeling
)

trainer.train()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save model

# COMMAND ----------

import time

time_now = time.time()
peft_model_path = os.path.join(output_directory, f"peft_model_{time_now}")
trainer.model.save_pretrained(peft_model_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference
# MAGIC
# MAGIC You can load the model from the path that you have saved to before, and ask the model to generate text based on our input before! 

# COMMAND ----------

from peft import PeftModel

loaded_model = PeftModel.from_pretrained(foundation_model, 
                                         peft_model_path, 
                                         is_trainable=False)

# COMMAND ----------

loaded_model_outputs = loaded_model.generate(
    input_ids=input1["input_ids"], 
    attention_mask=input1["attention_mask"], 
    max_new_tokens=7, 
    eos_token_id=tokenizer.eos_token_id
    )
print(tokenizer.batch_decode(loaded_model_outputs, skip_special_tokens=True))

# COMMAND ----------

# MAGIC %md
# MAGIC Well, it seems like our fine-tuned model is indeed getting closer to generating inspirational quotes. 
# MAGIC
# MAGIC
# MAGIC In fact, the input above is taken from the training dataset. 
# MAGIC <br>
# MAGIC <br>
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/llm/english_quote_example.png" width=500>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Text initialization
# MAGIC
# MAGIC Our fine-tuned, randomly initialized model did pretty well on the quote above. Let's now compare it with the text initialization method. 
# MAGIC
# MAGIC Notice that all we are changing is the `prompt_tuning_init` setting and we are also providing a concise text prompt. 
# MAGIC
# MAGIC API docs
# MAGIC * [prompt_tuning_init_text](https://huggingface.co/docs/peft/main/en/package_reference/tuners#peft.PromptTuningConfig.prompt_tuning_init_text)

# COMMAND ----------

text_peft_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    prompt_tuning_init=PromptTuningInit.TEXT,
    prompt_tuning_init_text="Generate inspirational quotes", # this provides a starter for the model to start searching for the best embeddings
    num_virtual_tokens=3, # this doesn't have to match the length of the text above
    tokenizer_name_or_path=model_name
)
text_peft_model = get_peft_model(foundation_model, text_peft_config)
print(text_peft_model.print_trainable_parameters())

# COMMAND ----------

text_trainer = Trainer(
    model=text_peft_model,
    args=training_args,
    train_dataset=train_sample,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

text_trainer.train()

# COMMAND ----------

# Save the model
time_now = time.time()
text_peft_model_path = os.path.join(output_directory, f"text_peft_model_{time_now}")
text_trainer.model.save_pretrained(text_peft_model_path)

# Load model 
loaded_text_model = PeftModel.from_pretrained(
    foundation_model, 
    text_peft_model_path, 
    is_trainable=False
)

# Generate output
text_outputs = text_peft_model.generate(
    input_ids=input1["input_ids"], 
    attention_mask=input1["attention_mask"], 
    max_new_tokens=7, 
    eos_token_id=tokenizer.eos_token_id
)
    
print(tokenizer.batch_decode(text_outputs, skip_special_tokens=True))

# COMMAND ----------

# MAGIC %md
# MAGIC You can see that text initialization doesn't necessarily perform better than random initialization. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Share model to HuggingFace hub (optional)
# MAGIC
# MAGIC If you have a model that you would like to share with the rest of the HuggingFace community, you can choose to push your model to the HuggingFace hub! 
# MAGIC
# MAGIC 1. You need to first create a free HuggingFace account! The signup process is simple. Go to the [home page](https://huggingface.co/) and click "Sign Up" on the top right corner.
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/llm/hf_homepage_signup.png" width=700>
# MAGIC
# MAGIC 2. Once you have signed up and confirmed your email address, click on your user icon on the top right and click the `Settings` button. 
# MAGIC
# MAGIC 3. Navigate to the `Access Token` tab and copy your token. 
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/llm/hf_token_page.png" width=500>
# MAGIC
# MAGIC
# MAGIC
# MAGIC API docs:
# MAGIC * [push_to_hub](https://huggingface.co/docs/transformers/main/en/model_sharing#share-a-model)

# COMMAND ----------

# MAGIC %md
# MAGIC Here, we use Databricks Secrets management tool to save your secrets to a secret scope on Databricks. For more documentation on how to manage secrets on Databricks, visit this page on [secret management](https://docs.databricks.com/en/security/secrets/index.html).

# COMMAND ----------

from huggingface_hub import login

os.environ["huggingface_key"] = dbutils.secrets.get("llm_scope", "huggingface_key")
hf_token = os.environ["huggingface_key"]
login(token=hf_token)

# COMMAND ----------

# MAGIC %md
# MAGIC Alternatively, you can use HuggingFace's helper login method. This login cell below will prompt you to enter your token

# COMMAND ----------

from huggingface_hub import notebook_login

notebook_login()

# COMMAND ----------

# TODO
hf_username = <FILL_IN_WITH_YOUR_HUGGINGFACE_USERNAME>
peft_model_id = f"{hf_username}/bloom_prompt_tuning_{time_now}"
trainer.model.push_to_hub(peft_model_id, use_auth_token=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Inference from model in HuggingFace hub

# COMMAND ----------

from peft import PeftModel, PeftConfig

config = PeftConfig.from_pretrained(peft_model_id)
foundation_model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path)
peft_random_model = PeftModel.from_pretrained(foundation_model, peft_model_id)

# COMMAND ----------

online_model_outputs = peft_random_model.generate(
    input_ids=input1["input_ids"], 
    attention_mask=input1["attention_mask"], 
    max_new_tokens=7, 
    eos_token_id=tokenizer.eos_token_id
    )
    
print(tokenizer.batch_decode(online_model_outputs, skip_special_tokens=True))

# COMMAND ----------

# MAGIC %md
# MAGIC Congrats on applying PEFT - prompt tuning for the first time! In the lab notebook, you will be applying LoRA. 

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2023 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
