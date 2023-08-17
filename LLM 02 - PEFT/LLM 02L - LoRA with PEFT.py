# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Low-Rank Adaption (LoRA)
# MAGIC This lab introduces how to apply low-rank adaptation (LoRA) to your model of choice using [Parameter-Efficient Fine-Tuning (PEFT) library developed by Hugging Face](https://huggingface.co/docs/peft/index). 
# MAGIC
# MAGIC
# MAGIC ### ![Dolly](https://files.training.databricks.com/images/llm/dolly_small.png) Learning Objectives
# MAGIC 1. Apply LoRA to a model
# MAGIC 1. Fine-tune on your provided dataset
# MAGIC 1. Save your model
# MAGIC 1. Conduct inference using the fine-tuned model

# COMMAND ----------

# MAGIC %pip install peft==0.4.0

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md
# MAGIC We will re-use the same dataset and model from the demo notebook. 

# COMMAND ----------

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "bigscience/bloomz-560m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
foundation_model = AutoModelForCausalLM.from_pretrained(model_name)

data = load_dataset("Abirate/english_quotes", cache_dir=DA.paths.datasets+"/datasets")
data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)
train_sample = data["train"].select(range(50))
display(train_sample) 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define LoRA configurations
# MAGIC
# MAGIC By using LoRA, you are unfreezing the attention `Weight_delta` matrix and only updating `W_a` and `W_b`. 
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/llm/lora.png" width=300>
# MAGIC
# MAGIC You can treat `r` (rank) as a hyperparameter. Recall from the lecture that, LoRA can perform well with very small ranks based on [Hu et a 2021's paper](https://arxiv.org/abs/2106.09685). GPT-3's validation accuracies across tasks with ranks from 1 to 64 are quite similar. From [PyTorch Lightning's documentation](https://lightning.ai/pages/community/article/lora-llm/):
# MAGIC
# MAGIC > A smaller r leads to a simpler low-rank matrix, which results in fewer parameters to learn during adaptation. This can lead to faster training and potentially reduced computational requirements. However, with a smaller r, the capacity of the low-rank matrix to capture task-specific information decreases. This may result in lower adaptation quality, and the model might not perform as well on the new task compared to a higher r.
# MAGIC
# MAGIC Other arguments:
# MAGIC - `lora_dropout`: 
# MAGIC   - Dropout is a regularization method that reduces overfitting by randomly and temporarily removing nodes during training. 
# MAGIC   - It works like this: <br>
# MAGIC     * Apply to most type of layers (e.g. fully connected, convolutional, recurrent) and larger networks
# MAGIC     * Temporarily and randomly remove nodes and their connections during each training cycle
# MAGIC     ![](https://files.training.databricks.com/images/nn_dropout.png)
# MAGIC     * See the original paper here: <a href="http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf" target="_blank">Dropout: A Simple Way to Prevent Neural Networks from Overfitting</a>
# MAGIC - `target_modules`:
# MAGIC   - Specifies the module names to apply to 
# MAGIC   - This is dependent on how the foundation model names its attention weight matrices. 
# MAGIC   - Typically, this can be:
# MAGIC     - `query`, `q`, `q_proj` 
# MAGIC     - `key`, `k`, `k_proj` 
# MAGIC     - `value`, `v` , `v_proj` 
# MAGIC     - `query_key_value` 
# MAGIC   - The easiest way to inspect the module/layer names is to print the model, like we are doing below.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Question 1
# MAGIC
# MAGIC Fill in `r=1` and `target_modules`. 
# MAGIC
# MAGIC Note:
# MAGIC - For `r`, any number is valid. The smaller the r is, the fewer parameters there are to update during the fine-tuning process.
# MAGIC
# MAGIC Hint: 
# MAGIC - For `target_modules`, what's the name of the **first** module within each `BloomBlock`'s `self_attention`? 
# MAGIC
# MAGIC Read more about [`LoraConfig` here](https://huggingface.co/docs/peft/conceptual_guides/lora#common-lora-parameters-in-peft).

# COMMAND ----------

# TODO
import peft
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=<FILL_IN>,
    lora_alpha=1, # a scaling factor that adjusts the magnitude of the weight matrix. Usually set to 1
    target_modules=["<FILL_IN>"],
    lora_dropout=0.05, 
    bias="none", # this specifies if the bias parameter should be trained. 
    task_type="CAUSAL_LM"
)

# COMMAND ----------

# Test your answer. DO NOT MODIFY THIS CELL.

dbTestQuestion2_1(lora_config.r, lora_config.target_modules)

# COMMAND ----------

# MAGIC %md
# MAGIC ###  Question 2
# MAGIC
# MAGIC Add the adapter layers to the foundation model to be trained

# COMMAND ----------

# TODO
peft_model = get_peft_model(<FILL_IN>, <FILL_IN>)
print(peft_model.print_trainable_parameters())

# COMMAND ----------

# Test your answer. DO NOT MODIFY THIS CELL.

dbTestQuestion2_2(peft_model)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define `Trainer` class for fine-tuning

# COMMAND ----------

# MAGIC %md
# MAGIC ### Question 3 
# MAGIC
# MAGIC Fill out the `Trainer` class. Feel free to tweak the `training_args` we provided, but remember that lowering the learning rate and increasing the number of epochs will increase training time significantly. If you change none of the defaults we set below, it could take ~15 mins to fine-tune.

# COMMAND ----------

# TODO
import transformers
from transformers import TrainingArguments, Trainer
import os

output_directory = os.path.join(DA.paths.working_dir, "peft_lab_outputs")
training_args = TrainingArguments(
    output_dir=output_directory,
    auto_find_batch_size=True,
    learning_rate= 3e-2, # Higher learning rate than full fine-tuning.
    num_train_epochs=5,
    no_cuda=True
)

trainer = Trainer(
    model=<FILL_IN>,
    args=<FILL_IN>,
    train_dataset=train_sample,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
trainer.train()

# COMMAND ----------

# Test your answer. DO NOT MODIFY THIS CELL.

dbTestQuestion2_3(trainer)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load model

# COMMAND ----------

# MAGIC %md
# MAGIC ### Question 4 
# MAGIC
# MAGIC Load the PEFT model using pre-defined LoRA configs and foundation model. We set `is_trainable=False` to avoid further training.

# COMMAND ----------

import time

time_now = time.time()

username = spark.sql("SELECT CURRENT_USER").first()[0]
peft_model_path = os.path.join(output_directory, f"peft_model_{time_now}")

trainer.model.save_pretrained(peft_model_path)

# COMMAND ----------

# TODO
from peft import PeftModel, PeftConfig

loaded_model = PeftModel.from_pretrained(<FILL_IN>, 
                                        <FILL_IN>, 
                                        is_trainable=False)

# COMMAND ----------

# Test your answer. DO NOT MODIFY THIS CELL.

dbTestQuestion2_4(loaded_model)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference

# COMMAND ----------

# MAGIC %md
# MAGIC ### Question 5
# MAGIC
# MAGIC Generate output tokens to the same input we provided in the demo notebook before. How do the outputs compare?

# COMMAND ----------

# TODO
inputs = tokenizer("Two things are infinite: ", return_tensors="pt")
outputs = peft_model.generate(
    input_ids=<FILL_IN>, 
    attention_mask=<FILL_IN>, 
    max_new_tokens=<FILL_IN>, 
    eos_token_id=tokenizer.eos_token_id
    )
print(tokenizer.batch_decode(<FILL_IN>, skip_special_tokens=True))

# COMMAND ----------

# Test your answer. DO NOT MODIFY THIS CELL.

dbTestQuestion2_5(outputs)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2023 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
