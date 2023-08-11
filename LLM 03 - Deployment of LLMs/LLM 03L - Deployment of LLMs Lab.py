# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Mixutre-of-Experts - Achieve Massively Scaled, but Efficient, LLM Peformance 
# MAGIC In this lab we will explore how to build our own, simplified version of a mixture-of-experts (MoE) LLM system. While this method often involves a complex training and transformer configuration, we can see some of the benefits of this approach in a pseudo-MoE that we will build with some open source LLMs. 
# MAGIC
# MAGIC
# MAGIC ### ![Dolly](https://files.training.databricks.com/images/llm/dolly_small.png) Learning Objectives
# MAGIC 1. Create your own MoE system using open source LLMs
# MAGIC 1. Build different gating mechanisms to direct different prompts to appropriate "expert models"

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Classroom Setup

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %pip install textblob==0.17.1

# COMMAND ----------

import torch

# COMMAND ----------

# MAGIC %md
# MAGIC # Section 1: An Overview of Mixture-of-Experts (MoE)
# MAGIC Mixture-of-Experts (MoE) is a machine learning architecture that incorporates the idea of "divide and conquer" to solve complex problems. In this approach, the model is composed of multiple individual models, referred to as "experts", each of which specializes in some aspect of the data. The model also includes a "gating" function that determines which expert or combination of experts to consult for a given input.
# MAGIC
# MAGIC The key feature of MoE models is that they can handle a diverse range of data patterns through the different areas of expertise of their component models. Each expert has its own set of parameters and is typically a simpler model than would be necessary to model the entire data set effectively. The gating mechanism then learns to recognize which expert is most likely to provide the best output for a particular input, thereby effectively dividing the problem space among the different experts.
# MAGIC
# MAGIC In a true MoE model, the experts and the gating function are trained together in an end-to-end manner. This joint training allows the experts to specialize on different parts of the input space and the gating function to learn how to best utilize the experts based on the input. It's a kind of "cooperative competition" among the experts, where they compete to contribute to the final output but cooperate in the sense that their combined expertise leads to a better overall model.
# MAGIC
# MAGIC An illustrative diagram would show the input being fed into the gating function, which then weights the contribution of each expert model to produce the final output. The expert models themselves would be shown as individual networks, each receiving the same input and producing its own output.
# MAGIC
# MAGIC The main advantage of MoE models is their efficiency in modeling complex functions with fewer parameters than a single large model. Since different experts can share parameters, this reduces the total number of parameters needed. This feature makes MoE models particularly useful in settings where data is diverse and complex, and a single model may struggle to capture all the different patterns present in the data.
# MAGIC
# MAGIC In this notebook, we will be creating a "pseudo" MoE model. This is not a true MoE model because we are not training the experts and the gating function together in an end-to-end manner. Instead, we will be using pre-trained models as our experts and defining our own simple gating function. While this approach does not fully capture the power of a true MoE model, it provides a useful introduction to the concept and allows us to explore how different experts and gating functions can affect the performance of the model. It also provides a foundation for understanding how a true MoE model might be implemented and trained.

# COMMAND ----------

# MAGIC %md
# MAGIC # Section 2: The Pseudo MoE Model
# MAGIC In this section, we'll implement a simplified version of an MoE model. Instead of training the experts and gating function together, we'll use pre-trained transformer models as our experts and a simple rule-based function as our gating function.
# MAGIC
# MAGIC We'll also look at different types of gating mechanisms - hard gating, soft gating, and top-k gating.

# COMMAND ----------

# Import the necessary libraries
# transformers is a state-of-the-art library for Natural Language Processing tasks, providing a wide range of pre-trained models
from transformers import GPT2LMHeadModel, GPT2Tokenizer, BertForSequenceClassification, BertTokenizer, T5ForConditionalGeneration, T5Tokenizer
# torch.nn.functional provides functions that don't have any parameters, such as activation functions, loss functions etc.
import torch.nn.functional as F

# Load the GPT2 model and tokenizer
# GPT2 is an autoregressive language model that uses transformer blocks and byte-pair encoding
gpt2 = GPT2LMHeadModel.from_pretrained("gpt2-XL", cache_dir=DA.paths.datasets+"/models")
# The tokenizer is responsible for turning input data into the format that the model expects
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2-XL", cache_dir=DA.paths.datasets+"/models")

# Load the BERT model and tokenizer
# BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based machine learning technique for natural language processing pre-training
bert = BertForSequenceClassification.from_pretrained("bert-base-uncased", cache_dir=DA.paths.datasets+"/models")
# The tokenizer is responsible for turning input data into the format that the model expects
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", cache_dir=DA.paths.datasets+"/models")

# Load the T5 model and tokenizer
# T5 (Text-to-Text Transfer Transformer) is a transformer model which treats every NLP problem as a text generation task
t5 = T5ForConditionalGeneration.from_pretrained("t5-base", cache_dir=DA.paths.datasets+"/models")
# The tokenizer is responsible for turning input data into the format that the model expects
t5_tokenizer = T5Tokenizer.from_pretrained("t5-base", cache_dir=DA.paths.datasets+"/models"+"/models")

# Define the "hard gating" function
# This function decides which model to use based on the length of the input
def hard_gating_function(input):
    if len(input) < 10:
        # For inputs less than 10 characters long, use the GPT2 model
        return "gpt2", gpt2, gpt2_tokenizer
    elif len(input) < 100:
        # For inputs less than 100 characters long but greater than 10 characters, use the T5 model
        return "t5" , t5, t5_tokenizer
    else:
        # For inputs greater than 100 characters, use the BERT model
        return "bert", bert, bert_tokenizer

# Define the "soft gating" function
# This function assigns a weight to each model based on the length of the input, and all models are used to a certain extent to generate the output
def soft_gating_function(input):
    # The weights for each model are calculated using the softmax function, which outputs a probability distribution
    weights = F.softmax(torch.tensor([len(input), 100 - len(input), len(input)], dtype=torch.float), dim=0)
    # The weights for each model are returned along with the models and their tokenizers
    return {"gpt2": (gpt2, gpt2_tokenizer, weights[0]),
            "bert": (bert, bert_tokenizer, weights[1]),
            "t5": (t5, t5_tokenizer, weights[2])}

# Define the pseudo MoE model
# This function uses the gating function to decide which model(s) to use for a given input
def pseudo_moe_model(input, gating_function):
    if gating_function == "hard":
        # If the hard gating function is used, only one model is used for a given input
        model_name, model, tokenizer = hard_gating_function(input)
        inputs = tokenizer(input, return_tensors="pt")
        if model_name == "t5":
            # For T5, create a decoder input sequence consisting of only the <BOS> token
            decoder_inputs = tokenizer(["<pad>"], return_tensors="pt")["input_ids"]
            outputs = model(**inputs, decoder_input_ids=decoder_inputs)
        else:
            outputs = model(**inputs)
        # The output of the model is decoded into a string
        decoded_output = tokenizer.decode(outputs.logits[0].argmax(-1).tolist())
        # The name of the model used and the decoded output are returned
        return model_name, decoded_output
    else:  # soft gating
        # If the soft gating function is used, all models are used to a certain extent to generate the output
        models = soft_gating_function(input)
        outputs = []
        for model_name, (model, tokenizer, weight) in models.items():
            inputs = tokenizer(input, return_tensors="pt")
            if model_name == "t5":
                # For T5, create a decoder input sequence consisting of only the <BOS> token
                decoder_inputs = tokenizer(["<pad>"], return_tensors="pt")["input_ids"]
                output = model(**inputs, decoder_input_ids=decoder_inputs)
            else:
                output = model(**inputs)
            # The output of each model is multiplied by its weight
            outputs.append((model_name, output.logits * weight))
        # The outputs of all models are added together to generate the final output
        decoded_outputs = [(model_name, tokenizer.decode(output[0].argmax(-1).tolist())) for model_name, output in outputs]
        # The decoded outputs are returned
        return decoded_outputs


# COMMAND ----------

# Test the hard gating function
example_1 = "Translate to german: This is a short input."
output = pseudo_moe_model(example_1, gating_function="hard")
print("Hard gating output:", output)

# Test the soft gating function
example_2 = "This is a longer input. We're adding more text here to make sure it's longer than 50 characters but shorter than 100 characters."
output = pseudo_moe_model(example_2, gating_function="soft")
print("Soft gating output:", output)


# COMMAND ----------

# MAGIC %md
# MAGIC # Section 3: Your Turn
# MAGIC Now it's your turn to experiment with the pseudo MoE model. Here are some exercises you can try:
# MAGIC
# MAGIC Implement a new gating function: Instead of just using the length of the input, can you use a basic sentiment analysis to determine which model to use? You can use the textblob library for the sentiment analysis.
# MAGIC
# MAGIC Add a new expert: Can you add a new expert to the pseudo MoE model? Try using the distilbert model, which is a smaller, faster, cheaper version of BERT.
# MAGIC
# MAGIC Test the updated pseudo MoE model: Once you've made your updates, test the pseudo MoE model with example inputs of different sentiment. What do you notice about the performance of the different experts (models) for different input text?

# COMMAND ----------

# TODO
# 1. Implement a new gating function

from textblob import TextBlob

def sentiment_based_gating_function(input):
    <FILL_IN>

# 2. Add a new expert

from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

distilbert = <FILL_IN>
distilbert_tokenizer = <FILL_IN>

# Update the gating function to include the new expert

def updated_gating_function(input):
    <FILL_IN>

# 3. Test the updated pseudo MoE model

test_input = "<FILL_IN>"
print(pseudo_moe_model(test_input, gating_function='hard'))

# COMMAND ----------

# Test your answer. DO NOT MODIFY THIS CELL.

dbTestQuestion3_1(distilbert, distilbert_tokenizer, test_input)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2023 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
