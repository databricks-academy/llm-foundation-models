# Databricks notebook source
print("Importing lab testing framework.")

# COMMAND ----------

# DEFINING HELPER FUNCTIONS

def createDirStructure():
  '''
  This creates the directories needed for the test handler.
  Note that `lesson_question_d` is lesson num: num of questions.
    Modify this when changing the number of questions
  '''
  from pathlib import Path

  lesson_question_d = {
    1: 3, # TODO: confirm these questions once tests are finalized 
    2: 6,
    3: 4,
    4: 9,
    5: 5,
  }
  path = getUsernameFromEnv("")

  for lesson, questions in lesson_question_d.items():
    for question in range(1, questions+1):
      final_path = f"{path}lesson{lesson}/question{question}"
      Path(final_path).mkdir(parents=True, exist_ok=True)

def questionPassed(userhome_for_testing, lesson, question):
  '''
  Helper function that writes an empty file named `PASSED` to the designated path
  '''
  from pathlib import Path

  print(f"\u001b[32mPASSED\x1b[0m: All tests passed for {lesson}, {question}")

  path = f"{userhome_for_testing}/{question}"
  Path(path).mkdir(parents=True, exist_ok=True)
  with open(f"{path}/PASSED", "wb") as handle:
      pass # just write an empty file
  
  print ("\u001b[32mRESULTS RECORDED\x1b[0m: Click `Submit` when all questions are completed to log the results.")

def getUsernameFromEnv(lesson):
  '''
  Exception handling for when the working directory is not in the scope
  (i.e. the Classroom-Setup was not run)
  '''
  try:
    return f"{DA.paths.working_dir}-testing-files/{lesson}"
  except NameError:
    raise NameError("Working directory not found. Please re-run the Classroom-Setup at the beginning of the notebook.")

createDirStructure()

# COMMAND ----------

# LLM 01L

def dbTestQuestion1_1(new_model):
  lesson, question = "lesson1", "question1"
  userhome_for_testing = getUsernameFromEnv(lesson)
  
  assert  str(type(new_model)) == "<class '__main__.TransformerEncoder'>", "Test NOT passed: Result should be of type `__main__.TransformerEncoder`"
  
  questionPassed(userhome_for_testing, lesson, question)

def dbTestQuestion1_2(words):
  lesson, question = "lesson1", "question2"
  userhome_for_testing = getUsernameFromEnv(lesson)

  assert bool(words) == True, "Test NOT passed: The list of words should be non-empty."
  assert len(words) >= 2, "Test NOT passed: The list of words should contain at least two words.."

  questionPassed(userhome_for_testing, lesson, question)

def dbTestQuestion1_3(sentence_q3, scrambled_sentence_q3):
  lesson, question = "lesson1", "question3"
  userhome_for_testing = getUsernameFromEnv(lesson)
  
  assert set(sentence_q3)==set(scrambled_sentence_q3), "Test NOT passed: The two sentences should contain the same words."
  assert (sentence_q3 == scrambled_sentence_q3) == False, "Test NOT passed: The two sentences should not be in the same order."
  
  questionPassed(userhome_for_testing, lesson, question)

def dbTestQuestion1_4(sentence_q4a, sentence_q4b):
  lesson, question = "lesson1", "question4"
  userhome_for_testing = getUsernameFromEnv(lesson)
  
  assert (str(type(sentence_q4a)) == "<class 'str'>" and str(type(sentence_q4b)) == "<class 'str'>"), "Test NOT passed: The two sentences should both be of type `string`."

  questionPassed(userhome_for_testing, lesson, question)

def dbTestQuestion1_5(sentence_q5):
  lesson, question = "lesson1", "question5"
  userhome_for_testing = getUsernameFromEnv(lesson)
  
  assert sentence_q5.count("[MASK]") >= 2, "Test NOT passed: `[MASK]` should appear at least twice in your sentence."

  questionPassed(userhome_for_testing, lesson, question) 

def dbTestQuestion1_6(sentence_q6):
  lesson, question = "lesson1", "question6"
  userhome_for_testing = getUsernameFromEnv(lesson)
  
  assert sentence_q6.count("[MASK]") >= 1, "Test NOT passed: `[MASK]` should appear at least once in your sentence."
  
  questionPassed(userhome_for_testing, lesson, question) 

def dbTestQuestion1_7(sentence_q7a, sentence_q7b):
  lesson, question = "lesson1", "question7"
  userhome_for_testing = getUsernameFromEnv(lesson)
  
  assert (sentence_q7a.count("[MASK]") >= 1 and sentence_q7b.count("[MASK]") >= 1), "Test NOT passed: `[MASK]` should appear in both sentences."
  
  questionPassed(userhome_for_testing, lesson, question) 

def dbTestQuestion1_8(sentence_q8):
  lesson, question = "lesson1", "question8"
  userhome_for_testing = getUsernameFromEnv(lesson)
  
  assert sentence_q8.count("[MASK]") >= 1, "Test NOT passed: `[MASK]` should appear at least once in your sentence."
  
  questionPassed(userhome_for_testing, lesson, question) 

def dbTestQuestion1_9(sentence_q9):
  lesson, question = "lesson1", "question9"
  userhome_for_testing = getUsernameFromEnv(lesson)

  assert sentence_q9.count("[MASK]") >= 1, "Test NOT passed: `[MASK]` should appear at least once in your sentence."
  
  questionPassed(userhome_for_testing, lesson, question) 

# COMMAND ----------

# LLM 02L

def dbTestQuestion2_1(r, target_modules):
  lesson, question = "lesson2", "question1"
  userhome_for_testing = getUsernameFromEnv(lesson)

  assert r == 1, "Test NOT passed: `r` should be equal to 1."
  assert target_modules == ["query_key_value"], "Test NOT passed: `target_modules` should be equal to `[\"query_key_value\"]`."
  questionPassed(userhome_for_testing, lesson, question)

def dbTestQuestion2_2(model):
  lesson, question = "lesson2", "question2"
  userhome_for_testing = getUsernameFromEnv(lesson)
  
  trainable_params = 0
  all_param = 0
  for _, param in peft_model.named_parameters():
      all_param += param.numel()
      if param.requires_grad:
          trainable_params += param.numel()

  assert trainable_params > 0, "Test NOT passed: Your new adapted model should have more than 0 training parameters."
  
  questionPassed(userhome_for_testing, lesson, question)

def dbTestQuestion2_3(trainer):
  lesson, question = "lesson2", "question3"
  userhome_for_testing = getUsernameFromEnv(lesson)

  assert str(type(trainer.args)) == "<class 'transformers.training_args.TrainingArguments'>", "Test NOT passed: `trainer.args` should be of type `transformers.training_args.TrainingArguments`."

  questionPassed(userhome_for_testing, lesson, question)

def dbTestQuestion2_4(loaded_model):
  lesson, question = "lesson2", "question4"
  userhome_for_testing = getUsernameFromEnv(lesson)

  assert str(type(loaded_model)) == "<class 'peft.peft_model.PeftModelForCausalLM'>", "Test NOT passed: `loaded_model` should be of class `peft.peft_model.PeftModelForCausalLM`."

  questionPassed(userhome_for_testing, lesson, question) 

def dbTestQuestion2_5(outputs):
  lesson, question = "lesson2", "question5"
  userhome_for_testing = getUsernameFromEnv(lesson)

  assert str(type(outputs)) == "<class 'torch.Tensor'>", "Test NOT passed: `outputs` should be of type `torch.Tensor`." 

  questionPassed(userhome_for_testing, lesson, question)

# COMMAND ----------

# LLM 03L

def dbTestQuestion3_1(distilbert, distilbert_tokenizer, test_input):
  lesson, question = "lesson3", "question1"
  userhome_for_testing = getUsernameFromEnv(lesson)

  assert str(type(distilbert)) == "<class 'transformers.models.distilbert.modeling_distilbert.DistilBertForSequenceClassification'>", "Test NOT passed: `distilbert` should be of class `ransformers.models.distilbert.modeling_distilbert.DistilBertForSequenceClassification`."

  assert str(type(distilbert_tokenizer)) == "<class 'transformers.models.distilbert.tokenization_distilbert.DistilBertTokenizer'>", "Test NOT passsed: `distilbert_tokenizer` should be of class `transformers.models.distilbert.tokenization_distilbert.DistilBertTokenizer`." 

  assert bool(test_input) == True, "Test NOT passsed: `test_input` should be non-empty."
  
  questionPassed(userhome_for_testing, lesson, question)

# COMMAND ----------

# LLM 04L

def dbTestQuestion4_1(indices):
  lesson, question = "lesson4", "question1"
  userhome_for_testing = getUsernameFromEnv(lesson)

  assert len(indices) == 32, "Test NOT passsed: `indices` length should be equal to 32."

  questionPassed(userhome_for_testing, lesson, question)

def dbTestQuestion4_2(model_name):
  lesson, question = "lesson4", "question2"
  userhome_for_testing = getUsernameFromEnv(lesson)

  assert model_name == "microsoft/xclip-base-patch16-zero-shot", "Test NOT passsed: `model_name` should equal 'microsoft/xclip-base-patch16-zero-shot'."

  questionPassed(userhome_for_testing, lesson, question)

def dbTestQuestion4_3(model_name, video_probs):
  lesson, question = "lesson4", "question3"
  userhome_for_testing = getUsernameFromEnv(lesson)

  assert text_description_list == ["play piano", "eat sandwich", "play football"], "Test NOT passsed: `text_description_list` should be `[\"play piano\", \"eat sandwich\", \"play football\"]`."

  assert str(type(video_probs)) == "<class 'torch.Tensor'>", "Test NOT passsed: `video_probs` should be of type `torch.Tensor`."
  
  questionPassed(userhome_for_testing, lesson, question)

