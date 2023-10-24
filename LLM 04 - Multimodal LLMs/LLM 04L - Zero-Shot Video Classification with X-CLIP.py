# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Zero-Shot Video Classification
# MAGIC In this lab, we are going to pass a video of choice to [X-CLIP](https://huggingface.co/docs/transformers/model_doc/xclip) and ask X-CLIP to assign probabilities to the provided text description. This model developed by [Ni et al 2022](https://arxiv.org/abs/2208.02816) aims to extend OpenAI's CLIP model that's focused on image-related tasks. From Hugging Face's documentation:
# MAGIC
# MAGIC >The model consists of a text encoder, a cross-frame vision encoder, a multi-frame integration Transformer, and a video-specific prompt generator. 
# MAGIC
# MAGIC ### ![Dolly](https://files.training.databricks.com/images/llm/dolly_small.png) Learning Objectives
# MAGIC 1. You will learn how to load a video from YouTube and do minor processing on the video for X-CLIP 
# MAGIC 1. Use X-CLIP to assign probabilities to text descriptions
# MAGIC
# MAGIC DISCLAIMER: The majority of this notebook's code is borrowed from Hugging Face's Tutorial GitHub Repo, specifically the["Transformers-Tutorials/X-CLIP"](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/X-CLIP/Zero_shot_classify_a_YouTube_video_with_X_CLIP.ipynb) notebook. 

# COMMAND ----------

# MAGIC %md
# MAGIC We will use [pytube](https://pytube.io/en/latest/index.html) to get videos from YouTube and load videos using [decord](https://github.com/dmlc/decord).

# COMMAND ----------

# MAGIC %pip install decord==0.6.0 openai==0.27.8 pytube==15.0.0

# COMMAND ----------

# MAGIC %run ../Includes/pytube_patch

# COMMAND ----------

# MAGIC %md
# MAGIC ## Classroom Setup

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md
# MAGIC Below, we will load a YouTube video of a piano performance.
# MAGIC
# MAGIC `streams.filter` method provides flexible ways for us to filter based on the type of stream that we're interested in. Refer to [documentation here](https://pytube.io/en/latest/user/streams.html#filtering-by-streaming-method).

# COMMAND ----------

from pytube import YouTube

# a video of piano performance 
youtube_url = "https://www.youtube.com/watch?v=-xKM3mGt2pE"
yt = YouTube(youtube_url)

streams = yt.streams.filter(file_extension="mp4")
print(streams)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's say that we only are interested in the first part of the video stream. We will download only the third portion and save it to our directory.

# COMMAND ----------

import os

output_dir = os.path.join(DA.paths.working_dir, "video")
file_path = streams[0].download(output_path=output_dir)
file_path

# COMMAND ----------

# MAGIC %md
# MAGIC Recall from the presentation that audio data is often split into chunks. The same applies to videos as well. Below we will split the video into different frames. 
# MAGIC
# MAGIC `frame_rate` is a common term in video processing to refer to # of pictures taken per second. For audio-only data, it's called `sampling_rate`.
# MAGIC
# MAGIC `VideoReader` helps us to access frames directly from the video files. Refer to [documentation here](https://github.com/dmlc/decord#videoreader).

# COMMAND ----------

from decord import VideoReader, cpu
import torch
import numpy as np
from huggingface_hub import hf_hub_download

np.random.seed(42)

# this does in-memory decoding of the video 
videoreader = VideoReader(file_path, num_threads=1, ctx=cpu(0))
print("Length of video frames: ", len(videoreader))

def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    
    # Since each frame length is 4 seconds, we need to find the total frame length if we want `clip_len` frames 
    converted_len = int(clip_len * frame_sample_rate)

    # Get a random frame to end on 
    end_idx = np.random.randint(converted_len, seg_len)
    # Find the starting frame, if the frame has length of clip_len
    start_idx = end_idx - converted_len

    # np.linspace returns evenly spaced numbers over a specified interval 
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices

# COMMAND ----------

# MAGIC %md
# MAGIC ## Question 1
# MAGIC
# MAGIC We want to retrieve 32 frames in total, with 4 seconds each. 

# COMMAND ----------

# TODO 
indices = sample_frame_indices(clip_len= <FILL_IN>, 
                               frame_sample_rate=<FILL_IN>, 
                               seg_len=len(videoreader))
print("Number of frames we will retrieve: ", len(indices))

# `get_batch` allows us to get multiple frames at once 
video = videoreader.get_batch(indices).asnumpy()

# COMMAND ----------

# Test your answer. DO NOT MODIFY THIS CELL.

dbTestQuestion4_1(indices)

# COMMAND ----------

# MAGIC %md
# MAGIC We will now randomly pick a video frame to inspect.

# COMMAND ----------

from PIL import Image

Image.fromarray(video[8])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Question 2
# MAGIC
# MAGIC We will now pass in XCLIP model to process our video frames and as our model to assign probabilities to text descriptions to the model. 
# MAGIC
# MAGIC The model we will use is `microsoft/xclip-base-patch16-zero-shot`.

# COMMAND ----------

# TODO 

from transformers import XCLIPProcessor, XCLIPModel

model_name = <FILL_IN>
processor = XCLIPProcessor.from_pretrained(model_name)
model = XCLIPModel.from_pretrained(model_name)

# COMMAND ----------

# Test your answer. DO NOT MODIFY THIS CELL.

dbTestQuestion4_2(model_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Question 3 
# MAGIC
# MAGIC We will provide a list of three text descriptions and ask the model to assign probabilities to each of them. 
# MAGIC
# MAGIC Let's use `text_description_list = ["play piano", "eat sandwich", "play football"]` 
# MAGIC
# MAGIC Hint:  for the `videos` argument: recall that we have a list of video frames we have processed in the cells above. 

# COMMAND ----------

# TODO 
import torch

text_description_list = <FILL_IN>

inputs = processor(text=<FILL_IN>, 
                   videos=<FILL_IN>, 
                   return_tensors="pt", 
                   padding=True)

# forward pass
# we are not going to train the model, hence we specify .no_grad()
with torch.no_grad():
    outputs = model(**inputs)

# we will get probabilities per video frame and calculate the softmax 
video_probs = outputs.logits_per_video.softmax(dim=1)
print(dict(zip(text_description_list, video_probs[0])))

# COMMAND ----------

# Test your answer. DO NOT MODIFY THIS CELL.

dbTestQuestion4_3(text_description_list, video_probs)

# COMMAND ----------

# MAGIC %md
# MAGIC Which text description has the highest probability? In the following optional section, you can play around with OpenAI's CLIP and Whisper API to generate image from text and get audio transcription.

# COMMAND ----------

# MAGIC %md
# MAGIC ## OPTIONAL (Non-graded): Using OpenAI's CLIP and Whisper
# MAGIC
# MAGIC
# MAGIC For this section to work, you need to generate an Open AI key. 
# MAGIC
# MAGIC Steps:
# MAGIC 1. You need to [create an account](https://platform.openai.com/signup) on OpenAI. 
# MAGIC 2. Generate an OpenAI [API key here](https://platform.openai.com/account/api-keys). 
# MAGIC
# MAGIC Note: OpenAI does not have a free option, but it gives you $5 as credit. Once you have exhausted your $5 credit, you will need to add your payment method. You will be [charged per token usage](https://openai.com/pricing). **IMPORTANT**: It's crucial that you keep your OpenAI API key to yourself. If others have access to your OpenAI key, they will be able to charge their usage to your account! 

# COMMAND ----------

# TODO
import os

os.environ["OPENAI_API_KEY"] = "<FILL IN>"

# COMMAND ----------

import openai

openai.api_key = os.environ["OPENAI_API_KEY"]

# COMMAND ----------

# MAGIC %md
# MAGIC ###  Using CLIP 
# MAGIC
# MAGIC OpenAI's CLIP can help you generate images from provided text. 

# COMMAND ----------

image_resp = openai.Image.create(prompt="robots play water balloons, modern painting", 
                                 n=1, 
                                 size="512x512")
image_resp
displayHTML(image_resp["data"][0]["url"])

# COMMAND ----------

# MAGIC %md
# MAGIC You can also use it to assign text caption probabilities based on provided image. 

# COMMAND ----------

from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
cat_image = Image.open(requests.get(url, stream=True).raw)
display(cat_image)

# COMMAND ----------

caption_list = ["eating pasta", "cats sleeping"]

inputs = clip_processor(text=caption_list, 
                        images=cat_image, 
                        return_tensors="pt", 
                        padding=True)

clip_outputs = clip_model(**inputs)
# This calculates image-text similarity score 
clip_logits_per_image = clip_outputs.logits_per_image 

# Use softmax to get caption probabilities 
image_probs = clip_logits_per_image.softmax(dim=1)
print(dict(zip(caption_list, image_probs[0])))

# COMMAND ----------

# MAGIC %md
# MAGIC You can see that the probability of the caption with "cats" is much higher than that of "pasta".

# COMMAND ----------

# MAGIC %md
# MAGIC ### Using Whisper
# MAGIC
# MAGIC OpenAI's Whisper Automatic Speech Recognition system is a simple and powerful tool for transcribing audio files. 
# MAGIC
# MAGIC If you'd like to browse interesting Whisper applications that people have been exploring, visit [this link](https://github.com/openai/whisper/discussions/categories/show-and-tell), notably [this web UI application](https://huggingface.co/spaces/aadnk/whisper-webui) and [this transcription + speaker identification discussion](https://github.com/openai/whisper/discussions/264).

# COMMAND ----------

import requests

# URL of the sample audio file (in this case, a simple English sentence)
audio_url = "https://audio-samples.github.io/samples/mp3/blizzard_primed/sample-1.mp3"

# Download the audio file
response = requests.get(audio_url)

audio_directory = os.path.join(DA.paths.working_dir, "sample_audio.mp3")
# Save the audio file to disk
with open(audio_directory, "wb") as audio_file:
    audio_file.write(response.content) 

print("Sample audio file 'sample_audio.wav' downloaded.")

# COMMAND ----------

audio_file = open(audio_directory, "rb")
transcript = openai.Audio.transcribe("whisper-1", audio_file)
print(transcript)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2023 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
