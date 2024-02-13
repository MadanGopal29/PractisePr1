#!/usr/bin/env python
# coding: utf-8

# import gradio as gr
# from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoModelForSequenceClassification, AutoTokenizer
# 
# # Initialize models and tokenizers
# model_summarization = T5ForConditionalGeneration.from_pretrained('t5-small')
# tokenizer_summarization = T5Tokenizer.from_pretrained('t5-small')
# 
# model_sentiment = AutoModelForSequenceClassification.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
# tokenizer_sentiment = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
# 
# # Define summarization function
# def summarize_text(text, out_length):
#     inputs = tokenizer_summarization.encode(
#         "summarize: " + text,
#         return_tensors='pt',
#         max_length=512,
#         truncation=True,
#         padding='max_length'
#     )
#     summary_ids = model_summarization.generate(
#         inputs,
#         max_length=out_length,
#         num_beams=5
#     )
#     return tokenizer_summarization.decode(summary_ids[0], skip_special_tokens=True)
# 
# # Define sentiment analysis function
# def predict_sentiment(text):
#     inputs = tokenizer_sentiment(text, return_tensors='pt')
#     outputs = model_sentiment(**inputs)
#     sentiment = 'Positive' if outputs.logits.argmax().item() ==   1 else 'Negative'
#     return sentiment
# 
# # Define a wrapper function that decides whether to summarize or analyze sentiment
# def summarize_or_analyze(text, task, out_length):
#     if task == 'summarize':
#         return summarize_text(text, out_length)
#     elif task == 'analyze sentiment':
#         return predict_sentiment(text)
#     else:
#         return "Invalid task selected."
# 
# # Create Gradio interface with conditional logic
# iface = gr.Interface(
#     fn=summarize_or_analyze,
#     inputs=[gr.Textbox(lines=10, placeholder='Enter Text Here...', label='Input text'),  
#             gr.Radio(['summarize', 'analyze sentiment'], label='Task')],
#     outputs=gr.Textbox(label='Result'),
#     title='Text Summarizer & Sentiment Analyzer'
# )
# 
# # Launch the Gradio app
# iface.launch()


#import subprocess

# Function to install a package using pip with sudo
#def install_with_sudo(package):
 #   subprocess.run(['sudo', 'pip', 'install', package], check=True)

# Change to the directory where your .py file is located
#subprocess.run(['cd', '/home/dai/Desktop/deploy_pipe'], check=True)

# Example usage
#install_with_sudo('gradio')  # Replace 'numpy' with the actual module name
#install_with_sudo('transformers')
#install_with_sudo('sentencepiece')
#install_with_sudo('torch')

import os
import subprocess

# Change to the desired directory
os.chdir('/home/dai/Desktop/deploy_pipe')

# Now you can run commands in the context of the new directory
subprocess.run(['sudo','pip', 'install', 'gradio'], check=True)

subprocess.run(['sudo','pip', 'install', 'transformers'], check=True)

subprocess.run(['sudo','pip', 'install', 'sentencepiece'], check=True)

subprocess.run(['sudo','pip', 'install', 'torch'], check=True)

# In[1]:


import gradio as gr
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoModelForSequenceClassification, AutoTokenizer


# In[2]:


# Initialize models and tokenizers
model_summarization = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer_summarization = T5Tokenizer.from_pretrained('t5-small')

model_sentiment = AutoModelForSequenceClassification.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
tokenizer_sentiment = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')


# In[3]:


# Define summarization function
def summarize_text(text, out_length):
    inputs = tokenizer_summarization.encode(
        "summarize: " + text,
        return_tensors='pt',
        max_length=512,
        truncation=True,
        padding='max_length'
    )
    summary_ids = model_summarization.generate(
        inputs,
        max_length=out_length,
        num_beams=5
    )
    return tokenizer_summarization.decode(summary_ids[0], skip_special_tokens=True)


# In[4]:


# Define sentiment analysis function
def predict_sentiment(text):
    inputs = tokenizer_sentiment(text, return_tensors='pt')
    outputs = model_sentiment(**inputs)
    sentiment = 'Positive' if outputs.logits.argmax().item() ==   1 else 'Negative'
    return sentiment


# In[5]:


# Define a wrapper function that decides whether to summarize or analyze sentiment
def summarize_or_analyze(text, task, out_length):
    if task == 'summarize':
        return summarize_text(text, out_length)
    elif task == 'analyze sentiment':
        return predict_sentiment(text)
    else:
        return "Invalid task selected."


# In[6]:


# Create Gradio interface with conditional logic
iface = gr.Interface(
    fn=summarize_or_analyze,
    inputs=[gr.Textbox(lines=10, placeholder='Enter Text Here...', label='Input text'),   
            gr.Radio(['summarize', 'analyze sentiment'], label='Task'),
            gr.Slider(minimum=10, maximum=100, step=1, label='Summary Length')],
    outputs=gr.Textbox(label='Result'),
    title='Text Summarizer & Sentiment Analyzer'
)


# In[7]:


# Launch the Gradio app
iface.launch()


# In[8]:


#from IPython.display import display, HTML

#url = "http://127.0.0.1:7860/"
#display(HTML(f'<a href="{url}" target="_blank">Open Link</a>'))


# In[ ]:




