import re
import os
import time
import json
import logging

from dotenv import load_dotenv
load_dotenv()

import warnings
warnings.filterwarnings('ignore')

import faiss
import numpy as np

import torch

import google.generativeai as genai

from transformers import pipeline
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from transformers.utils import is_flash_attn_2_available

from langchain_community.chat_models import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter

import pygame
from gtts import gTTS

import markdown
import html2text

import nltk
from nltk.tokenize import sent_tokenize

import streamlit as st
from pytube import YouTube

from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import letter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from QuizGen import create_question_paper

#Prompt Templates
with open("Prompts\\Summarization.txt") as fh: 
    summarization_template = fh.read()
with open("Prompts\\Question.txt") as fh: 
    question_template = fh.read()

with open("Prompts/Quiz/Mcqs.txt") as fh: 
    mcq_template = fh.read()
with open("Prompts/Quiz/FillBlanks.txt") as fh: 
    fib_template = fh.read()
with open("Prompts/Quiz/TrueFalse.txt") as fh: 
    tf_template = fh.read()
with open("Prompts/Quiz/ShortAnswer.txt") as fh: 
    sa_template = fh.read()


#Loading LLM Models
llama = ChatOllama(model='llama3')

API = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API)
gemini = genai.GenerativeModel('gemini-1.5-flash')

language_codes = {
    "English": "en",
    "Hindi": "hi",
    "Kannada": "kn"
}

llm_codes = {'Llama3 : 7B Instruct' : 0, 
    'Google Gemini' : 1
}

def markdown_to_normal(text):
    html = markdown.markdown(text)
    plain_text = html2text.html2text(html) 
    return plain_text.strip()

def llama_response(prompt): 
    Answer = llama.invoke(prompt)
    return(Answer.content)

def gemini_response(prompt): 
    Answer = gemini.generate_content(prompt)
    return(Answer.text)

class Transcript:
    def __init__(self,file):
        self.file = file 
        self.method = "automatic-speech-recognition"
        self.model = 'openai/whisper-tiny.en'
    
    def generate_transcript(self):
        pipe = pipeline(
            self.method,
            model=self.model,
            torch_dtype=torch.float16,
            device="cuda:0",
            model_kwargs={"attn_implementation": "flash_attention_2"} if is_flash_attn_2_available() else {"attn_implementation": "sdpa"},
            )
        
        start = time.time()

        output = pipe(
            self.file,
            chunk_length_s=10,
            batch_size=32,
            return_timestamps=False)
        
        end = time.time()
        total_time = round((end-start),2)

        with open("Content.txt","w+") as fh: 
            fh.write(output['text'])
        return(total_time)
    
def split_text_into_chunks_summary(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    total_words = len(text.split())
    
    if total_words > 10000:
        max_chunk_size = 4000  
        max_overlap = 800      

    elif total_words > 5000:
        max_chunk_size = 2000 
        max_overlap = 400      

    elif total_words > 3000:
        max_chunk_size = 1500  
        max_overlap = 300      

    elif total_words > 1000:
        max_chunk_size = 1000 
        max_overlap = 200      

    else:
        max_chunk_size = 500   
        max_overlap = 100      

    chunks = []
    current_chunk = []
    current_chunk_size = 0
    
    for sentence in sentences:
        words = sentence.split()
        sentence_length = len(words)
        
        if current_chunk_size + sentence_length > max_chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = current_chunk[-max_overlap:]
            current_chunk_size = len(current_chunk)
        
        current_chunk.extend(words)
        current_chunk_size += sentence_length
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def split_text_into_chunks_qa(text):
    def split_text_into_sentences(text):
        sentence_endings = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')
        sentences = sentence_endings.split(text)
        return sentences

    def create_chunks(sentences, chunk_size, chunk_overlap):
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence) + 1 
            if current_length + sentence_length > chunk_size:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        if chunk_overlap > 0:
            overlapping_chunks = []
            for i in range(len(chunks)):
                if i == 0:
                    overlapping_chunks.append(chunks[i])
                else:
                    overlap = ' '.join(chunks[i-1].split()[-chunk_overlap:])
                    new_chunk = overlap + ' ' + chunks[i]
                    overlapping_chunks.append(new_chunk)
            return overlapping_chunks
        else:
            return chunks

    length = len(text)
    if length >= 10000:
        chunk_size = 2000
        chunk_overlap = 500
    else:
        chunk_size = 500
        chunk_overlap = 300

    sentences = split_text_into_sentences(text)
    chunks = create_chunks(sentences, chunk_size, chunk_overlap)
    return chunks

def create_vector_store():
    with open("Content.txt") as file: 
        content = file.read()

    text_chunks = split_text_into_chunks_qa(content)
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    
    embeddings = model.encode(text_chunks, convert_to_numpy=True)

    dimension = embeddings.shape[1]  
    index = faiss.IndexFlatL2(dimension) 
    index.add(embeddings)  

    faiss.write_index(index, 'VectorDB.index')

def load_knowledgebase():
    vector_store_path = "VectorDB.index"
    index = faiss.read_index(vector_store_path)
    return index

def load_context(query,text_chunks, index):

    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    query_embedding = model.encode([query], convert_to_numpy=True)

    top_k = 2
    distances, indices = index.search(query_embedding, k=top_k)

    if indices.size == 0 or indices[0].size == 0: 
        return None
    
    relevant_chunks = []
    if indices.size > 0:
        for i in range(top_k):
            chunk_index = indices[0][i]
            if chunk_index < len(text_chunks): 
                relevant_chunks.append(text_chunks[chunk_index])
            else:
                pass

    return "\n".join(relevant_chunks) if relevant_chunks else None

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400&display=swap');
    
    .title {
        font-family: 'Roboto', sans-serif;
        font-size: 2rem;
        font-weight: bolder;
        color: #ffdab9;
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

if 'language' not in st.session_state:
    st.session_state.language = None
if 'path' not in st.session_state:
    st.session_state.path = None
if 'llm_model' not in st.session_state: 
    st.session_state.llm_model = None
if 'index' not in st.session_state:
    st.session_state.index = None
if 'qa_chunks' not in st.session_state: 
    st.session_state.qa_chunks = None
if 'status' not in st.session_state: 
    st.session_state.status = None

st.markdown('<p class="title">Whisper : Effortless Summarization And Question Answering✨</p>', unsafe_allow_html=True)

st.sidebar.markdown("### **🤖 LLM Model Selection​**")
selected_llm = st.sidebar.selectbox(
    "**Select Your Desired LLM To Work With**​",
    ['Llama3 : 7B Instruct', 'Google Gemini']
)
st.session_state.llm_model = llm_codes[selected_llm]

st.sidebar.markdown("<br>",unsafe_allow_html=True)

st.sidebar.markdown("### **⚙ LLM Model Parameters​**")
show_info = st.sidebar.checkbox('**Show Parameters Information**', value=False)

if show_info:
    st.sidebar.info("""
        **Parameters Info**:
        - **Temperature Sampling**: Adjusts response randomness. Lower values make responses more predictable (e.g., 0.1) while higher values increase diversity (e.g., 1.5).
        - **Top-k Sampling**: Limits sampling to the top `k` most probable tokens. This balances creativity and coherence.
        - **Top-p Sampling (Nucleus Sampling)**: Samples from the smallest set of tokens whose cumulative probability exceeds a threshold `p`. This provides a balance between diversity and coherence.
        - **Greedy Sampling**: Always chooses the most probable token at each step, leading to the most predictable and coherent output but potentially less diverse.
    """)

selected_llm = st.sidebar.selectbox(
    "**Select Your Sampling Method**​",
    ['Greedy Sampling', 'Top-k Sampling','Top-p Sampling']
)

col1, col2 = st.sidebar.columns([1, 1])
with col1:
    temperature = st.slider('**Temperature**', min_value=0.0, max_value=1.0, value=0.5, step=0.05)
with col2:
    sampling = st.slider('**Sampling**', min_value=0.0, max_value=1.0, value=0.5, step=0.05)

st.sidebar.markdown('<br><br>', unsafe_allow_html=True)

st.sidebar.markdown("### **🌐Language Selection​**")
selected_language = st.sidebar.selectbox(
    "**Select Your Desired Output Language**​",
    list(language_codes.keys())
)
st.session_state.language = language_codes[selected_language]

st.sidebar.markdown('<br>', unsafe_allow_html=True)

st.sidebar.markdown("### **📁File Uploading**") 
uploaded_file = st.sidebar.file_uploader("**Select Your Content File To Work With**", type=["mp3", "wav","mp4"])
upload_button = st.sidebar.button("### **Confirm Data File Upload!**")

if uploaded_file is None: 
    try:
        os.remove("Content.txt")
        os.remove("VectorDB.index")
        os.remove("QuestionBank.pdf")
        st.session_state.index = None
        st.session_state.qa_chunks = None
        st.session_state.path = None
        st.session_state.status = False

    except: 
        pass

if upload_button:
    if uploaded_file is not None:

        file_path = os.path.join("Downloads",uploaded_file.name)

        with open(file_path, "wb") as fh: 
            fh.write(uploaded_file.read())
            st.session_state.path = file_path

        with st.spinner('###### **Generating Transcript!**'):
            transcriber = Transcript(st.session_state.path)
            transcript_time = transcriber.generate_transcript()
            st.success(f"###### **Transcript Generated In {transcript_time}Secs!**")

        with open("Content.txt") as file: 
            content = file.read()
            st.session_state.qa_chunks = split_text_into_chunks_qa(content)

        with st.spinner("###### **Creating Vector Index / Database!**"):
            start = time.time()
            create_vector_store()
            end = time.time()

            total = round((end-start),2)
            st.session_state.index = load_knowledgebase()
            st.success(f"###### **Vector Index Created In {total}Secs!**")
        
        with open("Prompts/Instructions.txt") as file: 
            instructions = file.read()
        st.markdown(instructions,unsafe_allow_html=True)

        st.session_state.status = True

    else: 
        st.error("###### **No Content File To Work With! Please Upload File To Get Started!**")

st.sidebar.markdown("<br><br>", unsafe_allow_html=True)

st.sidebar.markdown("### **📛 Whisper Flagship Features**")

summarize_button = st.sidebar.button("### **Click The Button To Summarize Content!**")
if summarize_button:
    if st.session_state.status: 
        
        with st.spinner("###### **Generating Summary!**"):

            with open("Content.txt") as fh: 
                content = fh.read()

            chunks = split_text_into_chunks_summary(content)
            prev = None

            for chunk in chunks:
                prompt = summarization_template.format(previous_summary=prev, text=chunk)

                if st.session_state.llm_model == 0: 
                    summary = llama_response(prompt)
                    st.markdown(summary)
                    prev = summary
                
                else: 
                    try:
                        summary = gemini_response(prompt)
                        st.markdown(summary)
                        prev = summary

                    except Exception as e:
                        st.warning("###### **Change To Llama3 : 7B Instruct**")
                        st.error(f"###### **Gemini Throttling Error Code : {e}**")
                        break
            
            else: 
                st.success('**Content Summarized Succesfully!**')
    
    else: 
        st.error("###### **No Content File To Work With! Please Upload File To Get Started!**")

quiz_button = st.sidebar.button("### **Click The Button To Generate Quiz Based On Content!**")
if quiz_button:

    if st.session_state.status:

        if os.path.isfile("QuestionBank.pdf"):
            st.warning("Quiz already exists")

        else: 
            with open("Content.txt") as file: 
                content = file.read()
            
            if st.session_state.llm_model == 0:
                with st.spinner('###### **Generating MCQ Questions**'):
                    mcqs = json.loads(llama_response(mcq_template.format(context=content)))
                    st.success("**MCQs Generated Succesfully!**")

                with st.spinner('###### **Generating TRUE/FALSE Questions**'):
                    tfls = json.loads(llama_response(tf_template.format(context=content)))
                    st.success("**TRUE/FALSE Generated Succesfully!**")
                
                with st.spinner('###### **Generating FILL IN THE BLANKS Questions**'):
                    fib = json.loads(llama_response(fib_template.format(context=content)))
                    st.success("**FILL IN THE BLANKS Generated Succesfully!**")
                
                with st.spinner('###### **Generating Short Answer Questions**'):
                    sa = json.loads(llama_response(sa_template.format(context=content)))
                    st.success("**SHORT ANSWER Generated Succesfully!**")
                
                create_question_paper(mcqs,fib,tfls,sa)
                st.markdown("<br><br>", unsafe_allow_html=True)
                st.success("**QuestionBank Created And Saved To Local Directory!**")    

            
            else:
                try:
                    with st.spinner('###### **Generating MCQ Questions**'):
                        mcqs = json.loads(gemini_response(mcq_template.format(context=content)))
                        st.success("**MCQs Generated Succesfully!**")

                    with st.spinner('###### **Generating TRUE/FALSE Questions**'):
                        tfls = json.loads(gemini_response(tf_template.format(context=content)))
                        st.success("**TRUE/FALSE Generated Succesfully! ")
                    
                    with st.spinner('###### **Generating FILL IN THE BLANKS Questions**'):
                        fib = json.loads(gemini_response(fib_template.format(context=content)))
                        st.success("**FILL IN THE BLANKS Generated Succesfully!")
                    
                    with st.spinner('###### **Generating Short Answer Questions**'):
                        sa = json.loads(gemini_response(sa_template.format(context=content)))
                        st.success("**SHORT ANSWER Generated Succesfully!**")
                    
                    create_question_paper(mcqs,fib,tfls,sa)
                    st.success("**QuestionBank Created And Saved To Local Directory!**")

                except Exception as e:
                    st.warning("###### **Change To Llama3 : 7B Instruct**")
                    st.error(f"###### **Gemini Throttling Error Code : {e}**")
        
    else: 
        st.error("###### **No Content File To Work With! Please Upload File To Get Started!**")

question = st.sidebar.text_input("\t", placeholder="Type Your Question Here!", label_visibility='collapsed')
if st.sidebar.button("### **Get Answer**"):
    if question:

        if st.session_state.status: 
            context = load_context(question, st.session_state.qa_chunks, st.session_state.index)
            prompt = question_template.format(question=question, context=context)

            if st.session_state.llm_model == 0: 
                with st.spinner("Generating Answer"): 
                    answer = llama_response(prompt)
                    st.markdown(f"#### {question}")
                    st.markdown(answer)
            
            else: 
                try: 
                    with st.spinner("Generating Answer"): 
                        answer = gemini_response(prompt)
                        st.markdown(f"#### **{question}**")
                        st.markdown(answer)
                except: 
                    st.error("##### **Gemini API Limit Reached! Switch To Local LLM")                
        
        else: 
            st.error("###### **No Content File To Work With! Please Upload File To Get Started!**")

    if not question:
        st.error("###### **No Content File To Work With! Please Upload File To Get Started!**")