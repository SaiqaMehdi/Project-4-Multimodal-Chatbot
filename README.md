# 🌐 Multimodal AI Chatbots for *Wuthering Heights* Analysis

## 📖 Project Overview
This repository contains two multimodal AI chatbots for *Wuthering Heights*:

1. **Literary Analysis Chatbot**: Analyzes the text of *Wuthering Heights* to answer thematic and character-based questions.
2. **YouTube Review Analysis Chatbot**: Analyzes YouTube reviews on *Wuthering Heights*, allowing users to ask questions based on these insights.

Both models use LangChain for language processing, Pinecone for embedding storage and retrieval, and Gradio for a multimodal interface.

## Project 1: Literary Analysis Chatbot
### 📝 Project Overview
This chatbot provides detailed literary analysis on Wuthering Heights, using metadata-enriched embeddings and a Retrieval-Augmented Generation (RAG) pipeline. The model can answer in-depth questions about themes, symbolism, and narrative techniques based on the text of the novel.

## 🔑 Key Features
- **Data Preprocessing**: Converts Excel data on Wuthering Heights into JSON format with cleaned text and metadata.
- **Text Chunking and Embedding**: Splits the text into manageable chunks, embeds these chunks with metadata, and stores them in Pinecone.
- **RAG Pipeline**: Uses LangChain’s RAG pipeline to retrieve and analyze relevant text chunks, including context like chapter and paragraph details.
- **User Interface**: Gradio interface for text and audio input, with haunting audio responses for an immersive user experience.

## Code Outline

### 📂 Data Preprocessing
Load and clean Excel data, then convert it to JSON format.

```python
# Load data and clean text
import pandas as pd
df = pd.read_excel("Wuthering_Heights_Chapter1_Complete_Analysis_1.xlsx")
# Additional data cleaning steps...
```

### 🔍 Embedding and Storage in Pinecone
Embed text chunks with OpenAI embeddings and store them in Pinecone.

```python
from langchain.embeddings.openai import OpenAIEmbeddings
embed = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key="your_openai_api_key")
```

### ⚙️RAG Pipeline with LangChain Agent
Retrieve relevant text chunks using Pinecone, then generate answers.

```python
from langchain.agents import Tool
def retrieve_docs(question):
    # Retrieval logic here...
```

## 💻 User Interface
Gradio interface with support for audio transcription and haunting audio output, enables user interaction with the chatbot:

- **Text Input**: Users can type questions related to YouTube book reviews.
- **Voice Input**: Supports audio questions, which are transcribed using the Whisper model integrated via LangChain.User Interface

```python
import gradio as gr
# Gradio interface setup here...
```

## 📝 Project 2: YouTube Review Analysis Chatbot
### Project Overview
This chatbot provides analysis based on YouTube reviews of Wuthering Heights, allowing users to ask questions about themes, characters, and public opinions reflected in these reviews. The chatbot uses transcripts of YouTube reviews as data and employs a RAG pipeline for context-based answers.

## 🔑 Key Features
- **Data Conversion and Chunking**: Converts raw review transcripts to JSON format and splits the text into chunks for efficient embedding and retrieval.
- **Embeddings and Storage in Pinecone**: Stores chunk embeddings in Pinecone, enabling vector-based retrieval.
- **RAG and LangChain Agent**: Retrieves relevant chunks based on user queries, and uses GPT for answer generation.
- **User Interface**: Gradio interface with text and audio input, allowing for engaging interactions with YouTube review insights.

## 📑 Code Outline

### 📂 Data Conversion and Chunking
Load text file data, convert to JSON, and chunk for efficient retrieval.

```python
import json
# Read and chunk YouTube transcripts...
```

### 🗂️ Embedding and Storage in Pinecone
Generate embeddings for each text chunk and upload to Pinecone.

```python
from langchain.embeddings.openai import OpenAIEmbeddings
embed = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key="your_openai_api_key")
```

## ⚙️RAG Pipeline
Retrieve relevant review chunks and use LangChain agent to answer questions based on the review data.

```python
from langchain.agents import Tool
def retrieve_docs(question):
    # Retrieval logic here...
```

## 💻 User Interface
Gradio interface with support for audio transcription and haunting audio output, enables user interaction with the chatbot:

Text Input: Users can type questions related to YouTube book reviews.
Voice Input: Supports audio questions, which are transcribed using the Whisper model integrated via LangChain.

```python
import gradio as gr
# Gradio interface setup here...
```

## 🚀 Installation and Setup

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys Set up environment variables for OpenAI and Pinecone API keys:

```bash
export OPENAI_API_KEY='your_openai_api_key'
export PINECONE_API_KEY='your_pinecone_api_key'
```

### 3. Run the Chatbot Interfaces Launch each chatbot interface with Gradio.

```bash
python literary_analysis_chatbot.py  # For Literary Analysis Chatbot
python review_analysis_chatbot.py  # For YouTube Review Analysis Chatbot
```

## 📚 Future Enhancements
- **Expand to More Books and Reviews**: Add support for analyzing other Gothic literature and their corresponding reviews.
- **Enhanced Metadata Filtering**: Improve metadata handling for more precise retrieval.
- **Additional Language Models**: Incorporate models tailored for literary and thematic analysis.


## 👩‍💻 About Me
I am an AI engineer passionate about the intersection of artificial intelligence and literature. This project represents my fascination with *Wuthering Heights* and my dedication to exploring new applications of AI in literary analysis. By combining advanced NLP models, multimodal interaction through Gradio, and vector-based search with Pinecone, I aim to create interactive tools for exploring classic literature. My goal is to expand this project, adding more texts and analyses to build a comprehensive resource for readers, students, and researchers.

## 🤝 Contribution
Contributions are welcome! Please fork the repository, make your changes, and submit a pull request.

## 📜 License
This project is licensed under the MIT License. See the LICENSE.md file for details.







