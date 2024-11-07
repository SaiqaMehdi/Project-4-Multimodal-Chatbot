
#📚 Multimodal AI Chatbot for YouTube Book Review Analysis

## Project Overview
This project creates a multimodal chatbot interface using Gradio for both text and voice inputs. It’s designed to answer user questions based on YouTube video transcripts, using LangChain for advanced natural language processing and Pinecone for efficient vector-based querying and storage.

## 🔑 Key Features
Multimodal Interactions: Supports both text and voice inputs via Gradio.
Advanced NLP Capabilities: Utilizes LangChain with OpenAI and Pinecone for complex language processing, embedding generation, and retrieval.
Scalable Data Handling: Pinecone, a vector database, enables real-time, efficient querying and response generation.

## 📦 Installation
Ensure all dependencies are installed before running the project:

bash
Copy code
pip install gradio whisper langchain pinecone-client openai

## 📝 Data Preprocessing
Convert YouTube transcription text files into a structured JSON format and prepare them for embedding.

JSON Conversion: Transcription text is read from .txt files and converted into JSON format.
Text Chunking: Content is split into manageable chunks using LangChain’s RecursiveCharacterTextSplitter for optimized embedding.

## 🔍 Embedding Generation
Each text chunk is embedded using the OpenAIEmbeddings model from LangChain, creating dense vector representations that support efficient semantic search.

## 🗂️ Vector Database Setup
Embeddings are stored in Pinecone, allowing fast and accurate similarity searches that are essential for the chatbot’s retrieval-based functionality.

## ⚙️ Retrieval and Response Generation
Implements a Retrieval-Augmented Generation (RAG) system to produce accurate responses.

Retrieval: Queries are embedded and matched against chunk embeddings in Pinecone, retrieving relevant text segments.
Response Generation: The LangChain agent uses these retrieved chunks as context to generate accurate responses through GPT-4.

## 💻 Interface
A Gradio interface enables user interaction with the chatbot:

Text Input: Users can type questions related to YouTube book reviews.
Voice Input: Supports audio questions, which are transcribed using the Whisper model integrated via LangChain.

## 🧪 Testing
Local Testing: Run retrieval and response generation tests to ensure accuracy and relevance.
Automated Testing: Utilizes Giskard for automated testing and evaluation of chatbot responses.

## 🚀 Usage
To start the application, run:

bash
Copy code
python app.py
This command launches the Gradio interface, allowing users to interact with the chatbot through their web browser.

## 🤝 Contribution
Contributions are welcome! Please fork the repository, make your changes, and submit a pull request.

## 📜 License
This project is licensed under the MIT License. See the LICENSE.md file for details.

