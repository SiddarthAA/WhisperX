# WhisperX: Elevate Media Insights with Precision

![WhisperX Logo](https://i.pinimg.com/564x/b2/d6/0a/b2d60ad18a820681922defbbd0fccc10.jpg)

**WhisperX** is an advanced media processing tool designed to transform audio and video content into actionable insights. Leveraging cutting-edge technologies, WhisperX offers unparalleled **Summarization**, **Interactive Question Answering** through a sophisticated RAG-based approach, and dynamic **Quiz Generation**.

## 🌟 **Key Features**

- **Summarization**: Rapidly distill audio and video content into concise, actionable summaries.
- **Interactive Question Answering**: Utilize an advanced Retrieval-Augmented Generation (RAG) approach to generate precise and contextually relevant answers from transcribed content.
- **Quiz Generation**: Automatically create engaging quizzes from content summaries to support learning and assessment.

## 🔧 **Tech Stack**

- **Llama3 and Gemini**: Leading large language models (LLMs) providing robust language processing capabilities.
- **OpenAI Whisper/Tiny.en**: High-performance transcription tool for converting media to text.
- **All-MPNet-Base-V2**: Generates high-quality vector embeddings for enhanced semantic understanding.
- **FAISS**: Implements fast and scalable vector indexing and retrieval for efficient search operations.

## 🌐 **Architecture Overview**

![Architecture Diagram](https://i0.wp.com/www.phdata.io/wp-content/uploads/2023/11/image1-3.png)

**WhisperX** integrates a sophisticated architecture for seamless media processing:

1. **Transcription**: Convert audio and video to text using Whisper/Tiny.en.
2. **Embedding Creation**: Generate vector embeddings with All-MPNet-Base-V2.
3. **Indexing and Retrieval**: Utilize FAISS for managing and querying embeddings.
4. **Question Answering**: Employ the RAG-based approach to retrieve and generate precise answers.

## 🚀 **Getting Started**

### **Prerequisites**

Ensure you have **Ollama** installed. Pull the Llama3 model using:
```bash
ollama pull llama3
```

### **Setup Instructions**

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/whisperx.git
   cd whisperx
   ```

2. **Install Dependencies And Setup Environment**
   ```bash
   python setup.py
   ```

3. **Launch the Application**
   ```bash
   streamlit run App.py
   ```

## 🛠️ **Usage**

1. **Upload**: Add your audio or video file via the application interface.
2. **Select**: Choose the appropriate model for summarization, question answering, and quiz generation.
3. **Configure**: Adjust settings including model temperature and language options.
4. **Generate**: Obtain summaries, answers, and quizzes from the uploaded media.

## 🤝 **Collaborations**

We welcome contributions and collaborations to further enhance WhisperX. If you're interested in collaborating, please get in touch!

## 📜 **Acknowledgements**

- **Llama3 and Gemini Teams**: For their exceptional language models.
- **OpenAI**: For Whisper/Tiny.en, which powers our transcription capabilities.
- **All-MPNet and FAISS**: For their powerful embedding and indexing tools.
- **Our Contributors**: Heartfelt thanks to everyone who has supported the development of WhisperX.

## ✨ **Conclusion**

WhisperX redefines media analysis with its advanced features for summarizing, querying, and quiz generation. Embrace the power of Generative AI to enhance your media processing and learning experiences.

## 📫 **Contact**

For inquiries or support, please reach out:
- **Name**: Siddartha Aralakuppe Yogesha
- **Email**: siddartha_ay@protonmail.com
