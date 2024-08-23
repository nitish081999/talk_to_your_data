# Document Q&A with Chat History

This Streamlit application allows users to upload various types of documents (PDF, text, URL, images, Excel files, and even connect to databases) and ask questions about their content using natural language. The app uses OpenAI's language models and embeddings to provide accurate answers based on the document's content.

## Features

- Support for multiple input types:
  - PDF files
  - Text files
  - URLs (HTML and plain text)
  - Images (NEW)
  - Excel files (NEW)
  - Database connections (NEW)
- Natural language question answering
- Chat history tracking
- OpenAI API integration for powerful language understanding

## Prerequisites

Before running the application, make sure you have Python 3.7+ installed on your system.

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/document-qa-chat.git
   cd document-qa-chat
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key:
   - Sign up for an account at https://openai.com/
   - Obtain your API key from the OpenAI dashboard

## Usage

1. Run the Streamlit app:
   ```
   streamlit run chatbot.py
   ```

2. Open your web browser and go to the URL displayed in the terminal (usually http://localhost:8501)

3. Enter your OpenAI API key when prompted

4. Choose the input type (File Upload or URL)

5. Upload a document or enter a URL

6. Ask questions about the document in natural language

7. View the answers and chat history

## New Features (Coming Soon)

- **Image Upload**: Upload and analyze images using advanced computer vision models.
- **Excel File Support**: Ask questions about data stored in Excel spreadsheets.
- **Database Integration**: Connect to various databases and query their contents using natural language.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.



## Acknowledgments

- [Streamlit](https://streamlit.io/) for the web application framework
- [LangChain](https://github.com/hwchase17/langchain) for the document processing and Q&A capabilities
- [OpenAI](https://openai.com/) for the powerful language models and embeddings
