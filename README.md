# ENGenieAI

ENGenieAI is an interactive AI tutor designed to assist engineering students by providing Socratic questioning and document-based Q&A functionalities. It leverages advanced language models and allows seamless integration with PDF documents for context-aware assistance. Currently the socratic mode is available only for engineering topics

## Features

- **Socratic Mode**: Engages users with thought-provoking questions to facilitate learning.
- **Document-Based Q&A**: Upload PDF files and get context-specific answers.
- **Theme Selection**: Choose between a default and dark theme.
- **Chat History Management**: View and clear chat history.
- **Patch Notes and Support**: View patch notes and contact support via the settings.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/engenieai.git
    cd engenieai
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Set up environment variables:
    Create a `.env` file in the root directory and add your API keys:
    ```plaintext
    GOOGLE_API_KEY=your_google_api_key
    ```

5. Run the application:
    ```bash
    streamlit run frontend.py
    ```

## Usage

1. **Socratic Mode**:
    - Toggle the Socratic Mode checkbox on the main page to enable or disable Socratic questioning.
    - When enabled, the AI tutor will engage with questions to stimulate critical thinking.

2. **Document-Based Q&A**:
    - Upload PDF files using the "Upload Files" button.
    - Click "Document Embeddings" to process the documents.
    - Ask questions in the text input to get context-specific answers from the documents.
    - Use the "Delete Uploaded Files" button to remove documents and reset the session.

3. **Settings**:
    - **Patch Notes**: View the latest updates and changes.
    - **Support**: Contact support at `engenieai.query@gmail.com`.

4. **Chat History**:
    - View past interactions in the sidebar under "History".
    - Clear chat history using the "Clear Chat History" button.

## Project Structure

engenieai/

├── backend.py # Backend logic for document handling and LLM interactions

├── frontend.py # Streamlit application frontend

├── rag.py # Retrieval-Augmented Generation (RAG) logic

├── requirements.txt # Python dependencies

├── .env # Environment variables

└── README.md # Project documentation

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Make your changes and commit them with descriptive messages.
4. Push your changes to your fork.
5. Create a pull request to the main repository.

## Contact

For support or any questions, please contact us at:
**engenieai.query@gmail.com**
