# RAG (Retrieval-Augmented Generation) Application

A Django-based web application that uses LangChain and DeepSeek AI to provide intelligent question-answering based on a knowledge base document.

## Features

- **RAG System**: Uses LangChain to create a retrieval-augmented generation system
- **DeepSeek Integration**: Connects to DeepSeek AI for advanced language model capabilities
- **Document Processing**: Processes text documents and creates embeddings for semantic search
- **Web Interface**: Clean, responsive web interface built with Django and Tailwind CSS
- **Error Handling**: Graceful error handling for missing API keys and configuration issues

## Prerequisites

- Python 3.8+
- pip
- DeepSeek API key

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd rag-app/rag-d
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   - Copy the `.env.example` file to `.env`
   - Add your DeepSeek API key:
     ```
     DEEPSEEK_API_KEY=your_actual_api_key_here
     ```
   - Get your API key from [DeepSeek Platform](https://platform.deepseek.com/)

5. **Run database migrations**
   ```bash
   python manage.py migrate
   ```

## Usage

1. **Start the development server**
   ```bash
   python manage.py runserver
   ```

2. **Open your browser**
   - Navigate to `http://127.0.0.1:8000/rag/`
   - The application will show an error message if the API key is not configured

3. **Configure the API key**
   - Update the `.env` file with your actual DeepSeek API key
   - Restart the server

4. **Ask questions**
   - Type your question in the input field
   - The system will search the knowledge base and provide AI-generated answers

## Project Structure

```
rag-d/
├── myproject/          # Django project settings
│   ├── settings.py     # Django configuration
│   ├── urls.py         # Main URL routing
│   └── wsgi.py         # WSGI configuration
├── rag/                # RAG application
│   ├── views.py        # Main application logic
│   ├── urls.py         # App-specific URL routing
│   ├── templates/      # HTML templates
│   └── knowledge_base.txt  # Sample knowledge base
├── manage.py           # Django management script
├── requirements.txt    # Python dependencies
└── .env               # Environment variables
```

## Configuration

### Environment Variables

- `DEEPSEEK_API_KEY`: Your DeepSeek API key (required)

### Knowledge Base

The application uses `rag/knowledge_base.txt` as the source document. You can replace this file with your own content to customize the knowledge base.

## Troubleshooting

### Common Issues

1. **"ModuleNotFoundError: No module named 'langchain_deepseek'"**
   - Ensure you're in the virtual environment
   - Run `pip install -r requirements.txt`

2. **"Please set your DeepSeek API key"**
   - Check that the `.env` file exists
   - Verify the API key is correctly set
   - Restart the server after making changes

3. **"Failed to initialize RAG system"**
   - Check your internet connection
   - Verify the API key is valid
   - Check the console for detailed error messages

### Debug Mode

The application runs in debug mode by default. Check the Django console output for detailed error information.

## Dependencies

Key packages used:
- Django 5.2.5
- LangChain 0.3.27
- LangChain Community 0.3.29
- LangChain DeepSeek 0.1.4
- Sentence Transformers 5.1.0
- FAISS (for vector storage)
- Python-dotenv

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the Django console output
3. Ensure all dependencies are properly installed
4. Verify your API key configuration
