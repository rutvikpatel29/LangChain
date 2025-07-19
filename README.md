# LangChain Hello World Project

This project demonstrates a text summarization and question-answering system using LangChain and Hugging Face transformers.

## Features

- Text summarization using BART models
- Question-answering using RoBERTa model
- Interactive Q&A loop
- Chain-based processing with LangChain

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Hugging Face token (optional but recommended):**
   ```bash
   export HF_TOKEN="your_hugging_face_token_here"
   ```
   
   You can get a free token from [Hugging Face](https://huggingface.co/settings/tokens)

3. **Run the application:**
   ```bash
   python hello_world.py
   ```

## Usage

1. Enter the text you want to summarize when prompted
2. Choose the summary length (short/medium/long)
3. Ask questions about the generated summary
4. Type 'exit' to quit the Q&A loop

## Dependencies

The main dependencies include:
- `langchain` - Core LangChain functionality
- `transformers` - Hugging Face transformers library
- `torch` - PyTorch for deep learning
- `huggingface-hub` - Access to Hugging Face models

## Notes

- The script requires significant computational resources for model loading
- GPU acceleration is recommended for better performance
- Models will be downloaded automatically on first run

## Troubleshooting

If you encounter import errors:
1. Make sure all dependencies are installed: `pip install -r requirements.txt`
2. Check that you're using Python 3.8 or higher
3. For GPU issues, ensure CUDA is properly installed if using NVIDIA GPUs 