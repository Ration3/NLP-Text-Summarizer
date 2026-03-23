
# NLP Text Summarizer

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![Hugging Face Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-orange?style=flat-square&logo=huggingface)](https://huggingface.co/transformers/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A natural language processing (NLP) project for abstractive and extractive text summarization using state-of-the-art transformer models.

## Features
- Abstractive summarization with pre-trained models
- Modular design for easy integration
- Extensible for various NLP tasks

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
import torch
from src.summarizer import TextSummarizer

summarizer = TextSummarizer()
long_text = "Your long text here..."
summary = summarizer.summarize_text(long_text)
print(summary)
```

## Project Structure

```
. \
├── src\
│   └── summarizer.py
├── requirements.txt
└── README.md
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
