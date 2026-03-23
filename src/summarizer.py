
# src/summarizer.py - NLP Text Summarization Module

from transformers import pipeline

class TextSummarizer:
    def __init__(self, model_name="sshleifer/distilbart-cnn-12-6"):
        # Initialize a summarization pipeline using a pre-trained model
        # distilbart-cnn-12-6 is a good general-purpose summarization model
        self.summarizer = pipeline("summarization", model=model_name)

    def summarize_text(self, text, min_length=30, max_length=150):
        """
        Summarizes the input text using the initialized summarization model.

        Args:
            text (str): The input text to be summarized.
            min_length (int): The minimum length of the generated summary.
            max_length (int): The maximum length of the generated summary.

        Returns:
            str: The summarized text.
        """
        if not text or len(text.strip()) == 0:
            return "Input text cannot be empty."

        # The summarizer returns a list of dictionaries, we want the 'summary_text'
        summary = self.summarizer(text, min_length=min_length, max_length=max_length, do_sample=False)
        return summary[0]["summary_text"]

    def _extractive_summarize(self, text, num_sentences=3):
        """
        (Placeholder) Performs extractive summarization.
        In a real implementation, this would involve sentence tokenization, scoring,
        and selection of the most important sentences.
        """
        sentences = text.split(". ") # Simple split, needs proper sentence tokenizer
        if len(sentences) <= num_sentences:
            return text
        return ". ".join(sentences[:num_sentences]) + "."

if __name__ == "__main__":
    summarizer = TextSummarizer()
    
    long_text = """
    Artificial intelligence (AI) is intelligence—perceiving, synthesizing, and inferring information—demonstrated by machines, as opposed to intelligence displayed by animals or by humans. Example tasks in which AI is used include speech recognition, computer vision, translation between (natural) languages, and other mappings of inputs. AI applications include advanced web search engines (e.g., Google Search), recommendation systems (e.g., YouTube, Amazon, Netflix), understanding human speech (such as Siri and Alexa), self-driving cars (e.g., Waymo), generative or creative tools (ChatGPT and AI art), and competing at the highest level in strategic game systems (such as chess and Go). Artificial intelligence was founded as an academic discipline in 1956, and in the years since has experienced several waves of optimism, followed by disappointment and the loss of funding (known as an "AI winter"), followed by new approaches, success and renewed funding. For most of its history, AI research has been divided into subfields that often fail to communicate with each other. These sub-fields are based on technical considerations, such as goals (e.g. "robotics" or "machine learning"), the use of particular tools (e.g. "logic" or "neural networks"), or deep philosophical differences. Subfields have also been based on social factors (particular institutions or the work of individual researchers). The central problems (or goals) of AI research include reasoning, knowledge representation, planning, learning, natural language processing, perception, and the ability to move and manipulate objects. General intelligence (the ability to solve an arbitrary problem) is among the field's long-term goals. To solve these problems, AI researchers have adapted and integrated a wide range of problem-solving techniques, including search and mathematical optimization, formal logic, artificial neural networks, and methods based on statistics, probability and economics. AI also draws upon computer science, psychology, linguistics, philosophy, neuroscience, and artificial psychology. """

    print("\nOriginal Text:")
    print(long_text)
    
    abstractive_summary = summarizer.summarize_text(long_text, min_length=50, max_length=100)
    print("\nAbstractive Summary:")
    print(abstractive_summary)

    extractive_summary = summarizer._extractive_summarize(long_text, num_sentences=2)
    print("\nExtractive Summary (Placeholder):")
    print(extractive_summary)

    print("\nNLP Text Summarization module initialized.")

# This module provides functionality for text summarization using pre-trained transformer models.
# It leverages the Hugging Face transformers library for abstractive summarization.
# The TextSummarizer class encapsulates the summarization logic.
# It demonstrates how to use a pipeline for a common NLP task.
# The code is designed to be straightforward and easy to understand.
# It includes a placeholder for extractive summarization, which can be expanded.
# This project is ideal for showcasing NLP skills and practical applications.
# Further enhancements could include fine-tuning models, supporting multiple languages,
# and integrating with web interfaces.
# The use of pre-trained models significantly reduces the effort required for high-quality summarization.
# This is a great example of applying state-of-the-art NLP techniques.
# The comments explain the purpose of different sections and potential improvements.
# It's a valuable asset for anyone interested in natural language processing.
# Enjoy exploring the world of text summarization!
