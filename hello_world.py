# Import necessary libraries
from transformers.pipelines import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers.utils.logging import set_verbosity_error
import os

# Suppress detailed logging from the transformers library
set_verbosity_error()

# Get Hugging Face token from environment variable (set this in your environment)
# export HF_TOKEN="your_hugging_face_token_here"
my_secret_key = os.getenv('HF_TOKEN')  # Hugging Face token

# Initialize a summarization pipeline using Facebook's BART model (cnn-trained version)
# This model will generate an initial summary
summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn", device=0)
summarizer = HuggingFacePipeline(pipeline=summarization_pipeline)

# Initialize a second summarization pipeline for refining the summary using a different BART model
refinement_pipeline = pipeline("summarization", model="facebook/bart-large", device=0)
refiner = HuggingFacePipeline(pipeline=refinement_pipeline)

# Initialize a question-answering pipeline using the RoBERTa model fine-tuned on SQuAD2
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2", device=0)

# Define a prompt template for summarization length control
summary_template = PromptTemplate.from_template("Summarize the following text in a {length} way:\n\n{text}")

# Create a chain: Apply the template ➜ summarize ➜ refine
summarization_chain = summary_template | summarizer | refiner

# Take user input for the text to be summarized and preferred summary length
text_to_summarize = input("\nEnter text to summarize:\n")
length = input("\nEnter the length (short/medium/long): ")

# Execute the summarization chain using the user-provided input
summary = summarization_chain.invoke({"text": text_to_summarize, "length": length})

# Display the final summary
print("\n **Generated Summary:**")
print(summary)

# Interactive loop for asking questions about the generated summary
while True:
    question = input("\nAsk a question about the summary (or type 'exit' to stop):\n")
    if question.lower() == "exit":
        break

    # Use the QA pipeline to answer questions using the generated summary as context
    qa_result = qa_pipeline(question=question, context=summary)

    print("\n **Answer:**")
    if qa_result and hasattr(qa_result, 'get'):
        print(qa_result.get("answer", "No answer found"))
    else:
        print("Unable to generate answer")