# Langchain Text Splitter
This is a Python application that allows you to split and analyze text files using different methods, including character-based splitting, recursive character-based splitting, and token splitting using OpenAI's CL100K_BASE encoding scheme. It is designed to work with various programming languages and file formats.

## Features
- **File Upload**: You can upload a text file of your choice.
- **Splitting Methods**: Choose from different splitting methods, including:
    - ✅ Split code
        - MARKDOWN, PYTHON, PHP, HTML, CSS, JS
    - ✅ Split by character
    - ✅ Recursively split by character
    - Split by tokens
        - ✅ OpenAi Tiktoken
        - ⏳ spaCy (coming soon)
        - ⏳ SentenceTransformers (coming soon)
        - ⏳ NLTK (coming soon)
        - ⏳ Hugging Face tokenizer

- **Language Support**: The application supports various programming languages, including CPP, GO, JAVA, JS, PHP, PROTO, PYTHON, RST, RUBY, RUST, SCALA, SWIFT, MARKDOWN, LATEX, HTML, and SOL.
- **Chunk Size Control**: You can control the chunk size and overlap when splitting the text.
- **Analysis**: The application provides detailed analysis of the split text, including the number of chunks, tokens, and characters in each chunk.

## Usage
Upload a text file by dropping it into the designated area or clicking the upload button.
Select the splitting method based on your preference.
Adjust the chunk size and overlap if necessary.
If you chose the **"Split code"** method, select the programming language.
If you chose the "Split by character" method, specify the separator.
View the analysis results, including the number of chunks, tokens, and characters in each chunk.

## Dependencies
-   Streamlit: Used for creating the web-based user interface.
-   Tiktoken: Used for token encoding with the CL100K_BASE encoding scheme.
-   Pandas: Used for data manipulation and displaying results.
-   Re: This regex implementation

## Installation
To run this application, make sure you have Python installed on your system. Then, follow these steps:

1.  Clone this repository to your local machine.
2.  Install the required dependencies using pip: `pip install`
3.  Run the project usign `streamlit run chunkerizer.py`

## Author
This application was created by [Gustavo Espíndola](https://github.com/gustavoespindola). If you have any questions or feedback, please contact in our [CodeGPT Discord](https://discord.gg/mZf5aaYt) .

## License
This project is licensed under the MIT License.
