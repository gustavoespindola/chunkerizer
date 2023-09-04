import streamlit as st
import re

from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, Language

# Tokenize (OpenAI)
# This code creates a token encoding object for the CL100K_BASE
# encoding scheme. The CL100K_BASE encoding scheme is used to encode tokens for the CL100K dataset.
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

import pandas as pd

# From https://python.langchain.com/docs/modules/data_connection/document_transformers
splitters=[
  'Split code',
  'Split by character',
  'Recursively split by character',
  'Split by tokens (OpenAI)'
]

languages=['CPP', 'GO', 'JAVA', 'JS', 'PHP', 'PROTO', 'PYTHON', 'RST', 'RUBY', 'RUST', 'SCALA', 'SWIFT', 'MARKDOWN', 'LATEX', 'HTML', 'SOL']
files_types=[
  {
    "extension": "txt",
    "language": "MARKDOWN",
  },
  {
    "extension": "py",
    "language": "PYTHON",
  },
  {
    "extension": "php",
    "language": "PHP",
  },
  {
    "extension": "html",
    "language": "HTML",
  },
  {
    "extension": "js",
    "language": "JS",
  },
  {
    "extension": "css",
    "language": "CSS",
  },
  {
    "extension": "md",
    "language": "MARKDOWN",
  }
]

# Store the file extension in session state
if 'file_extension' not in st.session_state:
  st.session_state['file_extension'] = ''

# Store the file language in session state
if 'file_language' not in st.session_state:
  st.session_state['file_language'] = ''

# Store the tokenized content in session state
if 'tokenized_content' not in st.session_state:
  st.session_state['tokenized_content'] = 0


def file_upload():
  if uploaded_file is not None:
    # Get the file extension and save it in session state
    st.session_state['file_extension'] = uploaded_file.name.split(".")[-1]
    # Find the language of the file
    for file in files_types:
      if file["extension"] == st.session_state['file_extension']:
        st.session_state['file_language'] = file["language"].lower()
        break

    return uploaded_file.getvalue().decode("utf-8")

# Create Top KPI's
def metrics(chunks, tokens, characters):
  col1, col2, col3 = st.columns(3)
  col1.metric("Chunks", chunks)
  token_ratio=round(tokens/st.session_state['tokenized_content'], 2)
  col2.metric("Total Tokens", tokens, token_ratio, delta_color="inverse" if token_ratio!=1 else "off")
  col3.metric("Total Characters", characters)

# Create dataframe table
def create_dataframe(text_splitter, file_content):
  chunks=text_splitter.create_documents([file_content])
  st.session_state['tokenized_content']=len(enc.encode(file_content))
  df = pd.DataFrame(
    {
      "Text": [chunk.page_content for chunk in chunks],
      "Tokens": [len(enc.encode(chunk.page_content)) for chunk in chunks],
      "Characters": [len(chunk.page_content) for chunk in chunks],
    }
  )
  metrics(len(df), df['Tokens'].sum(), df['Characters'].sum())
  if len(df) < 1:
    st.error('No content found.')
  else:
    st.dataframe(df, use_container_width=True)


#
#
# MAIN
st.set_page_config(
  page_title="Chunkerizer",
  page_icon="✂️",
  layout="wide",
  initial_sidebar_state="expanded",
  menu_items={
      'Get Help': 'https://www.extremelycoolapp.com/help',
      'Report a bug': "https://www.extremelycoolapp.com/bug",
      'About': "# This is a header. This is an *extremely* cool app!"
  }
)
st.title("🦜️✂️ Chunkerizer")
st.markdown("""#### Using [@langchain](https://python.langchain.com/docs/modules/data_connection/document_transformers/) text splitter to chunk code and text files.""")
st.divider()

#
#
# SIDEBAR
uploaded_file=st.sidebar.file_uploader(
  "Drop a file here or click to upload",
  type=[file["extension"] for file in files_types],
  accept_multiple_files=False
)

file_content=file_upload()

if uploaded_file is not None:
  splitter=st.sidebar.selectbox(
    'Select a splitter',
    splitters,
    # if file extension is txt plain
    index=1 if st.session_state['file_extension']=="txt" else 0
  )
  splitter_index=splitters.index(splitter)
  if splitter_index==0:
    language_selector=st.sidebar.selectbox('Select a language', languages, index=languages.index(st.session_state['file_language'].upper()))
  if splitter_index==1:
    text_splitter_separator=st.sidebar.text_input('Separator', value="\n\n", help="By default the characters it tries to split", placeholder="Enter a separator like \\n\\n", max_chars=10, key="text_splitter_separator", type="default")
  st.sidebar.divider()
  chunk_size=st.sidebar.slider('Chunk Size', 1, 5000, 1000)
  chunk_overlap=st.sidebar.slider('Overlap', 0, chunk_size, 0)

  # Code
  if splitter_index==0:
    code_splitter = RecursiveCharacterTextSplitter.from_language(
      language=Language[language_selector],
      chunk_size=chunk_size,
      chunk_overlap=chunk_overlap
    )
    chunks=code_splitter.create_documents([file_content])
    st.session_state['tokenized_content']=len(enc.encode(file_content))
    df = pd.DataFrame(
      {
        "Text": [chunk.page_content for chunk in chunks],
        "Tokens": [len(enc.encode(chunk.page_content)) for chunk in chunks],
        "Characters": [len(chunk.page_content) for chunk in chunks],
      }
    )
    metrics(len(df), df['Tokens'].sum(), df['Characters'].sum())
    st.dataframe(df, use_container_width=True)


  # if is character splitter
  if splitter_index==1:
    # transform any like \\n\\n or \\r \\t into \n\n or \t
    text_splitter_separator=re.sub(r'\\', '', text_splitter_separator)
    # st.write(bytes(text_splitter_separator, 'ascii', 'ignore').decode("unicode_escape"))
    text_splitter=CharacterTextSplitter(
      separator=text_splitter_separator,
      chunk_size=chunk_size,
      chunk_overlap=chunk_overlap,
      # length_function=len,
      # is_separator_regex=False
    )
    create_dataframe(text_splitter, file_content)

  # if is recursive
  if splitter_index==2:
    text_splitter=RecursiveCharacterTextSplitter(
      chunk_size=chunk_size,
      chunk_overlap=chunk_overlap,
      # length_function=len,
      # is_separator_regex=False
    )
    create_dataframe(text_splitter, file_content)

  # if is token splitter
  if splitter_index==3:
    text_splitter=CharacterTextSplitter.from_tiktoken_encoder(
      chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    create_dataframe(text_splitter, file_content)

  with st.expander(f"View Original File {uploaded_file.name} – {st.session_state['file_language']}"):
    st.code(file_content, language=f"{st.session_state['file_language']}", line_numbers=True)

#Credits
st.sidebar.markdown("This project was created by [Gustavo Espíndola](https://github.com/gustavoespindola). 🆘 If you have any questions or feedback, please contact in our [Discord](https://discord.gg/mZf5aaYt) or [Github Repo](https://github.com/gustavoespindola/chunkerizer).")