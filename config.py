# CONFIGURATION
import os


# ==============================
# GOOGLE API KEY
GOOGLE_API_KEY = os.getenv("GOOGLEAPIKEY")

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY n'est pas configurée."
    )


# ==============================
# LANGSMITH API KEY
LANGSMITH_API_KEY = os.getenv("LANGCHAINKEY")

if not LANGSMITH_API_KEY:
    raise ValueError(
        "LANGSMITH_API_KEY n'est pas configurée."
    )


# ==============================
# CONFIGURATION LANGSMITH
os.environ["LANGSMITH_TRACING"] = "true"

os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY

os.environ["LANGSMITH_PROJECT"] = "NovaTech-RAG"


# ==============================
# CONFIGURATION DU PROJET
DOCUMENTS_PATH = "documents"

GEMINI_MODEL = "gemini-3.5-flash"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 400

CHUNK_OVERLAP = 50

TOP_K = 3