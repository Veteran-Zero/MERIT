"""
MERIT Configuration
Set paths and API credentials here, or override with environment variables.
"""
import os

# ─── Directory Layout ────────────────────────────────────────────────────────
BASE_DIR = os.environ.get("MERIT_BASE_DIR", ".")

DATA_DIR   = os.path.join(BASE_DIR, "data")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
IMAGE_DIR  = os.path.join(BASE_DIR, "images")

# ─── Dataset Paths ───────────────────────────────────────────────────────────
# Stage 1 input: raw merged train+valid split
TRAIN_FILE = os.path.join(DATA_DIR, "merged_train_valid.json")

# Stage 1 output: clustered data
CLUSTERED_FILE = os.path.join(DATA_DIR, "merged_train_valid_clustered.json")
CLUSTER_META_FILE = os.path.join(DATA_DIR, "cluster_metadata.json")

# Stage 1 intermediate: cached embeddings
EMBEDDING_CACHE_FILE = os.path.join(DATA_DIR, "embeddings_cache.npy")

# Stage 2 output: annotated memory bank
MEMORY_BANK_FILE = os.path.join(DATA_DIR, "annotated_memory_bank.json")

# Stage 3 input / output
TEST_FILE   = os.path.join(DATA_DIR, "test.json")
RESULT_FILE = os.path.join(OUTPUT_DIR, "test_results.json")

# BERTopic model save path
BERTOPIC_MODEL_PATH = os.path.join(MODEL_DIR, "bertopic_merit")

# ─── API Configuration ───────────────────────────────────────────────────────
# Provide via environment variable or fill in directly.
API_KEY  = os.environ.get("OPENAI_API_KEY", "your-api-key-here")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

# Stage 1 & 3 embedding model (must be OpenAI-compatible)
EMBEDDING_MODEL = os.environ.get("MERIT_EMBEDDING_MODEL", "text-embedding-3-large")

# Stage 2 annotation model (strong reasoning capability recommended)
ANNOTATION_MODEL = os.environ.get("MERIT_ANNOTATION_MODEL", "gemini-2.5-pro")

# Stage 3 inference model
INFERENCE_MODEL = os.environ.get("MERIT_INFERENCE_MODEL", "gemini-2.5-flash")

# ─── Stage 1: Clustering ─────────────────────────────────────────────────────
BERTOPIC_NR_TOPICS    = 20
BERTOPIC_MIN_TOPIC    = 50
EMBEDDING_BATCH_SIZE  = 100
EMBEDDING_CONCURRENCY = 50

# ─── Stage 2: Memory Bank Construction ───────────────────────────────────────
TOP_K_PER_CLUSTER   = 100   # representative prototypes per cognitive schema
ANNOTATION_MAX_CONC = 500

# ─── Stage 3 & 4: Retrieval + Inference ──────────────────────────────────────
TOP_K_RETRIEVAL        = 3     # number of memory entries retrieved
SIMILARITY_THRESHOLD   = 0.4   # minimum hybrid score to keep a candidate
LENGTH_DIFF_TOLERANCE  = 20    # max char-length diff for length-based filter
ALPHA                  = 0.7   # weight for dense (vector) score in hybrid search
BETA                   = 0.3   # weight for sparse (BM25) score in hybrid search
INFERENCE_MAX_CONC     = 500
