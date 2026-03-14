"""Central configuration constants for the project."""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
FILTERED_DIR = DATA_DIR / "filtered"
KD_DIR = DATA_DIR / "kd"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
CONFIGS_DIR = EXPERIMENTS_DIR / "configs"
RESULTS_DIR = EXPERIMENTS_DIR / "results"

# ── Models ───────────────────────────────────────────────────────────────────
BASE_MODEL = "CohereForAI/aya-expanse-8b"
TEACHER_MODEL = "CohereForAI/aya-expanse-32b"
NUM_LAYERS = 32

# ── Language pair ────────────────────────────────────────────────────────────
SRC_LANG = "ces"  # Czech (ISO 639-3)
TGT_LANG = "deu"  # German (ISO 639-3)
SRC_LANG_NAME = "Czech"
TGT_LANG_NAME = "German"

# ── Data filtering thresholds (following Moslem et al.) ──────────────────────
MAX_WORDS_PER_SEGMENT = 200
LENGTH_RATIO_MAX = 1.5
LANG_DETECT_THRESHOLD = 0.9
SEMANTIC_SIM_THRESHOLD = 0.7
SEMANTIC_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# ── Dataset splits ───────────────────────────────────────────────────────────
TRAIN_SIZE = 100_000
TEST_SIZE = 500

# ── KD filtering ─────────────────────────────────────────────────────────────
KD_COMET_THRESHOLD = 0.70

# ── Pruning targets ─────────────────────────────────────────────────────────
PRUNE_TARGETS = [8, 12, 16]  # Number of layers to remove

# ── Evaluation ───────────────────────────────────────────────────────────────
COMET_MODEL = "Unbabel/wmt22-comet-da"

# ── Fine-tuning defaults ────────────────────────────────────────────────────
FT_LEARNING_RATE = 2e-5
FT_EPOCHS = 3
FT_BATCH_SIZE = 4
FT_GRAD_ACCUM = 8
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# ── Translation prompt template ─────────────────────────────────────────────
TRANSLATION_PROMPT = "Translate the following {src_lang} text to {tgt_lang}.\n\n{src_lang}: {source}\n{tgt_lang}:"
