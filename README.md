#  Romanized Nepali Sentiment Analyzer (End-to-End NLP Pipeline)

An end-to-end, production-grade deep learning pipeline designed to clean, normalize, and classify sentiment patterns in code-mixed, phonetically noisy Romanized Nepali text. Built on a fine-tuned **DistilBERT Architecture** wrapped inside an interactive **Streamlit** frontend and fully containerized via **Docker**.

---

##  The Problem Statement & Business Context

Traditional Natural Language Processing (NLP) frameworks rely on uniform grammatical layouts and rigid vocabulary structures. In the South Asian digital market—specifically Nepal—e-commerce platforms, streaming comments, and customer support desks receive text primarily in **Romanized Nepali** (Nepali concepts typed phonetically using Latin letters). 

This introduces massive text-level noise:
* **Phonetic Variance:** One single word can be typed in multiple formats (e.g., *Ramro* as `rmro`, *Chha* as `xa`, `cha`, or `xha`). Standard models view these as completely distinct words, causing severe data sparsity.
* **Out-of-Vocabulary (OOV) Bloat:** Dialect combinations cause token spaces to explode, blinding deep learning models to critical client inputs.
* **Complex Negations:** Phrasings such as *"Ramro pani chhaina, naramro pani chhaina"* mix contradictory terms that confuse standard sentiment models.

### The Solution
This system tackles these challenges by introducing a **hybrid architecture**: a deterministic string-normalization pipeline paired with a sequence-classification deep learning Transformer model.

---

## Text Normalization Pipeline & Real-World Examples

To clean phonetic noise before it enters the neural network, the system routes raw user inputs through a custom Deterministic Preprocessing Engine. Below is how the engine maps chaotic inputs into standardized semantic anchors:

| Input Feature Type | Raw User Input Text | Cleaned & Normalized Output | Target Mapping Goal |
| :--- | :--- | :--- | :--- |
| **Phonetic Variance** | `Yo product ekdamai ramro xa` | `yo product ekdamai ramro chha` | Standardizes colloquial `xa` to `chha` |
| **Vowel Drop/Slang** | `baje ko delivery boy rmro thiana` | `baje ko delivery boy ramro thiana` | Corrects structural drop of `rmro` to `ramro` |
| **Double Negation** | `ramro pani chaina naramro pani chaina` | `product thik thak chha` | Intercepts contradiction, forcing a **Neutral** state |
| **Spelling Collision** | `bayo dherei ramro vayo` | `bhayo dherei ramro bhayo` | Normalizes split spellings (`bayo`/`vayo` ➔ `bhayo`) |

---

## End-to-End Machine Learning Engineering Lifecycle

This repository reflects a comprehensive Machine Learning Engineering (MLE) lifecycle, broken into five explicit operational phases:

* **Phase 1: Data Acquisition & Analysis**
  Investigated colloquial web commentary corpus structures (`RomanisedNepali.csv`) to isolate dialect variances, identifying core patterns of phonetic noise and spelling collisions.

* **Phase 2: Hybrid NLP Preprocessing & Normalization Design**
  Engineered an intermediate translation layer using regular expressions (Regex) to map variations to clean semantic roots. Built deterministic keyword overrides to handle edge-cases like double negations safely.

* **Phase 3: Transformer Model Fine-Tuning**
  Applied downstream fine-tuning on a `distilbert-base-multilingual-cased` Transformer backbone. Loaded normalized tokens, initialized categorical cross-entropy loss tracking, and updated weights via backpropagation.

* **Phase 4: Validation & Matrix Testing**
  Evaluated classification accuracy using Precision, Recall, and F1-Scores across separate test splits to ensure optimal multi-class performance before staging for production.

* **Phase 5: Containerization & Deployment**
  Packaged the Streamlit UI, system-level dependencies, and environment configurations into an isolated Docker container, ensuring identical application behavior across any cloud platform or host machine.

---

##  Core Technologies & Modeling Choices

### Why DistilBERT?
Instead of defaulting to massive, costly models, this system optimizes for production speed and efficiency by utilizing **DistilBERT** (`distilbert-base-multilingual-cased`):
* **Knowledge Distillation:** Replicates **95%** of the linguistic capability of native BERT layers while being **40% smaller** and **60% faster**.
* **WordPiece Tokenization:** Breaks unrecognized words down into sub-word token arrays (e.g., `naramro` ➔ `na` + `##ramro`), allowing the neural network to capture the semantic weight of negative prefixes perfectly without throwing out-of-vocabulary (`[UNK]`) errors.
* **Production Fleet Readiness:** The lightweight ~500MB checkpoint footprint enables real-time, low-latency prediction metrics using standard cloud CPU limits, avoiding the need for expensive GPU runtimes.

---

##  Project Structure & Execution

```text
├── .gitignore                      # Excludes local environments and system cache
├── Dockerfile                      # Builds the isolated production application container
├── README.md                       # Complete project overview and documentation
├── app.py                          # Streamlit UI implementation and inference pipeline
├── requirements.txt                # Fixed production framework dependencies
├── RomanisedNepali.csv             # Cleaned training corpus dataset
└── nepali_sentiment_distilbert/    # Fine-tuned Transformer weights and configurations
