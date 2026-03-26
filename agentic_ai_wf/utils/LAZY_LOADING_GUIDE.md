# Lazy Loading Implementation Guide

## Overview
Lazy loading improves application startup time by deferring heavy library imports until they're actually needed. This can reduce Django/Flask app startup time from several seconds to milliseconds.

## Performance Impact
- **Before**: All heavy libraries loaded at import time (~3-10 seconds startup)
- **After**: Libraries loaded on-demand (~50-200ms startup)

## How to Use

### 1. Basic Usage Pattern

**❌ Old way (eager loading):**
```python
# These imports happen immediately when module is loaded
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd

# Heavy model loaded at module import time
model = SentenceTransformer("all-MiniLM-L6-v2")

def process_data():
    # Model already loaded, but startup was slow
    embeddings = model.encode(texts)
```

**✅ New way (lazy loading):**
```python
# Lightweight imports
from agentic_ai_wf.utils.lazy_loader import get_sentence_model, get_faiss, get_pandas

def process_data():
    # Libraries only imported when function is called
    model = get_sentence_model()  # Loaded here on first use
    faiss = get_faiss()          # Loaded here on first use 
    pd = get_pandas()            # Loaded here on first use
    
    embeddings = model.encode(texts)
```

### 2. Migration Steps

#### Step 1: Replace Direct Imports
```python
# Before
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.metrics.pairwise import cosine_similarity

# After  
from agentic_ai_wf.utils.lazy_loader import (
    get_sentence_model, 
    get_faiss, 
    get_sklearn_cosine_similarity
)
```

#### Step 2: Update Global Variables
```python
# Before
MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# After - Remove global variables, use lazy loading in functions
# (No global MODEL variable needed)
```

#### Step 3: Update Function Implementations
```python
# Before
def analyze_text(texts):
    embeddings = MODEL.encode(texts)
    similarities = cosine_similarity(embeddings)
    
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

# After
def analyze_text(texts):
    model = get_sentence_model()
    cosine_sim = get_sklearn_cosine_similarity()
    faiss_lib = get_faiss()
    
    embeddings = model.encode(texts)
    similarities = cosine_sim(embeddings)
    
    index = faiss_lib.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
```

### 3. Available Lazy Loaders

| Function | Import | Use Case |
|----------|--------|----------|
| `get_sentence_model()` | SentenceTransformer | Text embeddings, semantic similarity |
| `get_faiss()` | faiss | Vector similarity search, indexing |
| `get_sklearn_cosine_similarity()` | sklearn.metrics.pairwise.cosine_similarity | Cosine similarity calculations |
| `get_transformers()` | transformers | Hugging Face transformers |
| `get_torch()` | torch | PyTorch deep learning |
| `get_numpy()` | numpy | Numerical computing |
| `get_pandas()` | pandas | Data manipulation |

### 4. Common Patterns

#### Pattern 1: Text Processing
```python
def process_pathways(pathway_texts):
    model = get_sentence_model()
    np = get_numpy()
    
    embeddings = model.encode(pathway_texts)
    return embeddings.astype(np.float32)
```

#### Pattern 2: Vector Search
```python
def similarity_search(query_embedding, database_embeddings):
    faiss_lib = get_faiss()
    
    # Build index
    index = faiss_lib.IndexFlatIP(database_embeddings.shape[1])
    index.add(database_embeddings)
    
    # Search
    distances, indices = index.search(query_embedding, k=10)
    return distances, indices
```

#### Pattern 3: Data Analysis
```python
def analyze_results(data_file):
    pd = get_pandas()
    np = get_numpy()
    
    df = pd.read_csv(data_file)
    summary_stats = df.describe()
    return summary_stats
```

### 5. Error Handling

The lazy loaders include automatic error handling:

```python
def safe_text_processing(texts):
    try:
        model = get_sentence_model()
        return model.encode(texts)
    except ImportError as e:
        print(f"SentenceTransformers not available: {e}")
        # Fallback to alternative method
        return simple_text_processing(texts)
```

### 6. Testing and Development

#### Clear Caches for Testing
```python
from agentic_ai_wf.utils.lazy_loader import clear_all_caches

def test_setup():
    clear_all_caches()  # Reset for clean testing
```

#### Mock Lazy Loaders in Tests
```python
def test_with_mock():
    from unittest.mock import patch
    
    with patch('agentic_ai_wf.utils.lazy_loader.get_sentence_model') as mock_model:
        mock_model.return_value = MockSentenceTransformer()
        result = your_function_using_lazy_loader()
        assert result is not None
```

## Files to Update

### High Priority (Heavy Libraries)
1. `agentic_ai_wf/pathway_agent/deduplication.py` ✅ **Done**
2. `agentic_ai_wf/pathway_agent/consolidation.py`
3. `agentic_ai_wf/drug_extraction_prioritization/tools/`
4. `agentic_ai_wf/gene_prioritization/`

### Medium Priority (Data Processing)
1. Any files importing `pandas`, `numpy` at module level
2. Files with heavy sklearn imports
3. Files using `matplotlib`, `seaborn` for plotting

### Low Priority (Framework Code)
1. Django models and views (usually fast to import)
2. Configuration files
3. Utility functions with lightweight imports

## Implementation Checklist

- [ ] ✅ Create `agentic_ai_wf/utils/lazy_loader.py`
- [ ] ✅ Update `deduplication.py`
- [ ] Update other pathway processing files
- [ ] Update gene prioritization modules  
- [ ] Update drug extraction modules
- [ ] Test application startup time improvement
- [ ] Update documentation and examples

## Performance Monitoring

### Before/After Measurement
```python
import time

# Measure startup time
start = time.time()
import your_heavy_module
end = time.time()
print(f"Startup time: {end - start:.2f}s")
```

### Expected Improvements
- **Django startup**: 5-10s → 200-500ms
- **CLI tools**: 3-5s → 50-100ms  
- **Memory usage**: Reduced by 200-500MB initially
- **Import time**: Heavy libraries load in ~1-2s when needed

## Troubleshooting

### Common Issues
1. **Function not found**: Check import path in lazy_loader.py
2. **Circular imports**: Ensure lazy_loader doesn't import application modules
3. **Cache not working**: Verify @lru_cache is present
4. **Tests failing**: Clear caches between test runs

### Debug Logging
```python
import logging
logging.getLogger('agentic_ai_wf.utils.lazy_loader').setLevel(logging.DEBUG)
```

This will show when libraries are actually loaded for the first time. 