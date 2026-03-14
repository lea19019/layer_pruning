# Data Card

## Source

- **Corpus:** News Commentary v18 (Czech-German)
- **URL:** https://data.statmt.org/news-commentary/v18/training/news-commentary-v18.cs-de.tsv.gz
- **Format:** Tab-separated values (gzip compressed)
- **Domain:** News commentary articles
- **License:** See http://data.statmt.org/news-commentary/

## Filtering Pipeline

Following Moslem et al. (2025), applied in order:

| Step | Method | Threshold | Pairs In | Pairs Out | Removed |
|------|--------|-----------|---------|----------|---------|
| 0. Raw load | — | — | — | 243,807 | — |
| 1. Deduplication | Exact pair match | — | 243,807 | 241,831 | 1,976 (0.8%) |
| 2. Length filter | Max words per segment | 200 | 241,831 | 225,680 | 16,151 (6.7%) |
| | Length ratio (max/min words) | ≤ 1.5 | | | |
| 3. Language detection | fastText lid.176.bin | ≥ 0.9 confidence | 225,680 | 214,532 | 11,148 (4.9%) |
| 4. Semantic similarity | paraphrase-multilingual-MiniLM-L12-v2 | ≥ 0.7 cosine | 214,532 | 208,812 | 5,720 (2.7%) |

**Total removed:** 34,995 pairs (14.3%)
**Total after filtering:** 208,812 pairs

## Splits

- **Train:** 100,000 pairs (random sample, seed=42)
- **Test:** 500 pairs (held out before train sampling)
- **Remaining (unused):** 108,312 pairs

## Dataset Statistics

### Train Set (100,000 pairs)

| Metric | Czech (src) | German (tgt) |
|--------|------------|-------------|
| Sentences | 100,000 | 100,000 |
| Total words | 2,102,181 | 2,343,172 |
| Avg words/sentence | 21.0 | 23.4 |
| Min words | 1 | 1 |
| Max words | 104 | 115 |
| p10 words | 9 | 10 |
| p25 words | 14 | 15 |
| p50 (median) words | 20 | 22 |
| p75 words | 27 | 30 |
| p90 words | 34 | 38 |
| p95 words | 39 | 44 |
| p99 words | 50 | 56 |
| Total chars | 14,297,350 | 17,302,012 |
| Avg chars/sentence | 143.0 | 173.0 |
| File size | 16.1 MB | 17.7 MB |

### Test Set (500 pairs)

| Metric | Czech (src) | German (tgt) |
|--------|------------|-------------|
| Sentences | 500 | 500 |
| Total words | 10,292 | 11,494 |
| Avg words/sentence | 20.6 | 23.0 |
| Min words | 4 | 4 |
| Max words | 68 | 80 |
| p50 (median) words | 19 | 21 |
| p95 words | 39 | 45 |
| Avg chars/sentence | 139.7 | 169.6 |
| File size | 78.9 KB | 86.9 KB |

### Length Ratios (train)

| Metric | Value |
|--------|-------|
| Mean ratio (max/min words) | 1.18 |
| Median ratio | 1.15 |
| p95 ratio | 1.43 |
| Max ratio | 1.50 |

## File Locations

```
data/
├── raw/
│   ├── news-commentary-v18.cs-de.tsv.gz   # 31.5 MB (compressed)
│   ├── news-commentary-v18.cs-de.tsv       # 77.8 MB (extracted)
│   └── lid.176.bin                         # 131.3 MB (fastText LID model)
├── filtered/
│   ├── nc18.cs                             # 33.7 MB (all filtered, Czech)
│   ├── nc18.de                             # 37.0 MB (all filtered, German)
│   ├── train.cs                            # 16.1 MB (100K training, Czech)
│   ├── train.de                            # 17.7 MB (100K training, German)
│   ├── test.cs                             # 78.9 KB (500 test, Czech)
│   └── test.de                             # 86.9 KB (500 test, German)
└── kd/                                     # (not yet generated)
```

## Filtering Tools

- **Language detection:** fastText lid.176.bin (176 languages, https://fasttext.cc/docs/en/language-identification.html)
- **Semantic similarity:** sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (PyTorch, 118M params, 50+ languages)

## Reproducibility

- Random seed: 42
- Test set is sampled first (first 500 after shuffle), train is the next 100K
- All filtering thresholds defined in `src/config.py`
