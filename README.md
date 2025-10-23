# bioRxiv Abstracts Classification Project

## Overview

This project classifies bioRxiv research abstracts into three infectious disease categories: COVID-19, Dengue, and Tuberculosis. I implemented and compared two approaches—traditional machine learning (TF-IDF + Logistic Regression) and deep learning (fine-tuned SciBERT transformer)—to determine which performs better for this biomedical text classification task.

The work demonstrates that well-engineered traditional features can outperform complex deep learning models when working with limited domain-specific data.

## Final Results

- **Logistic Regression**: 98.00% test accuracy (49/50 correct), 96.0% ± 1.2% cross-validation
- **SciBERT**: 96.00% test accuracy (48/50 correct), single train-test split
- **Dataset**: 246 balanced abstracts (82 per disease)
- **Split**: 196 training, 50 test (stratified)

The traditional approach won due to superior accuracy, faster training, smaller model size, and interpretability.

## Project Organization

The complete project lives in `bioRxiv_1/` with the following structure:

```
bioRxiv_1/
├── bioRxiv_Disease_Classification.ipynb          # Main analysis (57 cells, 10 phases)
├── Task_Instructions.txt                         # Original assignment requirements
├── README.md                                      # This file
├── venv_bioRxiv/                                  # Python virtual environment
│   ├── Include/
│   ├── Lib/
│   ├── Scripts/
│   ├── etc/
│   ├── share/
│   ├── .gitignore
│   └── pyvenv.cfg
└── outputs/                                       # All generated files (28 total)
    ├── file_manifest_20251021_141844.csv         # Complete file listing
    ├── PROJECT_SUMMARY.md        # Comprehensive report
    ├── data/                                      # 7 datasets
    │   ├── biorxiv_abstracts_raw_20251021_135633.csv
    │   ├── biorxiv_abstracts_preprocessed_20251021_135657.csv
    │   ├── biorxiv_dataset_final_20251021_141844.xlsx
    │   ├── model_comparison_20251021_141833.csv
    │   ├── feature_importance_analysis.csv
    │   ├── error_analysis_report.csv
    │   └── robustness_testing_report.csv
    ├── models/                                    # 3 trained models
    │   ├── tfidf_vectorizer_20251021_135717.pkl
    │   ├── logistic_regression_model_20251021_135728.pkl
    │   └── scibert_finetuned_20251021_141753/
    │       ├── config.json
    │       ├── model.safetensors
    │       ├── tokenizer_config.json
    │       ├── special_tokens_map.json
    │       ├── vocab.txt
    │       └── tokenizer.json
    └── plots/                                     # 14 visualizations
        ├── class_distribution_20251021_135704.png
        ├── text_length_analysis_20251021_135705.png
        ├── word_frequency_20251021_135707.png
        ├── confusion_matrix_lr_20251021_135740.png
        ├── classification_report_lr_20251021_135740.json
        ├── per_class_metrics_lr_20251021_135740.png
        ├── confusion_matrix_scibert_20251021_141821.png
        ├── classification_report_scibert_20251021_141821.json
        ├── per_class_metrics_scibert_20251021_141821.png
        ├── training_history_scibert_20251021_141821.png
        ├── model_comparison_20251021_141833.png
        ├── feature_importance_analysis.png
        ├── error_analysis_dashboard.png
        └── robustness_testing_dashboard.png
```

All outputs are timestamped (format: `YYYYMMDD_HHMMSS`) to preserve version history and prevent accidental overwrites.

## Notebook Structure: 10 Phases

The main notebook (`bioRxiv_Disease_Classification.ipynb`) is organized into 10 systematic phases:

### Phase 0: Environment Setup
- Install required libraries (pandas, numpy, scikit-learn, transformers, torch, nltk, spacy)
- Critical fixes: NLTK punkt_tab tokenizer, wordcloud library, accelerate for transformers
- Verification: Post-restart dependency checks to ensure all imports work
- Directory structure: Create `outputs/data/`, `outputs/plots/`, `outputs/models/`

### Phase 1: Data Collection
- Source: bioRxiv API with keyword-based search
- Keywords: "COVID-19", "SARS-CoV-2", "Dengue", "Tuberculosis", "TB", "Mycobacterium tuberculosis"
- Implemented custom `BioRxivCollector` class with rate limiting (1s delay) and retry logic
- Collected 80+ abstracts per disease, then applied DOI-based deduplication
- Class balancing: Stratified random downsampling to 82 abstracts per disease
- Output: `biorxiv_abstracts_raw_YYYYMMDD_HHMMSS.csv`

### Phase 2: Text Preprocessing
- **For Logistic Regression**: Aggressive preprocessing
  - Clean: lowercase, remove URLs/emails/special characters
  - Tokenize: NLTK `word_tokenize()`
  - Remove: English stopwords and tokens ≤ 2 characters
  - Lemmatize: spaCy `.lemma_` (used spaCy instead of NLTK for better accuracy)
  
- **For SciBERT**: Minimal preprocessing
  - Use original abstracts (transformers handle their own tokenization)
  - BERT tokenizer with 512 token max length

- Maintained both versions in separate columns: `abstract_preprocessed` and `abstract`
- Output: `biorxiv_abstracts_preprocessed_YYYYMMDD_HHMMSS.csv`

### Phase 3: Exploratory Data Analysis
- Class distribution: Bar/pie charts confirming 82-82-82 balance
- Text length analysis: Histograms and box plots of word counts per disease
- Word frequency: Top 15 words per disease using Counter
- Outputs: `class_distribution_*.png`, `text_length_analysis_*.png`, `word_frequency_*.png`

### Phase 4: Feature Engineering
- Train-test split: 80/20 (196 train, 50 test) with stratification
- TF-IDF vectorization:
  - Max 5000 features
  - Unigrams + bigrams (ngram_range=(1,2))
  - min_df=2, max_df=0.8
  - Sublinear TF scaling
- Saved vectorizer: `tfidf_vectorizer_*.pkl` for reproducible preprocessing

### Phase 5: Train Logistic Regression
- Multinomial Logistic Regression with L2 regularization (C=1.0)
- Solver: LBFGS, max iterations: 1000
- 5-fold stratified cross-validation: 96.0% ± 1.2% accuracy
- Training time: ~0.03 seconds
- Saved model: `logistic_regression_model_*.pkl`

### Phase 6: Evaluate Logistic Regression
- Test accuracy: 98.00% (49/50 correct)
- Per-class metrics: COVID-19 (P=1.00, R=1.00, F1=1.00), Dengue (P=0.94, R=1.00, F1=0.97), TB (P=1.00, R=0.94, F1=0.97)
- Confusion matrix: Saved as annotated heatmap
- **Advanced analyses** (beyond requirements):
  - Feature importance: Top 20 predictive words/bigrams per disease
  - Error analysis: Examined 1 misclassification (TB → Dengue, confidence=0.397)
  - Robustness testing: CV variance analysis, deployment readiness checks
- Outputs: `confusion_matrix_lr_*.png`, `classification_report_lr_*.json`, `feature_importance_analysis.png`, `error_analysis_dashboard.png`

### Phase 7: Train SciBERT Transformer
- Base: `allenai/scibert_scivocab_uncased` (pretrained on 1.14M scientific papers)
- Fine-tuning: 3 epochs, batch size 8, learning rate 2e-5, weight decay 0.01
- Used original abstracts (no heavy preprocessing)
- Training time: ~20 minutes (CPU)
- Saved: Complete model directory `scibert_finetuned_*/` with tokenizer

### Phase 8: Evaluate SciBERT
- Test accuracy: 96.00% (48/50 correct)
- Per-class metrics: COVID-19 (P=0.89, R=1.00, F1=0.94), Dengue (P=1.00, R=0.94, F1=0.97), TB (P=1.00, R=0.94, F1=0.97)
- Training history: Loss decreased from 1.05 → 0.12, validation accuracy plateaued at epoch 2
- Note: Single train-test split (no CV) due to computational cost (standard for transformers)
- Outputs: `confusion_matrix_scibert_*.png`, `training_history_*.png`

### Phase 9: Model Comparison
- Side-by-side comparison across metrics: accuracy, training time, model size, interpretability
- Winner: Logistic Regression (98% vs 96%, 40,000x faster, 200x smaller, interpretable)
- Why simpler model won: Limited data (246 samples), effective TF-IDF bigrams, linear separability
- Output: `model_comparison_*.png`, `model_comparison_*.csv`

### Phase 10: Final Documentation
- File manifest: Complete listing of all 28 output files
- Excel export: Multi-sheet workbook with preprocessed data, class distribution, model comparison
- Project summary: Comprehensive markdown report
- Outputs: `file_manifest_*.csv`, `biorxiv_dataset_final_*.xlsx`, `PROJECT_SUMMARY_*.md`

## Why I Made These Choices

**Dataset size (246 abstracts):**
I collected 82 per disease instead of the minimum 50 to account for deduplication and ensure robust class balance. This gave me enough data for reliable train-test splits while avoiding the time cost of collecting thousands of abstracts.

**Both traditional and deep learning:**
The assignment said "or" but I implemented both to see which performs better. This comparison revealed that traditional ML can outperform transformers on small domain-specific datasets—an important finding that challenges the "always use deep learning" mindset.

**Stratified sampling:**
I used stratification everywhere (train-test split, cross-validation, class balancing) to maintain the 33.3%-33.3%-33.3% distribution. This prevents class imbalance bias and makes accuracy a meaningful metric.

**spaCy lemmatization over NLTK:**
Even though I imported `WordNetLemmatizer`, I ended up using spaCy's `.lemma_` instead because it's more accurate for scientific text. The comment in my code says "more accurate than NLTK" because spaCy uses part-of-speech tagging for better lemmatization.

**Bigrams in TF-IDF:**
Including bigrams (2-word phrases) was crucial. Disease names like "dengue fever" and "mycobacterium tuberculosis" are better captured as single features than split words. This boosted accuracy significantly.

**No CV for SciBERT:**
Cross-validating transformers would take 5x the training time (20 min × 5 folds = 100 min). Since this is standard practice in the field and I already had CV results from Logistic Regression, I used a single train-test split for SciBERT.

**Advanced analyses:**
I added feature importance, error analysis, and robustness testing because I wanted to understand *why* the model works, not just *that* it works. These analyses would be essential in a real-world deployment scenario.

## Complete Output Files (28 Total)


### Datasets (outputs/data/) - 7 files
1. `biorxiv_abstracts_raw_20251021_135633.csv` - Raw abstracts from API
2. `biorxiv_abstracts_preprocessed_20251021_135657.csv` - Cleaned & lemmatized text
3. `biorxiv_dataset_final_20251021_141844.xlsx` - Excel with 3 sheets (data, distribution, comparison)
4. `model_comparison_20251021_141833.csv` - Performance metrics table
5. `feature_importance_analysis.csv` - Top 20 features per disease
6. `error_analysis_report.csv` - Misclassified samples breakdown
7. `robustness_testing_report.csv` - CV results and deployment checks

### Models (outputs/models/) - 3 files
1. `logistic_regression_model_20251021_135728.pkl` - Trained LR classifier (2 MB)
2. `tfidf_vectorizer_20251021_135717.pkl` - Feature extractor (saved for inference)
3. `scibert_finetuned_20251021_141753/` - Complete transformer directory (400 MB)

### Visualizations (outputs/plots/) - 14 files
1. `class_distribution_20251021_135704.png` - Bar & pie charts
2. `text_length_analysis_20251021_135705.png` - Word count distributions
3. `word_frequency_20251021_135707.png` - Top 15 words per disease
4. `confusion_matrix_lr_20251021_135740.png` - LR confusion matrix (98% accuracy)
5. `classification_report_lr_20251021_135740.json` - LR detailed metrics
6. `per_class_metrics_lr_20251021_135740.png` - LR precision/recall/F1 bars
7. `confusion_matrix_scibert_20251021_141821.png` - SciBERT confusion matrix (96%)
8. `classification_report_scibert_20251021_141821.json` - SciBERT detailed metrics
9. `per_class_metrics_scibert_20251021_141821.png` - SciBERT P/R/F1 bars
10. `training_history_scibert_20251021_141821.png` - Loss & accuracy curves
11. `model_comparison_20251021_141833.png` - Side-by-side comparison
12. `feature_importance_analysis.png` - Top features visualization
13. `error_analysis_dashboard.png` - 4-panel error diagnostics
14. `robustness_testing_dashboard.png` - CV stability & confidence analysis

### Documentation (in outputs/)
1. `PROJECT_SUMMARY_20251021_141846.md` - Comprehensive project report
2. `file_manifest_20251021_141844.csv` - Complete file listing

### Root Directory Files
1. `README.md` - This project overview
2. `bioRxiv_Disease_Classification.ipynb` - Main analysis notebook
3. `Task_Instructions.txt` - Assignment requirements

## Key Findings

**What worked:**
- TF-IDF with bigrams captured disease-specific terminology better than I expected
- 246 balanced samples was enough for traditional ML but too little for deep learning
- spaCy lemmatization improved preprocessing quality over NLTK
- 5-fold CV confirmed model stability (low variance = good generalization)

**What surprised me:**
- Logistic Regression beat SciBERT by 2% despite being 40,000x faster
- Single misclassification was genuinely ambiguous (discussed multiple diseases)
- Feature importance showed model learned real medical concepts (not spurious correlations)
- Bigrams like "mycobacterium tuberculosis" had higher weights than individual words

**What I learned:**
- More complex ≠ better (especially with limited data)
- Traditional ML still competitive for domain-specific tasks
- Proper preprocessing matters more than model choice sometimes
- Understanding errors is as important as measuring accuracy

## Technical Implementation Notes

**Reproducibility:**
- All random operations use `random_state=42` (train-test split, CV, class balancing)
- No hardcoded paths—everything uses `pathlib` for OS independence
- Timestamps on all outputs prevent accidental overwrites
- Complete dependency list in Phase 0 cells

**Code quality:**
- Progress bars (tqdm) for all long operations (data collection, preprocessing, training)
- Inline comments explaining "why" not just "what"
- Error handling with retry logic in API collector
- Validation checks after each major step

**Performance:**
- Total runtime: ~35-65 minutes (mostly SciBERT training)
- Logistic Regression: <1 second training, <1ms inference
- SciBERT: ~20 min training (CPU), ~100ms inference
- Data collection: ~3-8 minutes (depends on network)

## Project Deliverables Summary

**Required by assignment:**
-  Python code: `bioRxiv_Disease_Classification.ipynb` (57 cells)
-  Confusion matrices: 2 files (both models)
-  Accuracy/loss plots: 14 visualization files
-  Dataset: 7 files (CSV + Excel formats)

**Beyond requirements:**
-  Comparative analysis (traditional vs transformer)
-  Feature importance analysis
-  Error diagnostics with confidence scores
-  Robustness testing (CV, deployment readiness)
-  Comprehensive documentation (README + summary report)

**Total outputs:** 28 files (7 datasets + 3 models + 14 visualizations + 2 documentation + 2 root files) demonstrating end-to-end ML workflow from data collection to production-ready models.

---

*Project completed October 21, 2025. All code, data, and documentation available in this repository for evaluation.*

