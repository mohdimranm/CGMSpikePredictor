# Project Structure

## ✅ Final Clean Structure

```
sugarfit/
├── app/                          # Core application code
│   ├── api.py                    # FastAPI service (MAIN API)
│   ├── ohio_data_loader.py       # XML→CSV data loader
│   └── feature_engineering.py    # Feature engineering class
│
├── notebooks/                    # Jupyter notebooks for analysis
│   ├── explore_data.ipynb        # Data exploration & EDA
│   ├── feature_engineering_and_eda.ipynb  # Feature engineering
│   ├── train_model.ipynb         # Model training (USE THIS)
│   └── explanation_generator.ipynb # Test explanations
│
├── scripts/                      # Command-line utilities
│   ├── prepare_data.py           # Process XML files to CSV
│   └── train_model.py            # Train model from CLI
│
├── data/                         # Data files
│   ├── raw/                      # Original XML files
│   │   ├── train/
│   │   └── test/
│   └── processed/                # Generated CSV files
│       ├── train_data.csv
│       ├── test_data.csv
│       └── train_features.csv
│
├── models/                       # Trained models (generated)
│   ├── spike_predictor.pkl
│   ├── feature_names.txt
│   └── model_metadata.txt
│
├── docs/                         # Documentation
│   └── data_transformation_explained.md
│
├── requirements.txt              # Python dependencies
├── README.md                     # Main documentation
├── API_USAGE.md                  # API curl examples
├── test_api.py                   # API testing script
└── PROJECT_STRUCTURE.md          # This file
```

## 🗑️ Removed Files (No Longer Needed)

- `app/__init__.py` - Empty file
- `app/features.py` - Duplicate of feature_engineering.py
- `app/main.py` - Old FastAPI entry (replaced by api.py)
- `app/model.py` - Old model file
- `app/synth_data.py` - Synthetic data (not needed)
- `app/train.py` - Duplicate training script
- `notebook/diabetes.ipynb` - Old notebook (replaced by explore_data.ipynb)

## 📂 Directory Purposes

### `app/` - Production Code
Contains code used by the API service:
- `api.py` - FastAPI endpoints, request/response handling
- `ohio_data_loader.py` - Parse XML files, merge data sources
- `feature_engineering.py` - Create features from raw data

### `notebooks/` - Analysis & Development
Jupyter notebooks for interactive analysis:
- `explore_data.ipynb` - Understand the dataset, visualize distributions
- `feature_engineering_and_eda.ipynb` - Develop and test features
- `train_model.ipynb` - **Main training notebook** (use this to train)
- `explanation_generator.ipynb` - Test explanation generation

### `scripts/` - Batch Processing
Command-line tools for automation:
- `prepare_data.py` - One-time data processing (XML→CSV)
- `train_model.py` - Alternative to notebook for training

### `data/` - Data Storage
- `raw/` - Original OhioT1DM XML files (not in git)
- `processed/` - Generated CSV files (not in git, regenerate with scripts)

### `models/` - Trained Models
Generated files (not in git, regenerate by training):
- `spike_predictor.pkl` - Pickled sklearn model
- `feature_names.txt` - List of features in order
- `model_metadata.txt` - Performance metrics

## 🚀 Typical Workflow

### First Time Setup
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download OhioT1DM dataset to data/raw/

# 3. Process data
python scripts/prepare_data.py

# 4. Train model
jupyter notebook notebooks/train_model.ipynb
# OR: python scripts/train_model.py

# 5. Start API
uvicorn app.api:app --reload --port 8000

# 6. Test
python test_api.py
```

### Development Cycle
```bash
# 1. Make changes to feature_engineering.py

# 2. Test in notebook
jupyter notebook notebooks/feature_engineering_and_eda.ipynb

# 3. Retrain model
jupyter notebook notebooks/train_model.ipynb

# 4. Restart API
# (API auto-reloads with --reload flag)

# 5. Test
python test_api.py
```

## 📝 File Descriptions

| File | Purpose | When to Use |
|------|---------|-------------|
| **app/api.py** | FastAPI service | Run for production API |
| **app/ohio_data_loader.py** | Data loader | Used by prepare_data.py |
| **app/feature_engineering.py** | Feature engineering | Used by API and training |
| **notebooks/train_model.ipynb** | Model training | Retrain when features change |
| **scripts/prepare_data.py** | Data processing | Once after downloading dataset |
| **test_api.py** | API testing | Verify API works |
| **README.md** | Documentation | Learn about project |
| **API_USAGE.md** | API examples | Learn how to call API |

## 🔄 Dependencies

```
prepare_data.py → ohio_data_loader.py
train_model.ipynb → feature_engineering.py
api.py → feature_engineering.py + spike_predictor.pkl
```

## ⚙️ Configuration Files

- `requirements.txt` - Python package dependencies
- `.gitignore` - Files to exclude from git (data/, models/, etc.)

## 🎯 Key Takeaways

1. **`notebooks/train_model.ipynb`** is the main entry point for training
2. **`app/api.py`** is the production API
3. **`scripts/prepare_data.py`** only needs to run once
4. **`data/` and `models/`** folders are gitignored (too large)
5. All unnecessary files have been removed

