# CGM Spike Predictor

A machine learning system to predict glucose spikes in Type 1 Diabetes patients using Continuous Glucose Monitor (CGM) data from the OhioT1DM dataset.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Design Choices](#design-choices)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Model Performance](#model-performance)
- [Trade-offs](#trade-offs)
- [Testing](#testing)
- [Documentation](#documentation)

## 🎯 Overview

**Objective**: Predict whether a patient's glucose level will exceed 180 mg/dL in the next 2 hours.

**Key Features:**
- Binary classification with 2-hour prediction horizon
- Focus on glucose trends (velocity, delta, acceleration)
- Meal timing and carbohydrate intake analysis
- Insulin dosing context
- Human-readable explanations

**Dataset**: OhioT1DM 2018
- 6 Type 1 Diabetes patients
- ~70K training records, ~16K test records
- 24 features including CGM readings, meals, insulin, and wearable sensor data

## 🏗️ Architecture

1. **Data loading**: Parse OhioT1DM XML logs into a unified tabular format.
2. **Feature engineering**: Build trend-focused features (deltas, velocity, acceleration), meal/insulin timing, and interaction flags.
3. **Modeling**: Train baseline (logistic regression) and tree models; save the best model and feature list.
4. **Serving**: FastAPI `/predict` endpoint consumes recent CGM data and returns risk + explanation.

## ✅ Design Choices

- **Trend-first features**: Emphasize glucose dynamics over raw absolute values to catch spikes early.
- **Meal/insulin timing**: Include minutes since meal/bolus and recent carb/bolus windows for causal signals.
- **Simple baselines**: Use interpretable models for MVP speed and clarity.
- **Rule-based explanation**: Provide human-readable drivers aligned with the engineered features.

## 📁 Project Structure

```
sugarfit/
├── data/
│   ├── archive/                  # Raw OhioT1DM XML files
│   └── processed/                # Processed CSV files
│       ├── train_data.csv        # 69,255 records
│       ├── test_data.csv         # 15,970 records
│
├── models/                       # Trained models
│   ├── spike_predictor.pkl       # Best model
│   ├── feature_names.txt         # Feature list
│   └── model_metadata.txt        # Performance metrics
│
├── app/                          # Application code
│   ├── api.py                    # FastAPI service
│   ├── ohio_data_loader.py       # XML data loader
│   └── feature_engineering.py    # Feature engineering
│
├── notebook/                     # Analysis notebooks
│   ├── explore_data.ipynb
│   ├── feature_engineering_and_eda.ipynb
│   ├── train_model.ipynb
│   └── diabetes.ipynb
│
├── scripts/                      # Utility scripts
│   ├── prepare_data.py           # XML to CSV
│   └── train_model.py            # Train from CLI
│
├── requirements.txt
├── README.md
├── API_USAGE.md                  # API curl examples
└── test_api.py                   # API tests
```

## 🚀 Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download OhioT1DM dataset

Download from [OhioT1DM Dataset](http://smarthealth.cs.ohio.edu/bglp/OhioT1DM-dataset.zip)

Extract XML files to `data/archive/`

### 3. Process data

```bash
python scripts/prepare_data.py
```

### 4. Train model

```bash
# Option 1: Jupyter notebook (recommended)
jupyter notebook notebooks/train_model.ipynb

# Option 2: Command line
python scripts/train_model.py
```

### 5. Start API

```bash
uvicorn app.api:app --reload --port 8000
```

## 📊 Usage

### Quick Test

```bash
python test_api.py
```

### API Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "123",
    "recent_data": [
      {"timestamp": "2024-01-15T14:00:00", "glucose": 120},
      {"timestamp": "2024-01-15T14:05:00", "glucose": 125},
      ... (22 more readings - 2 hours total)
    ]
  }'
```

### Response

```json
{
  "user_id": "123",
  "will_spike": true,
  "risk_score": 0.85,
  "explanation": "Risk: VERY HIGH (85% probability of glucose spike above 180 mg/dL in next 2 hours). Recent glucose average is 165 mg/dL with rapidly rising velocity of 2.5 mg/dL/min (up 35 mg/dL in last 30 minutes). Recent significant carb intake of 60g consumed 15 minutes ago."
}
```

See [API_USAGE.md](API_USAGE.md) for detailed examples.

## 🔌 API Documentation

Interactive documentation available at: `http://localhost:8000/docs`

### Endpoints

- `GET /` - Health check
- `GET /health` - Detailed health status
- `POST /predict` - Spike prediction

### Request Requirements

- **Minimum**: 24 CGM readings (2 hours of data)
- **Frequency**: 5-minute intervals
- **Required fields**: `timestamp`, `glucose`
- **Optional fields**: `carbs`, `bolus_dose`, `heart_rate`, `steps`, etc.

## 📈 Model Performance

| Model | ROC-AUC | PR-AUC | Accuracy |
|-------|---------|--------|----------|
| Logistic Regression | 0.9995 | 0.9994 | 99% |
| Random Forest | 0.9956 | 0.9960 | 96% |
| Gradient Boosting | 0.9992 | 0.9993 | 99% |

**Selected Model**: Logistic Regression (best ROC-AUC, simplest)

### Top Features
1. glucose_mean_60min (30%)
2. glucose_mean_120min (21%)
3. glucose_delta_120min (6%)
4. glucose_velocity_60min (4%)
5. minutes_since_meal (3%)

## 🎨 Features

### Engineered Features (60+)

**Glucose Dynamics**:
- Velocity: Rate of change (mg/dL/min)
- Delta: Total change over time windows
- Acceleration: Change in velocity
- Direction: Rising/falling/stable indicators

**Meal Context**:
- Time since last meal
- Carbs from last meal
- Rolling carb sums (30/60/120 min)

**Insulin Context**:
- Time since last bolus
- Recent bolus dosage
- Rolling insulin sums

**Interaction Features**:
- meal_no_insulin: Meal without coverage
- high_carb_and_rising: High carbs + rising glucose
- fast_rise_after_meal: Rapid rise post-meal

## ⚖️ Trade-offs

- Used classical models (logistic regression / tree-based) instead of sequence models for MVP speed.
- Feature engineering focuses on recent trends; no personalization across users yet.
- No calibration step on probabilities; risk thresholds are fixed at 0.5.
- Explanation logic is rule-based rather than SHAP/LIME.
- Assumes 5-minute CGM cadence; irregular sampling is not fully handled.

## 🧪 Testing

- Run the local smoke test: `python test_api.py`
- Try a manual request: see `API_USAGE.md` for curl/Postman examples.

## 📝 Development

### Retraining Models

When features change:

```bash
jupyter notebook notebooks/train_model.ipynb
```

Or:

```bash
python scripts/train_model.py
```

### Adding New Features

1. Update `app/feature_engineering.py`
2. Retrain model
3. Update API if needed

## 📚 Documentation

- **[TECHNICAL_APPROACH.md](TECHNICAL_APPROACH.md)** - Comprehensive technical explanation:
  - Time-series data handling and nearest-timestamp merging
  - Feature engineering details (60+ features explained)
  - Model training and selection process
  - Explanation generation logic
  - Future improvements (LSTM, LLM-based explanations)
- **[API_USAGE.md](API_USAGE.md)** - API examples and curl commands
- **[TESTING.md](TESTING.md)** - Testing guide with test cases
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - Project organization
- **[EVALUATION_CRITERIA.md](EVALUATION_CRITERIA.md)** - Assignment criteria alignment

## 📚 References

- [OhioT1DM Dataset](http://smarthealth.cs.ohio.edu/bglp/OhioT1DM-dataset.html)
- [FastAPI Docs](https://fastapi.tiangolo.com/)

## 📄 License

Educational project for technical assessment.
# CGMSpikePredictor
