# Technical Approach - CGM Spike Predictor

This document explains how we addressed each evaluation criterion and potential improvements.

## 1. Data Thinking: Time-Series Data Handling & Feature Engineering

### Dataset Overview
- **Source**: OhioT1DM 2018 Dataset
- **Format**: XML files per patient with multiple data streams
- **Challenge**: Asynchronous data streams with different sampling frequencies

### Data Loading Strategy

#### Multi-Stream Integration
The OhioT1DM dataset contains separate data streams:
- **CGM readings**: Every 5 minutes (continuous)
- **Meal events**: Sporadic (user-logged with carb amounts)
- **Insulin bolus**: Sporadic (pump injections)
- **Finger stick glucose**: Occasional calibration readings
- **Exercise events**: User-logged activities
- **Physiological sensors**: Heart rate, GSR, skin/air temperature, steps

#### Timestamp Alignment Challenge
Each data stream has its own timestamps that don't align perfectly. For example:
```
CGM:   12:00, 12:05, 12:10, 12:15, 12:20, ...
Meal:        12:07,             12:23, ...
Bolus:            12:11,    12:19, ...
```

#### Solution: Nearest-Timestamp Merge (`pd.merge_asof`)


**Key Design Decisions**:
1. **CGM as base**: Use 5-minute CGM intervals as the primary timeline
2. **Nearest matching**: Find the closest meal/bolus/exercise event to each CGM reading
3. **Tolerance window**: Only match events within 30 minutes to avoid spurious associations
4. **Preserve original timestamps**: Keep `meal_timestamp`, `bolus_timestamp` separately to calculate time deltas

**Example Result**:
```
timestamp    glucose  carbs  meal_timestamp  bolus_dose  bolus_timestamp
12:00        120      0      NaT             0           NaT
12:05        122      0      NaT             0           NaT
12:10        125      45     12:07           0           12:11
12:15        130      45     12:07           5           12:11
12:20        138      0      NaT             0           NaT
```

### Feature Engineering (60+ Features)

#### 1. Glucose Trend Features
**Motivation**: Capture the dynamics of glucose change, not just absolute levels.

**Features Created**:
- **Glucose Mean** (15min, 30min, 60min, 120min windows)
- **Glucose Delta** (change over time)
- **Glucose Velocity** (rate of change)
- **Glucose Acceleration** (change in velocity)
- **Direction Indicators**

#### 2. Glucose Variability Features
**Motivation**: Unstable glucose suggests higher risk.
- **Standard Deviation** (rolling windows)
- **Range** (max - min)

#### 3. Meal Context Features
**Motivation**: Carbohydrate intake is the primary driver of glucose spikes.

- **Time Since Last Meal**
- **Carbs from Last Meal**
- **Rolling Carb Sums** (recent intake)

#### 4. Insulin Context Features

- **Time Since Last Bolus**
- **Bolus from Last Dose**
- **Rolling Insulin Sums**:


#### 5. Interaction Features
**Motivation**: Combinations of factors reveal risk patterns.

- **Meal Without Insulin**
- **High Carbs + Rising Glucose**
- **Fast Rise After Meal**:


#### 6. Target Variable
**Definition**: Will glucose exceed 180 mg/dL in the next 2 hours?



**Critical: Data Leakage Prevention**
- ✅ Only use past and current data for features
- ✅ Target is based on future data (what we're predicting)

### Why This Approach Works

1. **Captures Dynamics**: Velocity and acceleration features encode the glucose trajectory
2. **Context-Aware**: Meal and insulin timing provide causal context
3. **Multi-Scale**: Features at different time windows (15min to 2hr) capture both immediate and longer-term trends
4. **Domain Knowledge**: Features like "meal without insulin" encode diabetes management principles

---

## 2. Engineering Clarity: Code & API Design

### Code Organization

```
app/
├── ohio_data_loader.py       # Data ingestion (XML → DataFrame)
├── feature_engineering.py    # Feature computation (reusable)
└── api.py                    # FastAPI service (production)

notebooks/
├── explore_data.ipynb        # EDA and understanding
├── feature_engineering_and_eda.ipynb  # Feature development
├── train_model.ipynb         # Model training (primary)
└── explanation_generator.ipynb  # Explanation testing

scripts/
├── prepare_data.py           # Batch processing (XML → CSV)
└── train_model.py            # CLI training alternative
```

**Separation of Concerns**:
- **Data loading**: Isolated in `ohio_data_loader.py`
- **Feature engineering**: Reusable class in `feature_engineering.py`
- **Training**: Notebooks for experimentation, scripts for automation
- **Serving**: Clean FastAPI service in `api.py`

### API Design

#### Endpoint Structure
```
GET  /           # Quick health check
GET  /health     # Detailed status (model loaded, feature count)
POST /predict    # Spike prediction
```

#### Type-Safe Request/Response

**Request Schema**:
```python
class CGMDataPoint(BaseModel):
    timestamp: str = Field(..., description="ISO format")
    glucose: float = Field(..., description="mg/dL")
    carbs: Optional[float] = Field(0)
    bolus_dose: Optional[float] = Field(0)
    # ... 10 more optional fields

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2024-01-15T14:00:00",
                "glucose": 145
            }
        }

class PredictionRequest(BaseModel):
    user_id: str
    recent_data: List[CGMDataPoint] = Field(
        ...,
        min_length=24,  # Require 2 hours of data
        description="24 CGM readings (5-min intervals)"
    )
```

**Response Schema**:
```python
class PredictionResponse(BaseModel):
    user_id: str
    will_spike: bool
    risk_score: float = Field(..., ge=0, le=1)
    explanation: str
```

#### Error Handling
- **Validation errors**: Pydantic validates request structure automatically
- **Model errors**: Try-catch around prediction with informative messages
- **Missing data**: Default values for optional fields

### Model Training & Selection

#### Models Evaluated
1. **Logistic Regression** (baseline)
2. **Random Forest** (ensemble)
3. **Gradient Boosting** (XGBoost/LightGBM alternative)

#### Training Process
```python
# From train_model.ipynb

# Train multiple models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'Gradient Boosting': GradientBoostingClassifier()
}

```

#### Results
| Model | ROC-AUC | PR-AUC | Accuracy |
|-------|---------|--------|----------|
| **Logistic Regression** | **0.9995** | **0.9994** | **99%** |
| Random Forest | 0.9956 | 0.9960 | 96% |
| Gradient Boosting | 0.9992 | 0.9993 | 99% |

**Selected**: Logistic Regression for simplicity and interpretability.

#### Feature Importance (Random Forest)
```
glucose_mean_60min:     30.1%  ← Recent glucose level
glucose_mean_120min:    20.7%  ← Longer-term level
glucose_delta_120min:    6.2%  ← Total change
glucose_velocity_60min:  3.6%  ← Rate of change
minutes_since_meal:      2.8%  ← Meal timing
```

---

## 3. Product Sense: Explanation Quality

### Current Approach: Rule-Based Logic

#### Risk Level Classification
```python
def generate_explanation(probability, features, feature_names):
    # Risk categorization
    if probability >= 0.80:
        risk = "VERY HIGH"
    elif probability >= 0.60:
        risk = "HIGH"
    elif probability >= 0.40:
        risk = "MODERATE"
    elif probability >= 0.20:
        risk = "LOW"
    else:
        risk = "VERY LOW"
```

#### Narrative Components

**1. Glucose Trends**:
```python
if glucose_mean > 160:
    parts.append(f"Recent glucose average is {glucose_mean:.0f} mg/dL")

if velocity > 2.0:
    parts.append(f"with rapidly rising velocity of {velocity:.1f} mg/dL/min")
elif velocity > 1.0:
    parts.append(f"with rising velocity of {velocity:.1f} mg/dL/min")
```

**2. Meal Context**:
```python
minutes_meal = feat.get('minutes_since_meal', 999)
carbs = feat.get('carbs_from_last_meal', 0)

if minutes_meal < 120 and carbs > 0:
    time_desc = "just" if minutes_meal < 15 else f"{int(minutes_meal)} minutes ago"
    carb_desc = "significant" if carbs > 50 else "moderate"
    parts.append(f"{carb_desc} carb intake of {carbs:.0f}g {time_desc}")
```

**3. Insulin Context**:
```python
minutes_bolus = feat.get('minutes_since_bolus', 999)
bolus = feat.get('bolus_from_last_dose', 0)

if minutes_meal < 30 and carbs > 20 and bolus == 0:
    parts.append("with no insulin coverage yet")
```

#### Example Output
```
Risk: VERY HIGH (85% probability of glucose spike above 180 mg/dL in next 2 hours).
Recent glucose average is 165 mg/dL with rapidly rising velocity of 2.5 mg/dL/min
(up 35 mg/dL in last 30 minutes). Recent significant carb intake of 60g consumed
15 minutes ago with glucose already rising.
```

---

## 4. Future Improvements

### A. Better Models: LSTM for Time Series

#### Why LSTM?
Current approach uses hand-crafted features (velocity, delta, etc.) from time series. **LSTM can learn temporal patterns automatically**.

**Advantages**:
1. **Automatic feature learning**: No manual velocity/acceleration calculation
2. **Long-term dependencies**: Can capture patterns over hours
3. **Sequence-to-one prediction**: Natural fit for "given 2hr history, predict next 2hr"


#### Implementation Steps
1. **Reshape data**: From (N_samples, N_features) → (N_samples, 24_timesteps, raw_features)
2. **Use raw values**: Glucose, carbs, bolus, heart rate at each timestep
3. **Minimal preprocessing**: Just normalization, let LSTM learn trends
4. **Train with sequences**: Each sample is a 2-hour window

#### Expected Benefits
- **Better pattern recognition**: May detect subtle patterns we didn't engineer
- **Simpler pipeline**: Less feature engineering code to maintain
- **Personalization**: Fine-tune per patient with transfer learning

#### Trade-offs
- ⚠️ **More data needed**: Deep learning requires more training examples
- ⚠️ **Less interpretable**: Can't easily explain "glucose velocity" contribution
- ⚠️ **Slower inference**: Neural network vs. simple linear model
- ⚠️ **Infrastructure**: Requires TensorFlow/PyTorch in production

---

### B. LLM-Based Explanations

#### Current Limitation
Rule-based explanations are rigid:
```python
if velocity > 2.0:
    return "rapidly rising velocity"
elif velocity > 1.0:
    return "rising velocity"
```

#### LLM Approach: Personalized, Context-Aware Narratives

#### Example LLM Output
```
Your spike risk is high (85%) right now. Your glucose is rising faster than usual
after that 60g meal - you typically see 1.5 mg/dL/min, but you're at 2.5 mg/dL/min.
The 5-unit bolus you took 15 minutes ago hasn't peaked yet; it usually takes 30-45
minutes to start bringing you down. Consider a light walk or waiting 20 minutes
before adding more insulin.
```

#### Benefits
1. **Personalized**: References user's typical patterns
2. **Contextual**: Compares to past similar situations
3. **Actionable**: Suggests interventions based on history
4. **Natural language**: Reads like a coach, not a robot

#### Implementation Requirements
1. **User profile storage**: Store typical patterns (glucose range, meal responses)
2. **Episode history**: Save past predictions and outcomes
3. **Pattern analysis**: Compute user-specific metrics (avg velocity after meals, insulin timing)
4. **LLM API integration**: Call Claude/GPT with structured prompt
5. **Caching**: Cache explanations for similar situations to reduce API costs

---

## Summary

### What We Built
✅ **Data Pipeline**: Merged asynchronous data streams with nearest-timestamp matching
✅ **Feature Engineering**: 60+ features capturing glucose dynamics, meal context, insulin timing
✅ **Model Training**: Evaluated 3 models, selected Logistic Regression (99.95% ROC-AUC)
✅ **Production API**: Type-safe FastAPI with comprehensive validation
✅ **Explanations**: Rule-based narratives emphasizing glucose trends and context

### What Could Be Better
**LSTM Model**: Automatic temporal pattern learning, potentially better predictions
**LLM Explanations**: Personalized, context-aware narratives with user history
**Real-time Learning**: Update model with user feedback (did spike actually occur?)
