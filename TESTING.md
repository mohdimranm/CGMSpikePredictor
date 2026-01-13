# Testing Guide

This document provides instructions for testing the CGM Spike Predictor API.

## Prerequisites

1. **Start the API server**:
```bash
uvicorn app.api:app --reload --port 8000
```

2. **Verify API is running**:
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "feature_count": 60
}
```

## Test Cases

### 1. Health Check

**Endpoint**: `GET /`

```bash
curl http://localhost:8000/
```

**Expected Response**:
```json
{
  "message": "CGM Spike Predictor API is running",
  "version": "1.0",
  "endpoints": ["/health", "/predict"]
}
```

---

### 2. Full Prediction Request (High Risk Scenario)

**Scenario**: Patient with rising glucose after a 60g carb meal

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_patient_001",
    "recent_data": [
      {"timestamp": "2024-01-15T12:00:00", "glucose": 140, "carbs": 0, "bolus_dose": 0},
      {"timestamp": "2024-01-15T12:05:00", "glucose": 142, "carbs": 0, "bolus_dose": 0},
      {"timestamp": "2024-01-15T12:10:00", "glucose": 145, "carbs": 0, "bolus_dose": 0},
      {"timestamp": "2024-01-15T12:15:00", "glucose": 148, "carbs": 60, "bolus_dose": 5},
      {"timestamp": "2024-01-15T12:20:00", "glucose": 152, "carbs": 0, "bolus_dose": 0},
      {"timestamp": "2024-01-15T12:25:00", "glucose": 156, "carbs": 0, "bolus_dose": 0},
      {"timestamp": "2024-01-15T12:30:00", "glucose": 161, "carbs": 0, "bolus_dose": 0},
      {"timestamp": "2024-01-15T12:35:00", "glucose": 165, "carbs": 0, "bolus_dose": 0},
      {"timestamp": "2024-01-15T12:40:00", "glucose": 170, "carbs": 0, "bolus_dose": 0},
      {"timestamp": "2024-01-15T12:45:00", "glucose": 174, "carbs": 0, "bolus_dose": 0},
      {"timestamp": "2024-01-15T12:50:00", "glucose": 178, "carbs": 0, "bolus_dose": 0},
      {"timestamp": "2024-01-15T12:55:00", "glucose": 182, "carbs": 0, "bolus_dose": 0},
      {"timestamp": "2024-01-15T13:00:00", "glucose": 186, "carbs": 0, "bolus_dose": 0},
      {"timestamp": "2024-01-15T13:05:00", "glucose": 189, "carbs": 0, "bolus_dose": 0},
      {"timestamp": "2024-01-15T13:10:00", "glucose": 192, "carbs": 0, "bolus_dose": 0},
      {"timestamp": "2024-01-15T13:15:00", "glucose": 194, "carbs": 0, "bolus_dose": 0},
      {"timestamp": "2024-01-15T13:20:00", "glucose": 196, "carbs": 0, "bolus_dose": 0},
      {"timestamp": "2024-01-15T13:25:00", "glucose": 197, "carbs": 0, "bolus_dose": 0},
      {"timestamp": "2024-01-15T13:30:00", "glucose": 198, "carbs": 0, "bolus_dose": 0},
      {"timestamp": "2024-01-15T13:35:00", "glucose": 199, "carbs": 0, "bolus_dose": 0},
      {"timestamp": "2024-01-15T13:40:00", "glucose": 199, "carbs": 0, "bolus_dose": 0},
      {"timestamp": "2024-01-15T13:45:00", "glucose": 200, "carbs": 0, "bolus_dose": 0},
      {"timestamp": "2024-01-15T13:50:00", "glucose": 200, "carbs": 0, "bolus_dose": 0},
      {"timestamp": "2024-01-15T13:55:00", "glucose": 201, "carbs": 0, "bolus_dose": 0}
    ]
  }'
```

**Expected Response**:
```json
{
  "user_id": "test_patient_001",
  "will_spike": true,
  "risk_score": 0.95,
  "explanation": "Risk: VERY HIGH (95% probability of glucose spike above 180 mg/dL in next 2 hours). Recent glucose average is 185 mg/dL with rapidly rising velocity..."
}
```

---

### 3. Low Risk Scenario

**Scenario**: Stable glucose in normal range

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "stable_patient",
    "recent_data": [
      {"timestamp": "2024-01-15T12:00:00", "glucose": 105},
      {"timestamp": "2024-01-15T12:05:00", "glucose": 106},
      {"timestamp": "2024-01-15T12:10:00", "glucose": 105},
      {"timestamp": "2024-01-15T12:15:00", "glucose": 107},
      {"timestamp": "2024-01-15T12:20:00", "glucose": 106},
      {"timestamp": "2024-01-15T12:25:00", "glucose": 108},
      {"timestamp": "2024-01-15T12:30:00", "glucose": 107},
      {"timestamp": "2024-01-15T12:35:00", "glucose": 109},
      {"timestamp": "2024-01-15T12:40:00", "glucose": 108},
      {"timestamp": "2024-01-15T12:45:00", "glucose": 110},
      {"timestamp": "2024-01-15T12:50:00", "glucose": 109},
      {"timestamp": "2024-01-15T12:55:00", "glucose": 111},
      {"timestamp": "2024-01-15T13:00:00", "glucose": 110},
      {"timestamp": "2024-01-15T13:05:00", "glucose": 112},
      {"timestamp": "2024-01-15T13:10:00", "glucose": 111},
      {"timestamp": "2024-01-15T13:15:00", "glucose": 113},
      {"timestamp": "2024-01-15T13:20:00", "glucose": 112},
      {"timestamp": "2024-01-15T13:25:00", "glucose": 114},
      {"timestamp": "2024-01-15T13:30:00", "glucose": 113},
      {"timestamp": "2024-01-15T13:35:00", "glucose": 115},
      {"timestamp": "2024-01-15T13:40:00", "glucose": 114},
      {"timestamp": "2024-01-15T13:45:00", "glucose": 116},
      {"timestamp": "2024-01-15T13:50:00", "glucose": 115},
      {"timestamp": "2024-01-15T13:55:00", "glucose": 117}
    ]
  }'
```

**Expected Response**:
```json
{
  "user_id": "stable_patient",
  "will_spike": false,
  "risk_score": 0.05,
  "explanation": "Risk: VERY LOW (5% probability of glucose spike above 180 mg/dL in next 2 hours). Glucose is stable in healthy range..."
}
```

---

### 4. Minimal Data (Glucose Only)

**Test**: API accepts minimal required fields

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "minimal_test",
    "recent_data": [
      {"timestamp": "2024-01-15T12:00:00", "glucose": 150},
      {"timestamp": "2024-01-15T12:05:00", "glucose": 151},
      {"timestamp": "2024-01-15T12:10:00", "glucose": 152},
      {"timestamp": "2024-01-15T12:15:00", "glucose": 153},
      {"timestamp": "2024-01-15T12:20:00", "glucose": 154},
      {"timestamp": "2024-01-15T12:25:00", "glucose": 155},
      {"timestamp": "2024-01-15T12:30:00", "glucose": 156},
      {"timestamp": "2024-01-15T12:35:00", "glucose": 157},
      {"timestamp": "2024-01-15T12:40:00", "glucose": 158},
      {"timestamp": "2024-01-15T12:45:00", "glucose": 159},
      {"timestamp": "2024-01-15T12:50:00", "glucose": 160},
      {"timestamp": "2024-01-15T12:55:00", "glucose": 161},
      {"timestamp": "2024-01-15T13:00:00", "glucose": 162},
      {"timestamp": "2024-01-15T13:05:00", "glucose": 163},
      {"timestamp": "2024-01-15T13:10:00", "glucose": 164},
      {"timestamp": "2024-01-15T13:15:00", "glucose": 165},
      {"timestamp": "2024-01-15T13:20:00", "glucose": 166},
      {"timestamp": "2024-01-15T13:25:00", "glucose": 167},
      {"timestamp": "2024-01-15T13:30:00", "glucose": 168},
      {"timestamp": "2024-01-15T13:35:00", "glucose": 169},
      {"timestamp": "2024-01-15T13:40:00", "glucose": 170},
      {"timestamp": "2024-01-15T13:45:00", "glucose": 171},
      {"timestamp": "2024-01-15T13:50:00", "glucose": 172},
      {"timestamp": "2024-01-15T13:55:00", "glucose": 173}
    ]
  }'
```

---

### 5. Error Handling - Insufficient Data

**Test**: API validates minimum data requirement (24 readings)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "insufficient_test",
    "recent_data": [
      {"timestamp": "2024-01-15T12:00:00", "glucose": 120},
      {"timestamp": "2024-01-15T12:05:00", "glucose": 121},
      {"timestamp": "2024-01-15T12:10:00", "glucose": 122}
    ]
  }'
```

**Expected Response**: HTTP 422 (Validation Error)
```json
{
  "detail": [
    {
      "loc": ["body", "recent_data"],
      "msg": "ensure this value has at least 24 items",
      "type": "value_error.list.min_items"
    }
  ]
}
```

---

### 6. Full OhioT1DM Format (All Fields)

**Test**: API accepts all optional fields from OhioT1DM dataset

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "full_data_test",
    "recent_data": [
      {
        "timestamp": "2024-01-15T12:00:00",
        "glucose": 145,
        "carbs": 0,
        "bolus_dose": 0,
        "finger_stick_glucose": 0,
        "basal_rate": 0.8,
        "temp_basal_rate": 0,
        "hypo_event": 0,
        "stress_event": 0,
        "heart_rate": 72,
        "gsr": 0.5,
        "skin_temp_f": 92.3,
        "air_temp_f": 68.5,
        "steps": 120
      }
    ]
  }'
```

Note: Provide 24 full records for actual testing.

---

## Automated Testing

For automated testing, consider creating a Python script:

```python
import requests

def test_prediction():
    response = requests.post(
        "http://localhost:8000/predict",
        json={"user_id": "test", "recent_data": [...]}
    )
    assert response.status_code == 200
    assert "will_spike" in response.json()
    assert "risk_score" in response.json()
    assert "explanation" in response.json()
```

## Testing Checklist

- [ ] Health endpoint returns 200
- [ ] Prediction with high-risk scenario returns will_spike=true
- [ ] Prediction with low-risk scenario returns will_spike=false
- [ ] Minimal data (glucose only) is accepted
- [ ] Full OhioT1DM format (all fields) is accepted
- [ ] Insufficient data (<24 readings) returns validation error
- [ ] Explanation includes glucose trends
- [ ] Explanation includes meal context when carbs present
- [ ] Explanation includes insulin context when bolus present
- [ ] Risk score is between 0 and 1

## Interactive API Documentation

Visit `http://localhost:8000/docs` for interactive Swagger UI documentation where you can test endpoints directly in your browser.

## Troubleshooting

**API not responding**:
- Verify server is running: `ps aux | grep uvicorn`
- Check port 8000 is not in use: `lsof -i :8000`
- Restart API: `uvicorn app.api:app --reload --port 8000`

**Model not found error**:
- Train the model: `jupyter notebook notebooks/train_model.ipynb`
- Verify model file exists: `ls -la models/spike_predictor.pkl`

**Feature mismatch error**:
- Retrain model with current features
- Ensure feature_names.txt matches trained model
