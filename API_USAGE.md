# CGM Spike Predictor API Usage

## Starting the API

```bash
# Install dependencies
pip install fastapi uvicorn pydantic

# Start the server
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: `http://localhost:8000`

Interactive documentation: `http://localhost:8000/docs`

## API Endpoints

### 1. Health Check

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "feature_count": 60
}
```

### 2. Prediction Endpoint

**Endpoint:** `POST /predict`

**Required:** At least 24 CGM readings (2 hours of data at 5-minute intervals)

## Sample cURL Requests

### Simple Example (Minimal Fields)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "123",
    "recent_data": [
      {"timestamp": "2024-01-15T14:00:00", "glucose": 120},
      {"timestamp": "2024-01-15T14:05:00", "glucose": 122},
      {"timestamp": "2024-01-15T14:10:00", "glucose": 124},
      {"timestamp": "2024-01-15T14:15:00", "glucose": 126},
      {"timestamp": "2024-01-15T14:20:00", "glucose": 128},
      {"timestamp": "2024-01-15T14:25:00", "glucose": 130},
      {"timestamp": "2024-01-15T14:30:00", "glucose": 135, "carbs": 60, "bolus_dose": 5},
      {"timestamp": "2024-01-15T14:35:00", "glucose": 140},
      {"timestamp": "2024-01-15T14:40:00", "glucose": 145},
      {"timestamp": "2024-01-15T14:45:00", "glucose": 150},
      {"timestamp": "2024-01-15T14:50:00", "glucose": 155},
      {"timestamp": "2024-01-15T14:55:00", "glucose": 160},
      {"timestamp": "2024-01-15T15:00:00", "glucose": 165},
      {"timestamp": "2024-01-15T15:05:00", "glucose": 168},
      {"timestamp": "2024-01-15T15:10:00", "glucose": 171},
      {"timestamp": "2024-01-15T15:15:00", "glucose": 174},
      {"timestamp": "2024-01-15T15:20:00", "glucose": 177},
      {"timestamp": "2024-01-15T15:25:00", "glucose": 180},
      {"timestamp": "2024-01-15T15:30:00", "glucose": 183},
      {"timestamp": "2024-01-15T15:35:00", "glucose": 186},
      {"timestamp": "2024-01-15T15:40:00", "glucose": 189},
      {"timestamp": "2024-01-15T15:45:00", "glucose": 192},
      {"timestamp": "2024-01-15T15:50:00", "glucose": 195},
      {"timestamp": "2024-01-15T15:55:00", "glucose": 198}
    ]
  }'
```

### Full Example (With All OhioT1DM Fields)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "patient_588",
    "recent_data": [
      {
        "timestamp": "2024-01-15T14:00:00",
        "glucose": 120.0,
        "carbs": 0,
        "bolus_dose": 0,
        "meal_type": null,
        "finger_stick_glucose": 0,
        "basal_rate": 1.25,
        "temp_basal_rate": 1.25,
        "hypo_event": 0,
        "stress_event": 0,
        "exercise_duration": 0,
        "exercise_intensity": 0,
        "heart_rate": 73.0,
        "gsr": 0.000058,
        "skin_temp_f": 87.2,
        "air_temp_f": 85.1,
        "steps": 0
      },
      {
        "timestamp": "2024-01-15T14:05:00",
        "glucose": 122.0,
        "carbs": 0,
        "bolus_dose": 0,
        "basal_rate": 1.25,
        "temp_basal_rate": 1.25,
        "heart_rate": 74.0,
        "gsr": 0.000058,
        "skin_temp_f": 87.3,
        "air_temp_f": 85.2,
        "steps": 5
      },
      {
        "timestamp": "2024-01-15T14:10:00",
        "glucose": 124.0,
        "carbs": 0,
        "bolus_dose": 0,
        "basal_rate": 1.25,
        "temp_basal_rate": 1.25,
        "heart_rate": 75.0,
        "steps": 10
      },
      {
        "timestamp": "2024-01-15T14:15:00",
        "glucose": 126.0,
        "heart_rate": 76.0,
        "steps": 8
      },
      {
        "timestamp": "2024-01-15T14:20:00",
        "glucose": 128.0
      },
      {
        "timestamp": "2024-01-15T14:25:00",
        "glucose": 130.0
      },
      {
        "timestamp": "2024-01-15T14:30:00",
        "glucose": 135.0,
        "carbs": 60.0,
        "bolus_dose": 5.0,
        "meal_type": "Lunch"
      },
      {
        "timestamp": "2024-01-15T14:35:00",
        "glucose": 140.0
      },
      {
        "timestamp": "2024-01-15T14:40:00",
        "glucose": 145.0
      },
      {
        "timestamp": "2024-01-15T14:45:00",
        "glucose": 150.0
      },
      {
        "timestamp": "2024-01-15T14:50:00",
        "glucose": 155.0
      },
      {
        "timestamp": "2024-01-15T14:55:00",
        "glucose": 160.0
      },
      {
        "timestamp": "2024-01-15T15:00:00",
        "glucose": 165.0
      },
      {
        "timestamp": "2024-01-15T15:05:00",
        "glucose": 168.0
      },
      {
        "timestamp": "2024-01-15T15:10:00",
        "glucose": 171.0
      },
      {
        "timestamp": "2024-01-15T15:15:00",
        "glucose": 174.0
      },
      {
        "timestamp": "2024-01-15T15:20:00",
        "glucose": 177.0
      },
      {
        "timestamp": "2024-01-15T15:25:00",
        "glucose": 180.0
      },
      {
        "timestamp": "2024-01-15T15:30:00",
        "glucose": 183.0
      },
      {
        "timestamp": "2024-01-15T15:35:00",
        "glucose": 186.0
      },
      {
        "timestamp": "2024-01-15T15:40:00",
        "glucose": 189.0
      },
      {
        "timestamp": "2024-01-15T15:45:00",
        "glucose": 192.0
      },
      {
        "timestamp": "2024-01-15T15:50:00",
        "glucose": 195.0
      },
      {
        "timestamp": "2024-01-15T15:55:00",
        "glucose": 198.0
      }
    ]
  }'
```

## Expected Response

```json
{
  "user_id": "123",
  "will_spike": true,
  "risk_score": 0.85,
  "explanation": "Risk: VERY HIGH (85% probability of glucose spike above 180 mg/dL in next 2 hours). Recent glucose average is 165 mg/dL with rapidly rising velocity of 2.5 mg/dL/min (up 35 mg/dL in last 30 minutes). Recent significant carb intake of 60g consumed 30 minutes ago with glucose already rising."
}
```

## Data Requirements

### Minimum Required Fields
- `timestamp` (ISO 8601 format)
- `glucose` (mg/dL)

### Optional Fields (All Default to 0 if Not Provided)
- `carbs` - Carbohydrate intake in grams
- `bolus_dose` - Insulin bolus in units
- `meal_type` - Type of meal (Breakfast, Lunch, Dinner, Snack)
- `finger_stick_glucose` - Manual finger stick reading
- `basal_rate` - Basal insulin rate (u/hr)
- `temp_basal_rate` - Temporary basal rate (u/hr)
- `hypo_event` - Hypoglycemia event (0 or 1)
- `stress_event` - Stress event (0 or 1)
- `exercise_duration` - Exercise duration in minutes
- `exercise_intensity` - Exercise intensity (1-10)
- `heart_rate` - Heart rate in bpm
- `gsr` - Galvanic skin response
- `skin_temp_f` - Skin temperature in Fahrenheit
- `air_temp_f` - Air temperature in Fahrenheit
- `steps` - Step count per 5-min window

## Testing with Python

```python
import requests
import json

# Simple test
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "user_id": "123",
        "recent_data": [
            {"timestamp": f"2024-01-15T14:{i:02d}:00", "glucose": 120 + i}
            for i in range(24)
        ]
    }
)

print(json.dumps(response.json(), indent=2))
```

## Error Handling

### Insufficient Data
```bash
# Only 10 readings (need 24)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "123",
    "recent_data": [
      {"timestamp": "2024-01-15T14:00:00", "glucose": 120}
    ]
  }'
```

**Response:**
```json
{
  "detail": "Need at least 24 readings (2 hours). Received 1"
}
```

## Notes

- Timestamps should be in chronological order
- Data should be at 5-minute intervals (standard CGM frequency)
- The API uses the most recent reading for prediction
- All optional fields can be omitted; they default to 0
