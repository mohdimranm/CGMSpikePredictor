# OhioT1DM Data Transformation Explained

## The Challenge

Different data types in the XML have different timestamps and frequencies:

- **Glucose (CGM)**: Every 5 minutes (e.g., 08:00, 08:05, 08:10...)
- **Meals**: Irregular times (e.g., 08:17, 12:34, 19:45...)
- **Bolus insulin**: Irregular times (before meals typically)
- **Heart rate**: Every 5 minutes (may not align exactly with glucose)
- **Exercise**: Irregular start times with durations

## How We Merge Them

We use **`pd.merge_asof()`** with **`direction='backward'`**

### What does `merge_asof` do?

It's an **"as-of" merge** - for each timestamp in the left DataFrame, it finds the **closest matching timestamp** in the right DataFrame that is:
- **Less than or equal to** the left timestamp (when `direction='backward'`)

### Step-by-Step Example

#### Step 1: Base Data (Glucose readings every 5 minutes)
```
timestamp           glucose
08:00:00           100
08:05:00           105
08:10:00           110
08:15:00           115
08:20:00           125
08:25:00           135
08:30:00           140
```

#### Step 2: Meal Data (irregular times)
```
timestamp           meal_type    carbs
08:17:00           Breakfast    45
12:34:00           Lunch        60
```

#### Step 3: After merge_asof (backward)
```
timestamp           glucose    meal_type    carbs
08:00:00           100        None         0        # No meal before this
08:05:00           105        None         0        # No meal before this
08:10:00           110        None         0        # No meal before this
08:15:00           115        None         0        # No meal before this
08:20:00           125        Breakfast    45       # Meal at 08:17 carries forward
08:25:00           135        Breakfast    45       # Still using meal from 08:17
08:30:00           140        Breakfast    45       # Still using meal from 08:17
```

**Key Point**: The meal at 08:17 is "carried forward" to all glucose readings after it (08:20, 08:25, 08:30...) until a new meal occurs.

## Why This Makes Sense

This is the **correct approach** for CGM analysis because:

1. **Current state matters**: At any glucose reading, we want to know the most recent meal/insulin/exercise that could be affecting the current glucose level.

2. **Forward-fill logic**: If you ate at 08:17 with 45g carbs, that meal is still affecting your glucose at 08:20, 08:25, 08:30, etc. We want to capture that.

3. **Time-series nature**: Glucose responds to events that happened in the recent past, not future events.

## Real Example from the Dataset

Let's trace one glucose reading:

```python
# Glucose reading at 12:03:00
patient_id: 588
timestamp: 2021-08-30 12:03:00
glucose: 119.0

# Most recent meal (before 12:03)
meal_timestamp: 2021-08-30 12:03:00  # Exact match!
meal_type: Lunch
carbs: 37.0

# Most recent heart rate (before 12:03)
hr_timestamp: 2021-08-30 12:03:00  # Exact match!
heart_rate: 111.0

# Most recent bolus (before 12:03)
bolus_timestamp: 2021-08-30 11:45:00  # 18 minutes ago
bolus_dose: 4.5

# Result in unified DataFrame:
12:03:00, glucose=119, meal_type=Lunch, carbs=37,
         heart_rate=111, bolus_dose=4.5
```

## The Code That Does This

From `app/ohio_data_loader.py`:

```python
# Merge meal data
if not data['meal'].empty:
    df = pd.merge_asof(
        df.sort_values('timestamp'),           # Left: glucose readings
        data['meal'].sort_values('timestamp'),  # Right: meal events
        on='timestamp',                         # Join on timestamp
        by='patient_id',                        # Separate merge per patient
        direction='backward',                   # Find most recent event
        suffixes=('', '_meal')                  # Handle column name conflicts
    )
    df['carbs'] = df['carbs'].fillna(0)        # Fill missing with 0
```

## Why Timestamps Don't "Match" Exactly

You're correct that timestamps don't match exactly! Here's why:

### Example Scenario:

**Glucose readings:**
```
08:00:00  -> glucose = 100
08:05:00  -> glucose = 105
08:10:00  -> glucose = 110
```

**Meal event:**
```
08:07:00  -> meal = Breakfast, carbs = 50
```

**After merge_asof (backward):**
```
08:00:00  -> glucose = 100, carbs = 0     # No meal before
08:05:00  -> glucose = 105, carbs = 0     # No meal before
08:10:00  -> glucose = 110, carbs = 50    # Meal at 08:07 carries forward
```

The meal at 08:07 doesn't appear at 08:05 because `direction='backward'` only looks back in time.

## Implications for Machine Learning

This merging strategy is **exactly what we want** for prediction:

1. **Features represent the current state**: At any time point, we know what meal was eaten most recently, what the current heart rate is, etc.

2. **Temporal validity**: We never use future information to predict the current state (no data leakage).

3. **Realistic scenario**: In real-time prediction, you only know events that have already happened.

## Verification

You can verify this by looking at the data:

```python
# Load data
train_df = pd.read_csv('data/processed/train_data.csv', parse_dates=['timestamp'])

# Filter patient 588 around a meal time
patient_588 = train_df[train_df['patient_id'] == '588']
meal_window = patient_588[
    (patient_588['timestamp'] >= '2021-08-30 11:50:00') &
    (patient_588['timestamp'] <= '2021-08-30 12:30:00')
]

# You'll see the meal info at 12:03 carries forward to subsequent readings
print(meal_window[['timestamp', 'glucose', 'meal_type', 'carbs']])
```

## Summary

**The key insight**: We're not trying to match timestamps exactly. Instead, we're creating a **time-series feature set** where each glucose reading knows:
- The most recent meal eaten
- The most recent insulin dose
- The current heart rate
- The current step count
- etc.

This is the proper way to structure time-series data for machine learning!

---

# Complete Dataset Column Reference

## Overview

Our processed dataset has **18 columns** containing 69,255 training records and 15,970 test records from 6 Type 1 Diabetes patients.

**File locations:**
- Training: `data/processed/train_data.csv` (3.7 MB)
- Testing: `data/processed/test_data.csv` (857 KB)

---

## Column Descriptions

### 1. **patient_id** (String)
- **Description**: Unique identifier for each patient
- **Values**: '559', '563', '570', '575', '588', '591'
- **Total patients**: 6
- **Purpose**: Group data by individual; each patient has unique glucose patterns
- **Note**: Data from 2018 BGLP Challenge

### 2. **timestamp** (Datetime)
- **Description**: Date and time of the glucose reading
- **Format**: `YYYY-MM-DD HH:MM:SS`
- **Frequency**: Every 5 minutes (standard CGM sampling rate)
- **Range**:
  - Training: 2021-08-30 to 2022-01-17
  - Testing: 2021-10-15 to 2022-01-27
- **Purpose**: Primary time index; all other data is merged to glucose timestamps

### 3. **glucose** (Float)
- **Description**: Blood glucose level from Continuous Glucose Monitor (CGM)
- **Unit**: mg/dL (milligrams per deciliter)
- **Range**: 40 - 400 mg/dL
- **Statistics**:
  - Mean: 160.13 mg/dL
  - Std: 60.54 mg/dL
  - Target range: 70-180 mg/dL
- **Clinical significance**:
  - < 70: Hypoglycemia (low blood sugar)
  - 70-180: Target range
  - \> 180: Hyperglycemia (high blood sugar / "spike")
  - \> 250: Severe hyperglycemia
- **Purpose**: Primary outcome variable for spike prediction

---

## Meal & Nutrition Data

### 4. **meal_type** (String)
- **Description**: Type/category of meal consumed
- **Values**:
  - 'Breakfast', 'Lunch', 'Dinner', 'Snack'
  - 'None' (no recent meal)
- **Source**: Patient self-reported via app
- **Total events**: 1,091 meals in training set
- **Merge behavior**: Forward-filled from actual meal timestamp
- **Purpose**: Identify meal timing and type affecting glucose

### 5. **carbs** (Float)
- **Description**: Carbohydrate content of the meal
- **Unit**: grams (g)
- **Range**: 0 - 450g
- **Statistics** (when > 0):
  - Mean: 44.99g per meal
  - Median: ~35g
  - Typical ranges:
    - Small snack: 10-20g
    - Regular meal: 30-60g
    - Large meal: 60-100g
- **Source**: Patient-estimated carb count
- **Merge behavior**: 0 when no recent meal, forward-filled after meal
- **Purpose**: Primary predictor of glucose spikes (carbs → glucose increase)

---

## Insulin Data

### 6. **finger_stick_glucose** (Float)
- **Description**: Manual blood glucose check (finger prick test)
- **Unit**: mg/dL
- **Range**: Similar to CGM glucose (40-400)
- **Frequency**: Irregular (2,253 readings in training)
- **Purpose**:
  - Calibration reference for CGM
  - More accurate than CGM (gold standard)
  - Used before meals/insulin dosing
- **Merge behavior**: Forward-filled from most recent finger stick

### 7. **basal_rate** (Float)
- **Description**: Continuous background insulin infusion rate
- **Unit**: Units per hour (u/hr)
- **Range**: 0.5 - 2.5 u/hr typically
- **Source**: Insulin pump settings
- **Total events**: 681 rate changes
- **Purpose**:
  - Maintains baseline glucose control
  - Set by physician, changes infrequently
  - Affects glucose over hours
- **Merge behavior**: Forward-filled (rate continues until next change)

### 8. **temp_basal_rate** (Float)
- **Description**: Temporary basal rate override
- **Unit**: Units per hour (u/hr)
- **Range**: 0 (suspended) to 3.0 u/hr
- **Total events**: 118 temporary adjustments
- **Purpose**:
  - Short-term adjustments (exercise, illness)
  - 0 = basal insulin suspended
  - Overrides normal basal_rate
- **Merge behavior**: Defaults to basal_rate when no temp adjustment active

### 9. **bolus_dose** (Float)
- **Description**: Fast-acting insulin dose (meal coverage)
- **Unit**: Insulin units (u)
- **Range**: 0 - 15 units typically
- **Total events**: 1,455 bolus doses
- **Purpose**:
  - Cover carbs from meals
  - Correct high glucose
  - Delivered before/during meals
  - Acts within 15-60 minutes
- **Merge behavior**: 0 when no recent bolus, forward-filled after dose
- **Clinical note**: Counteracts glucose rise from carbs

---

## Exercise & Activity Data

### 10. **exercise_duration** (Float)
- **Description**: Duration of exercise session
- **Unit**: Minutes
- **Range**: 0 - 180 minutes
- **Total events**: 126 exercise sessions
- **Statistics** (when > 0):
  - Mean: 90.11 minutes
  - Typical: 30-60 minutes
- **Purpose**: Exercise lowers blood glucose
- **Merge behavior**: 0 when no recent exercise, forward-filled during session

### 11. **exercise_intensity** (Integer)
- **Description**: Subjective exercise intensity rating
- **Scale**: 1-10
  - 1-3: Light (walking)
  - 4-6: Moderate (jogging)
  - 7-10: Vigorous (running, sports)
- **Mean**: 4.96 when exercising
- **Purpose**: Higher intensity → greater glucose reduction
- **Merge behavior**: 0 when not exercising, forward-filled during session

---

## Event Markers

### 12. **hypo_event** (Integer - Binary)
- **Description**: Self-reported hypoglycemic episode
- **Values**: 0 (no event), 1 (hypoglycemia occurring)
- **Total events**: 74 in training set
- **Clinical definition**: Glucose < 70 mg/dL with symptoms
- **Symptoms**: Shakiness, sweating, confusion, dizziness
- **Purpose**: Identify dangerous low glucose events
- **Merge behavior**: 0 normally, 1 when hypo event active

### 13. **stress_event** (Integer - Binary)
- **Description**: Self-reported stress episode
- **Values**: 0 (no stress), 1 (stressed)
- **Total events**: 4 in training set (very rare)
- **Purpose**: Stress hormones can elevate glucose
- **Note**: Underreported (most stress not logged)
- **Merge behavior**: 0 normally, 1 during stress event

---

## Physiological Sensors (Basis Peak Wearable)

These fields come from a wearable sensor worn by patients. Not all patients have this data.

### 14. **heart_rate** (Float)
- **Description**: Heart rate from wearable sensor
- **Unit**: Beats per minute (bpm)
- **Range**: 40 - 180 bpm typically
- **Frequency**: Every 5 minutes (73,975 readings)
- **Statistics**:
  - Resting: 60-80 bpm
  - Active: 100-150 bpm
- **Purpose**:
  - Indicator of physical activity
  - Stress response
  - Sleep detection (lower HR)
- **Merge behavior**: Filled with mean when missing, otherwise exact match

### 15. **gsr** (Float - Galvanic Skin Response)
- **Description**: Skin conductance / electrodermal activity
- **Unit**: Microsiemens (μS)
- **Range**: 0 - 0.001 typically (scientific notation in data)
- **Frequency**: Every 5 minutes (73,166 readings)
- **Purpose**:
  - Measures sweat/moisture on skin
  - Indicator of stress/arousal
  - Higher during exercise/stress
- **Clinical relevance**: May correlate with hypoglycemia symptoms
- **Merge behavior**: 0 when missing, otherwise exact match

### 16. **skin_temp_f** (Float)
- **Description**: Skin surface temperature
- **Unit**: Degrees Fahrenheit (°F)
- **Range**: 85 - 98°F typically
- **Frequency**: Every 5 minutes (73,314 readings)
- **Statistics**:
  - Mean: ~92°F
  - Normal range: 88-95°F
- **Purpose**:
  - Indicator of circulation
  - Exercise (increases)
  - Sleep (decreases)
- **Merge behavior**: Filled with mean when missing

### 17. **air_temp_f** (Float)
- **Description**: Ambient air temperature
- **Unit**: Degrees Fahrenheit (°F)
- **Range**: 50 - 120°F
- **Frequency**: Every 5 minutes (73,314 readings)
- **Statistics**:
  - Mean: 84.56°F
  - Range: 50-119°F
- **Purpose**:
  - Environmental context
  - Extreme temps affect glucose
  - Indoor vs outdoor detection
- **Merge behavior**: Filled with mean when missing

### 18. **steps** (Float)
- **Description**: Step count per 5-minute window
- **Unit**: Steps
- **Range**: 0 - 160 steps per 5 min
- **Frequency**: Every 5 minutes (74,139 readings)
- **Statistics**:
  - Mean when active: 10-30 steps per 5 min
  - 0 = sedentary
  - \> 50 = walking/running
- **Purpose**:
  - Activity level detection
  - Exercise validation
  - Sedentary vs active periods
- **Conversion**: ~2,000 steps/mile, 10,000 steps/day target
- **Merge behavior**: 0 when no movement

---

## Data Quality Notes

### Missing Data Handling

1. **Sensor availability**: Not all patients have all sensors
   - Heart rate, GSR, temps, steps: Only available for patients with Basis Peak
   - Some patients: 0 readings for these fields

2. **Forward-fill fields** (retain last known value):
   - meal_type, carbs
   - basal_rate, temp_basal_rate, bolus_dose
   - exercise_duration, exercise_intensity
   - hypo_event, stress_event

3. **Filled with mean** (continuous physiological):
   - heart_rate
   - skin_temp_f
   - air_temp_f

4. **Filled with 0** (activity/events):
   - finger_stick_glucose
   - gsr
   - steps

### Temporal Alignment

- **Exact matches**: glucose, heart_rate, gsr, temps, steps (all 5-min intervals)
- **Forward-filled**: meals, insulin, exercise (irregular timing)
- **This is intentional**: Represents "current state" at each glucose reading

---

## Usage for Machine Learning

### Feature Engineering

From these 18 raw columns, we create ~50 engineered features:

1. **Glucose trends**: deltas, velocities, accelerations
2. **Time features**: hour, day of week, cyclical encoding
3. **Meal features**: time since meal, rolling carb sums
4. **Insulin features**: time since bolus, rolling doses
5. **Activity features**: exercise windows, step counts
6. **Physiological**: HR variability, temperature changes

### Target Variable

**Spike prediction**: Will glucose exceed 180 mg/dL in next 2 hours?
- Binary classification: 0 (no spike) or 1 (will spike)
- Calculated by looking ahead 24 readings (2 hours)

### Clinical Relevance

The 180 mg/dL threshold is based on:
- ADA (American Diabetes Association) guidelines
- Target for postprandial (after-meal) glucose
- Sustained levels > 180 increase diabetes complications