# Medical Diagnosis System using Q-Learning
This is a project that implements an intelligent medical diagnosis system using Q-Learning reinforcement learning to optimize symptom questioning and disease prediction.

## Overview

This project develops an AI-powered medical diagnosis system that learns to ask the most informative questions to efficiently diagnose diseases. The system uses Q-Learning to optimize the sequence of symptom inquiries, reducing the number of questions needed while maintaining high diagnostic accuracy. Features:

- **Q-Learning Optimization**: Uses reinforcement learning to learn optimal symptom questioning strategies
- **Multi-Disease Support**: Handles hundreds of diseases with their associated symptoms
- **Interactive Diagnosis**: Real-time patient questioning with intelligent follow-up
- **Non-Interactive Testing**: Automated evaluation on test datasets
- **Symptom Categorization**: Organized symptom mapping for better medical understanding
- **Confidence-Based Stopping**: Stops questioning when diagnosis confidence exceeds 90%

## Project Structure

```
├── Medical-Diagnosis/           # Main diagnosis system
│   ├── diagnose.py             # Interactive diagnosis interface
│   ├── diagnose_test.py        # Non-interactive testing and evaluation
│   ├── q_learning_train.py     # Q-Learning training implementation
│   ├── q_table.npy            # Pre-trained Q-table
│   ├── disease_symptom_prob.json # Disease-symptom probability mappings
│   ├── idx_to_disease.json    # Disease index mappings
│   ├── symptoms.json          # Symptom definitions
│   └── medical_diagnosis.csv  # Training dataset
├── qlearn.py                  # Basic Q-Learning implementation
├── full_qlearn.py            # Enhanced Q-Learning with preprocessing
├── dataset_loader.py         # Dataset processing utilities
├── preprocess.py             # Symptom mapping and preprocessing
├── download_db.py            # Dataset download utilities
├── data.csv                  # Full symptom database
├── data_reduced.csv          # Reduced symptom database
├── conditions_list.txt           # Raw disease-symptom data
└── Various disease datasets  # Specialized datasets (asthma, heart, etc.)
```

**Note datasets are not included in this repo due to size, contact me for data inquiries**

**Note my trained models stored in .npy are also not included, since the parameter decision tree is too large. Users will have to retrain model locally**

## How It Works

### 1. Data Preprocessing
- Raw disease-symptom data is processed and categorized into medical symptom groups
- Symptoms are mapped to standardized categories (e.g., "Localized Pain", "Respiratory Issues")
- Binary feature vectors are created for each disease-symptom combination

### 2. Q-Learning Training
- The system learns optimal questioning strategies through reinforcement learning
- States represent which symptoms have been asked about
- Actions represent which symptom to ask about next or when to make a diagnosis
- Rewards are based on diagnostic accuracy and efficiency

### 3. Diagnosis Process
- Interactive questioning based on learned Q-table
- Bayesian probability calculations for disease likelihood
- Early stopping when confidence threshold is reached
- Fallback to most likely diagnosis if confidence threshold not met

### Interactive Diagnosis
```python
# The system will ask questions like:
# "Do you have 'Localized Pain'? (1 for Yes, 0 for No): "
# "Do you have 'Fever'? (1 for Yes, 0 for No): "
# 
# After sufficient information is gathered:
# "Predicted Disease: Malaria (Probability: 0.92)"
```

### Non-Interactive Testing
```python
# Test on a patient with known symptoms
patient_symptoms = [1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
predicted_disease = diagnose_patient_non_interactive(
    q_table, symptoms, disease_symptom_prob, 
    idx_to_disease, n_diseases, len(symptoms), 
    patient_symptoms
)
```

## Performance

The system achieves high diagnostic accuracy while minimizing the number of questions asked:
- **Accuracy**: ~85-90% on test datasets
- **Efficiency**: Average 5-8 questions per diagnosis
- **Confidence**: 90% threshold for early stopping

## Dataset Information

### Main Dataset
- **Source**: Processed medical data with disease-symptom mappings
- **Size**: 6,000+ disease-symptom combinations
- **Symptoms**: 22 categorized symptom groups
- **Diseases**: 400+ different medical conditions

### Core Files
- **`diagnose.py`**: Main interactive diagnosis interface
- **`diagnose_test.py`**: Automated testing and evaluation framework
- **`q_learning_train.py`**: Q-Learning training implementation
- **`preprocess.py`**: Symptom mapping and data preprocessing
- **`dataset_loader.py`**: Dataset loading and processing utilities
- **`qlearn.py`**: Basic Q-Learning agent implementation
- **`full_qlearn.py`**: Enhanced Q-Learning with additional features
