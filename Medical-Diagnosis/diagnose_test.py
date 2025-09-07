import math
import numpy as np
import json
import pandas as pd
from sklearn.utils import shuffle

# Load pre-trained Q-table and related files
q_table = np.load('q_table.npy')
with open('disease_symptom_prob.json', 'r') as f:
    disease_symptom_prob = json.load(f)
with open('idx_to_disease.json', 'r') as f:
    idx_to_disease = json.load(f)
with open('symptoms.json', 'r') as f:
    symptoms = json.load(f)

n_diseases = len(idx_to_disease)
n_symptoms = len(symptoms)

# Helper function to calculate state index
def get_state_index(asked_symptoms):
    return int("".join(map(str, asked_symptoms.astype(int))), 2)

def diagnose_patient_non_interactive(q_table, symptoms, disease_symptom_prob, idx_to_disease, n_diseases, max_questions, patient_symptoms):
    """
    Non-interactive version of diagnose_patient.
    patient_symptoms: a binary list/array of length = len(symptoms) indicating if the patient has the symptom (1) or not (0).
    """
    n_symptoms = len(symptoms)
    symptoms_state = np.zeros(n_symptoms, dtype=int)  # Initially unknown
    asked_symptoms = np.zeros(n_symptoms, dtype=int)  # Track which symptoms asked

    for ques_num in range(max_questions):
        state_idx = get_state_index(asked_symptoms)

        # Choose action based on Q-table
        q_values = q_table[state_idx].copy()
        q_values[np.where(asked_symptoms == 1)[0]] = -np.inf
        action = np.argmax(q_values)

        # If action < n_symptoms, it means we are asking about that symptom
        if action < n_symptoms:
            # Mark symptom as asked
            if asked_symptoms[action] == 1:
                # If already asked, just continue
                continue
            asked_symptoms[action] = 1

            # Instead of asking user, we use patient_symptoms directly
            patient_response = patient_symptoms[action]
            symptoms_state[action] = patient_response

            # Compute disease probabilities after asking
            observed_symptoms = np.where(asked_symptoms == 1)[0]
            disease_log_likelihoods = []
            for disease_idx in range(n_diseases):
                disease_name = idx_to_disease[str(disease_idx)]
                epsilon_prob = 1e-6
                log_likelihood = 0
                for symptom_idx in observed_symptoms:
                    prob = disease_symptom_prob[disease_name][symptom_idx]
                    prob = min(max(prob, epsilon_prob), 1 - epsilon_prob)
                    if symptoms_state[symptom_idx] == 1:
                        log_likelihood += math.log(prob)
                    else:
                        log_likelihood += math.log(1 - prob)
                disease_log_likelihoods.append(log_likelihood)

            max_log_likelihood = max(disease_log_likelihoods)
            log_likelihoods_shifted = [ll - max_log_likelihood for ll in disease_log_likelihoods]
            exp_likelihoods = [math.exp(ll) for ll in log_likelihoods_shifted]
            total_likelihood = sum(exp_likelihoods)
            disease_probabilities = [exp_ll / total_likelihood for exp_ll in exp_likelihoods]

            # Check threshold
            max_probability = max(disease_probabilities)
            if max_probability >= 0.9:
                predicted_disease_idx = np.argmax(disease_probabilities)
                predicted_disease = idx_to_disease[str(predicted_disease_idx)]
                return predicted_disease

        else:
            # Action >= n_symptoms means model decides to predict now
            observed_symptoms = np.where(asked_symptoms == 1)[0]
            disease_log_likelihoods = []
            for disease_idx in range(n_diseases):
                disease_name = idx_to_disease[str(disease_idx)]
                epsilon_prob = 1e-6
                log_likelihood = 0
                for symptom_idx in observed_symptoms:
                    prob = disease_symptom_prob[disease_name][symptom_idx]
                    prob = min(max(prob, epsilon_prob), 1 - epsilon_prob)
                    if symptoms_state[symptom_idx] == 1:
                        log_likelihood += math.log(prob)
                    else:
                        log_likelihood += math.log(1 - prob)
                disease_log_likelihoods.append(log_likelihood)

            max_log_likelihood = max(disease_log_likelihoods)
            log_likelihoods_shifted = [ll - max_log_likelihood for ll in disease_log_likelihoods]
            exp_likelihoods = [math.exp(ll) for ll in log_likelihoods_shifted]
            total_likelihood = sum(exp_likelihoods)
            disease_probabilities = [exp_ll / total_likelihood for exp_ll in exp_likelihoods]

            predicted_disease_idx = np.argmax(disease_probabilities)
            predicted_disease = idx_to_disease[str(predicted_disease_idx)]
            return predicted_disease

    # If max questions reached without high confidence
    # Make a forced prediction based on asked symptoms (similar to the else block above)
    observed_symptoms = np.where(asked_symptoms == 1)[0]
    if len(observed_symptoms) > 0:
        disease_log_likelihoods = []
        for disease_idx in range(n_diseases):
            disease_name = idx_to_disease[str(disease_idx)]
            epsilon_prob = 1e-6
            log_likelihood = 0
            for symptom_idx in observed_symptoms:
                prob = disease_symptom_prob[disease_name][symptom_idx]
                prob = min(max(prob, epsilon_prob), 1 - epsilon_prob)
                if symptoms_state[symptom_idx] == 1:
                    log_likelihood += math.log(prob)
                else:
                    log_likelihood += math.log(1 - prob)
            disease_log_likelihoods.append(log_likelihood)

        max_log_likelihood = max(disease_log_likelihoods)
        log_likelihoods_shifted = [ll - max_log_likelihood for ll in disease_log_likelihoods]
        exp_likelihoods = [math.exp(ll) for ll in log_likelihoods_shifted]
        total_likelihood = sum(exp_likelihoods)
        disease_probabilities = [exp_ll / total_likelihood for exp_ll in exp_likelihoods]
        predicted_disease_idx = np.argmax(disease_probabilities)
        predicted_disease = idx_to_disease[str(predicted_disease_idx)]
        return predicted_disease

    # If no symptoms asked (unlikely), just predict the most probable disease overall (or None)
    return None


# --- EVALUATION ---

df = pd.read_csv("medical_diagnosis.csv")

test_df = df.sample(n=17, random_state=285) 

csv_symptoms = test_df.columns[1:].tolist()

correct_predictions = 0
total_cases = 0

for idx, row in test_df.iterrows():
    actual_disease = row['disease']
    patient_symptom_data = row[csv_symptoms].values.astype(int)

    # Diagnose the patient non-interactively
    predicted_disease = diagnose_patient_non_interactive(q_table, symptoms, disease_symptom_prob, idx_to_disease, n_diseases, len(symptoms), patient_symptom_data)

    if predicted_disease == actual_disease:
        correct_predictions += 1
    # else:
    #     print(f"Patient with symptoms {patient_symptom_data} was diagnosed as {predicted_disease}, but actually had {actual_disease}")
    total_cases += 1

accuracy = correct_predictions / total_cases * 100
print(f"Accuracy on sampled cases: {accuracy:.2f}%")
