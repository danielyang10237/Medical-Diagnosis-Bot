import numpy as np
import pandas as pd
import random

class QLearningAgent(object):
    def __init__(self, small_file='freakinatorDB_reduced.csv', large_file='freakinatorDB.csv'):
        self.small_data = pd.read_csv(small_file)
        self.full_data = pd.read_csv(large_file)

        self.diseases = self.small_data['disease'].unique()
        self.small_symptoms = self.small_data.columns.tolist()[1:]
        self.full_symptoms = self.full_data.columns.tolist()[1:]
        
        self.small_symptom_to_index = {symptom: idx for idx, symptom in enumerate(self.small_symptoms)}
        self.small_index_to_symptom = {idx: symptom for symptom, idx in self.small_symptom_to_index.items()}

        self.full_symptom_to_index = {symptom: idx for idx, symptom in enumerate(self.full_symptoms)}
        self.full_index_to_symptom = {idx: symptom for symptom, idx in self.full_symptom_to_index.items()}

        self.disease_to_index = {disease: idx for idx, disease in enumerate(self.diseases)}
        self.index_to_disease = {idx: disease for disease, idx in self.disease_to_index.items()}
    
    def qlearn(self, symptoms, data, num_diseases, alpha=0.1, gamma=0.9, epsilon=0.1, num_episodes=1000):
        num_symptoms = len(symptoms)
        q_table = np.zeros((2**num_symptoms, num_diseases))

        data = data.to_numpy()

        print(data.dtypes if isinstance(data, pd.DataFrame) else type(data))

        print(q_table.shape, "SHAPE")

        for episode in range(num_episodes):
            state = random.randint(0, 2**num_symptoms - 1)
            if isinstance(data, pd.DataFrame):
                data = data.to_numpy()  # Convert to NumPy for zero-based indexing

            while True:
                if random.uniform(0, 1) < epsilon:
                    action = random.randint(0, num_diseases - 1)
                else:
                    action = np.argmax(q_table[state])

                next_state = random.randint(0, 2**num_symptoms - 1)
                reward = 0

                if next_state < data.shape[0] and action < data.shape[1]:
                    reward = data[next_state, action]
                else:
                    print(f"Index out of bounds: next_state={next_state}, action={action}")
                    continue  # Skip invalid states

                q_table[state][action] = q_table[state][action] + alpha * (
                    reward + gamma * np.max(q_table[next_state]) - q_table[state][action]
                )
                state = next_state

                if reward == 1:
                    break
        
        return q_table
    
    def calculate_probabilities(self, observed_symptoms, q_table):
        
        def get_state_index(symptom_vector):
            state = 0
            for idx, val in enumerate(symptom_vector):
                state += val * (2**idx)
            return state

        state = get_state_index(observed_symptoms)
        probabilities = q_table[state]
        total = sum(probabilities)
        return [p/total for p in probabilities]
    
    def symptom_screen(self, symptom_length):
        observed_symptoms = [-1] * symptom_length
        print("Answer the following questions with 'yes' or 'no':\n")

        for action in range(len(self.small_symptoms)):
            symptom_name = self.small_index_to_symptom[action]
            answer = input(f"Do you have '{symptom_name}'? (yes/no): ").strip().lower()
            if answer == 'yes' or answer == 'y':
                observed_symptoms[action] = 1
            else:
                observed_symptoms[action] = 0
        
        return observed_symptoms
        
    
    def narrow_down(self, candidate_diseases):
        disease_list = [self.index_to_disease[x[0]] for x in candidate_diseases]
        filtered_data = self.full_data[self.full_data['disease'].isin(disease_list)]

        filtered_data = filtered_data.dropna(axis=1, how='all')

        symptoms = filtered_data.columns.tolist()[1:]

        symptom_to_index = {symptom: idx for idx, symptom in enumerate(symptoms)}

        data = filtered_data.to_numpy()

        q_table = self.qlearn(symptoms, data, len(candidate_diseases), num_episodes=1000)

        observed_symptoms = self.second_screen(symptoms)

        probabilities = self.calculate_probabilities(observed_symptoms, q_table)

        for idx, prob in enumerate(probabilities):
            print(f"{self.index_to_disease[idx]}: {prob:.2f}")
        

    def diagnose(self):
        initial_symptoms = self.symptom_screen(len(self.small_symptoms))
        first_q_table = self.qlearn(self.small_symptoms, self.small_data, len(self.diseases), num_episodes=1000)
        probabilities = self.calculate_probabilities(initial_symptoms, first_q_table)

        for idx, prob in enumerate(probabilities):
            print(f"{self.index_to_disease[idx]}: {prob:.2f}")
        
        print("Would you like to narrow it down?")
        answer = input("yes/no: ").strip().lower()

        if answer == 'yes' or answer == 'y':
            top_candidates = [(i, x) for i, x in enumerate(probabilities) if x > 0.1]

            # second_q_table = self.qlearn(self.full_symptoms, self.full_data, num_episodes=1000)
            # probabilities = self.calculate_probabilities(initial_symptoms, second_q_table)

            # for idx, prob in enumerate(probabilities):
            #     print(f"{self.index_to_disease[idx]}: {prob:.2f}")
        else:
            print("Have a nice day!")

qlearn = QLearningAgent()

qlearn.diagnose()