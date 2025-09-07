import numpy as np
import pandas as pd
import random
import os

class QLearningAgent(object):
    def __init__(self, small_file='freakinatorDB_reduced.csv', large_file='freakinatorDB.csv'):
        self.small_data = pd.read_csv(small_file)
        self.full_data = pd.read_csv(large_file)

        drop_columns = ['Miscellaneous', 'urge to move']

        self.small_data = self.small_data.drop(columns=drop_columns)

        self.diseases = self.small_data['disease'].unique()
        self.small_symptoms = self.small_data.columns.tolist()[1:]
        self.full_symptoms = self.full_data.columns.tolist()[1:]
        
        self.small_symptom_to_index = {symptom: idx for idx, symptom in enumerate(self.small_symptoms)}
        self.small_index_to_symptom = {idx: symptom for symptom, idx in self.small_symptom_to_index.items()}

        self.full_symptom_to_index = {symptom: idx for idx, symptom in enumerate(self.full_symptoms)}
        self.full_index_to_symptom = {idx: symptom for symptom, idx in self.full_symptom_to_index.items()}

        self.disease_to_index = {disease: idx for idx, disease in enumerate(self.diseases)}
        self.index_to_disease = {idx: disease for disease, idx in self.disease_to_index.items()}
    
    def qlearn(self, symptoms, data, num_diseases, initial=True, alpha=0.2, gamma=0.9, epsilon=0.1, num_episodes=1000):
        # check if q_table.npy exists
        if initial and os.path.exists('q_table.npy'):
            q_table = np.load('q_table.npy')
            return q_table

        # Convert data to NumPy once at the start
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()  # First column: disease name, rest: symptom indicators
        
        num_symptoms = len(symptoms)
        q_table = np.zeros((2**num_symptoms, num_diseases))

        for episode in range(num_episodes):
            # Start with a random state
            state = random.randint(0, 2**num_symptoms - 1)
            
            while True:
                # Epsilon-greedy action selection
                if random.uniform(0, 1) < epsilon:
                    action = random.randint(0, num_diseases - 1)
                else:
                    action = np.argmax(q_table[state])

                # Randomly choose a next_state (this is arbitrary and may not be meaningful)
                next_state = random.randint(0, 2**num_symptoms - 1)

                # Define reward:
                # The first column of data contains the disease name.
                # If the chosen action corresponds to the correct disease for next_state, reward = 1, else 0.
                if next_state < data.shape[0]:
                    actual_disease_name = data[next_state, 0]  # The actual disease in that row
                    chosen_disease = self.index_to_disease[action]
                    reward = 1 if actual_disease_name == chosen_disease else 0
                else:
                    # If somehow next_state is out of bounds, skip this iteration
                    continue

                # Q-learning update
                q_table[state, action] = q_table[state, action] + alpha * (
                    reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
                )

                state = next_state

                # If we got a positive reward (correct disease guess), end this episode step
                if reward == 1:
                    break

        if initial:
            np.save('q_table.npy', q_table)
        
        return q_table
    
    def calculate_probabilities(self, observed_symptoms, q_table):
        
        def get_state_index(symptom_vector):
            # Convert a binary vector of symptom presence/absence into a state index
            # Assumption: symptom_vector is a list of 0/1 values
            state = 0
            for idx, val in enumerate(symptom_vector):
                state += val * (2**idx)
            return state

        state = get_state_index(observed_symptoms)
        probabilities = q_table[state]
        total = sum(probabilities)
        # Avoid division by zero if total is zero
        if total == 0:
            # If no probabilities, return uniform or zero
            return [0]*len(probabilities)
        
        return [p/total for p in probabilities]
    
    def symptom_screen(self, symptom_length):
        observed_symptoms = [-1] * symptom_length
        print("Answer the following questions with 'yes' or 'no':\n")

        for action in range(len(self.small_symptoms)):
            symptom_name = self.small_index_to_symptom[action]
            answer = input(f"Do you have '{symptom_name}'? (yes/no): ").strip().lower()
            if answer in ['yes', 'y']:
                observed_symptoms[action] = 1
            else:
                observed_symptoms[action] = 0
        
        return observed_symptoms
        
    
    def narrow_down(self, candidate_diseases):
        disease_list = [self.index_to_disease[x[0]] for x in candidate_diseases]
        filtered_data = self.full_data[self.full_data['disease'].isin(disease_list)]

        # Drop columns with all NaN
        # Filter columns with all NaN values
        filtered_data = filtered_data.dropna(axis=1, how='all')

        # Drop columns where all values are the same
        filtered_data = filtered_data.loc[:, filtered_data.nunique() > 1]
        symptoms = filtered_data.columns.tolist()[1:]

        print("candidate diseases:", disease_list)
        print("symptoms:", symptoms)

        symptom_to_index = {symptom: idx for idx, symptom in enumerate(symptoms)}

        data = filtered_data
        q_table = self.qlearn(symptoms, data, len(candidate_diseases), num_episodes=1000)

        # This method is called but not defined in the original code:
        # We need a method self.second_screen(symptoms) or reuse symptom_screen
        # For now, let's assume second_screen is similar to symptom_screen:
        observed_symptoms = self.second_screen(symptoms)  # You need to implement this method similarly to symptom_screen

        probabilities = self.calculate_probabilities(observed_symptoms, q_table)

        for idx, prob in enumerate(probabilities):
            print(f"{self.index_to_disease[idx]}: {prob:.2f}")
        
    def second_screen(self, symptoms):
        # Similar to symptom_screen but adapted for a new set of symptoms
        observed_symptoms = [-1] * len(symptoms)
        print("Narrowing down: Answer the following with 'yes' or 'no':\n")

        for i, symptom_name in enumerate(symptoms):
            answer = input(f"Do you have '{symptom_name}'? (yes/no): ").strip().lower()
            observed_symptoms[i] = 1 if answer in ['yes', 'y'] else 0
        return observed_symptoms

    def diagnose(self):
        initial_symptoms = self.symptom_screen(len(self.small_symptoms))
        first_q_table = self.qlearn(self.small_symptoms, self.small_data, len(self.diseases), num_episodes=1000)
        probabilities = self.calculate_probabilities(initial_symptoms, first_q_table)

        top_candidates = [(i, x) for i, x in enumerate(probabilities) if x > 0.1]

        for idx, prob in top_candidates:
            if prob > 0:
                print(f"{self.index_to_disease[idx]}: {prob:.2f}")
        
        print("Would you like to narrow it down?")
        answer = input("yes/no: ").strip().lower()

        if answer in ['yes', 'y']:
            if len(top_candidates) == 1:
                print("The most likely diagnosis is:")
                print(self.index_to_disease[top_candidates[0][0]])
            elif len(top_candidates) == 0:
                print("No likely diagnosis found.")
            else:
                self.narrow_down(top_candidates)
            
        else:
            print("Have a nice day!")


# Example usage:
qlearn = QLearningAgent()
qlearn.diagnose()
