import csv
import random

def load_dataset(file_path, destination_path):
    symptoms_idx = 0
    symptoms2idx = {}
    dataset = []

    with open(file_path, 'r') as file:
        for line in file:
            colon_idx = line.index(':')
            disease = line[:colon_idx]
            symptoms = line[colon_idx + 1:].strip().split(',')

            for symptom in symptoms:
                if "(occasional)" in symptom:
                    symptom = symptom.replace("(occasional)", "").strip()

                if ":" in symptom:
                    symptom = symptom[symptom.index(':')+1:].strip()

                
                symptom = symptom.strip()
                
                if symptom not in symptoms2idx:
                    symptoms2idx[symptom] = symptoms_idx
                    symptoms_idx += 1
        
        num_entries = len(symptoms2idx)
        header = ["disease"] + [''] * num_entries
        for symptom, idx in symptoms2idx.items():
            header[idx + 1] = symptom
        
        file.seek(0)

        dataset.append(header)
        for line in file:
            colon_idx = line.index(':')
            disease = line[:colon_idx]
            symptoms = line[colon_idx + 1:].strip().split(',')

            for _ in range(16):
                entry = [disease] + ['0'] * num_entries
                for symptom in symptoms:
                    threshold = 90
                    if "(occasional)" in symptom:
                        symptom = symptom.replace("(occasional)", "").strip()
                        threshold = 50

                    if ":" in symptom:
                        symptom = symptom[symptom.index(':')+1:].strip()
                    
                    symptom = symptom.strip()
                    
                    if random.randint(0, 100) < threshold:
                        entry[symptoms2idx[symptom] + 1] = '1'
                
                dataset.append(entry)
    
    # scramble the order of dataset after the header
    dataset = dataset[:1] + random.sample(dataset[1:], len(dataset) - 1)
        
    with open(destination_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(dataset)


print(load_dataset("freakinator_reduced.txt", "freakinatorDB_reduced.csv"))