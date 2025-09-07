import kagglehub

# Download latest version
# path = kagglehub.dataset_download("uom190346a/disease-symptoms-and-patient-profile-dataset")
# path = kagglehub.dataset_download("rabieelkharoua/asthma-disease-dataset")
# path = kagglehub.dataset_download("amirmahdiabbootalebi/heart-disease")
path = kagglehub.dataset_download("rabieelkharoua/alzheimers-disease-dataset")

print("Path to dataset files:", path)