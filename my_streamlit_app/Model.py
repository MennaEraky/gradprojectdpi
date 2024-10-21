import gdown
import pickle
import pandas as pd

def load_model():
    # Load model from Google Drive
    model_id = 'YOUR_MODEL_FILE_ID'  # Replace with the actual file ID from Google Drive
    url = f'https://drive.google.com/uc?id={model_id}'
    gdown.download(url, 'vehicle_price_model.pkl', quiet=False)
    with open('vehicle_price_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

def preprocess_input(input_data, model):
    input_df = pd.DataFrame([input_data])
    return input_df  # Modify according to your preprocessing
