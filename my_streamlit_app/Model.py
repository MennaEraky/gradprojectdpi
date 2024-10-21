import gdown
import pickle
import pandas as pd

def load_model():
    # Load model from Google Drive
    model_id = '11btPBNR74na_NjjnjrrYT8RSf8ffiumo'  # Google Drive file ID
    url = f'https://drive.google.com/uc?id={model_id}'
    
    try:
        # Download the model file
        gdown.download(url, 'vehicle_price_model.pkl', quiet=False)
        
        # Load the model using pickle
        with open('vehicle_price_model.pkl', 'rb') as file:
            model = pickle.load(file)
        
        return model
    
    except gdown.exceptions.FileURLRetrievalError as e:
        print(f"Error downloading file: {e}")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def preprocess_input(input_data, model):
    # Convert input data to DataFrame for preprocessing
    input_df = pd.DataFrame([input_data])
    # Perform additional preprocessing steps as needed
    return input_df  # Modify according to your preprocessing requirements
