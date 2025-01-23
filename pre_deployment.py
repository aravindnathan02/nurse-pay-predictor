from main import NursePayPipeline
import os

def prepare_deployment():
    # Ensure saved_models directory exists
    os.makedirs("saved_models", exist_ok=True)
    
    # Initialize pipeline
    pipeline = NursePayPipeline()
    
    # Generate synthetic data
    print("Generating synthetic data...")
    data = pipeline.generate_synthetic_data()
    data.to_csv("Synthetic_Nurse_Pay_Data.csv", index=False)
    
    # Preprocess data
    print("Preprocessing data...")
    data = pipeline.preprocess_data(data)
    
    # Train models
    print("Training traditional models...")
    pipeline.train_traditional_models(data)
    
    print("\nTraining LSTM model...")
    pipeline.train_lstm_model(data)

    # Save models
    print("\nSaving models...")
    pipeline.save_models()

if __name__ == "__main__":
    prepare_deployment()    