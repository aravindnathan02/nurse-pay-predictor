import pickle
import bz2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential, save_model, load_model # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
import category_encoders as ce
from datetime import datetime, timedelta


class NursePayPipeline():
    def __init__(self):
        random.seed(42)
        np.random.seed(42)
        self.locations = [
            "Dallas, TX", "Atlanta, GA", "New York, NY", "Philadelphia, PA", 
            "Washington, DC", "San Fransisco, CA", "Los Angeles, CA", "Seattle, WA", 
            "Chicago, IL", "San Diego, CA", "Miami, FL", "Boston, MA", 
            "Detroit, MI", "Phoenix, AZ", "Houston, TX"
        ]
        
        self.job_titles = [
            "RegisteredNurse_ICU", "RegisteredNurse_MedSurg", "RegisteredNurse_Telemetry",
            "RegisteredNurse_Oncology", "RegisteredNurse_Pediatric", "PhysioTherapist", 
            "LabTechnician", "RegisteredNurse_CriticalCare", "RegisteredNurse_Cardiology",
            "RegisteredNurse_Surgery"
        ]
        
        self.hospital_suffixes = ["Corporate", "NonProfit", "Community", "Veterans", "Govt"]
        
        self.base_rates = {
            "RegisteredNurse_ICU": 40, "RegisteredNurse_MedSurg": 35,
            "RegisteredNurse_Telemetry": 38, "RegisteredNurse_Oncology": 42,
            "RegisteredNurse_Pediatric": 37, "PhysioTherapist": 45,
            "LabTechnician": 30, "RegisteredNurse_CriticalCare": 45,
            "RegisteredNurse_Cardiology": 43, "RegisteredNurse_Surgery": 50
        }
        
        self.desirability_scores = {
            "Dallas": 70, "Atlanta": 65, "New York": 50, "Philadelphia": 60,
            "Washington": 75, "San Fransisco": 40, "Los Angeles": 55, "Seattle": 60,
            "Chicago": 55, "San Diego": 70, "Miami": 65, "Boston": 75,
            "Detroit": 50, "Phoenix": 60, "Houston": 65
        }
        
        self.rf_model = None
        self.xgb_model = None
        self.lstm_model = None
        self.scaler = None
        self.encoder = None
        
    def generate_hospital_name(self, location):
        """Generate hospital name from location and suffix"""
        city = location.split(",")[0]
        suffix = random.choice(self.hospital_suffixes)
        return f"{city} {suffix} Hospital"
    
    def generate_hourly_pay_rate(self, base_rate, season):
        """Generate hourly pay rate based on season"""
        multipliers = {"normal": 1.0, "flu": 1.2, "holiday": 1.3}
        return round(np.random.normal(base_rate * multipliers[season], 2), 2)
    
    def generate_synthetic_data(self, num_rows=250000):
        """Generate synthetic nurse pay data"""
        data = []
        
        for _ in range(num_rows):
            location = random.choice(self.locations)
            job_title = random.choice(self.job_titles)
            hospital_name = self.generate_hospital_name(location)
            
            start_date = datetime(2023, 1, 1) + timedelta(days=random.randint(0, 730))
            end_date = start_date + timedelta(weeks=random.randint(1, 13))
            
            month = start_date.month
            season = "holiday" if month == 12 else "flu" if month in [10, 11, 1, 2, 3, 4, 5] else "normal"
            
            base_rate = self.base_rates[job_title]
            hourly_pay_rate = self.generate_hourly_pay_rate(base_rate, season)
            
            data.append([job_title, location, hospital_name, start_date.date(), end_date.date(), season, hourly_pay_rate])
        
        columns = ["Job_Title", "Location", "Hospital_Name", "Contract_Start", "Contract_End", "Season", "Hourly_Pay"]
        return pd.DataFrame(data, columns=columns)
    
    def preprocess_data(self, data):
        """Preprocess data for ML models"""
        # Add city and specialization
        data['City'] = data['Location'].apply(lambda x: x.split(",")[0])
        data['Specialization'] = data['Job_Title'].apply(lambda x: 'Specialization' if any(s in x for s in ['Oncology', 'Cardiology', 'Surgery']) else 'Other')
        
        # Add desirability scores
        data['Desirability_Score'] = data['City'].map(self.desirability_scores)
        
        # Convert dates
        data['Contract_Start'] = pd.to_datetime(data['Contract_Start'])
        data['Contract_End'] = pd.to_datetime(data['Contract_End'])
        data['Contract_Duration'] = (data['Contract_End'] - data['Contract_Start']).dt.days
        
        return data
    
    def train_traditional_models(self, data):
        """Train Random Forest and XGBoost models"""
        # Prepare features
        categorical_features = ['Job_Title', 'Location', 'Hospital_Name', 'Season', 'City', 'Specialization']
        X = data.drop(columns=['Hourly_Pay'])
        y = data['Hourly_Pay']
        
        # Encode categorical features
        for feature in categorical_features:
            le = LabelEncoder()
            X[feature] = le.fit_transform(X[feature])
        
        # Convert dates to timestamps
        X['Contract_Start'] = pd.to_datetime(X['Contract_Start']).astype(int) / 10**9
        X['Contract_End'] = pd.to_datetime(X['Contract_End']).astype(int) / 10**9
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Random Forest
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.rf_model.fit(X_train, y_train)
        rf_predictions = self.rf_model.predict(X_test)
        
        # Train XGBoost
        self.xgb_model = xgb.XGBRegressor(n_estimators=100, objective='reg:squarederror', random_state=42)
        self.xgb_model.fit(X_train, y_train)
        xgb_predictions = self.xgb_model.predict(X_test)
        
        # Calculate metrics
        metrics = {}
        for name, pred in [('Random Forest', rf_predictions), ('XGBoost', xgb_predictions)]:
            metrics[name] = {
                'MAE': mean_absolute_error(y_test, pred),
                'MSE': mean_squared_error(y_test, pred),
                'R2': r2_score(y_test, pred)
            }
        
        return metrics
    
    def train_lstm_model(self, data):
        """Train LSTM model for time series prediction"""
        # Prepare data
        data_grouped = data.groupby('Contract_Start')['Hourly_Pay'].mean().reset_index()
        
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        data_normalized = self.scaler.fit_transform(data_grouped['Hourly_Pay'].values.reshape(-1, 1))
        
        # Create sequences
        time_step = 10
        X, y = [], []
        for i in range(len(data_normalized) - time_step - 1):
            X.append(data_normalized[i:(i + time_step), 0])
            y.append(data_normalized[i + time_step, 0])
        
        X = np.array(X).reshape(-1, time_step, 1)
        y = np.array(y)
        
        # Build and train model
        self.lstm_model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        
        self.lstm_model.compile(optimizer='adam', loss='mean_squared_error')
        self.lstm_model.fit(X, y, epochs=100, batch_size=32, verbose=1)
        
        return X, data_grouped
    
    def create_visualizations(self, data):
        """Create various visualizations for analysis"""
        # Hourly pay rates by metro
        plt.figure(figsize=(12, 8))
        sns.boxplot(data, x='City', y='Hourly_Pay')
        plt.title('Variations of Hourly Pay Rates Across Major Metros')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()
        
        # Seasonal pay rates
        seasonal_pay = data.groupby('Season')['Hourly_Pay'].mean().reset_index()
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(seasonal_pay, x='Season', y='Hourly_Pay')
        plt.title('Average Hourly Pay Rates During Different Seasons')
        ax.bar_label(ax.containers[0], fmt='%.2f')
        plt.tight_layout()
        plt.show()
        
        # Pay rates vs desirability
        avg_pay = data['Hourly_Pay'].mean()
        plt.figure(figsize=(10, 6))
        sns.lineplot(data, x='Desirability_Score', y='Hourly_Pay', alpha=0.5)
        plt.title('Hourly Pay Rates Against City Desirability')
        plt.axhline(y=avg_pay, color='r', linestyle='--', 
                   label=f'Avg. Pay Rate = {avg_pay:.2f}')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def save_models(self):
        """Save all trained models"""
        # Use the current directory for saving models
        models_dir = os.path.join(os.path.dirname(__file__), "saved_models")
        os.makedirs(models_dir, exist_ok=True)
        
        # Save XGBoost model
        if self.xgb_model:
            with open(os.path.join(models_dir, "xgb_model.pkl"), "wb") as f:
                pickle.dump(self.xgb_model, f)
        
        # Save Random Forest model (compressed)
        # if self.rf_model:
        #     with bz2.BZ2File(os.path.join(models_dir, "rf_model.pkl.bz2"), "wb") as f:
        #         pickle.dump(self.rf_model, f)
            
            # Save LSTM model
        if self.lstm_model:
            self.lstm_model.save(os.path.join(models_dir, "lstm_model.h5"))
        
            # Save scaler
        if self.scaler:
            with open(os.path.join(models_dir, "scaler.pkl"), "wb") as f:
                pickle.dump(self.scaler, f)

    def load_models(self):
        """Load saved models"""
        models_dir = os.path.join(os.path.dirname(__file__), "saved_models")
    
        try:
            # Load XGBoost model
            with open(os.path.join(models_dir, "xgb_model.pkl"), "rb") as f:
                self.xgb_model = pickle.load(f)
            
            # Load Random Forest model (compressed)
            # with bz2.BZ2File(os.path.join(models_dir, "rf_model.pkl.bz2"), "rb") as f:
            #     self.rf_model = pickle.load(f)
                
                # Load LSTM model
            self.lstm_model = load_model(os.path.join(models_dir, "lstm_model.h5"))
            
            # Load scaler
            with open(os.path.join(models_dir, "scaler.pkl"), "rb") as f:
                self.scaler = pickle.load(f)
            
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False

def main():
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
    metrics = pipeline.train_traditional_models(data)
    for model_name, model_metrics in metrics.items():
        print(f"\n{model_name} Metrics:")
        for metric_name, value in model_metrics.items():
            print(f"{metric_name}: {value:.4f}")
    
    print("\nTraining LSTM model...")
    pipeline.train_lstm_model(data)

    # Save models
    print("\nSaving models...")
    pipeline.save_models()
    
    # Create visualizations
    print("\nCreating visualizations...")
    pipeline.create_visualizations(data)

    # Streamlit app
    print("\nYou can now run the Streamlit app by running 'streamlit run streamlit_app.py'")

if __name__ == "__main__":
    main()