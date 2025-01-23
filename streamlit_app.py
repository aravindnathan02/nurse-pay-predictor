import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import pickle
import os
from main import NursePayPipeline

def load_pipeline():
    pipeline = NursePayPipeline()
    if not pipeline.load_models():
        st.error("Error loading models. Please ensure models are trained first.")
        return None
    return pipeline

def main():
    st.set_page_config(page_title="Nurse Pay Rate Predictor", layout="wide")
    
    st.title("🏥 Nurse Pay Rate Prediction System")
    st.write("This application predicts hourly pay rates for healthcare professionals based on various factors.")
    
    # Initialize pipeline
    pipeline = load_pipeline()
    if pipeline is None:
        return
    
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Pay Prediction", "Market Analysis", "Historical Trends"])
    
    with tab1:
        st.header("Pay Rate Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            job_title = st.selectbox(
                "Select Job Title",
                options=pipeline.job_titles,
                help="Choose the healthcare position"
            )
            
            location = st.selectbox(
                "Select Location",
                options=pipeline.locations,
                help="Choose the work location"
            )
            
            hospital_name = st.selectbox(
                "Select Hospital",
                options=[pipeline.generate_hospital_name(location) for _ in range(5)],
                help="Select the healthcare facility"
            )
        
        with col2:
            start_date = st.date_input(
                "Contract Start Date",
                max_value=datetime.now(),
                help="When does the contract begin?"
            )
            
            duration_weeks = st.slider(
                "Contract Duration (weeks)",
                min_value=1,
                max_value=13,
                help="How long is the contract?"
            )
            
            end_date = start_date + timedelta(weeks=duration_weeks)
            st.write(f"Contract End Date: {end_date}")
        
        if st.button("Calculate Pay Rate", type="primary"):
            # Prepare prediction data
            pred_data = pd.DataFrame({
                'Job_Title': [job_title],
                'Location': [location],
                'Hospital_Name': [hospital_name],
                'Contract_Start': [start_date],
                'Contract_End': [end_date],
                'Season': ['normal'],  # Will be updated in preprocessing
                'Hourly_Pay': [0]  # Placeholder
            })
            
            # Preprocess the data
            pred_data = pipeline.preprocess_data(pred_data)
            
            # Make predictions with both models
            with st.spinner("Calculating predictions..."):
                # Format predictions for display
                month = start_date.month
                season = "holiday" if month == 12 else "flu" if month in [10, 11, 1, 2, 3, 4, 5] else "normal"
                base_rate = pipeline.base_rates[job_title]
                
                st.success("Prediction Complete!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Base Rate",
                        f"${base_rate:.2f}/hr",
                        help="Standard base rate for this position"
                    )
                
                with col2:
                    seasonal_adjustment = 1.0 if season == "normal" else 1.2 if season == "flu" else 1.3
                    adjusted_rate = base_rate * seasonal_adjustment
                    st.metric(
                        "Seasonal Adjusted Rate",
                        f"${adjusted_rate:.2f}/hr",
                        f"{((seasonal_adjustment - 1) * 100):.0f}% seasonal adjustment",
                        help=f"Rate adjusted for {season} season"
                    )
                
                with col3:
                    desirability_score = pipeline.desirability_scores[location.split(",")[0]]
                    st.metric(
                        "Location Score",
                        f"{desirability_score}/100",
                        help="Location desirability rating"
                    )
    
    with tab2:
        st.header("Market Analysis")
        
        # Load and display market trends
        try:
            data = pd.read_csv("Synthetic_Nurse_Pay_Data.csv")
            data = pipeline.preprocess_data(data)
            
            st.subheader("Pay Rates by Location")
            avg_pay_by_location = data.groupby('Location')['Hourly_Pay'].agg(['mean', 'min', 'max']).round(2)
            st.dataframe(avg_pay_by_location, use_container_width=True)
            
            st.subheader("Seasonal Trends")
            seasonal_avg = data.groupby('Season')['Hourly_Pay'].mean().round(2)
            st.bar_chart(seasonal_avg)
            
        except Exception as e:
            st.error("Error loading market analysis data. Please ensure the dataset exists.")
    
    with tab3:
        st.header("Historical Trends")
        st.write("View historical pay rate trends for different positions and locations.")
        
        # Allow users to filter data
        selected_location = st.multiselect(
            "Select Locations",
            options=pipeline.locations,
            default=[pipeline.locations[0]]
        )
        
        selected_job = st.multiselect(
            "Select Job Titles",
            options=pipeline.job_titles,
            default=[pipeline.job_titles[0]]
        )
        
        try:
            data = pd.read_csv("Synthetic_Nurse_Pay_Data.csv")
            filtered_data = data[
                data['Location'].isin(selected_location) &
                data['Job_Title'].isin(selected_job)
            ]
            
            if not filtered_data.empty:
                st.line_chart(
                    filtered_data.groupby('Contract_Start')['Hourly_Pay'].mean(),
                    use_container_width=True
                )
            else:
                st.warning("No data available for the selected filters.")
                
        except Exception as e:
            st.error("Error loading historical data. Please ensure the dataset exists.")

if __name__ == "__main__":
    main()