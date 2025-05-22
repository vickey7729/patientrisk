from model import PatientRiskModel
import pandas as pd

def get_user_input():
    """Get patient information from user input"""
    print("\n=== Patient Risk Prediction System ===")
    
    # Get demographic information
    print("\nDEMOGRAPHIC INFORMATION:")
    age = int(input("Enter patient age: "))
    gender = input("Enter patient gender (M/F): ").strip().upper()
    
    # Get chronic conditions
    print("\nCHRONIC CONDITIONS:")
    print("Enter patient conditions (comma-separated list).")
    print("Examples: Diabetes, COPD, Heart Failure, Asthma, Hypertension")
    conditions_input = input("Enter conditions: ")
    
    # Process conditions
    if conditions_input.strip():
        chronic_conditions = [cond.strip() for cond in conditions_input.split(",")]
    else:
        chronic_conditions = []
    
    # Create patient dictionary
    patient = {
        "demographics": {
            "age": age,
            "gender": gender
        },
        "conditions": {
            "chronic_conditions": chronic_conditions
        }
    }
    
    return patient

def main():
    # Load data
    try:
        df = pd.read_csv('data/PatientRecords.csv')
        print("Data loaded successfully!")
    except FileNotFoundError:
        print("Error: Could not find data file. Checking alternate locations...")
        try:
            df = pd.read_csv('C:/Users/Swathi/Downloads/patient-risk-prediction-main/patient-risk-prediction-main/data/PatientRecords.csv')
            print("Data loaded from alternate location!")
        except FileNotFoundError:
            print("Error: Patient data file not found. Please check file path.")
            return

    # Create and train model
    print("Training model...")
    model = PatientRiskModel()
    model.train(df)
    print("Model training complete!")

    while True:
        # Get patient data from user
        patient = get_user_input()
        
        # Make prediction
        prediction = model.predict(patient)
        
        # Display results
        print("\nRESULTS:")
        print(f"Prediction: {prediction}")
        
        # Ask if user wants to continue
        continue_choice = input("\nPredict for another patient? (y/n): ").lower().strip()
        if continue_choice != 'y':
            break
    
    print("Thank you for using the Patient Risk Prediction System!")

if __name__ == "__main__":
    main()