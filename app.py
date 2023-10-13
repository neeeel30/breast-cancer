# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# from sklearn.preprocessing import MinMaxScaler

# # Load the trained model and scaler
# loaded_model = joblib.load(r'C:\Users\NEEL\Desktop\Coding\Anaconda_Data_Science\Projects\Project-4\my_model.pkl')
# scaler = joblib.load(r'C:\Users\NEEL\Desktop\Coding\Anaconda_Data_Science\Projects\Project-4\scaler.pkl')

# # Page title and description
# st.title("Cancer Classification App")
# st.write("Predict whether a tumor is benign or malignant using a trained neural network model.")

# # Sidebar
# st.sidebar.header("User Input")
# features = []

# # Create input sliders for each feature
# for i in range(30):
#     feature = st.sidebar.slider(f"Feature {i + 1}", float(0), float(1), float(0.5))
#     features.append(feature)

# # Predict button
# if st.sidebar.button("Predict"):
#     # Preprocess user input using the scaler
#     scaled_input = scaler.transform([features])
    
#     # Make predictions using the loaded model
#     prediction = loaded_model.predict(scaled_input)
    
#     # Display the prediction
#     st.subheader("Prediction")
#     if prediction[0] == 0:
#         st.write("The tumor is benign (non-cancerous).")
#     else:
#         st.write("The tumor is malignant (cancerous).")

# # Display dataset (optional)
# if st.sidebar.checkbox("Show Dataset"):
#     df = pd.read_csv(r'C:\Users\NEEL\Desktop\Coding\Anaconda_Data_Science\Projects\Project-4\cancer_classification.csv')
#     st.subheader("Dataset")
#     st.write(df)

# # Main content
# st.header("Model Training and Evaluation")
# # Add visualizations and model evaluation results here as needed
# # You can include plots, performance metrics, etc.

# # Footer
# st.sidebar.markdown("### About")
# st.sidebar.text("This app is for educational and demonstration purposes. The model may not be suitable for clinical use.")






# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# from sklearn.preprocessing import MinMaxScaler

# # Load the trained model and scaler
# loaded_model = joblib.load(r'C:\Users\NEEL\Desktop\Coding\Anaconda_Data_Science\Projects\Project-4\my_model.pkl')
# scaler = joblib.load(r'C:\Users\NEEL\Desktop\Coding\Anaconda_Data_Science\Projects\Project-4\scaler.pkl')

# # Page title and description
# st.title("Cancer Classification App")
# st.write("Predict whether a tumor is benign or malignant using a trained neural network model.")

# # Sidebar
# st.sidebar.header("User Input")
# features = {}

# # Create input sliders for each feature
# feature_names = [
#     "mean radius",
#     "mean texture",
#     "mean perimeter",
#     "mean area",
#     "mean smoothness",
#     "mean compactness",
#     "mean concavity",
#     "mean concave points",
#     "mean symmetry",
#     "mean fractal dimension",
#     "radius error",
#     "texture error",
#     "perimeter error",
#     "area error",
#     "smoothness error",
#     "compactness error",
#     "concavity error",
#     "concave points error",
#     "symmetry error",
#     "fractal dimension error",
#     "worst radius",
#     "worst texture",
#     "worst perimeter",
#     "worst area",
#     "worst smoothness",
#     "worst compactness",
#     "worst concavity",
#     "worst concave points",
#     "worst symmetry",
#     "worst fractal dimension"
# ]

# for feature_name in feature_names:
#     feature_value = st.sidebar.slider(f"{feature_name}", float(0), float(1), float(0.5))
#     features[feature_name] = feature_value

# # Predict button
# if st.sidebar.button("Predict"):
#     # Preprocess user input using the scaler
#     user_input = [features[feature_name] for feature_name in feature_names]
#     scaled_input = scaler.transform([user_input])
    
#     # Make predictions using the loaded model
#     prediction = loaded_model.predict(scaled_input)
    
#     # Display the prediction
#     st.subheader("Prediction")
#     if prediction[0] == 0:
#         st.write("The tumor is benign (non-cancerous).")
#     else:
#         st.write("The tumor is malignant (cancerous).")

# # Display dataset (optional)
# if st.sidebar.checkbox("Show Dataset"):
#     df = pd.read_csv(r'C:\Users\NEEL\Desktop\Coding\Anaconda_Data_Science\Projects\Project-4\cancer_classification.csv')
#     st.subheader("Dataset")
#     st.write(df)

# # Main content
# st.header("Model Training and Evaluation")
# # Add visualizations and model evaluation results here as needed
# # You can include plots, performance metrics, etc.

# # Footer
# st.sidebar.markdown("### About")
# st.sidebar.text("This app is for educational and demonstration purposes. The model may not be suitable for clinical use.")


import streamlit as st
import pandas as pd
import pickle


# Load the trained model and scaler
loaded_model = pickle.load(open(r'C:\Users\NEEL\Desktop\Coding\Anaconda_Data_Science\Projects\Project-4 - Copy\model.pkl', 'rb'))
scaler = pickle.load(open(r'C:\Users\NEEL\Desktop\Coding\Anaconda_Data_Science\Projects\Project-4 - Copy\scaler.pkl', 'rb'))
st.markdown(
    """
    <style>
    .main {
        background-image: url(https://images.unsplash.com/photo-1576086213369-97a306d36557?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8Y2FuY2VyfGVufDB8fDB8fHww&w=1000&q=80);  /* Replace with your image URL */
        background-size: cover;
        opacity: 0.1
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Page title and description
st.title("Cancer Classification App")
st.write("Predict whether a tumor is benign or malignant using a trained neural network model.")

# Sidebar
st.sidebar.header("User Input")
features = {}

# Create text input fields for each feature
feature_names = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points",
    "mean symmetry",
    "mean fractal dimension",
    "radius error",
    "texture error",
    "perimeter error",
    "area error",
    "smoothness error",
    "compactness error",
    "concavity error",
    "concave points error",
    "symmetry error",
    "fractal dimension error",
    "worst radius",
    "worst texture",
    "worst perimeter",
    "worst area",
    "worst smoothness",
    "worst compactness",
    "worst concavity",
    "worst concave points",
    "worst symmetry",
    "worst fractal dimension"
]

for feature_name in feature_names:
    feature_value = st.sidebar.text_input(f"{feature_name}", "0.5")  # Default value is 0.5
    features[feature_name] = feature_value

# Predict button
if st.sidebar.button("Predict"):
    # Preprocess user input using the scaler
    user_input = [float(features[feature_name]) for feature_name in feature_names]
    scaled_input = scaler.transform([user_input])
    
    # Make predictions using the loaded model
    prediction = loaded_model.predict(scaled_input)
    
    # Display the prediction
    st.subheader("Prediction")
    if prediction[0] == 0:
        st.write("The tumor is benign (non-cancerous).")
    else:
        st.write("The tumor is malignant (cancerous).")

# Display dataset (optional)
if st.sidebar.checkbox("Show Dataset"):
    df = pd.read_csv(r'C:\Users\NEEL\Desktop\Coding\Anaconda_Data_Science\Projects\Project-4\cancer_classification.csv')
    st.subheader("Dataset")
    st.write(df)

# Main content
st.header("Model Training and Evaluation")
# Add visualizations and model evaluation results here as needed
# You can include plots, performance metrics, etc.

# Footer
st.sidebar.markdown("### About")
st.sidebar.text("This app is for educational and demonstration purposes. The model may not be suitable for clinical use.")
