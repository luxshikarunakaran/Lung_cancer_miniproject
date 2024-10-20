import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from sqlite3 import Error
import pyttsx3
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential, load_model
from sklearn.svm import SVC
import plotly.express as px
import tensorflow as tf
from gtts import gTTS
import io
import pickle
import streamlit as st
from PIL import Image
import numpy as np
from keras.models import load_model
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
from gtts import gTTS
import tempfile
import os


#-------------------------------------------------------------------------------------------------------
# Load the dataset
@st.cache_data
def load_data():
    encodings = ['utf-8', 'latin-1', 'iso-8859-1']
    for encoding in encodings:
        try:
            data = pd.read_csv('survey_lung_cancer.csv', encoding=encoding)
            return data
        except UnicodeDecodeError:
            continue
    st.error("Could not decode file. Please check the file encoding.")
    return None

data = load_data()

#-------------------------------------------------------------------------------------------------------

# # # Load pre-trained models and scaler
# model_detection = tf.keras.models.load_model('best_model_effinetb0.h5')
# scaler = pickle.load(open('scaler.pkl', 'rb'))

# Load data and models
#data = load_data()
best_model = pickle.load(open('best_model_k_fold.pkl', 'rb'))
#best_model = pickle.load(open('numerical_best_model.pkl', 'rb'))
#scaler = pickle.load(open('scaler.pkl', 'rb'))


# If data loaded successfully, preprocess and split data
if data is not None:
    # Initialize Encoder and Scaler
    label_encoder = LabelEncoder()
    scaler = StandardScaler()

    # Label Encoding for categorical columns
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        data[col] = label_encoder.fit_transform(data[col])

    # Fill missing values
    data.fillna(method='ffill', inplace=True)

    # Split data into features and target variable
    X = data.drop('LUNG_CANCER', axis=1)
    y = data['LUNG_CANCER']

    # Scale the features
    X_scaled = scaler.fit_transform(X)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)




#------------------------------------------------------------------------------------------------------------

# Function to compute sensitivity and specificity
# def compute_sensitivity_specificity(conf_matrix):
#     TN, FP, FN, TP = conf_matrix.ravel()
#     sensitivity = TP / (TP + FN)
#     specificity = TN / (TN + FP)
#     return sensitivity, specificity


# Function to preprocess the data
def preprocess_data(df):
    # Handle missing values
    imputer = SimpleImputer(strategy='most_frequent')
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # Convert categorical columns to numeric
    categorical_columns = df.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        df[column] = LabelEncoder().fit_transform(df[column])

    # Ensure all columns are numeric
    df = df.apply(pd.to_numeric)


    return df

#-----------------------------------------------------------------------------------------------------------
# Define the login function
# def login():
#     # Display the login button
#     if st.button("Login to the Page"):
#         st.session_state['logged_in'] = True  # Set logged-in state
#         return True  # Indicates successful login
#     return False  # Indicates not logged in

def login():
    # Check if the user is already logged in
    if 'logged_in' in st.session_state and st.session_state['logged_in']:
        st.success("You are already logged in.")
        return True

    # Input fields for username and password
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    # Display the login button
    if st.button("Login to the Page"):
        # Check if both fields are filled (no need for strict validation)
        if username and password:
            st.session_state['logged_in'] = True  # Set logged-in state
            st.success("Login successful!")
            return True  # Indicates successful login
        else:
            st.error("Please enter both a username and password!")
            return False  # Login failed

    return False  # Not logged in yet

def logout():
    # Display the logout button
    if st.button("Logout"):
        st.session_state['logged_in'] = False  # Set logged-in state to False
        st.success("Successfully logged out!")  # Display success message
     #   st.experimental_rerun()  # Refresh the app to show login screen
        return True
    return False

def main():
    # Initialize session state if not present
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False  # User starts as logged out

    st.title("Welcome to the Lung Cancer Prediction App")

    # Check if the user is logged in
    if st.session_state['logged_in']:
        # Display the sidebar for navigation
        st.sidebar.title("Lung Cancer Prediction")
        menu1 = st.sidebar.selectbox("Menu",
                                     ["Home", "About", "Dataset", "Prediction", "Image Segmentation", "Prevention", "Logout"])

        # Navigate to different pages based on the selection
        if menu1 == "Home":
            home_page()  # Replace with your actual function
        elif menu1 == "About":
            about_page()  # Replace with your actual function
        elif menu1 == "Dataset":
            dataset_page()  # Replace with your actual function
        elif menu1 == "Prediction":
            lung_cancer_prediction_page()  # Replace with your actual function
        elif menu1 == "Image Segmentation":
            segmentation_page()  # Replace with your actual function
        elif menu1 == "Prevention":
            tips_page()  # Replace with your actual function
        elif menu1 == "Logout":
            logout()

    else:
        # Prompt for login if not logged in
        if login():
            st.success("Successfully logged in!")  # Display success message
        else:
            st.write("Please log in to continue.")  # Prompt for login





#---------------------------------------------------------------------------------------------------------------------

# Define pages
import streamlit as st


def home_page():
    st.write("------------------------------------------------")
    st.title("Home")
    st.write("----------------------------------------------")

    # Define the layout for the images (2 columns, with the left column split into 2 rows)
    col1, col2 = st.columns([2, 1])  # Left column smaller, right column larger

    with col1:
        # First smaller image
        st.image("lung_cancer_image.gif", use_column_width=True)
        # Second smaller image

    with col2:
        # Larger image on the right


        st.image("failure_lung.gif", use_column_width=True)

        st.image("lung_cancer_2.png", use_column_width=True)

    st.write("----------------------------------------------")

    # After displaying images, display the introduction paragraph
    st.write("""
        ## Introduction
        Lung cancer is one of the most common and serious types of cancer. This application predicts the lung cancer status of a patient based on various features using Machine Learning models.

        This app uses several algorithms to predict whether a person is likely to have lung cancer or not based on their symptoms and health data.
    """)
    st.write("----------------------------------------------")

    st.video("lung_cancer_video.mp4", start_time=0, muted=1, autoplay=1)

    st.write("----------------------------------------------")

    st.write(
        "@Lung Cancer: Official Website - American Cancer Society : https://en.wikipedia.org/wiki/American_Cancer_Society")


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def about_page():
    st.write("--------------------------------")
    st.write("# About")
    st.image("lung_cancer_infor.webp")
    st.write("--------------------------------")
    st.write("""
    Lung cancer is the leading cause of cancer deaths worldwide. Smoking is the biggest risk factor, accounting for 85% of cases. Other factors include exposure to radon, asbestos, and other carcinogens, as well as genetic predispositions.

    Early detection is crucial as lung cancer is more treatable when diagnosed in its initial stages. Symptoms may include a persistent cough, shortness of breath, and chest pain, but many cases are detected at advanced stages when symptoms become more pronounced.
    """
    )
    st.write("----------------------------------")
    st.write(
        "More Information: https://www.cancer.org/cancer/lung-cancer.html")

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def dataset_page():
    st.write("--------------------------------")
    st.title("Dataset")
    st.write("---------------------------------")
    st.title("Numerical Dataset")
    data_view = st.selectbox("Choose a view", ["View Data", "Lung Cancer Patients", "Non-Cancer Patients"])

    if data_view == "View Data":
        st.dataframe(data)

    elif data_view == "Lung Cancer Patients":
        cancer_data = data[data['LUNG_CANCER'] == 1]
        st.dataframe(cancer_data)
        st.write(f"Total Lung Cancer Patients: {len(cancer_data)}")

        # Pie chart for Lung Cancer Patients
        cancer_count = len(cancer_data)
        non_cancer_count = len(data) - cancer_count
        labels = ['Lung Cancer', 'No Lung Cancer']
        values = [cancer_count, non_cancer_count]
        fig = px.pie(names=labels, values=values, title='Lung Cancer Patients')
        st.plotly_chart(fig)

    elif data_view == "Non-Cancer Patients":
        non_cancer_data = data[data['LUNG_CANCER'] == 0]
        st.dataframe(non_cancer_data)
        st.write(f"Total Non-Cancer Patients: {len(non_cancer_data)}")

        # Pie chart for Non-Cancer Patients
        non_cancer_count = len(non_cancer_data)
        cancer_count = len(data) - non_cancer_count
        labels = ['No Lung Cancer', 'Lung Cancer']
        values = [non_cancer_count, cancer_count]
        fig = px.pie(names=labels, values=values, title='Non-Cancer Patients')
        st.plotly_chart(fig)

    st.write("-----------------")
    # Dictionary of features and their descriptions
    st.write("Dataset Features")
    st.write("-----------------")
    features = {
        "1. GENDER": "Gender of the patient [M/F]",
        "2. AGE": "Age of the patient",
        "3. SMOKING": "Whether the patient is a smoker [Y/N]",
        "4. YELLOW_FINGERS": "Yellow fingers due to smoking [Y/N]",
        "5. ANXIETY": "Anxiety levels [Y/N]",
        "6. PEER_PRESSURE": "Peer pressure experienced [Y/N]",
        "7. CHRONIC DISEASE": "Presence of chronic diseases [Y/N]",
        "8. FATIGUE": "Fatigue levels [Y/N]",
        "9. ALLERGY": "Presence of allergies [Y/N]",
        "10. WHEEZING": "Wheezing sounds [Y/N]",
        "11. ALCOHOL CONSUMING": "Alcohol consumption [Y/N]",
        "12. COUGHING": "Presence of coughing [Y/N]",
        "13. SHORTNESS OF BREATH": "Shortness of breath [Y/N]",
        "14. SWALLOWING DIFFICULTY": "Difficulty swallowing [Y/N]",
        "15. CHEST PAIN": "Chest pain [Y/N]",
        "16. LUNG_CANCER": "Lung cancer diagnosis [Y/N]"
    }



    # Display features and their descriptions using a loop
    for feature, description in features.items():
        st.write(f"{feature}: {description}")

    st.write("-----------------")


    # Set the image directory
    image_directory = 'Data/test'

    # Image dataset view
    st.title("Lung Cancer Image Data")
    image_type = st.selectbox("Choose Image Type",
                              ["Adenocarcinoma", "Large Cell Carcinoma", "Normal", "Squamous Cell Carcinoma"])

    # Dictionary to match selection to folder name
    image_type_to_folder = {
        "Adenocarcinoma": "adenocarcinoma",
        "Large Cell Carcinoma": "large.cell.carcinoma",
        "Normal": "normal",
        "Squamous Cell Carcinoma": "squamous.cell.carcinoma"
    }

    # Get folder path for the selected image type
    selected_folder = image_type_to_folder[image_type]
    folder_path = os.path.join(image_directory, selected_folder)

    # List all images in the selected folder
    images = os.listdir(folder_path)

    if images:
        # Add a slider to navigate through images
        image_index = st.slider("Select Image", 0, len(images) - 1, 0)

        # Display the selected image
        image_path = os.path.join(folder_path, images[image_index])
        img_color = Image.open(image_path)  # Open the image

        # Ensure the original image is in RGB format
        img_color = img_color.convert("RGB")

        # Convert the image to grayscale
        img_gray = img_color.convert("L")

        # Resize both images to make them smaller
        img_color = img_color.resize((200, 200))
        img_gray = img_gray.resize((200, 200))

        # Convert grayscale to a colormap for visual similarity with the uploaded image
        img_gray_np = np.array(img_gray)  # Convert grayscale PIL image to numpy array
        img_colormap = cv2.applyColorMap(img_gray_np,
                                         cv2.COLORMAP_VIRIDIS)  # Apply colormap similar to your uploaded image

        # Convert back to PIL image for display in Streamlit
        img_colormap_pil = Image.fromarray(img_colormap)

        # Display images side by side using Streamlit columns
        col1, col2 = st.columns(2)

        with col1:
            st.image(img_colormap_pil, caption=f"{image_type}  (Colormap)", use_column_width=True)

        with col2:
            st.image(img_color, caption=f"{image_type}  (graycolor)", use_column_width=True)

    else:
        st.write(f"No images found for {image_type}.")

    st.write("-----------------")
    st.write(
        "Dataset Source: [Lung Cancer numerical Dataset](https://www.kaggle.com/datasets/akashnath29/lung-cancer-dataset)")
    st.write(
        "Dataset Source: [Lung Cancer Image Dataset](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images)")

    st.write("-----------------")

#-------------------------------------------------------------------------------------------------------------------------------

def lung_cancer_prediction_page():
    st.write("--------------------------------")
    st.title("Prediction")
    st.write("Enter the patient details to predict the lung cancer status.")

    # Collect user input for all features
    input_features = {
        "GENDER": st.selectbox("Gender", [1, 0]),
        "AGE": st.number_input("Age", 1, 100),
        "SMOKING": st.selectbox("Smoking", [1, 2]),
        "YELLOW_FINGERS": st.selectbox("Yellow Fingers", [1, 2]),
        "ANXIETY": st.selectbox("Anxiety", [1, 2]),
        "PEER_PRESSURE": st.selectbox("Peer Pressure", [1, 2]),
        "CHRONIC_DISEASE": st.selectbox("Chronic Disease", [1, 2]),
        "FATIGUE": st.selectbox("Fatigue", [1, 2]),
        "ALLERGY": st.selectbox("Allergy", [1, 2]),
        "WHEEZING": st.selectbox("Wheezing", [1, 2]),
        "ALCOHOL_CONSUMING": st.selectbox("Alcohol Consuming", [1, 2]),
        "COUGHING": st.selectbox("Coughing", [1, 2]),
        "SHORTNESS_OF_BREATH": st.selectbox("Shortness of Breath", [1, 2]),
        "SWALLOWING_DIFFICULTY": st.selectbox("Swallowing Difficulty", [1, 2]),
        "CHEST_PAIN": st.selectbox("Chest Pain", [1, 2])
    }

    input_df = pd.DataFrame([input_features])

    # Preprocess the input data to match the training data
    X = data.drop("LUNG_CANCER", axis=1)
    y = data["LUNG_CANCER"]

    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    input_scaled = scaler.transform(input_df)

    # Train the SVC model with probability=True
    model = SVC(probability=True)
    model.fit(X_train, y_train)

    # Make predictions
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0]

    if st.button("Predict"):
        if prediction == 1:
            st.success("The patient is likely to have lung cancer.")
            #st.image("heart.gif")
            st.image("broken-heart.gif")
        else:
            st.error("The patient is not likely to have lung cancer.")
            st.balloons()



        # Voice Output
        engine = pyttsx3.init()
        result = "likely to have lung cancer" if prediction == 1 else "not likely to have lung cancer"
        engine.say(f"The patient is predicted to be {result}")
        engine.runAndWait()

        # Display prediction probabilities
        st.write(f"Prediction Probability: Lung Cancer = {prediction_proba[1]:.2f}, No Lung Cancer = {prediction_proba[0]:.2f}")

        # Display model performance on test data
        y_pred_test = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred_test)
        st.write(f"Model Accuracy on Test Data: {accuracy:.2%}")

        # # Confusion matrix
        # conf_matrix = confusion_matrix(y_test, y_pred_test)
        # st.write("Confusion Matrix:")
        # st.write(conf_matrix)


#----------------------------------------------------------------------------------------------------------------

# Load your pre-trained EfficientNetB0 model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('best_model_effinetb0.h5')
    return model

# Function to preprocess image for model input
def preprocess_image(image):
    image = np.array(image)
    resized_image = cv2.resize(image, (350, 350))  # Resize to a smaller size for display
    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)  # Ensure the image has 3 channels (RGB)
    final_image = rgb_image / 255.0  # Normalize pixel values to [0, 1]
    return np.expand_dims(final_image, axis=0)  # Add batch dimension

# Function for filtering (Smoothing or sharpening)
def filter_image(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

# Edge Detection function using Canny Edge Detection
def edge_detection(image):
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)  # Convert image to uint8 if it's not already
    return cv2.Canny(image, 100, 200)

# Simple segmentation (Thresholding)
def segment_image(image):
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)  # Ensure uint8 type before thresholding
    _, segmented_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    return segmented_image

# Streamlit app structure for Image Segmentation Page
def segmentation_page():
    st.title("Lung Cancer Image Segmentation and Detection")

    # Initialize session state
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None
        st.session_state.preprocessed_image = None
        st.session_state.filtered_image = None
        st.session_state.edge_image = None
        st.session_state.segmented_image = None
        st.session_state.prediction = None


    # Upload new image if no image has been uploaded or after reset
    uploaded_file = st.file_uploader("Upload a lung CT scan image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file

    # Process the uploaded image if it exists in session state
    if st.session_state.uploaded_file is not None:
        input_image = Image.open(st.session_state.uploaded_file)

        # Display input image
        st.image(input_image, caption="Input Image", use_column_width=True)

        # Preprocessing step
        st.subheader("Processing Steps:")

        # Create columns to display images side by side
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            # Preprocessing (Resizing and RGB Conversion)
            preprocessed_image = preprocess_image(input_image)
            st.image(preprocessed_image[0], caption="Preprocessed", use_column_width=True)

        with col2:
            # Filtering step (Gaussian Blur)
            filtered_image = filter_image(preprocessed_image[0])
            st.image(filtered_image, caption="Filtered", use_column_width=True, clamp=True)

        with col3:
            # Edge detection step
            edge_image = edge_detection(filtered_image)
            st.image(edge_image, caption="Edge Detection", use_column_width=True, clamp=True)

        with col4:
            # Segmentation step (Binary threshold)
            segmented_image = segment_image(filtered_image)
            st.image(segmented_image, caption="Segmentation", use_column_width=True, clamp=True)

        # Load the model
        model = load_model()

        # Prediction based on the preprocessed image
        prediction = model.predict(preprocessed_image)
        class_idx = np.argmax(prediction, axis=1)

        # Map the class index to cancer type
        cancer_types = ['Adenocarcinoma', 'Large Cell Carcinoma', 'Normal', 'Squamous Cell Carcinoma']
        predicted_cancer = cancer_types[class_idx[0]]

        # Display prediction result
        st.subheader(f"Detection Result: {predicted_cancer}")

        # Handle reset button
        if st.button("Reset"):
            st.session_state.uploaded_file = None
            st.session_state.preprocessed_image = None
            st.session_state.filtered_image = None
            st.session_state.edge_image = None
            st.session_state.segmented_image = None
            st.session_state.prediction = None
            st.success("All results cleared!")


# #-------------------------------------------------------------------------------------------------------------------

def tips_page():
    st.write("--------------------------------")
    st.write("""
        ## Tips for Preventing Lung Cancer
        ----------------------------------------------
        1. Avoid smoking and exposure to secondhand smoke.
        2. Test your home for radon and reduce exposure.
        3. Avoid exposure to asbestos and other carcinogens.
        4. Maintain a healthy diet and regular exercise.
        5. Get regular check-ups and report any symptoms to your doctor.
    """)
    st.write("---------------------")
    st.image("lung_cancer.jpg", width=500)
    st.write("---------------------")
    st.write("@Lung cancer : https://www.cdc.gov/lung-cancer/prevention/index.html")


# Run the main app
if __name__ == '__main__':
    main()

