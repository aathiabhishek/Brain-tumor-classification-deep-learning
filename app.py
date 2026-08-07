# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import json

import streamlit as st
import numpy as np
import tensorflow as tf
import requests
from bs4 import BeautifulSoup

from PIL import Image

from utils.auth import check_login


# ============================================================
# 2. PROJECT PATHS
# ============================================================

# Get the directory where this app.py file is located.
# This makes file paths reliable regardless of where
# Streamlit is launched from.

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# Main project folders
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_images")


# Create folders if they don't already exist.
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


# ============================================================
# 3. CLASS NAMES
# ============================================================

# These class names must match the class folder names
# used during model training.

class_names = [
    "Glioma Tumor",
    "Meningioma Tumor",
    "No Tumor",
    "Pituitary Tumor"
]


# ============================================================
# 4. IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):
    """
    Preprocess an MRI image exactly according to
    the preprocessing used during model training.

    Training preprocessing was:

        target_size = (224, 224)
        rescale = 1 / 255

    Therefore, we DO NOT apply:
        - CLAHE
        - Denoising
        - Center cropping

    because those operations were not used during training.
    """

    # Convert image to RGB
    image = image.convert("RGB")

    # Resize exactly as flow_from_directory() does
    image = image.resize((224, 224))

    # Convert PIL image to NumPy array
    img_array = np.array(image, dtype=np.float32)

    # Normalize pixel values from 0-255 to 0-1.
    # This matches:
    #
    # ImageDataGenerator(rescale=1.0 / 255)
    #
    img_array = img_array / 255.0

    # Add batch dimension.
    #
    # Before:
    # (224, 224, 3)
    #
    # After:
    # (1, 224, 224, 3)

    img_array = np.expand_dims(img_array, axis=0)

    return image, img_array



# ============================================================
# ONLINE TUMOR INFORMATION
# ============================================================

def get_tumor_information(tumor):
    """
    Retrieve educational information from trusted
    medical websites.

    NCI (National Cancer Institute) is used as the
    primary information source.
    """

    tumor_pages = {

        "Glioma Tumor": {
            "url": "https://www.cancer.gov/types/brain",
            "source": "National Cancer Institute (NCI)"
        },

        "Meningioma Tumor": {
            "url": "https://www.cancer.gov/rare-brain-spine-tumor/tumors/meningioma",
            "source": "National Cancer Institute (NCI)"
        },

        "Pituitary Tumor": {
            "url": "https://www.cancer.gov/types/pituitary",
            "source": "National Cancer Institute (NCI)"
        }
    }

    # No online information needed for this class
    if tumor == "No Tumor":
        return {
            "title": "No Tumor Detected",
            "description": (
                "The selected classification indicates that "
                "the uploaded image was classified as No Tumor "
                "by the selected AI model."
            ),
            "source": None,
            "url": None
        }

    if tumor not in tumor_pages:
        return None

    page = tumor_pages[tumor]

    try:

        # Request the medical webpage
        response = requests.get(
            page["url"],
            timeout=10,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        response.raise_for_status()

        # Parse webpage
        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        # Remove unnecessary HTML elements
        for element in soup(
            ["script", "style", "nav", "footer"]
        ):
            element.decompose()

        # Get page text
        paragraphs = soup.find_all("p")

        text = []

        for paragraph in paragraphs:

            content = paragraph.get_text(
                " ",
                strip=True
            )

            if content:
                text.append(content)

        # Keep a reasonable amount of information
        description = " ".join(text[:15])

        return {
            "title": tumor,
            "description": description,
            "source": page["source"],
            "url": page["url"]
        }

    except requests.RequestException as e:

        return {
            "title": tumor,
            "description": (
                "Unable to retrieve the latest information "
                "from the medical source. Please use the "
                "official source link below."
            ),
            "source": page["source"],
            "url": page["url"]
        }

# ============================================================
# 5. MODEL LOADING
# ============================================================

@st.cache_resource
def load_selected_model(model_name):
    """
    Load the selected trained TensorFlow/Keras model.

    All models are stored inside the models/ folder.
    """

    model_paths = {

        # Custom CNN
        "CNN": os.path.join(
            MODELS_DIR,
            "brain_tumor_model_cnn.h5"
        ),

        # Tuned CNN
        #
        # IMPORTANT:
        # Training code saves this as:
        # brain_tumor_model_tuned_cnn.h5
        #
        "CNN (Hypertuned)": os.path.join(
            MODELS_DIR,
            "brain_tumor_model_tuned.h5"
        ),

        # VGG16
        "VGG16": os.path.join(
            MODELS_DIR,
            "brain_tumor_model_vgg16.h5"
        ),

        # Autoencoder classifier
        "Autoencoder": os.path.join(
            MODELS_DIR,
            "brain_tumor_model_autoencoder.h5"
        ),

        # MobileNetV2
        "MobileNetv2": os.path.join(
            MODELS_DIR,
            "brain_tumor_model_mobilenetv2.h5"
        ),

        # ResNet50
        "ResNet50": os.path.join(
            MODELS_DIR,
            "brain_tumor_model_resnet50.h5"
        )
    }


    # Get path for selected model
    model_path = model_paths[model_name]


    # Check if model exists
    if not os.path.exists(model_path):

        raise FileNotFoundError(
            f"""
Model file not found:

{model_path}

Please make sure the trained model exists
inside the models/ folder.
"""
        )


    # Load trained Keras model
    model = tf.keras.models.load_model(model_path)

    return model


# ============================================================
# 6. LOAD FOOD DATA
# ============================================================

@st.cache_data
def load_food_data():
    """
    Load tumor information and food recommendations
    from data/food_data.json.
    """

    food_data_path = os.path.join(
        DATA_DIR,
        "food_data.json"
    )

    if not os.path.exists(food_data_path):

        raise FileNotFoundError(
            f"Food data file not found: {food_data_path}"
        )


    with open(
        food_data_path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


food_data = load_food_data()


# ============================================================
# 7. DEFAULT MODEL
# ============================================================

# Load VGG16 as the default model when the application
# starts for the first time.

if "current_model" not in st.session_state:

    st.session_state.current_model = "VGG16"

    st.session_state.model = load_selected_model(
        "VGG16"
    )


# ============================================================
# 8. UI STYLING
# ============================================================

st.markdown(
    """
    <style>

    [data-testid="stSidebar"] {
        width: 300px !important;
    }

    [data-testid="stMainContent"] {
        margin-left: 310px !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 9. AUTHENTICATION
# ============================================================

def login():
    """
    Display login form and authenticate
    using utils.auth.check_login().
    """

    st.markdown("## 🔐 Secure Login")

    with st.form("login_form"):

        username = st.text_input(
            "👤 Username"
        )

        password = st.text_input(
            "🔒 Password",
            type="password"
        )

        submitted = st.form_submit_button(
            "Login"
        )


        if submitted:

            if check_login(
                username,
                password
            ):

                st.session_state.logged_in = True

                st.session_state.username = username

                st.success(
                    "✅ Logged in successfully!"
                )

                st.rerun()

            else:

                st.error(
                    "❌ Invalid credentials"
                )


# Initialize login state
if "logged_in" not in st.session_state:

    st.session_state.logged_in = False


# Show login page if user is not logged in
if not st.session_state.logged_in:

    login()

    st.stop()


# ============================================================
# 10. SIDEBAR
# ============================================================

st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/2922/2922510.png",
    width=100
)

st.sidebar.markdown(
    f"**👋 Welcome, "
    f"{st.session_state.username.capitalize()}**"
)


page = st.sidebar.radio(
    "📂 Menu",
    [
        "Home",
        "Upload MRI",
        "Tumor Info",
        "Diet Plan",
        "Logout"
    ]
)


# ============================================================
# 11. LOGOUT
# ============================================================

if page == "Logout":

    st.session_state.logged_in = False

    # Remove previous prediction
    st.session_state.pop(
        "last_pred",
        None
    )

    st.rerun()











# ============================================================
# HOME PAGE
# ============================================================

if page == "Home":

    # --------------------------------------------------------
    # PAGE TITLE
    # --------------------------------------------------------

    st.title("🧠 NeuroScan Diagnostics")

    st.subheader(
        "AI-Powered Brain MRI Analysis Platform"
    )

    st.write(
        """
        An educational deep-learning platform designed to
        analyze brain MRI images and classify them into
        different tumor categories.
        """
    )

    st.write(
        """
        Upload an MRI scan, select a trained AI model,
        review the prediction, explore tumor information,
        and access general nutrition resources.
        """
    )

    st.divider()


    # ========================================================
    # WHAT NEUROSCAN PROVIDES
    # ========================================================

    st.header("🔬 What NeuroScan Provides")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.subheader("🔬 AI MRI Analysis")

        st.write(
            """
            Upload a brain MRI image and use trained
            deep learning models to classify the scan
            into one of four categories.
            """
        )

        st.info(
            "Upload → Analyze → Get Prediction"
        )


    with col2:

        st.subheader("🧬 Tumor Information")

        st.write(
            """
            Explore educational information about
            glioma, meningioma, and pituitary tumors,
            including symptoms, diagnosis, and
            treatment information.
            """
        )

        st.info(
            "Learn about tumor types and symptoms"
        )


    with col3:

        st.subheader("🍎 Nutrition Resources")

        st.write(
            """
            Explore general food and nutrition
            information associated with the selected
            classification.
            """
        )

        st.info(
            "Explore general nutrition information"
        )


    st.divider()


    # ========================================================
    # HOW IT WORKS
    # ========================================================

    st.header("⚙️ How It Works")

    step1, step2, step3 = st.columns(3)

    with step1:

        st.subheader("1️⃣ Upload")

        st.write(
            """
            Upload a JPG, JPEG, or PNG brain MRI image
            using the Upload MRI section.
            """
        )


    with step2:

        st.subheader("2️⃣ Analyze")

        st.write(
            """
            Select one of the available deep-learning
            models and process the MRI image.
            """
        )


    with step3:

        st.subheader("3️⃣ Review")

        st.write(
            """
            Review the predicted category, confidence
            score, class probabilities, and downloadable
            report.
            """
        )


    st.divider()


    # ========================================================
    # AVAILABLE AI MODELS
    # ========================================================

    st.header("🧠 Available AI Models")

    model_col1, model_col2 = st.columns(2)


    with model_col1:

        st.subheader("🧠 Custom CNN")

        st.write(
            """
            A convolutional neural network designed
            for brain MRI image classification.
            """
        )

        st.subheader("⚙️ Tuned CNN")

        st.write(
            """
            An optimized CNN architecture developed
            through hyperparameter experimentation.
            """
        )

        st.subheader("🔷 VGG16")

        st.write(
            """
            ImageNet-pretrained convolutional architecture
            used through transfer learning.
            """
        )


    with model_col2:

        st.subheader("📦 Autoencoder Classifier")

        st.write(
            """
            Uses learned feature representations to
            support multi-class image classification.
            """
        )

        st.subheader("⚡ MobileNetV2")

        st.write(
            """
            A lightweight architecture designed for
            efficient deep-learning inference.
            """
        )

        st.subheader("🔗 ResNet50")

        st.write(
            """
            A deep residual network using skip connections
            to improve feature learning.
            """
        )


    st.divider()


    # ========================================================
    # CLASSIFICATION CATEGORIES
    # ========================================================

    st.header("🔎 Classification Categories")

    category1, category2, category3, category4 = st.columns(4)

    with category1:

        st.metric(
            label="🧬 Glioma",
            value="Classification"
        )


    with category2:

        st.metric(
            label="🧬 Meningioma",
            value="Classification"
        )


    with category3:

        st.metric(
            label="🧬 Pituitary",
            value="Classification"
        )


    with category4:

        st.metric(
            label="✅ No Tumor",
            value="Classification"
        )


    st.divider()


    # ========================================================
    # QUICK START
    # ========================================================

    st.header("🚀 Get Started")

    st.write(
        """
        Ready to analyze an MRI image?
        """
    )

    st.info(
        """
        Go to **Upload MRI** from the sidebar,
        upload your MRI image, select a model,
        and review the classification results.
        """
    )


    # ========================================================
    # MEDICAL DISCLAIMER
    # ========================================================

    st.divider()

    st.warning(
        """
        ⚠️ **Important Medical Notice**

        NeuroScan Diagnostics is an educational and
        research-oriented AI application.

        Its predictions should NOT be considered a
        medical diagnosis.

        MRI results should always be reviewed by a
        qualified healthcare professional or radiologist.
        """
    )


    # ========================================================
    # FOOTER
    # ========================================================

    st.divider()

    st.subheader("🏥 NeuroScan Diagnostics")

    st.write(
        """
        AI-assisted brain MRI analysis and educational
        healthcare information.
        """
    )

    st.caption(
        "📍 Toronto, Ontario, Canada"
    )

    st.caption(
        "📧 support@neuroscan.ai"
    )

    st.caption(
        "This application is intended for educational "
        "and demonstration purposes."
    )





# ============================================================
# 13. UPLOAD MRI PAGE
# ============================================================

elif page == "Upload MRI":

    st.title("📤 Upload MRI Scan")


    # --------------------------------------------------------
    # MODEL SELECTION
    # --------------------------------------------------------

    new_model = st.selectbox(
        "Select Model Architecture",
        [
            "CNN",
            "CNN (Hypertuned)",
            "VGG16",
            "Autoencoder",
            "MobileNetv2",
            "ResNet50"
        ],
        index=2,
        key="model_selector"
    )


    # --------------------------------------------------------
    # LOAD SELECTED MODEL
    # --------------------------------------------------------

    if new_model != st.session_state.current_model:

        with st.spinner(
            f"Loading {new_model} model..."
        ):

            st.session_state.model = (
                load_selected_model(new_model)
            )

            st.session_state.current_model = new_model


        st.success(
            f"✅ Switched to {new_model} model"
        )


    st.info(
        f"ℹ️ Currently using: "
        f"{st.session_state.current_model}"
    )


    # --------------------------------------------------------
    # FILE UPLOAD
    # --------------------------------------------------------

    uploaded_file = st.file_uploader(
        "Upload brain MRI image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )


    if uploaded_file:

        with st.spinner(
            "Processing MRI image..."
        ):


            # ------------------------------------------------
            # STEP 1: READ IMAGE
            # ------------------------------------------------

            original_image = Image.open(
                uploaded_file
            ).convert("RGB")


            # ------------------------------------------------
            # STEP 2: SAVE ORIGINAL IMAGE
            # ------------------------------------------------

            # Generate a safe filename
            original_filename = os.path.basename(
                uploaded_file.name
            )


            original_path = os.path.join(
                UPLOADS_DIR,
                original_filename
            )


            # Save original uploaded MRI
            original_image.save(
                original_path
            )


            # ------------------------------------------------
            # STEP 3: DISPLAY ORIGINAL IMAGE
            # ------------------------------------------------

            col1, col2 = st.columns(2)


            with col1:

                st.image(
                    original_image,
                    caption="Original MRI",
                    use_container_width=True
                )


            # ------------------------------------------------
            # STEP 4: PREPROCESS IMAGE
            # ------------------------------------------------

            # IMPORTANT:
            #
            # Training used:
            #
            # target_size=(224,224)
            # rescale=1/255
            #
            # Therefore we use exactly the same
            # preprocessing here.

            processed_image, img_array = (
                preprocess_image(
                    original_image
                )
            )


            # ------------------------------------------------
            # STEP 5: SAVE PROCESSED IMAGE
            # ------------------------------------------------

            processed_filename = (
                "processed_"
                + os.path.splitext(
                    original_filename
                )[0]
                + ".png"
            )


            processed_path = os.path.join(
                PROCESSED_DIR,
                processed_filename
            )


            processed_image.save(
                processed_path
            )


            # ------------------------------------------------
            # STEP 6: DISPLAY PROCESSED IMAGE
            # ------------------------------------------------

            with col2:

                st.image(
                    processed_image,
                    caption="Model Input (224 × 224)",
                    use_container_width=True
                )


            # ------------------------------------------------
            # STEP 7: DISPLAY IMAGE SHAPE
            # ------------------------------------------------

            st.caption(
                f"Model input shape: {img_array.shape}"
            )


            # ------------------------------------------------
            # STEP 8: MODEL PREDICTION
            # ------------------------------------------------

            prediction = (
                st.session_state.model.predict(
                    img_array,
                    verbose=0
                )
            )


            # ------------------------------------------------
            # STEP 9: FIND PREDICTED CLASS
            # ------------------------------------------------

            class_index = int(
                np.argmax(prediction[0])
            )


            # Check that the output contains
            # exactly four classes.

            if len(prediction[0]) != len(class_names):

                st.error(
                    "Model output does not match "
                    "the expected four classes."
                )

                st.stop()


            predicted_tumor = (
                class_names[class_index]
            )


            # ------------------------------------------------
            # STEP 10: CONFIDENCE
            # ------------------------------------------------

            confidence = (
                float(
                    np.max(
                        prediction[0]
                    )
                ) * 100
            )


            # ------------------------------------------------
            # STEP 11: DISPLAY RESULT
            # ------------------------------------------------

            st.success(
                f"**Prediction:** {predicted_tumor}"
            )


            st.info(
                f"**Confidence:** "
                f"{confidence:.2f}%"
            )


            # ------------------------------------------------
            # STEP 12: DISPLAY ALL CLASS PROBABILITIES
            # ------------------------------------------------

            st.subheader(
                "📊 Class Probabilities"
            )


            probabilities = prediction[0] * 100


            for i, class_name in enumerate(
                class_names
            ):

                st.write(
                    f"**{class_name}: "
                    f"{probabilities[i]:.2f}%**"
                )


            # ------------------------------------------------
            # STEP 13: LOW CONFIDENCE WARNING
            # ------------------------------------------------

            if confidence < 60:

                st.warning(
                    "⚠️ Low model confidence. "
                    "The prediction should not be "
                    "considered reliable."
                )


            # ------------------------------------------------
            # STEP 14: SAVE PREDICTION
            # ------------------------------------------------

            st.session_state.last_pred = (
                predicted_tumor
            )


            # ------------------------------------------------
            # STEP 15: DOWNLOAD REPORT
            # ------------------------------------------------

            report = f"""
NeuroScan Diagnostics Report
============================

Prediction:
{predicted_tumor}

Confidence:
{confidence:.2f}%

Model Used:
{st.session_state.current_model}

Input Image:
{original_filename}

Image Size:
224 x 224

Note:
This AI output is for educational/research
purposes and should not replace professional
medical diagnosis.
"""


            st.download_button(
                "📥 Download Report",
                report,
                file_name="neuroscan_report.txt",
                mime="text/plain"
            )


# ============================================================
# TUMOR INFORMATION PAGE
# ============================================================

elif page == "Tumor Info":

    st.title("🧬 Brain Tumor Information")

    st.markdown(
        """
        Explore educational information about different
        brain tumor classifications, including symptoms,
        possible effects, diagnosis, and treatment.
        """
    )

    # --------------------------------------------------------
    # Select tumor
    # --------------------------------------------------------

    tumor = st.selectbox(
        "Select Tumor Type",
        class_names
    )

    st.divider()

    # --------------------------------------------------------
    # No Tumor
    # --------------------------------------------------------

    if tumor == "No Tumor":

        st.success("✅ No Tumor Classification")

        st.write(
            """
            The AI model classified the uploaded MRI image
            as **No Tumor**.

            This classification does not replace a medical
            examination or professional radiological review.
            """

        )

        st.warning(
            """
            If you have symptoms or medical concerns,
            consult a qualified healthcare professional
            regardless of an AI model's prediction.
            """
        )

    # --------------------------------------------------------
    # Tumor information
    # --------------------------------------------------------

    else:

        with st.spinner(
            "🌐 Retrieving medical information..."
        ):

            tumor_info = get_tumor_information(
                tumor
            )

        if tumor_info:

            # ------------------------------------------------
            # About
            # ------------------------------------------------

            st.header(
                f"🧬 About {tumor}"
            )

            st.write(
                tumor_info["description"]
            )

            st.divider()

            # ------------------------------------------------
            # Possible Symptoms
            # ------------------------------------------------

            st.header(
                "⚠️ Common Symptoms & Possible Effects"
            )

            st.markdown(
                """
                Symptoms and effects can vary significantly
                depending on the tumor's type, size, grade,
                and location.

                Possible symptoms associated with brain tumors
                may include:

                • Headaches  
                • Seizures  
                • Vision changes  
                • Speech or communication difficulties  
                • Memory or concentration changes  
                • Problems with balance or coordination  
                • Nausea or vomiting  
                • Changes in mood or behavior  
                • Weakness or numbness  

                Not every person with a brain tumor will
                experience these symptoms.
                """
            )

            st.divider()

            # ------------------------------------------------
            # Diagnosis
            # ------------------------------------------------

            st.header(
                "🔬 Diagnosis"
            )

            st.write(
                """
                Brain tumors may be evaluated using medical
                history, neurological examinations, imaging
                such as MRI, and other tests. The exact
                diagnostic process depends on the suspected
                tumor type and the individual patient.
                """
            )

            st.divider()

            # ------------------------------------------------
            # Treatment
            # ------------------------------------------------

            st.header(
                "🩺 Treatment Overview"
            )

            st.write(
                """
                Treatment depends on the tumor type, grade,
                size, location, growth characteristics, and
                the patient's overall health.

                Depending on the situation, treatment may
                involve observation, surgery, radiation
                therapy, medication, chemotherapy, or other
                specialized treatments.

                Treatment decisions should always be made
                with an appropriate medical specialist.
                """
            )

            st.divider()

            # ------------------------------------------------
            # Educational Video
            # ------------------------------------------------

            st.header(
                "🎥 Educational Resources"
            )

            video_links = {

    "Glioma Tumor":
        "https://www.youtube.com/results?search_query=glioma+brain+tumor+National+Cancer+Institute",

    "Meningioma Tumor":
        "https://www.youtube.com/results?search_query=meningioma+brain+tumor+National+Cancer+Institute",

    "Pituitary Tumor":
        "https://www.youtube.com/results?search_query=pituitary+tumor+National+Cancer+Institute"
}

                
            if tumor in video_links:

              st.info(
                "Watch educational videos from trusted medical "
                "and healthcare organizations."
              )

              st.link_button(
             "🎥 Watch Educational Videos on YouTube",
             video_links[tumor]
            )

            st.divider()

            # ------------------------------------------------
            # Medical Source
            # ------------------------------------------------

            st.header(
                "📚 Trusted Medical Source"
            )

            st.markdown(
                f"""
                **Source:** {tumor_info["source"]}
                """
            )

            st.link_button(
                "🔗 Read Full Information",
                tumor_info["url"]
            )

            st.divider()

            # ------------------------------------------------
            # Disclaimer
            # ------------------------------------------------

            st.warning(
                """
                ⚕️ **Medical Disclaimer**

                This page provides educational information
                and is not intended to diagnose, treat, or
                replace advice from a qualified healthcare
                professional.

                The AI prediction from this application
                should not be considered a medical diagnosis.
                """
            )

        else:

            st.error(
                "Unable to retrieve information for "
                "this tumor type."
            )

# ============================================================
# 15. DIET PLAN PAGE
# ============================================================

elif page == "Diet Plan":

    st.title(
        "🍽️ Personalized Diet Plan"
    )


    # Get latest prediction
    tumor = st.session_state.get(
        "last_pred",
        "No Tumor"
    )


    st.subheader(
        f"Recommendations for: {tumor}"
    )


    if tumor == "No Tumor":

        st.info(
            "Maintain a balanced diet "
            "for overall health."
        )


    else:

        if tumor not in food_data:

            st.warning(
                "Diet information is not available "
                "for this prediction."
            )

        else:

            foods = food_data[tumor]["foods"]


            for food in foods:

                with st.expander(
                    f"🍎 {food['name']}"
                ):

                    st.markdown(
                        f"""
                        **Nutrients:** {food['nutrients']}

                        **Calories:** {food['calories']} kcal

                        **Benefits:** {
                            food.get(
                                'benefits',
                                'General brain health'
                            )
                        }
                        """
                    )


            st.success(
                "Consult a qualified healthcare "
                "professional or nutritionist for "
                "personalized dietary advice."
            )
            
            


