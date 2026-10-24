import os
import io
import json
from datetime import datetime
import sqlite3
import hashlib
import hmac
import secrets

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    BASE_DIR, "models", "brain_tumor_mobilenetv2.keras"
)
CLASS_PATH = os.path.join(
    BASE_DIR, "models", "mobilenet_class_names.json"
)
IMG_SIZE = 160
DB_PATH = os.path.join(BASE_DIR, "neuroscan_users.db")


def get_db_connection():
    return sqlite3.connect(DB_PATH)


def initialize_database():
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)


def hash_password(password, salt=None):
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), 200000
    ).hex()
    return f"{salt}${digest}"


def verify_password(password, stored_hash):
    try:
        salt, expected = stored_hash.split("$", 1)
        actual = hash_password(password, salt).split("$", 1)[1]
        return hmac.compare_digest(actual, expected)
    except (ValueError, AttributeError):
        return False


def create_user(name, email, password):
    try:
        with get_db_connection() as conn:
            conn.execute(
                "INSERT INTO users (name, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
                (name.strip(), email.strip().lower(), hash_password(password),
                 datetime.now().isoformat(timespec="seconds"))
            )
        return True, "Account created. Please log in."
    except sqlite3.IntegrityError:
        return False, "An account with this email already exists. Please log in."


def authenticate_user(email, password):
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT id, name, email, password_hash FROM users WHERE email = ?",
            (email.strip().lower(),)
        ).fetchone()
    if row and verify_password(password, row[3]):
        return {"id": row[0], "name": row[1], "email": row[2]}
    return None


st.set_page_config(
    page_title="NeuroScan AI | Login and MRI Dashboard",
    page_icon="🧠",
    layout="wide"
)

initialize_database()
if "user" not in st.session_state:
    st.session_state.user = None
if "page" not in st.session_state:
    st.session_state.page = "Home"


# =========================================================
# PAGE DESIGN
# =========================================================

st.markdown("""
<style>
.stApp {
    background: #f3f7fc;
}
h1, h2, h3 {
    color: #073b66;
}
div[data-testid="stMetric"] {
    background: white;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #dce7f2;
}
.notice {
    background: #fff2cc;
    color: #684b00;
    padding: 15px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL AND CLASS NAMES
# =========================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "preprocess_input": preprocess_input,
            "function": preprocess_input,
        },
        compile=False
    )


def load_classes():
    with open(CLASS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        try:
            return [
                data[k] for k in sorted(data, key=lambda x: int(x))
            ]
        except (ValueError, TypeError):
            return list(data.values())

    raise ValueError("Unsupported class-names JSON format.")


# =========================================================
# IMAGE PREPARATION
# =========================================================

def prepare_image(image):
    image = image.convert("RGB")
    resized = image.resize((IMG_SIZE, IMG_SIZE))
    array = np.asarray(resized, dtype=np.float32)
    return np.expand_dims(array, axis=0)


def get_scores(model, image, classes):
    batch = prepare_image(image)

    # The saved model contains its own preprocessing layer.
    # Keep pixel values in the 0-255 range here.
    output = np.asarray(
        model.predict(batch, verbose=0)[0],
        dtype=np.float64
    )

    if len(output) != len(classes):
        raise ValueError(
            "Model output count does not match the class names."
        )

    # Support either probabilities or raw logits.
    if (
        np.all(output >= 0)
        and np.isfinite(output).all()
        and np.isclose(output.sum(), 1.0, atol=0.01)
    ):
        scores = output / output.sum()
    else:
        shifted = output - np.max(output)
        exp_output = np.exp(shifted)
        scores = exp_output / exp_output.sum()

    index = int(np.argmax(scores))
    return classes[index], scores


# =========================================================
# IMAGE FILTERS
# =========================================================

def apply_filter(image, filter_name):
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    if filter_name == "Original":
        return rgb

    if filter_name == "Grayscale":
        return gray

    if filter_name == "Gaussian Blur":
        return cv2.GaussianBlur(gray, (5, 5), 0)

    if filter_name == "Sharpen":
        blurred = cv2.GaussianBlur(gray, (0, 0), 2)
        return cv2.addWeighted(gray, 1.7, blurred, -0.7, 0)

    if filter_name == "Edge Detection":
        return cv2.Canny(gray, 60, 150)

    if filter_name == "Contrast Enhancement":
        return cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        ).apply(gray)

    return rgb


# =========================================================
# GRAD-CAM
# =========================================================

def create_gradcam(model, image, class_index):
    """
    Compute Grad-CAM for the nested MobileNetV2 backbone.
    This is a visualization of model influence, not a tumour mask.
    """

    backbone = model.get_layer("mobilenetv2_1.00_160")

    # Find the last four-dimensional feature layer in the backbone.
    conv_layer = None

    for layer in reversed(backbone.layers):
        try:
            if len(layer.output.shape) == 4:
                conv_layer = layer
                break
        except (AttributeError, TypeError):
            continue

    if conv_layer is None:
        raise ValueError(
            "Could not find a spatial feature layer for Grad-CAM."
        )

    feature_model = tf.keras.Model(
        inputs=backbone.inputs,
        outputs=[conv_layer.output, backbone.output]
    )

    input_tensor = tf.convert_to_tensor(
        prepare_image(image),
        dtype=tf.float32
    )

    # Reproduce the saved model's preprocessing path.
    augmentation = model.get_layer("data_augmentation")
    preprocessing = model.get_layer("mobilenet_preprocessing")

    with tf.GradientTape() as tape:
        augmented = augmentation(input_tensor, training=False)
        processed = preprocessing(augmented)

        feature_maps, backbone_output = feature_model(
            processed, training=False
        )

        x = backbone_output

        # Run the saved classification head after the backbone.
        backbone_position = next(
            i for i, layer in enumerate(model.layers)
            if layer.name == "mobilenetv2_1.00_160"
        )

        for layer in model.layers[backbone_position + 1:]:
            try:
                x = layer(x, training=False)
            except TypeError:
                x = layer(x)

        class_score = x[:, class_index]

    gradients = tape.gradient(class_score, feature_maps)

    if gradients is None:
        raise ValueError(
            "Gradients were unavailable for the selected feature layer."
        )

    weights = tf.reduce_mean(gradients, axis=(1, 2))
    feature_maps = feature_maps[0]
    weights = weights[0]

    heatmap = tf.reduce_sum(
        feature_maps * weights[tf.newaxis, tf.newaxis, :],
        axis=-1
    )

    heatmap = tf.maximum(heatmap, 0)
    max_value = tf.reduce_max(heatmap)

    heatmap = heatmap / (max_value + tf.keras.backend.epsilon())
    heatmap = heatmap.numpy()

    heatmap = cv2.resize(
        heatmap, image.size, interpolation=cv2.INTER_LINEAR
    )

    heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0, 1))
    colour_map = cv2.applyColorMap(
        heatmap_uint8, cv2.COLORMAP_JET
    )
    colour_map = cv2.cvtColor(
        colour_map, cv2.COLOR_BGR2RGB
    )

    original = np.asarray(image.convert("RGB"))

    overlay = cv2.addWeighted(
        original, 0.55, colour_map, 0.45, 0
    )

    return heatmap_uint8, overlay


# =========================================================
# DOWNLOAD HELPERS
# =========================================================

def png_bytes(array):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


# =========================================================
# NAVIGATION AND ACCOUNT PAGES
# =========================================================

with st.sidebar:
    st.title("🧠 NeuroScan AI")
    st.caption("Brain MRI classification — educational demo")
    st.markdown("**Navigation**")
    if st.session_state.user:
        st.success(f"Signed in as {st.session_state.user['name']}")
        page_options = ["Home", "Dashboard", "Profile"]
    else:
        page_options = ["Home", "Register", "Login"]
    for page_option in page_options:
        if st.button(page_option, use_container_width=True,
                             type="primary" if st.session_state.page == page_option else "secondary"):
            st.session_state.page = page_option
            st.rerun()
    if st.session_state.user and st.button("Log out", use_container_width=True):
        st.session_state.user = None
        st.session_state.page = "Home"
        st.rerun()
    st.divider()
    st.caption("Predictions are experimental and are not a medical diagnosis.")

page = st.session_state.page

if page == "Home":
    st.title("🧠 NeuroScan AI")
    st.caption("AI-assisted brain MRI image classification | Student project")
    st.subheader("Welcome to NeuroScan AI")
    st.markdown("### Explore the platform")
    st.write("Register or log in to access the protected dashboard, MRI image analysis, and scanner-style camera demo.")
    st.markdown("---")
    st.subheader("Brain Tumor Image Classification Using MobileNetV2")
    st.markdown("""
    <div class="notice"><b>Educational-use notice:</b> This project demonstrates an AI image-classification workflow. It is not a clinical diagnostic system, and its predictions may be wrong. Do not use it to make medical decisions.</div>
    """, unsafe_allow_html=True)
    st.write("Start by creating an account or logging in. After sign-in, your Dashboard provides access to MRI image analysis and the scanner-style camera demonstration.")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Create account", type="primary", use_container_width=True):
            st.session_state.page = "Register"
            st.rerun()
    with c2:
        if st.button("Log in", use_container_width=True):
            st.session_state.page = "Login"
            st.rerun()
    st.stop()

elif page == "Register":
    st.title("Create your account")
    st.write("Register to access the NeuroScan AI dashboard.")
    with st.form("register_form", clear_on_submit=False):
        name = st.text_input("Full name", max_chars=80)
        email = st.text_input("Email address", max_chars=254)
        password = st.text_input("Password", type="password", help="Use at least 8 characters.")
        confirm_password = st.text_input("Confirm password", type="password")
        submitted = st.form_submit_button("Register", type="primary", use_container_width=True)
    if submitted:
        if not name.strip() or not email.strip() or not password:
            st.error("Please complete all fields.")
        elif "@" not in email or "." not in email.rsplit("@", 1)[-1]:
            st.error("Enter a valid email address.")
        elif len(password) < 8:
            st.error("Password must contain at least 8 characters.")
        elif password != confirm_password:
            st.error("The passwords do not match.")
        else:
            ok, message = create_user(name, email, password)
            (st.success if ok else st.error)(message)
            if ok:
                st.session_state.page = "Login"
                st.rerun()
    if st.button("Already registered? Go to Login"):
        st.session_state.page = "Login"
        st.rerun()
    st.stop()

elif page == "Login":
    st.title("Login to NeuroScan AI")
    with st.form("login_form"):
        email = st.text_input("Email address")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login", type="primary", use_container_width=True)
    if submitted:
        user = authenticate_user(email, password)
        if user:
            st.session_state.user = user
            st.session_state.page = "Dashboard"
            st.success("Login successful.")
            st.rerun()
        else:
            st.error("Email or password is incorrect.")
    if st.button("New here? Create an account"):
        st.session_state.page = "Register"
        st.rerun()
    st.stop()

elif page == "Profile":
    if not st.session_state.user:
        st.session_state.page = "Login"
        st.rerun()
    st.title("My Profile")
    st.write(f"**Name:** {st.session_state.user['name']}")
    st.write(f"**Email:** {st.session_state.user['email']}")
    st.caption("This local demo stores account details in neuroscan_users.db on this computer.")
    st.stop()

elif page == "Dashboard":
    if not st.session_state.user:
        st.warning("Please log in to open the dashboard.")
        st.session_state.page = "Login"
        st.stop()
    st.title(f"Welcome, {st.session_state.user['name']} 👋")
    st.subheader("NeuroScan AI Dashboard")
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Model", "MobileNetV2")
    metric2.metric("Input size", "160 × 160")
    metric3.metric("Classes", "4")
    st.info("Choose MRI Image Analysis to upload an existing brain MRI. Scanner Camera Demo only captures a normal webcam photo and does not send it to the MRI classifier.")
    dash_a, dash_b = st.columns(2)
    with dash_a:
        st.markdown("#### 🧠 MRI Image Analysis")
        st.write("Upload an existing brain MRI image to view the model's four-class prediction and class scores.")
    with dash_b:
        st.markdown("#### 📷 Scanner Camera Demo")
        st.write("Capture an ordinary photo for the interface demonstration. A webcam cannot scan the brain.")

# =========================================================
# IMAGE SOURCE: MRI ANALYSIS OR CAMERA DEMO
# =========================================================

st.header("1. Choose a Mode")

mode = st.radio(
    "Select how you want to use NeuroScan AI",
    ["MRI Image Analysis", "Scanner Camera Demo"],
    horizontal=True,
)

image = None

if mode == "MRI Image Analysis":
    st.subheader("Upload an existing brain MRI image")
    st.info(
        "Only an existing brain MRI image should be submitted to the "
        "brain MRI classifier. Use images that you have permission to use."
    )

    uploaded = st.file_uploader(
        "Choose an MRI image",
        type=["png", "jpg", "jpeg"],
        key="mri_uploader",
    )

    if uploaded is not None:
        try:
            image = Image.open(uploaded).convert("RGB")
        except Exception:
            st.error("Could not read this image file.")

else:
    st.subheader("Scanner-style webcam")
    st.markdown(
        """
        <div class="notice">
        <b>Camera limitation:</b> This webcam captures an ordinary photograph.
        It cannot see inside the head or create an X-ray/MRI image. For safety,
        webcam photos are displayed only and are not sent to the brain MRI model.
        </div>
        """,
        unsafe_allow_html=True,
    )

    captured_photo = st.camera_input(
        "Position the camera and capture a photo",
        key="webcam_demo",
    )

    if captured_photo is not None:
        try:
            camera_image = Image.open(captured_photo).convert("RGB")
            st.image(
                camera_image,
                caption="Captured webcam photo — not a brain MRI",
                use_container_width=True,
            )
            st.success("Photo captured successfully.")
            st.caption(
                "To run the educational classifier, switch to "
                "'MRI Image Analysis' and upload an existing brain MRI image."
            )
        except Exception:
            st.error("Could not read the captured photo.")

# =========================================================
# FILTERS
# =========================================================

if image is not None:
    st.divider()
    st.header("2. MRI Image Filters")

    filter_names = [
        "Original",
        "Grayscale",
        "Gaussian Blur",
        "Sharpen",
        "Edge Detection",
        "Contrast Enhancement"
    ]

    selected_filter = st.selectbox(
        "Choose a filter",
        filter_names
    )

    filtered = apply_filter(image, selected_filter)

    left, right = st.columns(2)

    with left:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with right:
        st.subheader(selected_filter)
        st.image(filtered, use_container_width=True)

    st.download_button(
        "Download Filtered Image",
        data=png_bytes(filtered),
        file_name="neuroscan_filtered.png",
        mime="image/png"
    )

    st.caption(
        "Filters are for visual inspection. Classification below uses "
        "the original image, not the filtered version."
    )

    # =====================================================
    # CLASSIFICATION
    # =====================================================

    st.divider()
    st.header("3. AI Classification")

    if st.button(
        "Run Prediction and Generate Heatmap",
        type="primary",
        use_container_width=True
    ):
        if not os.path.isfile(MODEL_PATH):
            st.error(f"Model file not found: {MODEL_PATH}")

        elif not os.path.isfile(CLASS_PATH):
            st.error(f"Class-names file not found: {CLASS_PATH}")

        else:
            try:
                with st.spinner(
                    "Loading model and analysing image..."
                ):
                    model = load_model()
                    classes = load_classes()

                    predicted_class, scores = get_scores(
                        model, image, classes
                    )

                    class_index = int(np.argmax(scores))

                st.subheader("Predicted Model Class")
                st.info(str(predicted_class))

                st.metric(
                    "Model score for this class",
                    f"{float(scores[class_index]) * 100:.2f}%"
                )

                st.caption(
                    "This is a model score, not clinical certainty "
                    "or a confirmed medical diagnosis."
                )

                st.subheader("Scores for All Classes")

                for idx in np.argsort(scores)[::-1]:
                    st.write(
                        f"**{classes[idx]}:** "
                        f"{float(scores[idx]) * 100:.2f}%"
                    )
                    st.progress(
                        float(np.clip(scores[idx], 0, 1))
                    )

                # Grad-CAM
                st.divider()
                st.header("4. Grad-CAM Heatmap")

                with st.spinner("Generating Grad-CAM visualization..."):
                    heatmap, overlay = create_gradcam(
                        model, image, class_index
                    )

                c1, c2, c3 = st.columns(3)

                with c1:
                    st.subheader("Original")
                    st.image(image, use_container_width=True)

                with c2:
                    st.subheader("Heatmap")
                    st.image(
                        heatmap,
                        caption="Influence intensity",
                        use_container_width=True
                    )

                with c3:
                    st.subheader("Overlay")
                    st.image(
                        overlay,
                        caption="Heatmap over original image",
                        use_container_width=True
                    )

                st.caption(
                    "Warm colours indicate regions with stronger "
                    "relative Grad-CAM response for the selected class. "
                    "This does not establish the exact tumour location "
                    "or boundary."
                )

                st.download_button(
                    "Download Heatmap",
                    data=png_bytes(heatmap),
                    file_name="neuroscan_gradcam_heatmap.png",
                    mime="image/png"
                )

                st.download_button(
                    "Download Heatmap Overlay",
                    data=png_bytes(overlay),
                    file_name="neuroscan_gradcam_overlay.png",
                    mime="image/png"
                )

                # Educational report
                report_lines = [
                    "NeuroScan AI - Educational Model Report",
                    f"Date: {datetime.now().isoformat(timespec='seconds')}",
                    f"Predicted model class: {predicted_class}",
                    "",
                    "Model class scores:"
                ]

                for idx, label in enumerate(classes):
                    report_lines.append(
                        f"{label}: {float(scores[idx]) * 100:.2f}%"
                    )

                report_lines.extend([
                    "",
                    "This is an educational model output, not a clinical",
                    "diagnosis. Grad-CAM is not a verified tumour mask.",
                    "Do not use this report to make medical decisions."
                ])

                st.download_button(
                    "Download Prediction Report",
                    data="\n".join(report_lines),
                    file_name="neuroscan_report.txt",
                    mime="text/plain"
                )

            except Exception as error:
                st.error(
                    "Analysis failed. Please copy the error below "
                    "so the model integration can be corrected."
                )
                st.code(str(error))


# =========================================================
# ABOUT
# =========================================================

st.divider()
st.header("About NeuroScan AI")

st.write("""
This student project demonstrates deep-learning image classification,
basic image processing, and Grad-CAM visualization using MobileNetV2.

A highlighted region is not proof of a tumour. Exact tumour outlining
requires a suitable segmentation model and appropriate validation.
""")

st.caption("NeuroScan AI | Educational Deep Learning Project")
