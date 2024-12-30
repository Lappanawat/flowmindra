import streamlit as st
import librosa
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter
import pandas as pd
import os
import io

# Get the directory where this script is running
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the model file
MODEL_PATH = os.path.join(BASE_DIR, "feature_list.pkl")        # Currently loads the feature list (list of column names)
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")             # Loads your StandardScaler object
FEATURE_LIST_PATH = os.path.join(BASE_DIR, "trained_urineflow_model.pkl")  # Loads your actual RandomForestRegressor model

# Load your model and supporting files
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_list = joblib.load(FEATURE_LIST_PATH)

def extract_audio_features_with_rms(y, sr, smoothing_window=201, downsample_factor=20):
    """
    Extract smoothed RMS features (downsampled) in memory.
    Returns:
        rms_downsampled: 1D NumPy array of RMS values
        time_series: 1D NumPy array of time indices corresponding to the RMS
        duration: float, duration of the audio
    """
    try:
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Compute RMS (frame_length=2048, hop_length=512).
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]

        # Apply Savitzky-Golay filter for smoothing
        rms_smoothed = savgol_filter(rms, smoothing_window, polyorder=3)

        # Downsample for fewer points
        rms_downsampled = rms_smoothed[::downsample_factor]
        time_series = np.linspace(0, duration, len(rms_downsampled))

        return rms_downsampled, time_series, duration
    except Exception as e:
        st.error(f"Error extracting RMS features: {e}")
        return None, None, None


def predict_audio_metrics(y, sr):
    """
    Given the raw audio signal (y) and sample rate (sr),
    extract features, scale them, and predict using the loaded model.
    
    Returns a dict of predictions if successful, or None on failure.
    """
    try:
        # Optionally apply a pre-emphasis filter
        y = librosa.effects.preemphasis(y)

        duration = librosa.get_duration(y=y, sr=sr)
        rms = np.mean(librosa.feature.rms(y=y))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
        spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spec_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        spec_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        spec_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
        
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=30), axis=1)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr), axis=1)

        # Combine into a single row of features
        features = [
            duration,
            rms,
            zcr,
            spec_centroid,
            spec_bandwidth,
            spec_rolloff,
            spec_contrast,
        ]
        features += mfccs.tolist() + chroma.tolist() + tonnetz.tolist()

        # Create DataFrame and scale features
        features_df = pd.DataFrame([features], columns=feature_list)
        features_scaled = scaler.transform(features_df)

        # Predict with your loaded model
        predictions = model.predict(features_scaled)

        # Define your target columns in the same order as your model output
        target_columns = [
            "maximum_flow",
            "average_flow",
            "voiding_time",
            "flow_time",
            "time_to_max_flow",
            "flow_at_2_seconds",
            "acceleration",
            "voided_volume",
        ]
        
        # Convert prediction array to dictionary
        return dict(zip(target_columns, predictions.flatten()))

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None


def generate_flow_graph_with_rms(predictions, y, sr):
    """
    Generate a matplotlib figure in memory showing the predicted flow graph 
    based on RMS features scaled to the predicted 'maximum_flow'.
    Returns a BytesIO buffer containing the PNG image.
    """
    # Extract RMS features
    rms, time_series, duration = extract_audio_features_with_rms(y, sr)
    if rms is None:
        return None

    # Normalize RMS to match predicted maximum flow
    max_flow = predictions["maximum_flow"]
    mms = MinMaxScaler(feature_range=(0, max_flow))
    flow_rate = mms.fit_transform(rms.reshape(-1, 1)).flatten()

    # Optional: Taper start and end for smoother curve
    taper_len = len(flow_rate) // 6
    taper_start = np.linspace(0, 1, taper_len)
    taper_end = np.linspace(1, 0, taper_len)
    flow_rate[:taper_len] *= taper_start
    flow_rate[-taper_len:] *= taper_end

    # Plot in memory (Matplotlib default background is white in most Streamlit themes)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time_series, flow_rate, label="Sound-Based Urine Flow Graph", linewidth=2, color="blue")
    ax.set_title("Sound-Based Uroflowmetry Graph")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Flow Rate (ml/s)")
    ax.set_xlim(0, 35)
    ax.set_ylim(0, 50)
    ax.grid(True)

    # Text annotation of predictions
    numerical_data = (
        f"Max Flow Rate: {predictions['maximum_flow']:.2f} ml/s\n"
        f"Average Flow Rate: {predictions['average_flow']:.2f} ml/s\n"
        f"Voiding Duration: {predictions['voiding_time']:.2f} s\n"
        f"Voided Volume: {predictions['voided_volume']:.2f} ml\n"
        f"Time to Max Flow: {predictions['time_to_max_flow']:.2f} s\n"
        f"Flow at 2s: {predictions['flow_at_2_seconds']:.2f} ml/s\n"
        f"Acceleration: {predictions['acceleration']:.2f} ml/s²"
    )
    ax.text(
        36, 45, numerical_data, fontsize=10, va="top", ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
    )
    ax.legend(loc="upper left")

    # Save figure to a BytesIO buffer
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return buf


def main():
    # 1. Display a logo (optional) and custom headings
    LOGO_PATH = os.path.join(BASE_DIR, "logo.jpg")
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_column_width=True)
    else:
        # Alternatively, if you have a GitHub RAW link, you could do:
        # st.image("https://raw.githubusercontent.com/username/repo/branch/logo.jpg", use_column_width=True)
        pass

    st.title("FLOWMIND-RA")
    st.subheader("“PERSONALIZED UROLOGY CARE”")
    st.markdown("""
    ### Sound Uroflowmetry
    **Home-Based PROM and Sound Uroflowmetry**
    """)

    # 2. UI for uploading audio
    st.write("Upload an audio file to get urine flow predictions and a generated flow graph.")
    audio_file = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a", "ogg", "flac"])
    
    if audio_file is not None:
        file_bytes = audio_file.read()
        
        try:
            # Load audio from in-memory
            y, sr = librosa.load(io.BytesIO(file_bytes), sr=None, res_type="kaiser_fast")

            # 3. Predict
            predictions = predict_audio_metrics(y, sr)
            if predictions is None:
                st.error("Prediction failed. Check logs for details.")
                return

            # 4. Translate the keys to Thai before displaying
            key_translation = {
                "maximum_flow": "ปัสสาวะไหลแรงที่สุด",
                "average_flow": "ปัสสาวะไหลเฉลี่ย",
                "voiding_time": "เวลาที่ใช้ในการปัสสาวะทั้งหมด",
                "flow_time": "เวลาที่ปัสสาวะไหลออก",
                "time_to_max_flow": "เวลาที่ใช้จนปัสสาวะไหลแรงที่สุด",
                "flow_at_2_seconds": "ปัสสาวะไหลใน 2 วินาทีแรก",
                "acceleration": "ความเร็วในการไหลของปัสสาวะที่เพิ่มขึ้น",
                "voided_volume": "ปริมาณปัสสาวะที่ปัสสาวะออก"
            }

            translated_predictions = {}
            for eng_key, value in predictions.items():
                # If there's a translation, use it; otherwise keep the original key
                if eng_key in key_translation:
                    translated_predictions[key_translation[eng_key]] = value
                else:
                    translated_predictions[eng_key] = value

            # 5. Display the translated predictions
            st.subheader("Predicted Metrics (ข้อมูลที่คาดการณ์)")
            st.json(translated_predictions)

            # 6. Generate and display the flow graph
            buf = generate_flow_graph_with_rms(predictions, y, sr)
            if buf is not None:
                st.subheader("Generated Flow Graph")
                st.image(buf, caption="Sound-Based Uroflowmetry Graph")
                
                # -----------------------------------------------------------------
                # 7. Add a link to your external website below the graph
                # -----------------------------------------------------------------
                st.markdown("[Visit Flowmind-RA site](https://flowmind-ra.my.canva.site/)")
            else:
                st.error("Failed to generate flow graph.")

        except Exception as e:
            if "Format not recognised" in str(e) or "Unsupported format" in str(e):
                st.error(
                    "Error loading or processing the file: Format not recognized. "
                    "Please try a different audio format (e.g. WAV, MP3, M4A, OGG, FLAC)."
                )
            else:
                st.error(f"Error loading or processing the file: {e}")


if __name__ == "__main__":
    main()