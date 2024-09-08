import io
import torch
import soundfile as sf
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from audio_classification.api.process_test import preprocess_single_audio
from audio_classification.data_preprocess.mlflow_utils import load_model, fetch_latest_model_version
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

# Initialize FastAPI app
app = FastAPI()

# Initialize Prometheus metrics
INPUT_DATA_MEAN = Gauge('input_data_mean', 'Mean of the input data')
INPUT_DATA_VARIANCE = Gauge('input_data_variance', 'Variance of the input data')
MODEL_ACCURACY = Gauge('model_accuracy', 'Accuracy of the model')
MODEL_PRECISION = Gauge('model_precision', 'Precision of the model')
MODEL_RECALL = Gauge('model_recall', 'Recall of the model')
MODEL_F1_SCORE = Gauge('model_f1_score', 'F1 Score of the model')
PREDICTION_DRIFT = Histogram('prediction_drift', 'Drift in model predictions')
RESPONSE_TIME = Histogram('http_response_duration_seconds', 'Histogram of response duration for HTTP requests')

# Initialize Instrumentator
instrumentator = Instrumentator().instrument(app).expose(app)

# Initial load of the model from MLflow or local storage
app.state.model = load_model()
app.state.model_version = fetch_latest_model_version()  # Track the current version of the model

# CORS middleware to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

def check_for_model_update():
    """
    Check if the model version has changed and reload the model if necessary.
    """
    current_version = fetch_latest_model_version()
    if current_version != app.state.model_version:
        print(f"Model has changed from version {app.state.model_version} to {current_version}. Loading new model.")
        app.state.model = load_model()
        app.state.model_version = current_version
    else:
        print(f"Model hasn't changed.")

@app.middleware("http")
async def track_response_time(request: Request, call_next):
    start_time = time.time()  # Record start time
    response = await call_next(request)
    duration = time.time() - start_time  # Calculate duration
    RESPONSE_TIME.observe(duration)  # Record duration in histogram
    return response

@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    """
    Upload an audio file (WAV format) and predict the intent or class.
    """
    # Check if the model has been updated and reload if necessary
    check_for_model_update()

    # Read the file and convert it to a format usable by your model
    audio_bytes = await file.read()
    audio_buffer = io.BytesIO(audio_bytes)

    # Load the audio file (assuming it's in WAV format)
    audio_data, sample_rate = sf.read(audio_buffer)

    # Preprocess the audio to extract features (similar to your training preprocessing)
    X_processed = preprocess_single_audio(audio_data, sample_rate)

    # Convert the preprocessed features to a tensor (assuming your model uses PyTorch)
    X_tensor = torch.tensor(X_processed).unsqueeze(0).float()  # Add batch dimension and convert to float32

    # Perform prediction
    with torch.no_grad():
        result = app.state.model(X_tensor)

    # Example true labels for demonstration; replace with actual data
    true_labels = [1]  # Replace with actual true labels from a dataset or another source
    predicted_labels = [result.argmax(dim=1).item()]

    # Calculate performance metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')

    # Example drift measurement (dummy value; replace with actual computation)
    prediction_drift_measurement = 0.0

    # Update metrics for model performance
    MODEL_ACCURACY.set(accuracy)
    MODEL_PRECISION.set(precision)
    MODEL_RECALL.set(recall)
    MODEL_F1_SCORE.set(f1)

    # Log prediction drift
    PREDICTION_DRIFT.observe(prediction_drift_measurement)

    # Assuming the model returns a class label or some output
    predicted_class = result.argmax(dim=1).item()  # Example for classification

    return {'predicted_class': predicted_class}

@app.get("/")
def root():
    return {'greeting': 'Hello'}
