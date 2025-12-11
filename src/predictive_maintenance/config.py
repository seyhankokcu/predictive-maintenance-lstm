from pyprojroot import here
from datetime import timedelta


class Config:
    PROJECT_ROOT = here()

    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"
    REPORTS_DIR = PROJECT_ROOT / "reports"
    FIGURES_DIR = PROJECT_ROOT / "figures"

    RAW_DATA_PATH = DATA_DIR / "raw"
    PROCESSED_DATA_PATH = DATA_DIR / "processed"
    DATA_PATH = PROCESSED_DATA_PATH / "IMS_Parquet_Data"

    MODEL_SAVE_PATH = MODELS_DIR / "ims_lstm_model.keras"
    SCALER_SAVE_PATH = MODELS_DIR / "scaler.pkl"

    OUTPUTS_DIR = PROJECT_ROOT / "outputs"
    RESULTS_SAVE_PATH = OUTPUTS_DIR / "results.csv"

    WINDOW = 128
    BATCH = 128
    EPOCHS = 40
    LR = 0.001
    FEATURES = 6
    DATA_SAMPLE_RATIO = 0.20
    TRAIN_RATIO = 0.80

    CALIBRATION_HOURS = 24
    SECURITY_MARGIN = 0.20
    FIXED_Y_LIMIT = 0.25

    SAMPLING_RATE = 20000  # Hz
    GAP_THRESHOLD = timedelta(minutes=15)

    GROUND_TRUTHS = {
    1: {3: "2003-11-21 09:14:00", 4: "2003-11-19 12:14:00"},
    2: {1: "2004-02-17 18:42:00"},
    3: {3: "2004-04-15 11:57:00"},
}
