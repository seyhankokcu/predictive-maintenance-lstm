import logging
import pickle
from typing import Optional

import tensorflow as tf
from keras import callbacks, layers, models, optimizers

from predictive_maintenance import Config
from predictive_maintenance import IMSDataETL

# Logger configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def build_model() -> models.Model:
    """Builds the LSTM Autoencoder architecture."""
    inputs = layers.Input(shape=(Config.WINDOW, Config.FEATURES), name="input_layer")

    # Encoder
    x = layers.LSTM(64, return_sequences=True, name="encoder_lstm_1")(inputs)
    x = layers.LSTM(32, return_sequences=False, name="encoder_lstm_2")(x)
    x = layers.RepeatVector(Config.WINDOW, name="repeat_vector")(x)

    # Decoder
    x = layers.LSTM(32, return_sequences=True, name="decoder_lstm_1")(x)
    x = layers.LSTM(64, return_sequences=True, name="decoder_lstm_2")(x)

    outputs = layers.TimeDistributed(
        layers.Dense(Config.FEATURES), name="output_layer"
    )(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="ims_lstm_autoencoder")

    # Using slightly higher learning rate or scheduler is managed in callbacks/config
    model.compile(optimizer=optimizers.Adam(learning_rate=Config.LR), loss="mse")

    return model


class IMSAnomalyDetector:
    """
    Manages the lifecycle of the LSTM Autoencoder: ETL, Training, and Artifact Saving.
    """

    def __init__(self):
        self.etl = IMSDataETL()
        self.scaler = self.etl.scaler
        self.model: Optional[models.Model] = None

        # Ensure output directory exists
        Config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _get_dataset_size(dataset: tf.data.Dataset) -> int:
        """Efficiently determines dataset cardinality."""
        cardinality = tf.data.experimental.cardinality(dataset)

        if cardinality == tf.data.experimental.INFINITE_CARDINALITY:
            raise ValueError("Infinite datasets are not supported.")

        if cardinality == tf.data.experimental.UNKNOWN_CARDINALITY:
            logger.info("Calculating dataset size...")
            return dataset.reduce(0, lambda x, _: x + 1).numpy()

        return int(cardinality)

    def _save_artifacts(self) -> None:
        """Saves model weights and the scaler object."""
        try:
            if self.model:
                self.model.save(Config.MODEL_SAVE_PATH)
                logger.info(f"✔ Model saved: {Config.MODEL_SAVE_PATH}")

            with open(Config.SCALER_SAVE_PATH, "wb") as f:
                pickle.dump(self.scaler, f)
            logger.info(f"✔ Scaler saved: {Config.SCALER_SAVE_PATH}")

        except Exception as e:
            logger.error(f"Artifact save failed: {e}")

    def run(self) -> None:
        """Executes the full training pipeline with subset sampling."""
        logger.info(">>> Pipeline started: Generating Dataset...")
        full_ds = self.etl.generate_training_dataset()

        # 1. Toplam veri setinin büyüklüğünü al
        total_available = self._get_dataset_size(full_ds)

        # 2. ADIM: Veri Azaltma (Downsampling)
        # 100X verinin sadece 20X'ini (Config.DATA_SAMPLE_RATIO) alıyoruz.
        usage_limit = int(total_available * Config.DATA_SAMPLE_RATIO)

        logger.info(
            f"Total Available: {total_available} | Using Subset: {usage_limit} (Ratio: {Config.DATA_SAMPLE_RATIO})")

        # Sadece belirlenen limitteki veriyi al (Örn: İlk 7 batch)
        subset_ds = full_ds.take(usage_limit)

        # 3. ADIM: Train / Val Ayrımı
        # Seçtiğimiz o 20X'lik veriyi kendi içinde bölüyoruz.
        train_size = int(usage_limit * Config.TRAIN_RATIO)
        val_size = usage_limit - train_size

        logger.info(f"Splitting Subset -> Train: {train_size} | Val: {val_size}")

        # Train: Subset'in başından train_size kadar al
        train_ds = subset_ds.take(train_size).cache().prefetch(tf.data.AUTOTUNE)

        # Val: Subset'in içinden train kısmını atla, kalanını al
        val_ds = subset_ds.skip(train_size).cache().prefetch(tf.data.AUTOTUNE)

        # Model Building
        logger.info(">>> Building Model...")
        self.model = build_model()

        cbs = [
            callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=2, verbose=1, min_lr=1e-6
            ),
        ]

        logger.info(">>> Training Started...")
        self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=Config.EPOCHS,
            callbacks=cbs,
            verbose="auto",
        )

        logger.info(">>> Saving Artifacts...")
        self._save_artifacts()