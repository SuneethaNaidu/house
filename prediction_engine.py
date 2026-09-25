import joblib
import numpy as np
import pandas as pd


class HousePriceEngine:

    def __init__(
        self,
        model_path="house_price_model.pkl"
    ):

        self.artifact = joblib.load(
            model_path
        )

        self.models = (
            self.artifact["models"]
        )

        self.weights = (
            self.artifact["weights"]
        )

        self.features = (
            self.artifact["features"]
        )

        self.residual_std = (
            self.artifact["residual_std"]
        )

    # ---------------------------------------------------------
    # FEATURE PREPARATION
    # ---------------------------------------------------------

    def prepare_features(
        self,
        data
    ):

        df = pd.DataFrame(
            [data]
        )

        # Ensure every required feature exists
        for feature in self.features:

            if feature not in df.columns:

                df[feature] = 0

        # Keep only trained features
        df = df[
            self.features
        ]

        # Numeric conversion
        for col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        df = df.fillna(0)

        return df

    # ---------------------------------------------------------
    # PREDICTION
    # ---------------------------------------------------------

    def predict(
        self,
        data
    ):

        X = self.prepare_features(
            data
        )

        model_predictions = {}

        final_prediction = 0

        for name, model in self.models.items():

            prediction = float(
                model.predict(X)[0]
            )

            model_predictions[name] = (
                prediction
            )

            final_prediction += (
                self.weights[name]
                * prediction
            )

        # Prediction interval
        margin = (
            1.96 *
            self.residual_std
        )

        lower = max(
            0,
            final_prediction - margin
        )

        upper = (
            final_prediction + margin
        )

        return {

            "prediction":
                final_prediction,

            "lower":
                lower,

            "upper":
                upper,

            "model_predictions":
                model_predictions
        }
