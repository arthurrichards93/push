import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


EXPERIMENTS = (
    {"fit_intercept": True, "positive": False},
    {"fit_intercept": False, "positive": False},
    {"fit_intercept": True, "positive": True},
)
ARTIFACT_PATH = Path("artifacts/best_model.joblib")


def run_experiments(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    experiments: Sequence[Mapping[str, bool]] = EXPERIMENTS,
) -> list[dict[str, object]]:
    results = []

    for experiment_number, parameters in enumerate(experiments, start=1):
        model = LinearRegression(**parameters)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        metrics = {
            "model_name": f"Experiment {experiment_number}",
            "test_mean_squared_error": round(
                mean_squared_error(y_test, predictions), 4
            ),
            "test_r_squared": round(r2_score(y_test, predictions), 4),
        }
        result = {
            "experiment": experiment_number,
            "parameters": dict(parameters),
            "intercept": float(model.intercept_),
            "coefficients": model.coef_.tolist(),
            **metrics,
        }
        results.append(result)

        print(f"\n--- Experiment {experiment_number} ---")
        print(f"parameters: {json.dumps(result['parameters'], sort_keys=True)}")
        print(f"intercept: {result['intercept']:.6f}")
        print(f"coefficients: {json.dumps(result['coefficients'])}")
        print(f"test_mean_squared_error: {metrics['test_mean_squared_error']}")
        print(f"test_r_squared: {metrics['test_r_squared']}")

    return results


def train_best_model(
    X: pd.DataFrame,
    y: np.ndarray,
    experiment_results: Sequence[Mapping[str, object]],
    artifact_path: Path = ARTIFACT_PATH,
) -> dict[str, object]:
    best_experiment = min(
        experiment_results,
        key=lambda result: float(result["test_mean_squared_error"]),
    )
    parameters = best_experiment["parameters"]
    if not isinstance(parameters, Mapping):
        raise TypeError("Experiment parameters must be a mapping.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LinearRegression(**parameters)
    model.fit(X_scaled, y)

    artifact = {
        "model": model,
        "scaler": scaler,
        "feature_columns": list(X.columns),
        "parameters": dict(parameters),
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, artifact_path)

    print(
        "Selected parameters: "
        f"{json.dumps(artifact['parameters'], sort_keys=True)} "
        f"(test MSE: {best_experiment['test_mean_squared_error']})"
    )
    print(f"Saved model artifact to {artifact_path}")
    return artifact


def main() -> None:
    np.random.seed(42)

    tips = sns.load_dataset("tips")
    df = pd.get_dummies(tips, drop_first=True)
    X = df.drop(columns=["tip"])
    y = df["tip"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    metadata = {
        "train_shape": X_train_scaled.shape,
        "test_shape": X_test_scaled.shape,
        "num_features": X_train_scaled.shape[1],
    }
    print(f"Pipeline Data Diagnostics: {metadata}")

    X_train_sm = sm.add_constant(X_train_scaled)
    model_sm = sm.OLS(y_train, X_train_sm).fit()
    print(model_sm.summary())

    results = run_experiments(X_train_scaled, X_test_scaled, y_train, y_test)
    train_best_model(X, y, results)

print(f"End of training experiments.")

if __name__ == "__main__":
    main()


print("Training pipeline execution completed.")
print("GitHub Actions CI/CD push test - 22 October 2026")
