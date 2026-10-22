import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from starter import baseline_model


def test_evaluate_regression_model_returns_metrics():
	model = DummyRegressor(strategy="constant", constant=0)
	model.fit([[0], [1]], [0, 0])

	metrics = baseline_model.evaluate_regression_model(
		model, [[0], [1]], [1, 3], name="Dummy"
	)

	assert metrics == {
		"model_name": "Dummy",
		"test_mean_squared_error": 5.0,
		"test_r_squared": -4.0,
	}


def test_baseline_pipeline_with_synthetic_data():
	np.random.seed(42)
	row_count = 40
	rng = np.random.default_rng(42)
	tips = pd.DataFrame({
		"total_bill": rng.uniform(10, 50, size=row_count),
		"tip": rng.uniform(1, 8, size=row_count),
		"sex": rng.permutation(["Female", "Male"] * (row_count // 2)),
		"smoker": rng.permutation(["No", "Yes"] * (row_count // 2)),
		"day": rng.permutation(["Thur", "Fri", "Sat", "Sun"] * (row_count // 4)),
		"time": rng.permutation(["Lunch", "Dinner"] * (row_count // 2)),
		"size": rng.integers(1, 7, size=row_count),
	})

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
	assert metadata == {
		"train_shape": (32, 8),
		"test_shape": (8, 8),
		"num_features": 8,
	}
	np.testing.assert_allclose(X_train_scaled.mean(axis=0), 0, atol=1e-10)

	X_train_sm = sm.add_constant(X_train_scaled)
	model_sm = sm.OLS(y_train, X_train_sm).fit()
	assert np.isfinite(model_sm.params).all()

	reg = LinearRegression()
	reg.fit(X_train_scaled, y_train)

	baseline_metrics = baseline_model.evaluate_regression_model(
		reg, X_test_scaled, y_test, name="Scikit-Learn OLS"
	)

	assert baseline_metrics["model_name"] == "Scikit-Learn OLS"
	assert baseline_metrics["test_mean_squared_error"] == round(
		mean_squared_error(y_test, reg.predict(X_test_scaled)), 4
	)
	assert baseline_metrics["test_r_squared"] == round(
		r2_score(y_test, reg.predict(X_test_scaled)), 4
	)
	summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
	if summary_file:
		with open(summary_file, "a", encoding="utf-8") as summary:
			print(
				"### `test_baseline_pipeline_with_synthetic_data` passed",
				file=summary,
			)

def test_github_push():
    """Simple test to confirm GitHub Actions runs."""
    assert 2 + 2 == 4