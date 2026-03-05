from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd


TARGET_COLUMN = "Target"
MODEL_RELATIVE_PATH = Path("saved_models/Random_Forest_best_model.pkl")
SCALER_RELATIVE_PATH = Path("saved_models/standard_scaler.pkl")
TRAIN_RELATIVE_PATH = Path("Train_Set.csv")
TEST_RELATIVE_PATH = Path("Test_Set.csv")


@dataclass(frozen=True)
class RuntimeAssets:
    project_dir: Path
    model: Any
    scaler: Any
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    model_features: list[str]
    scaler_features: list[str]


def load_runtime_assets(project_dir: str | Path) -> RuntimeAssets:
    base_dir = Path(project_dir).resolve()
    model = joblib.load(base_dir / MODEL_RELATIVE_PATH)
    scaler = joblib.load(base_dir / SCALER_RELATIVE_PATH)
    # 在受限环境下并行预测可能触发权限问题，默认降为单线程推理。
    try:
        params = model.get_params(deep=False)
        if "n_jobs" in params:
            model.set_params(n_jobs=1)
    except Exception:
        pass

    train_df = pd.read_csv(base_dir / TRAIN_RELATIVE_PATH, encoding="utf-8-sig")
    test_df = pd.read_csv(base_dir / TEST_RELATIVE_PATH, encoding="utf-8-sig")

    model_features = list(getattr(model, "feature_names_in_", []))
    scaler_features = list(getattr(scaler, "feature_names_in_", []))
    if not model_features:
        raise ValueError("随机森林模型中缺少 feature_names_in_，无法保证特征顺序。")
    if not scaler_features:
        raise ValueError("标准化器中缺少 feature_names_in_，无法保证缩放列映射。")

    return RuntimeAssets(
        project_dir=base_dir,
        model=model,
        scaler=scaler,
        train_df=train_df,
        test_df=test_df,
        model_features=model_features,
        scaler_features=scaler_features,
    )


def _is_categorical_like(series: pd.Series) -> bool:
    clean = series.dropna()
    if clean.empty:
        return False
    unique_count = clean.nunique()
    all_integer_like = np.all(np.isclose(clean.astype(float) % 1, 0.0))
    return unique_count <= 10 and all_integer_like


def build_feature_specs(train_df: pd.DataFrame, model_features: list[str]) -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    for feature in model_features:
        s = train_df[feature].dropna().astype(float)
        if s.empty:
            raise ValueError(f"训练集特征 {feature} 为空，无法创建页面控件。")

        default = float(np.median(s))
        f_min = float(np.min(s))
        f_max = float(np.max(s))
        q1 = float(np.percentile(s, 25))
        q3 = float(np.percentile(s, 75))

        is_categorical = _is_categorical_like(s)
        options = sorted({int(v) for v in s.tolist()}) if is_categorical else None

        # slider 默认范围不直接用极值，避免极端异常值导致 UI 滑动不便。
        slider_min = f_min
        slider_max = f_max
        if not is_categorical:
            iqr = q3 - q1
            if iqr > 0:
                bounded_min = q1 - 1.5 * iqr
                bounded_max = q3 + 1.5 * iqr
                slider_min = max(f_min, bounded_min)
                slider_max = min(f_max, bounded_max)
                if slider_min >= slider_max:
                    slider_min, slider_max = f_min, f_max

        step = 1.0 if _is_integer_range(s) else max((slider_max - slider_min) / 200.0, 0.01)

        specs[feature] = {
            "is_categorical": is_categorical,
            "options": options,
            "default": default,
            "data_min": f_min,
            "data_max": f_max,
            "slider_min": float(slider_min),
            "slider_max": float(slider_max),
            "step": float(step),
        }
    return specs


def _is_integer_range(series: pd.Series) -> bool:
    clean = series.dropna()
    if clean.empty:
        return False
    return np.all(np.isclose(clean.astype(float) % 1, 0.0))


def scale_selected_features(
    raw_feature_values: Mapping[str, float],
    scaler: Any,
    model_features: list[str],
) -> pd.DataFrame:
    feature_index = {name: idx for idx, name in enumerate(scaler.feature_names_in_)}
    row: dict[str, float] = {}
    for feature in model_features:
        if feature not in raw_feature_values:
            raise KeyError(f"缺少特征输入: {feature}")
        if feature not in feature_index:
            raise KeyError(f"标准化器未找到特征: {feature}")
        idx = feature_index[feature]
        mean = float(scaler.mean_[idx])
        scale = float(scaler.scale_[idx]) if float(scaler.scale_[idx]) != 0 else 1.0
        raw_value = float(raw_feature_values[feature])
        row[feature] = (raw_value - mean) / scale

    return pd.DataFrame([row], columns=model_features)


def transform_dataset_for_model(
    df: pd.DataFrame,
    scaler: Any,
    model_features: list[str],
) -> pd.DataFrame:
    if df.empty:
        raise ValueError("输入数据集为空，无法执行标准化。")
    raw_part = df.loc[:, model_features].astype(float)
    feature_index = {name: idx for idx, name in enumerate(scaler.feature_names_in_)}
    means = np.array([float(scaler.mean_[feature_index[f]]) for f in model_features], dtype=float)
    scales = np.array([float(scaler.scale_[feature_index[f]]) for f in model_features], dtype=float)
    scales = np.where(scales == 0.0, 1.0, scales)
    scaled_values = (raw_part.to_numpy(dtype=float) - means) / scales
    return pd.DataFrame(scaled_values, columns=model_features, index=raw_part.index)


def compute_risk_cutoffs(
    train_probabilities: np.ndarray,
    low_quantile: float = 0.33,
    high_quantile: float = 0.67,
) -> tuple[float, float]:
    low = float(np.quantile(train_probabilities, low_quantile))
    high = float(np.quantile(train_probabilities, high_quantile))
    if low >= high:
        low, high = 0.33, 0.67
    return low, high


def classify_risk(probability: float, cutoffs: tuple[float, float]) -> str:
    low, high = cutoffs
    if probability < low:
        return "Low"
    if probability < high:
        return "Medium"
    return "High"


def predict_with_risk_stratification(
    model: Any,
    scaler: Any,
    model_features: list[str],
    raw_input: Mapping[str, float],
    cutoffs: tuple[float, float],
) -> dict[str, Any]:
    scaled_input = scale_selected_features(raw_input, scaler=scaler, model_features=model_features)
    probability = float(model.predict_proba(scaled_input)[0, 1])
    prediction = int(model.predict(scaled_input)[0])
    return {
        "probability": probability,
        "prediction": prediction,
        "risk_level": classify_risk(probability, cutoffs),
        "scaled_input": scaled_input,
    }


def calculate_net_benefit_model(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    net_benefits: list[float] = []
    n = len(y_true)
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        weight = thresh / (1.0 - thresh)
        net_benefit = (tp / n) - (fp / n) * weight
        net_benefits.append(float(net_benefit))
    return np.array(net_benefits, dtype=float)


def calculate_net_benefit_all(y_true: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    net_benefits: list[float] = []
    n = len(y_true)
    prevalence = np.sum(y_true == 1) / n
    for thresh in thresholds:
        weight = thresh / (1.0 - thresh)
        net_benefit = prevalence - (1.0 - prevalence) * weight
        net_benefits.append(float(net_benefit))
    return np.array(net_benefits, dtype=float)
