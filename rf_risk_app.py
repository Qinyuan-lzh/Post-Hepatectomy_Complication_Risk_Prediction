from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
import streamlit.components.v1 as components

from rf_risk_core import (
    build_feature_specs,
    compute_risk_cutoffs,
    load_runtime_assets,
    predict_with_risk_stratification,
    transform_dataset_for_model,
)

try:
    import lime.lime_tabular

    HAS_LIME = True
except ImportError:
    HAS_LIME = False


PROJECT_DIR = Path(__file__).resolve().parent
SEED = 888

st.set_page_config(
    page_title="Post-Hepatectomy Complication Risk Prediction (Random Forest)",
    page_icon=":hospital:",
    layout="wide",
)


def _extract_shap_values(shap_obj: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = shap_obj.values
    base_values = shap_obj.base_values
    data = shap_obj.data

    if values.ndim == 3:
        values = values[:, :, 1]
        if np.ndim(base_values) == 2:
            base_values = base_values[:, 1]
    if np.ndim(base_values) == 0:
        base_values = np.repeat(float(base_values), repeats=values.shape[0])

    return np.array(values, dtype=float), np.array(base_values, dtype=float), np.array(data, dtype=float)


def _single_case_explanation(
    explainer: Any,
    scaled_input: pd.DataFrame,
    feature_names: list[str],
) -> shap.Explanation:
    case_obj = explainer(scaled_input)
    case_values, case_base_values, case_data = _extract_shap_values(case_obj)
    return shap.Explanation(
        values=case_values[0],
        base_values=float(case_base_values[0]),
        data=case_data[0],
        feature_names=feature_names,
    )


@st.cache_resource(show_spinner=False)
def load_context() -> dict[str, Any]:
    assets = load_runtime_assets(PROJECT_DIR)

    x_train_scaled = transform_dataset_for_model(assets.train_df, assets.scaler, assets.model_features)
    train_prob = assets.model.predict_proba(x_train_scaled)[:, 1]
    cutoffs = compute_risk_cutoffs(train_prob)
    feature_specs = build_feature_specs(assets.train_df, assets.model_features)

    explainer = shap.TreeExplainer(assets.model)

    if HAS_LIME:
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=x_train_scaled.to_numpy(dtype=float),
            feature_names=assets.model_features,
            class_names=["Negative", "Positive"],
            mode="classification",
            random_state=SEED,
            verbose=False,
        )
    else:
        lime_explainer = None

    return {
        "assets": assets,
        "cutoffs": cutoffs,
        "feature_specs": feature_specs,
        "shap_explainer": explainer,
        "lime_explainer": lime_explainer,
    }


def render_input_form(context: dict[str, Any]) -> tuple[bool, dict[str, float]]:
    assets = context["assets"]
    specs = context["feature_specs"]
    user_input: dict[str, float] = {}

    st.subheader("Patient Input")
    st.caption("Categorical features use dropdowns. Numeric features support both slider and manual input.")

    with st.form("predict_form", clear_on_submit=False):
        for feature in assets.model_features:
            spec = specs[feature]
            if spec["is_categorical"]:
                options = spec["options"] or [int(round(spec["default"]))]
                default_value = int(round(spec["default"]))
                default_index = options.index(default_value) if default_value in options else 0
                selected = st.selectbox(
                    label=f"{feature} (categorical)",
                    options=options,
                    index=default_index,
                    help="Choose one value observed in the training dataset.",
                )
                user_input[feature] = float(selected)
            else:
                st.markdown(f"**{feature} (numeric)**")
                col1, col2 = st.columns([2, 1])
                slider_value = col1.slider(
                    label=f"{feature} slider",
                    min_value=float(spec["slider_min"]),
                    max_value=float(spec["slider_max"]),
                    value=float(np.clip(spec["default"], spec["slider_min"], spec["slider_max"])),
                    step=float(spec["step"]),
                    key=f"{feature}_slider",
                )
                number_value = col2.number_input(
                    label=f"{feature} value",
                    min_value=float(spec["data_min"]),
                    max_value=float(spec["data_max"]),
                    value=float(slider_value),
                    step=float(spec["step"]),
                    key=f"{feature}_number",
                )
                user_input[feature] = float(number_value)

        submit = st.form_submit_button("Predict Risk and Generate Explanations", type="primary")
    return submit, user_input


def plot_shap_section(context: dict[str, Any], scaled_input: pd.DataFrame) -> None:
    assets = context["assets"]
    explainer = context["shap_explainer"]

    st.markdown("#### SHAP Explanations")
    st.caption("Select one single-case SHAP plot to display.")

    case_exp = _single_case_explanation(explainer, scaled_input, assets.model_features)
    plot_choice = st.selectbox(
        "SHAP plot",
        options=["Force Plot", "Waterfall", "Contribution Bar"],
        index=0,
        key="shap_plot_choice",
    )

    if plot_choice == "Force Plot":
        try:
            base_value = float(np.asarray(case_exp.base_values).reshape(-1)[0])
            force_obj = shap.force_plot(
                base_value,
                np.asarray(case_exp.values, dtype=float),
                np.asarray(case_exp.data, dtype=float),
                feature_names=assets.model_features,
            )
            force_html = f"<head>{shap.getjs()}</head><body>{force_obj.html()}</body>"
            components.html(force_html, height=320, scrolling=True)
        except Exception:
            fig_force = plt.figure(figsize=(11, 3.6))
            try:
                shap.force_plot(
                    float(np.asarray(case_exp.base_values).reshape(-1)[0]),
                    np.asarray(case_exp.values, dtype=float),
                    np.asarray(case_exp.data, dtype=float),
                    feature_names=assets.model_features,
                    matplotlib=True,
                    show=False,
                )
                plt.title("Single-Case SHAP Force Plot")
                st.pyplot(fig_force, clear_figure=True)
            except Exception as exc:
                plt.close(fig_force)
                st.error(f"Force plot rendering failed: {exc}")
        return

    if plot_choice == "Waterfall":
        fig_waterfall = plt.figure(figsize=(9, 4.8))
        shap.plots.waterfall(case_exp, show=False, max_display=8)
        plt.title("Single-Case SHAP Waterfall")
        st.pyplot(fig_waterfall, clear_figure=True)
        return

    contrib = pd.Series(case_exp.values, index=assets.model_features).sort_values(key=lambda s: np.abs(s), ascending=False)
    fig_bar, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(contrib.index, contrib.values, color=["#D62728" if v > 0 else "#1F77B4" for v in contrib.values])
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("SHAP value")
    ax.set_ylabel("Feature")
    ax.set_title("Single-Case SHAP Contribution Bar")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_bar, clear_figure=True)


def plot_lime_section(context: dict[str, Any], scaled_input: pd.DataFrame) -> None:
    st.markdown("#### LIME Local Explanation")
    if not HAS_LIME or context["lime_explainer"] is None:
        st.warning("The `lime` package is not available in this environment.")
        return

    assets = context["assets"]
    lime_explainer = context["lime_explainer"]

    exp = lime_explainer.explain_instance(
        data_row=scaled_input.iloc[0].to_numpy(dtype=float),
        predict_fn=assets.model.predict_proba,
        num_features=min(8, len(assets.model_features)),
    )
    fig_lime = exp.as_pyplot_figure()
    fig_lime.set_size_inches(9, 4.5)
    plt.title("Single-Case LIME Explanation")
    plt.tight_layout()
    st.pyplot(fig_lime, clear_figure=True)


def main() -> None:
    st.title("Post-Hepatectomy Overall Complication Risk Prediction (Random Forest)")

    context = load_context()
    submitted, user_input = render_input_form(context)

    assets = context["assets"]
    result_state_key = "rf_last_prediction_result"
    if submitted:
        st.session_state[result_state_key] = predict_with_risk_stratification(
            model=assets.model,
            scaler=assets.scaler,
            model_features=assets.model_features,
            raw_input=user_input,
            cutoffs=context["cutoffs"],
        )

    if result_state_key not in st.session_state:
        st.info("Fill in all fields and click 'Predict Risk and Generate Explanations'.")
        return

    if not submitted:
        st.caption("Showing the last submitted prediction result.")

    result = st.session_state[result_state_key]

    probability = result["probability"]
    risk_level = result["risk_level"]
    prediction = "Positive" if result["prediction"] == 1 else "Negative"

    c1, c2, c3 = st.columns(3)
    c1.metric("Complication Risk Probability", f"{probability:.2%}")
    c2.metric("Risk Stratum", risk_level)
    c3.metric("Model Classification", prediction)
    st.progress(min(max(probability, 0.0), 1.0), text=f"Risk probability: {probability:.2%}")

    tabs = st.tabs(["SHAP", "LIME"])
    with tabs[0]:
        plot_shap_section(context, scaled_input=result["scaled_input"])
    with tabs[1]:
        plot_lime_section(context, scaled_input=result["scaled_input"])


if __name__ == "__main__":
    main()
