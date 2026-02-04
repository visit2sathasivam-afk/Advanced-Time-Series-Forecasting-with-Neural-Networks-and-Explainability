import shap
import matplotlib.pyplot as plt


def run_shap_explainability(model, X_train, X_test, output_path="shap_summary.png"):
    """
    Runs SHAP explainability for an LSTM model and saves summary plot.

    Parameters:
        model: Trained keras model
        X_train: Training data
        X_test: Test data
        output_path: File path to save SHAP plot

    Returns:
        shap_values
    """
    print("Running SHAP Explainability...")

    # Use a small background sample for performance
    background = X_train[:100]
    test_sample = X_test[:10]

    explainer = shap.KernelExplainer(model.predict, background)
    shap_values = explainer.shap_values(test_sample)

    shap.summary_plot(shap_values, test_sample, show=False)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    print(f"SHAP summary plot saved to: {output_path}")

    return shap_values
