# ============================================================================
# PART A: ADD THIS AFTER YOUR THRESHOLD TUNING
# ============================================================================
# Add this after your threshold tuning code (after you've calculated best_thresh)

import mlflow
import mlflow.lightgbm

# Set experiment name
mlflow.set_experiment("fraud-detection")

# Start MLflow run
with mlflow.start_run(run_name="lgb_baseline_v1"):
    
    # ---- Log Parameters ----
    mlflow.log_params({
        "objective": params["objective"],
        "metric": params["metric"],
        "learning_rate": params["learning_rate"],
        "num_leaves": params["num_leaves"],
        "min_child_samples": params["min_child_samples"],
        "feature_fraction": params["feature_fraction"],
        "bagging_fraction": params["bagging_fraction"],
        "lambda_l1": params["lambda_l1"],
        "lambda_l2": params["lambda_l2"],
        "num_boost_round": model.best_iteration,
    })
    
    # ---- Log Metrics ----
    mlflow.log_metric("test_roc_auc", roc_auc)
    mlflow.log_metric("test_pr_auc", pr_auc)
    mlflow.log_metric("best_threshold", best_thresh)
    mlflow.log_metric("best_f1", f1s[best_idx])
    mlflow.log_metric("test_precision", float(classification_report(
        y_test, test_pred, output_dict=True)["1"]["precision"]))
    mlflow.log_metric("test_recall", float(classification_report(
        y_test, test_pred, output_dict=True)["1"]["recall"]))
    
    # ---- Log the Model ----
    mlflow.lightgbm.log_model(
        model,
        artifact_path="fraud_model",
        registered_model_name="fraud-detector",
        input_example=X_test[:5],
        pip_requirements=["lightgbm>=4.0", "numpy>=1.20", "mlflow>=2.0"],
    )
    
    run_id = mlflow.active_run().info.run_id
    print(f"\n{'='*60}")
    print(f"✓ Model logged to MLflow!")
    print(f"{'='*60}")
    print(f"Run ID:      {run_id}")
    print(f"Experiment:  fraud-detection")
    print(f"Model Name:  fraud-detector")
    print(f"ROC-AUC:     {roc_auc:.4f}")
    print(f"PR-AUC:      {pr_auc:.4f}")
    print(f"Threshold:   {best_thresh:.4f}")
    print(f"{'='*60}")
    print(f"\nNext steps:")
    print(f"1. Start MLflow UI: mlflow ui --host 0.0.0.0 --port 5001")
    print(f"2. Promote model to 'Production' in MLflow UI")
    print(f"3. Run API: python app.py")
    print(f"4. Test: python test_api.py")


# ============================================================================
# PART B: ALTERNATIVE - If using MLflow Model Registry with Transitions
# ============================================================================
# Use this if you want to programmatically manage model versions

import mlflow
from mlflow.client import MlflowClient

client = MlflowClient()

# Get the latest model version
model_version = client.search_model_versions("name='fraud-detector'")[0]

# Transition to Production
client.transition_model_version_stage(
    name="fraud-detector",
    version=model_version.version,
    stage="Production",
)

print(f"✓ Model version {model_version.version} transitioned to Production")


# ============================================================================
# PART C: VERIFY MODEL WAS SAVED (Run in a separate cell after logging)
# ============================================================================

import mlflow.pyfunc

# Test loading the model
model_uri = "models:/fraud-detector/Production"
loaded_model = mlflow.pyfunc.load_model(model_uri)

# Make a test prediction
test_sample = X_test[:1].astype(np.float32)
pred = loaded_model.predict(test_sample)
print(f"✓ Model loaded successfully!")
print(f"  Sample prediction: {pred[0]:.4f}")
print(f"  Model ready for API!")
