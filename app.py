import os
import numpy as np
import mlflow.pyfunc
from flask import Flask, request, jsonify
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ============================================================================
# MODEL LOADING
# ============================================================================

MODEL_NAME = "fraud-detector"
MODEL_STAGE = "Production"  # Change to "Staging" or "None" if needed

try:
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    model = mlflow.pyfunc.load_model(model_uri)
    logger.info(f"✓ Loaded model from: {model_uri}")
except Exception as e:
    logger.warning(f"⚠ Could not load model from registry: {e}")
    logger.info("  To fix: Make sure MLflow is running and model is registered.")
    logger.info("  OR set MLFLOW_TRACKING_URI and ensure model exists in registry.")
    model = None


# ============================================================================
# FEATURE CONFIGURATION
# ============================================================================
# ⚠️ CRITICAL: This MUST match your training features exactly

FEATURE_COLS = [
    # Transaction basics
    "type_enc", "is_risky_type", "log_amount", "step",
    "hour_of_day", "hour_of_week", "day", "is_night",

    # Raw balances (log-transformed)
    "log_oldbal_orig", "log_newbal_orig",
    "log_oldbal_dest", "log_newbal_dest",

    # Balance engineering (highest signal)
    "orig_balance_error", "dest_balance_error",
    "orig_zeroed", "dest_zeroed",
    "sufficient_balance", "dest_was_empty",
    "amount_ratio_orig", "amount_ratio_dest",
    "exact_drain",

    # Account type
    "orig_is_customer", "dest_is_customer", "dest_is_merchant",
]

# Update this to your tuned threshold after running threshold tuning
THRESHOLD = 0.5


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.route("/", methods=["GET"])
def index():
    """API info page."""
    return jsonify({
        "name": "Fraud Detection API",
        "version": "1.0",
        "model": MODEL_NAME,
        "endpoints": {
            "GET /health": "Health check",
            "POST /predict": "Single prediction",
            "POST /predict_batch": "Batch predictions",
            "GET /features": "Get required features",
        }
    }), 200


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "model_name": MODEL_NAME,
        "model_stage": MODEL_STAGE,
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict fraud probability for a single transaction.
    
    Request JSON:
    {
        "type_enc": 1,
        "is_risky_type": 1,
        "log_amount": 5.5,
        "step": 100,
        ...
        "dest_is_merchant": 0
    }
    
    Response:
    {
        "fraud_probability": 0.92,
        "is_fraud": true,
        "threshold": 0.5,
        "model": "fraud-detector"
    }
    """
    if model is None:
        return jsonify({
            "error": "Model not loaded",
            "status": "service_unavailable"
        }), 503
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "Empty request body"}), 400
        
        # Validate required fields
        missing = [f for f in FEATURE_COLS if f not in data]
        if missing:
            return jsonify({
                "error": "Missing required features",
                "missing": missing,
                "required": FEATURE_COLS,
            }), 400
        
        # Extract features in correct order
        try:
            X = np.array([
                [float(data[col]) for col in FEATURE_COLS]
            ], dtype=np.float32)
        except (ValueError, TypeError) as e:
            return jsonify({
                "error": f"Feature conversion error: {str(e)}",
                "hint": "All feature values must be numeric (int or float)"
            }), 400
        
        # Get prediction
        proba = model.predict(X)[0]
        pred_class = int(proba >= THRESHOLD)
        
        return jsonify({
            "fraud_probability": float(proba),
            "is_fraud": bool(pred_class),
            "threshold": THRESHOLD,
            "model": MODEL_NAME,
        }), 200
        
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 400


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    """
    Predict fraud for multiple transactions.
    
    Request JSON:
    {
        "data": [
            {"type_enc": 1, "is_risky_type": 1, ..., "dest_is_merchant": 1},
            {"type_enc": 0, "is_risky_type": 0, ..., "dest_is_merchant": 0}
        ]
    }
    
    Response:
    {
        "predictions": [
            {"index": 0, "fraud_probability": 0.92, "is_fraud": true},
            {"index": 1, "fraud_probability": 0.05, "is_fraud": false}
        ]
    }
    """
    if model is None:
        return jsonify({
            "error": "Model not loaded",
            "status": "service_unavailable"
        }), 503
    
    try:
        payload = request.get_json()
        
        if not payload:
            return jsonify({"error": "Empty request body"}), 400
        
        transactions = payload.get("data", [])
        
        if not transactions:
            return jsonify({"error": "Empty data array"}), 400
        
        if not isinstance(transactions, list):
            return jsonify({"error": "data must be a list"}), 400
        
        results = []
        
        for i, tx in enumerate(transactions):
            if not isinstance(tx, dict):
                results.append({
                    "index": i,
                    "error": "Transaction must be a dict"
                })
                continue
            
            # Validate features
            missing = [f for f in FEATURE_COLS if f not in tx]
            if missing:
                results.append({
                    "index": i,
                    "error": f"Missing features: {missing}"
                })
                continue
            
            try:
                X = np.array([
                    [float(tx[col]) for col in FEATURE_COLS]
                ], dtype=np.float32)
                
                proba = model.predict(X)[0]
                results.append({
                    "index": i,
                    "fraud_probability": float(proba),
                    "is_fraud": bool(proba >= THRESHOLD),
                })
            except (ValueError, TypeError) as e:
                results.append({
                    "index": i,
                    "error": f"Feature conversion error: {str(e)}"
                })
            except Exception as e:
                results.append({
                    "index": i,
                    "error": str(e)
                })
        
        return jsonify({
            "predictions": results,
            "total": len(transactions),
            "successful": len([r for r in results if "error" not in r]),
        }), 200
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}", exc_info=True)
        return jsonify({"error": f"Batch prediction failed: {str(e)}"}), 400


@app.route("/features", methods=["GET"])
def get_features():
    """Return required feature names and order."""
    return jsonify({
        "feature_columns": FEATURE_COLS,
        "count": len(FEATURE_COLS),
        "note": "Features must be provided in this exact order for predictions",
    }), 200


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"error": "Method not allowed"}), 405


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}", exc_info=True)
    return jsonify({"error": "Internal server error"}), 500


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Starting Fraud Detection API")
    logger.info("=" * 60)
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Features: {len(FEATURE_COLS)}")
    logger.info(f"Threshold: {THRESHOLD}")
    logger.info("=" * 60)
    logger.info("Access at: http://localhost:5000")
    logger.info("=" * 60)
    
    app.run(host="0.0.0.0", port=5000, debug=False)
