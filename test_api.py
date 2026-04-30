"""
Test script for Fraud Detection API
Run this after starting the Flask server: python app.py
"""

import requests
import json
import sys

BASE_URL = "http://localhost:5000"

# Sample transaction features (legitimate example)
LEGIT_TRANSACTION = {
    "type_enc": 0,              # PAYMENT (low fraud)
    "is_risky_type": 0,         # Not TRANSFER/CASH_OUT
    "log_amount": 4.5,          # log(90)
    "step": 100,
    "hour_of_day": 14,
    "hour_of_week": 62,
    "day": 5,
    "is_night": 0,
    "log_oldbal_orig": 10.5,
    "log_newbal_orig": 10.2,    # Balance preserved
    "log_oldbal_dest": 8.5,
    "log_newbal_dest": 8.8,
    "orig_balance_error": 0.05,  # Small error
    "dest_balance_error": 0.02,
    "orig_zeroed": 0,           # Account not zeroed
    "dest_zeroed": 0,
    "sufficient_balance": 1,    # Had funds
    "dest_was_empty": 0,        # Recipient had balance
    "amount_ratio_orig": 0.05,  # Small fraction
    "amount_ratio_dest": 0.3,
    "exact_drain": 0,           # Not exact drain
    "orig_is_customer": 1,
    "dest_is_customer": 1,
    "dest_is_merchant": 0,
}

# Sample transaction features (suspicious example - likely fraud)
FRAUD_TRANSACTION = {
    "type_enc": 1,              # TRANSFER (high fraud risk)
    "is_risky_type": 1,         # TRANSFER is risky
    "log_amount": 8.5,          # Large amount log(5000)
    "step": 50,
    "hour_of_day": 3,           # Night time
    "hour_of_week": 20,
    "day": 2,
    "is_night": 1,              # Night transaction
    "log_oldbal_orig": 11.5,
    "log_newbal_orig": 0.0,     # BALANCE ZEROED OUT ⚠️
    "log_oldbal_dest": 0.0,
    "log_newbal_dest": 8.5,
    "orig_balance_error": 0.0,  # Perfect drain
    "dest_balance_error": 0.0,
    "orig_zeroed": 1,           # ⚠️ Account zeroed
    "dest_zeroed": 0,
    "sufficient_balance": 1,
    "dest_was_empty": 1,        # ⚠️ Destination was empty (mule account)
    "amount_ratio_orig": 1.0,   # Sent 100% of balance
    "amount_ratio_dest": 99.0,
    "exact_drain": 1,           # ⚠️ EXACT DRAIN (fraud pattern)
    "orig_is_customer": 1,
    "dest_is_customer": 0,      # Destination is not customer
    "dest_is_merchant": 1,      # Destination is merchant
}


def test_health():
    """Test health endpoint."""
    print("\n" + "="*60)
    print("TEST: Health Check")
    print("="*60)
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"Status: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_features():
    """Test features endpoint."""
    print("\n" + "="*60)
    print("TEST: Get Features")
    print("="*60)
    try:
        response = requests.get(f"{BASE_URL}/features", timeout=5)
        data = response.json()
        print(f"Status: {response.status_code}")
        print(f"Total features: {data['count']}")
        print("Features:", data['feature_columns'])
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_single_prediction(name, transaction, expected_fraud=False):
    """Test single prediction."""
    print("\n" + "="*60)
    print(f"TEST: Single Prediction — {name}")
    print("="*60)
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=transaction,
            timeout=10
        )
        print(f"Status: {response.status_code}")
        data = response.json()
        print(json.dumps(data, indent=2))
        
        if response.status_code == 200:
            is_fraud = data.get("is_fraud", False)
            prob = data.get("fraud_probability", 0)
            print(f"\n✓ Fraud score: {prob:.4f}")
            print(f"✓ Classification: {'FRAUD' if is_fraud else 'LEGIT'}")
            if is_fraud == expected_fraud:
                print(f"✓ Matches expected result!")
            else:
                print(f"⚠ Expected {'FRAUD' if expected_fraud else 'LEGIT'}, got {'FRAUD' if is_fraud else 'LEGIT'}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_batch_prediction():
    """Test batch prediction."""
    print("\n" + "="*60)
    print("TEST: Batch Prediction")
    print("="*60)
    try:
        payload = {
            "data": [LEGIT_TRANSACTION, FRAUD_TRANSACTION]
        }
        response = requests.post(
            f"{BASE_URL}/predict_batch",
            json=payload,
            timeout=10
        )
        print(f"Status: {response.status_code}")
        data = response.json()
        print(json.dumps(data, indent=2))
        
        if response.status_code == 200:
            for pred in data.get("predictions", []):
                idx = pred.get("index")
                is_fraud = pred.get("is_fraud")
                prob = pred.get("fraud_probability")
                print(f"\n  TX {idx}: {prob:.4f} — {'FRAUD' if is_fraud else 'LEGIT'}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_missing_features():
    """Test error handling for missing features."""
    print("\n" + "="*60)
    print("TEST: Error Handling — Missing Features")
    print("="*60)
    try:
        incomplete_tx = {k: v for k, v in LEGIT_TRANSACTION.items() if k != "type_enc"}
        response = requests.post(
            f"{BASE_URL}/predict",
            json=incomplete_tx,
            timeout=10
        )
        print(f"Status: {response.status_code}")
        data = response.json()
        print(json.dumps(data, indent=2))
        
        if response.status_code == 400 and "missing" in data:
            print(f"✓ Correctly caught missing features")
            return True
        else:
            print(f"⚠ Expected 400 error with missing fields")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# FRAUD DETECTION API — TEST SUITE")
    print("#"*60)
    
    results = {
        "Health Check": test_health(),
        "Features Endpoint": test_features(),
        "Legitimate Transaction": test_single_prediction(
            "Legitimate transaction", LEGIT_TRANSACTION, expected_fraud=False
        ),
        "Fraudulent Transaction": test_single_prediction(
            "Fraudulent transaction", FRAUD_TRANSACTION, expected_fraud=True
        ),
        "Batch Prediction": test_batch_prediction(),
        "Missing Features Error": test_missing_features(),
    }
    
    print("\n" + "#"*60)
    print("# TEST SUMMARY")
    print("#"*60)
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} | {test_name}")
    
    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
