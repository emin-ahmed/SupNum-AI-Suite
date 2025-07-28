import requests
import mlflow
import os
import subprocess
import json
from urllib.parse import urlparse

def debug_mlflow_server(mlflow_uri):
    """Comprehensive MLflow server debugging"""
    print("=== MLflow Server Debug ===")
    print(f"Target URI: {mlflow_uri}")
    
    # Parse the URI
    parsed = urlparse(mlflow_uri)
    print(f"Scheme: {parsed.scheme}")
    print(f"Host: {parsed.hostname}")
    print(f"Port: {parsed.port}")
    print(f"Path: {parsed.path}")
    
    # Test 1: Basic connectivity
    print("\n1. Testing Basic Connectivity...")
    try:
        response = requests.get(mlflow_uri, timeout=10)
        print(f"✅ HTTP Status: {response.status_code}")
        print(f"✅ Response headers: {dict(response.headers)}")
        
        if response.text:
            print(f"Response content length: {len(response.text)} characters")
            if len(response.text) < 500:
                print(f"Response content: {response.text[:500]}")
        else:
            print("❌ Empty response body")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed - server might be down")
        return False
    except requests.exceptions.Timeout:
        print("❌ Connection timeout")
        return False
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False
    
    # Test 2: MLflow API endpoints
    print("\n2. Testing MLflow API Endpoints...")
    api_endpoints = [
        "/api/2.0/mlflow/experiments/search",
        "/health",
        "/version",
        "/api/2.0/mlflow/runs/search"
    ]
    
    for endpoint in api_endpoints:
        try:
            url = mlflow_uri.rstrip('/') + endpoint
            response = requests.get(url, timeout=5)
            print(f"✅ {endpoint}: {response.status_code}")
            if endpoint == "/version" and response.text:
                print(f"   Version info: {response.text}")
        except Exception as e:
            print(f"❌ {endpoint}: Failed - {e}")
    
    # Test 3: MLflow Python client
    print("\n3. Testing MLflow Python Client...")
    try:
        mlflow.set_tracking_uri(mlflow_uri)
        experiments = mlflow.search_experiments()
        print(f"✅ MLflow client connected successfully")
        print(f"✅ Found {len(experiments)} experiments")
        
        if experiments:
            print("Experiments:")
            for exp in experiments[:3]:  # Show first 3
                print(f"   - {exp.name} (ID: {exp.experiment_id})")
        
    except Exception as e:
        print(f"❌ MLflow client failed: {e}")
    
    # Test 4: Check for common issues
    print("\n4. Common Issues Check...")
    
    # Check if it's a localhost/127.0.0.1 issue
    if parsed.hostname in ['localhost', '127.0.0.1']:
        print("⚠️  Warning: Using localhost - may not work in distributed environments")
    
    # Check default port
    if parsed.port == 5000:
        print("ℹ️  Using default MLflow port (5000)")
    
    return True

def test_mlflow_ui_assets(mlflow_uri):
    """Test if MLflow UI static assets are loading"""
    print("\n=== Testing MLflow UI Assets ===")
    
    # Common static asset paths
    static_paths = [
        "/static-files/css/app.css",
        "/static-files/js/app.js",
        "/static/css/app.css",
        "/static/js/app.js"
    ]
    
    for path in static_paths:
        try:
            url = mlflow_uri.rstrip('/') + path
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✅ {path}: Available")
            else:
                print(f"❌ {path}: Status {response.status_code}")
        except Exception as e:
            print(f"❌ {path}: Failed - {e}")

def check_browser_console_equivalent():
    """Simulate browser console checks"""
    print("\n=== Browser Console Equivalent Checks ===")
    print("If using a browser, check the developer console (F12) for:")
    print("1. JavaScript errors")
    print("2. Failed network requests (404, 500 errors)")
    print("3. CORS errors")
    print("4. Mixed content warnings (HTTP vs HTTPS)")

def suggest_fixes():
    """Suggest common fixes"""
    print("\n=== Common Fixes ===")
    fixes = [
        "1. Restart MLflow server: mlflow server --host 0.0.0.0 --port 5000",
        "2. Check if port is already in use: netstat -tulpn | grep :5000",
        "3. Try different browser or incognito mode",
        "4. Clear browser cache",
        "5. Check firewall settings",
        "6. Verify MLflow installation: pip show mlflow",
        "7. Check server logs for errors",
        "8. Try accessing via IP instead of hostname",
        "9. Ensure no proxy/VPN interference"
    ]
    
    for fix in fixes:
        print(fix)

def start_local_mlflow_server():
    """Helper to start a local MLflow server for testing"""
    print("\n=== Starting Local MLflow Server ===")
    print("Run this command in a separate terminal:")
    print("mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns")
    print("\nOr with specific backend:")
    print("mlflow server --host 0.0.0.0 --port 5000 --default-artifact-root s3://project-mlops-bucket/mlruns")

def main():
    # Get MLflow URI from environment or user input
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    
    if not mlflow_uri:
        mlflow_uri = input("Enter your MLflow server URI (e.g., http://13.49.226.14:5000): ")
    
    if not mlflow_uri:
        mlflow_uri = "http://13.62.48.156:5000"  # Default
    
    print(f"Debugging MLflow server at: {mlflow_uri}")
    
    # Run diagnostics
    server_ok = debug_mlflow_server(mlflow_uri)
    
    if server_ok:
        test_mlflow_ui_assets(mlflow_uri)
    
    check_browser_console_equivalent()
    suggest_fixes()
    
    if not server_ok:
        start_local_mlflow_server()

if __name__ == "__main__":
    main()