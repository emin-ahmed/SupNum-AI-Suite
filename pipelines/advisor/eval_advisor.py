import yaml
import mlflow
import boto3
import os
import json
import ast
from rouge import Rouge

def download_from_s3(bucket, key, local_path):
    """Download file from S3 with proper AWS session handling"""
    print("Access key:", os.getenv("AWS_ACCESS_KEY_ID"))
    print("Secret key:", os.getenv("AWS_SECRET_ACCESS_KEY"))
    session = boto3.session.Session(
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    )
    s3 = session.client("s3")
    s3.download_file(bucket, key, local_path)

def fix_jsonl_format(file_path):
    """Convert Python dict format to proper JSONL format"""
    with open(file_path, "r") as infile:
        lines = infile.readlines()

    with open(file_path, "w") as outfile:
        for line in lines:
            try:
                data = ast.literal_eval(line.strip())   # safely parse Python dict
                outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
            except Exception as e:
                print("Failed line:", line)
                print("Error:", e)
    
    print("Test file successfully fixed and overwritten as proper JSONL")

def generate_answer_with_mlflow_model(model, question_ar, question_fr=""):
    """Generate answer using the MLflow model (same format as backend)"""
    # Format input exactly like the backend expects
    input_data = {
        "question_ar": question_ar,
        "question_fr": question_fr
    }
    
    # Use the MLflow model's predict method
    result = model.predict(input_data)
    
    # Handle the response format (list of dicts)
    if isinstance(result, list) and len(result) > 0:
        if isinstance(result[0], dict):
            return result[0].get('response', str(result[0]))
        else:
            return str(result[0])
    
    return str(result)

if __name__ == "__main__":
    params = yaml.safe_load(open("params.yaml"))["advisor"]

    # Download test set
    os.makedirs("data", exist_ok=True)
    local_test = "data/test.jsonl"
    download_from_s3(params["bucket"], params["test_key"], local_test)
    
    # Fix JSONL format (same as in training script)
    fix_jsonl_format(local_test)
    
    # Load test data
    with open(local_test) as f:
        test_data = [json.loads(line) for line in f]

    # Load model from MLflow (same way as backend)
    mlflow.set_tracking_uri(params["mlflow_tracking_uri"])
    model_name = params.get("mlflow_model_name", "advisor-model")
    
    # Load the model using MLflow PyFunc (exactly like your backend)
    try:
        model = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")
        print(f"Successfully loaded model: {model_name}/Production")
    except Exception as e:
        print(f"Failed to load Production model, trying latest version: {e}")
        # Fallback to latest version if Production doesn't exist
        model = mlflow.pyfunc.load_model(f"models:/{model_name}/latest")
        print(f"Successfully loaded model: {model_name}/latest")

    # Generate predictions
    predictions = []
    references = []
    
    print(f"Evaluating on {len(test_data)} samples...")
    
    for i, sample in enumerate(test_data):
        try:
            # Generate prediction using the same method as backend
            pred = generate_answer_with_mlflow_model(
                model, 
                sample["question_ar"], 
                sample.get("question_fr", "")
            )
            
            # Create reference in the same format as training
            reference = f"جواب: {sample['answer_ar']}  Réponse: {sample.get('answer_fr', '')}"
            
            predictions.append(str(pred))
            references.append(reference)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(test_data)} samples")
                
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            print(f"Sample: {sample}")
            # Add empty strings to maintain alignment
            predictions.append("")
            references.append(f"جواب: {sample['answer_ar']}  Réponse: {sample.get('answer_fr', '')}")

    # Filter out empty predictions for ROUGE calculation
    valid_pairs = [(p, r) for p, r in zip(predictions, references) if p.strip()]
    
    if not valid_pairs:
        print("No valid predictions generated!")
        metrics = {"rouge_l_f": 0.0, "valid_predictions": 0, "total_samples": len(test_data)}
    else:
        valid_predictions, valid_references = zip(*valid_pairs)
        
        # Calculate ROUGE scores
        rouge = Rouge()
        try:
            rouge_scores = rouge.get_scores(list(valid_predictions), list(valid_references), avg=True)
            metrics = {
                "rouge_l_f": rouge_scores["rouge-l"]["f"],
                "rouge_1_f": rouge_scores["rouge-1"]["f"],
                "rouge_2_f": rouge_scores["rouge-2"]["f"],
                "valid_predictions": len(valid_pairs),
                "total_samples": len(test_data),
                "success_rate": len(valid_pairs) / len(test_data)
            }
        except Exception as e:
            print(f"Error calculating ROUGE scores: {e}")
            metrics = {
                "rouge_l_f": 0.0,
                "rouge_1_f": 0.0,
                "rouge_2_f": 0.0,
                "valid_predictions": len(valid_pairs),
                "total_samples": len(test_data),
                "success_rate": len(valid_pairs) / len(test_data),
                "error": str(e)
            }

    # Save metrics
    os.makedirs("metrics", exist_ok=True)
    with open("metrics/eval_metrics.json", "w") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Log metrics to MLflow
    with mlflow.start_run():
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"eval_{key}", value)
        
        # Log some sample predictions for inspection
        sample_results = []
        for i in range(min(5, len(test_data))):
            sample_results.append({
                "question_ar": test_data[i]["question_ar"],
                "question_fr": test_data[i].get("question_fr", ""),
                "expected_answer": f"{test_data[i]['answer_ar']} {test_data[i].get('answer_fr', '')}",
                "predicted_answer": predictions[i] if i < len(predictions) else "N/A"
            })
        
        with open("sample_predictions.json", "w") as f:
            json.dump(sample_results, f, ensure_ascii=False, indent=2)
        mlflow.log_artifact("sample_predictions.json")

    print("Evaluation completed!")
    print("Metrics:", json.dumps(metrics, indent=2, ensure_ascii=False))
    
    # Print some sample results
    print("\n--- Sample Predictions ---")
    for i in range(min(3, len(test_data))):
        print(f"\nSample {i+1}:")
        print(f"Question (AR): {test_data[i]['question_ar']}")
        print(f"Question (FR): {test_data[i].get('question_fr', 'N/A')}")
        print(f"Expected: {test_data[i]['answer_ar']} {test_data[i].get('answer_fr', '')}")
        print(f"Predicted: {predictions[i] if i < len(predictions) else 'N/A'}")
        print("-" * 50)
