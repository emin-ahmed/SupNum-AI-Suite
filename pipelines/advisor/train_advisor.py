import os
import boto3
import yaml
import mlflow
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel
from rouge import Rouge
from transformers import logging as hf_logging
import mlflow.pyfunc
from transformers import DataCollatorForSeq2Seq
import shutil

hf_logging.set_verbosity_info()

# --- GPU CHECK ---
print("CUDA Available:", torch.cuda.is_available())
print("Device Count:", torch.cuda.device_count())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CPUCompatibleLoRAAdvisorModel(mlflow.pyfunc.PythonModel):
    """CPU-Compatible MLflow PyFunc wrapper for LoRA-adapted T5 model"""
    
    def load_context(self, context):
        """Load the model when MLflow loads the artifact"""
        import torch
        from transformers import AutoTokenizer, T5ForConditionalGeneration
        from peft import PeftModel
        
        # Force CPU usage
        torch.set_default_tensor_type(torch.FloatTensor)
        
        # Get the base model name from the context
        with open(context.artifacts["config"], 'r') as f:
            config = json.load(f)
        
        base_model_name = config["base_model"]
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model with explicit CPU device mapping
        base_model = T5ForConditionalGeneration.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        
        # Load LoRA adapter with CPU device mapping
        self.model = PeftModel.from_pretrained(
            base_model, 
            context.artifacts["adapter"],
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        
        # Ensure CPU device
        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded on device: {self.device}")
    
    def predict(self, context, model_input):
        """
        Make predictions on input data
        
        Args:
            model_input: Can be:
                - pandas DataFrame with columns: question, language
                - dict with keys: question, language (optional, defaults to 'ar')
                - list of dicts
                - string (treated as Arabic question)
        """
        import pandas as pd
        
        # Handle different input formats
        if isinstance(model_input, pd.DataFrame):
            questions = model_input.to_dict('records')
        elif isinstance(model_input, dict):
            questions = [model_input]
        elif isinstance(model_input, list):
            if len(model_input) > 0 and isinstance(model_input[0], str):
                # Handle list of strings (assume Arabic)
                questions = [{"question": q, "language": "ar"} for q in model_input]
            else:
                questions = model_input
        elif isinstance(model_input, str):
            # Handle single string input (assume Arabic)
            questions = [{"question": model_input, "language": "ar"}]
        else:
            raise ValueError("Input must be DataFrame, dict, list, or string")
        
        results = []
        
        for question in questions:
            # Ensure question has required keys
            if isinstance(question, str):
                question = {"question": question, "language": "ar"}
            
            question_text = question.get("question", "")
            language = question.get("language", "ar")
            
            # Format input based on language
            if language == "ar":
                input_text = f"سؤال: {question_text}"
            elif language == "fr":
                input_text = f"Question: {question_text}"
            else:
                # Default to Arabic format
                input_text = f"سؤال: {question_text}"
            
            # Tokenize
            inputs = self.tokenizer(
                input_text,
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=False
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            results.append({
                "input": input_text,
                "response": response,
                "language": language
            })
        
        return results

def download_from_s3(bucket, key, local_path):
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

def preprocess_function_single_lang(examples, tokenizer):
    """
    Process dataset to create separate training examples for each language
    Each row will be duplicated: one for Arabic and one for French
    """
    inputs = []
    targets = []
    
    # Process each example
    for i in range(len(examples["question_ar"])):
        # Arabic example
        ar_input = f"سؤال: {examples['question_ar'][i]}"
        ar_target = f"جواب: {examples['answer_ar'][i]}"
        inputs.append(ar_input)
        targets.append(ar_target)
        
        # French example
        fr_input = f"Question: {examples['question_fr'][i]}"
        fr_target = f"Réponse: {examples['answer_fr'][i]}"
        inputs.append(fr_input)
        targets.append(fr_target)
    
    # Tokenize inputs and targets
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding=True)
    labels = tokenizer(targets, max_length=512, truncation=True, padding=True)
    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs

def compute_metrics(eval_preds, tokenizer):
    preds, labels = eval_preds
    
    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Clean up labels (remove padding)
    decoded_labels = [label.replace(tokenizer.pad_token, "").strip() for label in decoded_labels]
    decoded_preds = [pred.strip() for pred in decoded_preds]
    
    # Calculate ROUGE scores
    rouge = Rouge()
    try:
        rouge_scores = rouge.get_scores(decoded_preds, decoded_labels, avg=True)
        return {"rouge_l_f": rouge_scores["rouge-l"]["f"]}
    except:
        return {"rouge_l_f": 0.0}

def expand_dataset_for_single_language(dataset):
    """
    Expand the dataset to create separate examples for each language
    This doubles the dataset size: each original row becomes 2 rows (AR + FR)
    """
    expanded_data = []
    
    for example in dataset:
        # Arabic example
        ar_example = {
            "department": example["department"],
            "question": example["question_ar"],
            "answer": example["answer_ar"],
            "language": "ar"
        }
        expanded_data.append(ar_example)
        
        # French example
        fr_example = {
            "department": example["department"],
            "question": example["question_fr"],
            "answer": example["answer_fr"],
            "language": "fr"
        }
        expanded_data.append(fr_example)
    
    return expanded_data

def preprocess_expanded_dataset(examples, tokenizer):
    """
    Preprocess the expanded dataset where each example is single-language
    """
    inputs = []
    targets = []
    
    for i in range(len(examples["question"])):
        language = examples["language"][i]
        question = examples["question"][i]
        answer = examples["answer"][i]
        
        if language == "ar":
            input_text = f"سؤال: {question}"
            target_text = f"جواب: {answer}"
        else:  # French
            input_text = f"Question: {question}"
            target_text = f"Réponse: {answer}"
        
        inputs.append(input_text)
        targets.append(target_text)
    
    # Tokenize
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding=True)
    labels = tokenizer(targets, max_length=512, truncation=True, padding=True)
    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs

if __name__ == "__main__":
    params = yaml.safe_load(open("params.yaml"))["advisor"]

    mlflow.set_tracking_uri(params["mlflow_tracking_uri"])
    mlflow.set_experiment(params["mlflow_experiment"])
    print("Model : ", params["base_model"])
    os.makedirs("data", exist_ok=True)
    local_train = "data/train.jsonl"
    download_from_s3(params["bucket"], params["train_key"], local_train)
    
    import ast
    import json

    # Read the Python-dict style lines, convert to valid JSON, and overwrite
    with open(local_train, "r") as infile:
        lines = infile.readlines()

    with open(local_train, "w") as outfile:
        for line in lines:
            try:
                data = ast.literal_eval(line.strip())   # safely parse Python dict
                outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
            except Exception as e:
                print("Failed line:", line)
                print("Error:", e)

    print("File successfully fixed and overwritten as proper JSONL")

    # Load original dataset
    original_dataset = load_dataset("json", data_files={"train": local_train})
    
    # Expand dataset for single-language training
    print("Expanding dataset for single-language training...")
    expanded_data = expand_dataset_for_single_language(original_dataset["train"])
    
    # Save expanded dataset
    expanded_file = "data/expanded_train.jsonl"
    with open(expanded_file, "w") as f:
        for example in expanded_data:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    # Load expanded dataset
    dataset = load_dataset("json", data_files={"train": expanded_file})
    print(f"Original dataset size: {len(original_dataset['train'])}")
    print(f"Expanded dataset size: {len(dataset['train'])}")
    
    tokenizer = AutoTokenizer.from_pretrained(params["base_model"])
    
    # Add padding token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = T5ForConditionalGeneration.from_pretrained(params["base_model"])
    model.to(device)

    # Use the new preprocessing function for expanded dataset
    dataset = dataset.map(lambda x: preprocess_expanded_dataset(x, tokenizer), batched=True)

    lora_config = LoraConfig(
        r=params["lora_r"],
        lora_alpha=params["lora_alpha"],
        target_modules=["q", "v"],
        lora_dropout=params["lora_dropout"],
        task_type="SEQ_2_SEQ_LM"
    )
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir="./output",
        per_device_train_batch_size=params["batch_size"],
        num_train_epochs=params["epochs"],
        learning_rate=float(params["lr"]),
        logging_dir="./logs",
        logging_strategy="steps",
        logging_steps=1,
        save_strategy="no",
        save_total_limit=0,
        report_to=["mlflow"],
        fp16=torch.cuda.is_available()
    )

    # Create compute_metrics function with tokenizer
    def compute_metrics_with_tokenizer(eval_preds):
        return compute_metrics(eval_preds, tokenizer)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        compute_metrics=compute_metrics_with_tokenizer,
        data_collator=data_collator
    )

    with mlflow.start_run() as run:
        trainer.train()

        mlflow.log_params({
            "epochs": params["epochs"],
            "lr": params["lr"],
            "batch_size": params["batch_size"],
            "lora_r": params["lora_r"],
            "lora_alpha": params["lora_alpha"],
            "base_model": params["base_model"],
            "training_approach": "single_language_per_example"
        })

        # Save metrics
        os.makedirs("metrics", exist_ok=True)
        with open("metrics/train_metrics.json", "w") as f:
            json.dump(trainer.state.log_history, f, ensure_ascii=False)

        # Log training metrics to MLflow
        final_metrics = trainer.state.log_history
        for record in final_metrics:
            if "loss" in record:
                mlflow.log_metric("loss", record["loss"], step=record.get("step", 0))
            if "eval_loss" in record:
                mlflow.log_metric("eval_loss", record["eval_loss"], step=record.get("step", 0))

        # ---- CRITICAL: Convert to CPU for CPU-compatible saving ----
        print("Moving model to CPU for CPU-compatible saving...")
        
        # Move model to CPU
        model.to("cpu")
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Set default tensor type to FloatTensor (CPU)
        torch.set_default_tensor_type(torch.FloatTensor)
        
        # Force garbage collection
        import gc
        gc.collect()
        
        print("Model successfully moved to CPU")
        
        # Save adapter with CPU state
        adapter_dir = "/tmp/adapter"
        if os.path.exists(adapter_dir):
            shutil.rmtree(adapter_dir)
        
        # Save the LoRA adapter
        model.save_pretrained(adapter_dir)
        print(f"Adapter saved to {adapter_dir}")
        
        # Save model configuration for inference
        config_path = "/tmp/model_config.json"
        config = {
            "base_model": params["base_model"],
            "lora_r": params["lora_r"],
            "lora_alpha": params["lora_alpha"],
            "target_modules": ["q", "v"],
            "task_type": "SEQ_2_SEQ_LM",
            "torch_dtype": "float32",
            "device_map": "cpu"
        }
        with open(config_path, 'w') as f:
            json.dump(config, f)
        print(f"Config saved to {config_path}")

        temp_model_path = "./temp_advisor_model"
        if os.path.exists(temp_model_path):
            shutil.rmtree(temp_model_path)
        print(f"Removed existing directory: {temp_model_path}")

        # Save model with CPU-compatible PyFunc wrapper
        mlflow.pyfunc.save_model(
            path=temp_model_path,
            python_model=CPUCompatibleLoRAAdvisorModel(),
            artifacts={
                "adapter": adapter_dir,
                "config": config_path
            },
            pip_requirements=[
                "torch",
                "transformers",
                "peft",
                "accelerate"
            ]
        )
        print("Model saved with CPU-compatible wrapper")

        # Log to MLflow
        mlflow.log_artifacts(temp_model_path, "advisor_model")
        print("Artifacts logged to MLflow")

        # Register the model
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/advisor_model"
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name="advisor-model"
        )

        print(f"Model registered: {registered_model.name}, Version: {registered_model.version}")
        
        # Test the saved model locally (optional)
        try:
            print("Testing saved model...")
            test_model = mlflow.pyfunc.load_model(temp_model_path)
            
            # Test Arabic
            test_input_ar = {"question": "ما هو الاستثمار؟", "language": "ar"}
            test_result_ar = test_model.predict(test_input_ar)
            print(f"Arabic test result: {test_result_ar}")
            
            # Test French
            test_input_fr = {"question": "Qu'est-ce que l'investissement?", "language": "fr"}
            test_result_fr = test_model.predict(test_input_fr)
            print(f"French test result: {test_result_fr}")
            
            print("Model test successful!")
        except Exception as e:
            print(f"Model test failed: {e}")
        
    print("Training completed and CPU-compatible model logged to MLflow.")
    print(f"Model URI: runs:/{run.info.run_id}/advisor_model")