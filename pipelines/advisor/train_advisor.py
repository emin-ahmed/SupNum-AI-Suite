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


hf_logging.set_verbosity_info()

# --- GPU CHECK ---
print("CUDA Available:", torch.cuda.is_available())
print("Device Count:", torch.cuda.device_count())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LoRAAdvisorModel(mlflow.pyfunc.PythonModel):
    """Custom MLflow PyFunc wrapper for LoRA-adapted T5 model"""
    
    def load_context(self, context):
        """Load the model when MLflow loads the artifact"""
        import torch
        from transformers import AutoTokenizer, T5ForConditionalGeneration
        from peft import PeftModel
        
        # Get the base model name from the context (we'll save it as an artifact)
        with open(context.artifacts["config"], 'r') as f:
            config = json.load(f)
        
        base_model_name = config["base_model"]
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        base_model = T5ForConditionalGeneration.from_pretrained(base_model_name)
        
        # Load LoRA adapter
        self.model = PeftModel.from_pretrained(base_model, context.artifacts["adapter"])
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, context, model_input):
        """
        Make predictions on input data
        
        Args:
            model_input: Can be:
                - pandas DataFrame with columns: question_ar, question_fr
                - dict with keys: question_ar, question_fr
                - list of dicts
        """
        import pandas as pd
        
        # Handle different input formats
        if isinstance(model_input, pd.DataFrame):
            questions = model_input.to_dict('records')
        elif isinstance(model_input, dict):
            questions = [model_input]
        elif isinstance(model_input, list):
            questions = model_input
        else:
            raise ValueError("Input must be DataFrame, dict, or list of dicts")
        
        results = []
        
        for question in questions:
            # Format input like during training
            input_text = f"سؤال: {question['question_ar']}  Question_FR: {question['question_fr']}"
            
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
                "response": response
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

def preprocess_function(examples, tokenizer):
    inputs = [f"سؤال: {q_ar}  Question_FR: {q_fr}" 
              for q_ar, q_fr in zip(examples["question_ar"], examples["question_fr"])]
    targets = [f"جواب: {a_ar}  Réponse: {a_fr}" 
               for a_ar, a_fr in zip(examples["answer_ar"], examples["answer_fr"])]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)
    labels = tokenizer(targets, max_length=512, truncation=True)
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

if __name__ == "__main__":
    params = yaml.safe_load(open("params.yaml"))["advisor"]

    mlflow.set_tracking_uri(params["mlflow_tracking_uri"])
    mlflow.set_experiment(params["mlflow_experiment"])

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





    dataset = load_dataset("json", data_files={"train": local_train})
    tokenizer = AutoTokenizer.from_pretrained(params["base_model"])
    
    # Add padding token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = T5ForConditionalGeneration.from_pretrained(params["base_model"])
    model.to(device)

    dataset = dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)

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
            "base_model": params["base_model"]
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

        # Save adapter
        adapter_dir = "/tmp/adapter"
        model.save_pretrained(adapter_dir)
        
        # Save model configuration for inference
        config_path = "/tmp/model_config.json"
        config = {
            "base_model": params["base_model"],
            "lora_r": params["lora_r"],
            "lora_alpha": params["lora_alpha"],
            "target_modules": ["q", "v"],
            "task_type": "SEQ_2_SEQ_LM"
        }
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        # Log model with custom PyFunc wrapper
        mlflow.pyfunc.log_model(
            artifact_path="advisor_model",
            python_model=LoRAAdvisorModel(),
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

    print("Training completed and model logged to MLflow.")
    print(f"Model URI: runs:/{run.info.run_id}/advisor_model")