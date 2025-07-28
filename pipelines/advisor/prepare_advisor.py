import boto3, yaml, os
import pandas as pd
from sklearn.model_selection import train_test_split
from io import StringIO

def read_csv_from_s3(bucket, key):
    s3 = boto3.client("s3")
    csv_obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(StringIO(csv_obj['Body'].read().decode('utf-8')))

def write_jsonl_to_s3(df, bucket, key):
    s3 = boto3.client("s3")
    df_records = df.to_dict(orient="records")
    data_str = "\n".join([str(r) for r in df_records])
    s3.put_object(Bucket=bucket, Key=key, Body=data_str.encode('utf-8'))

if __name__ == "__main__":
    params = yaml.safe_load(open("params.yaml"))["advisor"]

    df = read_csv_from_s3(params["bucket"], params["raw_key"])
    df = df.drop_duplicates().dropna()

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    write_jsonl_to_s3(train_df, params["bucket"], os.path.join(params["processed_prefix"], "train.jsonl"))
    write_jsonl_to_s3(test_df, params["bucket"], os.path.join(params["processed_prefix"], "test.jsonl"))
    print("Preprocessing complete → train/test saved to S3.")
