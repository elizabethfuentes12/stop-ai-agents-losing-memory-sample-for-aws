"""Custom Resource handler, S3 Vectors bucket+index and DynamoDB table+vector-index.

CloudFormation calls this Lambda on Create/Update/Delete.
All three resources use APIs too new for native CloudFormation support (as of 2025),
so we manage their lifecycle here using boto3>=1.43.72 (bundled as a Lambda layer).

ResourceType field in Properties selects which resource to manage:
  "S3VectorBucket" , create/delete the vector bucket
  "S3VectorIndex"  , create/delete one vector index inside the bucket
  "DynamoDBVectorTable", create/delete a DynamoDB table with a vector index
"""
import json
import threading
import time
import os
import boto3
import urllib.request
from urllib.parse import urlparse

REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")


def send_response(event, context, status, reason, physical_id, data=None):
    body = json.dumps({
        "Status": status,
        "Reason": reason,
        "PhysicalResourceId": physical_id,
        "StackId": event["StackId"],
        "RequestId": event["RequestId"],
        "LogicalResourceId": event["LogicalResourceId"],
        "Data": data or {},
    }).encode()
    response_url = event["ResponseURL"]
    if urlparse(response_url).scheme not in ("https", "http"):
        raise ValueError(f"Unexpected scheme in ResponseURL: {response_url!r}")
    req = urllib.request.Request(
        url=response_url,
        data=body,
        method="PUT",
        headers={"Content-Type": "", "Content-Length": len(body)},
    )
    urllib.request.urlopen(req)


def handler(event, context):
    props = event.get("ResourceProperties", {})
    resource_type = props["ResourceType"]
    request_type = event["RequestType"]

    try:
        if resource_type == "S3VectorBucket":
            result = handle_s3v_bucket(request_type, props)
        elif resource_type == "S3VectorIndex":
            result = handle_s3v_index(request_type, props)
        elif resource_type == "DynamoDBVectorTable":
            result = handle_dynamodb_table(request_type, props)
        else:
            raise ValueError(f"Unknown ResourceType: {resource_type}")

        send_response(event, context, "SUCCESS", "OK",
                      physical_id=result["PhysicalResourceId"],
                      data=result.get("Data", {}))
    except Exception as exc:
        send_response(event, context, "FAILED", str(exc),
                      physical_id=event.get("PhysicalResourceId", "UNKNOWN"))


# ── S3 Vectors bucket ─────────────────────────────────────────────────────────

def handle_s3v_bucket(request_type, props):
    client = boto3.client("s3vectors", region_name=REGION)
    bucket = props["BucketName"]

    if request_type in ("Create", "Update"):
        try:
            client.get_vector_bucket(vectorBucketName=bucket)
        except client.exceptions.NotFoundException:
            client.create_vector_bucket(vectorBucketName=bucket)
            print(f"Created S3 Vectors bucket: {bucket}")
        return {"PhysicalResourceId": bucket, "Data": {"BucketName": bucket}}

    if request_type == "Delete":
        try:
            # Delete all indexes first, then the bucket.
            resp = client.list_indexes(vectorBucketName=bucket)
            for idx in resp.get("indexes", []):
                client.delete_index(vectorBucketName=bucket, indexName=idx["indexName"])
            client.delete_vector_bucket(vectorBucketName=bucket)
            print(f"Deleted S3 Vectors bucket: {bucket}")
        except client.exceptions.NotFoundException:
            pass
        return {"PhysicalResourceId": bucket}


# ── S3 Vectors index ──────────────────────────────────────────────────────────

def handle_s3v_index(request_type, props):
    client = boto3.client("s3vectors", region_name=REGION)
    bucket = props["BucketName"]
    index = props["IndexName"]
    dims = int(props.get("Dimensions", 1024))
    metric = props.get("DistanceMetric", "cosine")

    physical_id = f"{bucket}/{index}"

    if request_type in ("Create", "Update"):
        try:
            client.get_index(vectorBucketName=bucket, indexName=index)
        except client.exceptions.NotFoundException:
            client.create_index(
                vectorBucketName=bucket, indexName=index,
                dimension=dims, distanceMetric=metric, dataType="float32",
            )
            print(f"Created S3 Vectors index: {index} in {bucket}")
        return {"PhysicalResourceId": physical_id,
                "Data": {"BucketName": bucket, "IndexName": index}}

    if request_type == "Delete":
        try:
            client.delete_index(vectorBucketName=bucket, indexName=index)
            print(f"Deleted S3 Vectors index: {index}")
        except client.exceptions.NotFoundException:
            pass
        return {"PhysicalResourceId": physical_id}


# ── DynamoDB table with vector index ─────────────────────────────────────────

def handle_dynamodb_table(request_type, props):
    client = boto3.client("dynamodb", region_name=REGION)
    table = props["TableName"]
    index = props["VectorIndexName"]
    attr = props.get("VectorAttribute", "embedding")
    dims = int(props.get("Dimensions", 1024))

    if request_type in ("Create", "Update"):
        try:
            client.describe_table(TableName=table)
        except client.exceptions.ResourceNotFoundException:
            client.create_table(
                TableName=table,
                BillingMode="PAY_PER_REQUEST",
                KeySchema=[{"AttributeName": "memory_key", "KeyType": "HASH"}],
                AttributeDefinitions=[{"AttributeName": "memory_key", "AttributeType": "S"}],
                VectorIndexes=[{
                    "IndexName": index,
                    "VectorAttribute": {"AttributeName": attr},
                    "Dimensions": dims,
                    "DistanceFunction": "COSINE",
                    "Projection": {"ProjectionType": "ALL"},
                }],
            )
            print(f"Created DynamoDB table: {table} with vector index: {index}")

        # Wait until ACTIVE + index ACTIVE.
        _wait_table_active(client, table)
        _wait_index_active(client, table, index)
        desc = client.describe_table(TableName=table)["Table"]
        return {
            "PhysicalResourceId": table,
            "Data": {"TableName": table, "TableArn": desc["TableArn"]},
        }

    if request_type == "Delete":
        try:
            client.delete_table(TableName=table)
            _wait_table_gone(client, table)
            print(f"Deleted DynamoDB table: {table}")
        except client.exceptions.ResourceNotFoundException:
            pass
        return {"PhysicalResourceId": table}


def _wait_table_active(client, table, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = client.describe_table(TableName=table)["Table"]["TableStatus"]
        if status == "ACTIVE":
            return
        threading.Event().wait(2)
    raise TimeoutError(f"Table {table!r} not ACTIVE after {timeout}s")


def _wait_index_active(client, table, index, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        desc = client.describe_table(TableName=table)["Table"]
        for vi in desc.get("VectorIndexes", []):
            if vi["IndexName"] == index:
                if vi["IndexStatus"] == "ACTIVE" and not vi.get("Backfilling", False):
                    return
                break
        threading.Event().wait(2)
    raise TimeoutError(f"Vector index {index!r} not ACTIVE after {timeout}s")


def _wait_table_gone(client, table, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            client.describe_table(TableName=table)
            threading.Event().wait(3)
        except client.exceptions.ResourceNotFoundException:
            return
    raise TimeoutError(f"Table {table!r} not deleted after {timeout}s")
