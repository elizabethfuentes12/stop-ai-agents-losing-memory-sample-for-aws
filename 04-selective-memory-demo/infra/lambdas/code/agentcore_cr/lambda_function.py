"""Custom Resource handler, AgentCore Memory and S3 Vectors indexes for Demo 04.

CloudFormation calls this Lambda on Create/Update/Delete.
AgentCore Memory and S3 Vectors have no native CloudFormation support (2025),
so lifecycle is managed here using boto3>=1.43.72 bundled as a Lambda layer.

ResourceType field in Properties selects which resource to manage:
  "S3VectorBucket" , create/delete the vector bucket
  "S3VectorIndex"  , create/delete one vector index inside the bucket
  "AgentCoreMemory", create/delete an AgentCore Memory with 4 built-in strategies
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
    # nosemgrep: dynamic-urllib-use-detected  # ResponseURL is the CloudFormation-provided
    # pre-signed S3 callback URL, and its scheme is validated to http/https above.
    urllib.request.urlopen(req)  # nosec B310


def handler(event, context):
    props = event.get("ResourceProperties", {})
    resource_type = props["ResourceType"]
    request_type = event["RequestType"]
    # PhysicalResourceId from a previous Create/Update, needed for Delete.
    existing_physical_id = event.get("PhysicalResourceId", "UNKNOWN")

    try:
        if resource_type == "S3VectorBucket":
            result = handle_s3v_bucket(request_type, props)
        elif resource_type == "S3VectorIndex":
            result = handle_s3v_index(request_type, props)
        elif resource_type == "AgentCoreMemory":
            result = handle_agentcore_memory(request_type, props, existing_physical_id)
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
    physical_id = f"{bucket}/{index}"

    if request_type in ("Create", "Update"):
        try:
            client.get_index(vectorBucketName=bucket, indexName=index)
        except client.exceptions.NotFoundException:
            client.create_index(
                vectorBucketName=bucket, indexName=index,
                dimension=dims, distanceMetric="cosine", dataType="float32",
            )
            print(f"Created S3 Vectors index: {index}")
        return {"PhysicalResourceId": physical_id,
                "Data": {"BucketName": bucket, "IndexName": index}}

    if request_type == "Delete":
        try:
            client.delete_index(vectorBucketName=bucket, indexName=index)
        except client.exceptions.NotFoundException:
            pass
        return {"PhysicalResourceId": physical_id}


# ── AgentCore Memory ──────────────────────────────────────────────────────────

def handle_agentcore_memory(request_type, props, existing_physical_id="UNKNOWN"):
    ctrl = boto3.client("bedrock-agentcore-control", region_name=REGION)
    memory_name = props["MemoryName"]

    if request_type in ("Create", "Update"):
        # Check if this memory already exists.
        memory_id = None
        for m in ctrl.list_memories(maxResults=50).get("memories", []):
            if m["id"].startswith(memory_name + "-"):
                memory_id = m["id"]
                break

        if memory_id is None:
            resp = ctrl.create_memory(
                name=memory_name,
                description="Selective-memory demo, 4 built-in strategies",
                eventExpiryDuration=7,
                memoryStrategies=[
                    {"semanticMemoryStrategy": {"name": "facts"}},
                    {"userPreferenceMemoryStrategy": {"name": "preferences"}},
                    {"summaryMemoryStrategy": {"name": "tripSummary"}},
                    {"episodicMemoryStrategy": {
                        "name": "episodes",
                        "reflectionConfiguration": {
                            "namespaces": ["/strategies/{memoryStrategyId}/actors/{actorId}"]
                        },
                    }},
                ],
            )
            memory_id = resp["memory"]["id"]
            print(f"Created AgentCore Memory: {memory_id}")

        # Wait for ACTIVE (can take several minutes).
        memory_id = _wait_memory_active(ctrl, memory_id)
        return {
            "PhysicalResourceId": memory_id,
            "Data": {"MemoryId": memory_id, "MemoryName": memory_name},
        }

    if request_type == "Delete":
        memory_id = existing_physical_id
        try:
            ctrl.delete_memory(memoryId=memory_id)
            print(f"Deleted AgentCore Memory: {memory_id}")
        except ctrl.exceptions.ResourceNotFoundException:
            pass
        return {"PhysicalResourceId": memory_id}


def _wait_memory_active(ctrl, memory_id, timeout=540):
    deadline = time.time() + timeout
    while time.time() < deadline:
        detail = ctrl.get_memory(memoryId=memory_id)["memory"]
        if detail["status"] == "ACTIVE":
            return memory_id
        threading.Event().wait(15)
    raise TimeoutError(f"AgentCore Memory {memory_id} not ACTIVE after {timeout}s")
