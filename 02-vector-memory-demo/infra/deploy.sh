#!/usr/bin/env bash
# deploy.sh — deploys the AgentMemoryVectorStoreStack for Demo 02.
#
# What it creates:
#   - Amazon S3 Vectors bucket + traveler-memories index (Tests 3)
#   - Amazon DynamoDB table with native vector index (Test 4)
#
# After deploy, VECTOR_BUCKET, VECTOR_INDEX, DYNAMODB_TABLE are written
# to ../.env automatically. Run `cdk destroy` from this folder to tear down.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Confirm the target account (never hardcoded) ─────────────────────────────
ACCOUNT=$(aws sts get-caller-identity --query Account --output text 2>/dev/null) || {
  echo "ERROR: could not resolve AWS account. Configure credentials (aws configure or AWS_PROFILE)." >&2
  exit 1
}
REGION=$(aws configure get region 2>/dev/null || echo "us-east-1")

echo "=========================================="
echo "  Demo 02 — Vector store deploy"
echo "  Account : $ACCOUNT"
echo "  Region  : $REGION"
echo "=========================================="
echo "Press Ctrl+C within 5 s to cancel."
sleep 5

# ── Python venv + CDK deps ───────────────────────────────────────────────────
python3 -m venv .venv
source .venv/bin/activate
if command -v uv &>/dev/null; then
  uv pip install -q -r requirements.txt
else
  pip install -q -r requirements.txt
fi

# ── Build the boto3 layer (pure Python — no Docker needed) ───────────────────
# This layer overrides the Lambda runtime's boto3 to get SearchVectors (DynamoDB)
# and the S3 Vectors APIs, both added in boto3>=1.43.72.
echo "Building boto3>=1.43.72 Lambda layer..."
rm -rf layers/boto3-layer/python
mkdir -p layers/boto3-layer/python
pip install boto3==1.43.72 \
  -t layers/boto3-layer/python \
  --no-deps \
  --quiet
pip install botocore==1.43.72 s3transfer urllib3 jmespath python-dateutil six \
  -t layers/boto3-layer/python \
  --quiet
echo "Layer built."

# ── CDK bootstrap (idempotent — safe to run every time) ─────────────────────
cdk bootstrap "aws://$ACCOUNT/$REGION" --quiet

# ── Deploy ───────────────────────────────────────────────────────────────────
cdk deploy AgentMemoryVectorStoreStack \
  --require-approval never \
  --outputs-file cdk-outputs.json

# ── Write outputs to the demo's .env ─────────────────────────────────────────
python3 write_env.py cdk-outputs.json ..

echo ""
echo "Deploy complete. VECTOR_BUCKET, VECTOR_INDEX, DYNAMODB_TABLE written to ../.env"
