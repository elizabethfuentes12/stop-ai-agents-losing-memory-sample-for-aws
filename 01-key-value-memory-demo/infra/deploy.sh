#!/usr/bin/env bash
# deploy.sh — deploys the AgentMemorySessionsStack for Demo 01.
#
# What it creates:
#   - Private S3 bucket for Strands S3SessionManager (Test 4)
#
# After deploy, SESSIONS_BUCKET is written to ../.env automatically.
# Run `cdk destroy` from this folder to tear everything down cleanly.
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
echo "  Demo 01 — Sessions bucket deploy"
echo "  Account : $ACCOUNT"
echo "  Region  : $REGION"
echo "=========================================="
echo "Press Ctrl+C within 5 s to cancel."
sleep 5

# ── Python venv + deps ───────────────────────────────────────────────────────
python3 -m venv .venv
source .venv/bin/activate
if command -v uv &>/dev/null; then
  uv pip install -q -r requirements.txt
else
  pip install -q -r requirements.txt
fi

# ── CDK bootstrap (idempotent — safe to run every time) ─────────────────────
cdk bootstrap "aws://$ACCOUNT/$REGION" --quiet

# ── Deploy ───────────────────────────────────────────────────────────────────
cdk deploy AgentMemorySessionsStack \
  --require-approval never \
  --outputs-file cdk-outputs.json

# ── Write outputs to the demo's .env ─────────────────────────────────────────
python3 write_env.py cdk-outputs.json ..

echo ""
echo "Deploy complete. SESSIONS_BUCKET written to ../. env"
