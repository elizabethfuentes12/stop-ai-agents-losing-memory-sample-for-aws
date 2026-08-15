# agentcore_cr

CloudFormation Custom Resource handler for Demo 04 AWS infrastructure.

## Trigger
CloudFormation Custom Resource lifecycle events (Create / Update / Delete).
Invoked by the `AgentCoreStack` during `cdk deploy` and `cdk destroy`.

## Input
`ResourceProperties.ResourceType` selects the resource:

| ResourceType | Creates |
|---|---|
| `S3VectorBucket` | S3 Vectors bucket |
| `S3VectorIndex` | S3 Vectors index (one per memory type: facts, prefs, summary, episodes) |
| `AgentCoreMemory` | AgentCore Memory with 4 built-in strategies (semantic, userPreference, summary, episodic) |

## Output
`Data` dict with resource name/ID (used as CDK stack outputs).

## Environment variables
| Variable | Set in | Purpose |
|---|---|---|
| `AWS_DEFAULT_REGION` | Lambda runtime | Region for boto3 clients |

## Permissions
- `s3vectors:CreateVectorBucket`, `GetVectorBucket`, `DeleteVectorBucket`, `ListIndexes`
- `s3vectors:CreateIndex`, `GetIndex`, `DeleteIndex`
- `bedrock-agentcore-control:CreateMemory`, `GetMemory`, `ListMemories`, `DeleteMemory`

## Layers / dependencies
boto3-layer (boto3>=1.43.72) — required for S3 Vectors and AgentCore APIs.

## Timeout
9 minutes — AgentCore Memory creation is asynchronous and can take several minutes
to become ACTIVE. The Lambda polls until ACTIVE or timeout.
