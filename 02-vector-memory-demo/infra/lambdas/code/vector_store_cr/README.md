# vector_store_cr

CloudFormation Custom Resource handler for Demo 02 AWS infrastructure.

## Trigger
CloudFormation Custom Resource lifecycle events (Create / Update / Delete).
Invoked by the `VectorStoreStack` during `cdk deploy` and `cdk destroy`.

## Input
`ResourceProperties.ResourceType` selects the resource:

| ResourceType | Creates |
|---|---|
| `S3VectorBucket` | S3 Vectors bucket (new service, no native CFn support) |
| `S3VectorIndex` | S3 Vectors index inside the bucket |
| `DynamoDBVectorTable` | DynamoDB table with a native vector index |

## Output
`Data` dict with the created resource's name/ARN (used as CDK stack outputs).

## Environment variables
| Variable | Set in | Purpose |
|---|---|---|
| `AWS_DEFAULT_REGION` | Lambda runtime | Region for boto3 clients |

## Permissions
- `s3vectors:CreateVectorBucket`, `GetVectorBucket`, `DeleteVectorBucket`, `ListIndexes`
- `s3vectors:CreateIndex`, `GetIndex`, `DeleteIndex`
- `dynamodb:CreateTable`, `DescribeTable`, `DeleteTable`

## Layers / dependencies
boto3-layer (boto3>=1.43.72) — required for S3 Vectors and DynamoDB Vector Search APIs.
