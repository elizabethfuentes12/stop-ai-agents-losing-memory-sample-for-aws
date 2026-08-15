import aws_cdk as cdk
from aws_cdk import Stack, CfnOutput, aws_lambda as lambda_
from constructs import Construct
from lambdas import VectorStoreLambdas
from custom_resources import VectorStoreResources


class VectorStoreStack(Stack):
    """Stack for Demo 02 — S3 Vectors bucket+index and DynamoDB table+vector-index.

    Both services use Lambda-backed Custom Resources because native CloudFormation
    support was not available when these APIs were introduced (mid-2025).
    The Lambda bundles boto3>=1.43.72 via a pre-built layer.

    Account and region are resolved at deploy time from the active credentials.
    """

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        stk = Stack.of(self)

        # boto3 layer built by deploy.sh before cdk deploy
        boto3_layer = lambda_.LayerVersion(
            self, "Boto3Layer",
            code=lambda_.Code.from_asset("layers/boto3-layer"),
            compatible_runtimes=[lambda_.Runtime.PYTHON_3_12],
            compatible_architectures=[lambda_.Architecture.ARM_64],
            description="boto3>=1.43.72 — S3 Vectors and DynamoDB Vector Search",
        )

        lambdas = VectorStoreLambdas(self, "Lambdas", boto3_layer=boto3_layer)

        resources = VectorStoreResources(
            self, "VectorStore",
            cr_handler=lambdas.cr_handler,
            # No account ID hardcoded — CDK Fn.sub resolves ${AWS::AccountId} at deploy time.
            vector_bucket_name=cdk.Fn.sub("agent-memory-demo-vectors-${AWS::AccountId}"),
            vector_index_name="traveler-memories",
            dynamodb_table_name="agent-memory-demo-ddb",
            dynamodb_vector_index="memory-vector-index",
            embed_dims=1024,
        )

        CfnOutput(self, "VectorBucketName",
                  value=resources.vector_bucket_name,
                  description="Set as VECTOR_BUCKET in the demo .env file")

        CfnOutput(self, "VectorIndexName",
                  value="traveler-memories",
                  description="Set as VECTOR_INDEX in the demo .env file")

        CfnOutput(self, "DynamoDBTableName",
                  value=resources.dynamodb_table_name,
                  description="Set as DYNAMODB_TABLE in the demo .env file")
