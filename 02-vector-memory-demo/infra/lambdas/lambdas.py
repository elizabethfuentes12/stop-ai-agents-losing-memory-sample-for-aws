from aws_cdk import Duration, aws_lambda as lambda_, aws_iam as iam
from constructs import Construct


class VectorStoreLambdas(Construct):
    """Custom Resource handler Lambda for S3 Vectors and DynamoDB Vector APIs.

    boto3>=1.43.72 is bundled as a layer — those APIs aren't in older runtime versions.
    """

    def __init__(self, scope: Construct, construct_id: str,
                 boto3_layer: lambda_.LayerVersion, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.cr_handler = lambda_.Function(
            self, "CRHandler",
            description="Custom Resource handler for S3 Vectors and DynamoDB vector tables",
            runtime=lambda_.Runtime.PYTHON_3_12,
            architecture=lambda_.Architecture.ARM_64,
            handler="lambda_function.handler",
            code=lambda_.Code.from_asset("lambdas/code/vector_store_cr"),
            layers=[boto3_layer],
            timeout=Duration.minutes(5),
            tracing=lambda_.Tracing.ACTIVE,
        )

        # S3 Vectors permissions
        self.cr_handler.add_to_role_policy(iam.PolicyStatement(
            actions=[
                "s3vectors:CreateVectorBucket",
                "s3vectors:GetVectorBucket",
                "s3vectors:DeleteVectorBucket",
                "s3vectors:ListIndexes",
                "s3vectors:CreateIndex",
                "s3vectors:GetIndex",
                "s3vectors:DeleteIndex",
            ],
            resources=["*"],
        ))

        # DynamoDB table management
        self.cr_handler.add_to_role_policy(iam.PolicyStatement(
            actions=[
                "dynamodb:CreateTable",
                "dynamodb:DescribeTable",
                "dynamodb:DeleteTable",
            ],
            resources=["*"],
        ))
