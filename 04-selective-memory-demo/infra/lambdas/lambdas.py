from aws_cdk import Duration, aws_lambda as lambda_, aws_iam as iam
from constructs import Construct


class AgentCoreLambdas(Construct):
    """Custom Resource handler Lambda for AgentCore Memory and S3 Vectors (Demo 04).

    Timeout is 9 minutes because AgentCore Memory creation is async and can
    take several minutes to become ACTIVE. boto3>=1.43.72 bundled as a layer.
    """

    def __init__(self, scope: Construct, construct_id: str,
                 boto3_layer: lambda_.LayerVersion, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.cr_handler = lambda_.Function(
            self, "CRHandler",
            description="Custom Resource handler for AgentCore Memory and S3 Vectors (Demo 04)",
            runtime=lambda_.Runtime.PYTHON_3_12,
            architecture=lambda_.Architecture.ARM_64,
            handler="lambda_function.handler",
            code=lambda_.Code.from_asset("lambdas/code/agentcore_cr"),
            layers=[boto3_layer],
            timeout=Duration.minutes(9),
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

        # AgentCore Memory control-plane permissions
        self.cr_handler.add_to_role_policy(iam.PolicyStatement(
            actions=[
                "bedrock-agentcore:CreateMemory",
                "bedrock-agentcore:GetMemory",
                "bedrock-agentcore:ListMemories",
                "bedrock-agentcore:DeleteMemory",
            ],
            resources=["*"],
        ))
