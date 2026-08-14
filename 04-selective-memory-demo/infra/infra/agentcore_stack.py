import aws_cdk as cdk
from aws_cdk import Stack, CfnOutput, aws_lambda as lambda_
from constructs import Construct
from lambdas import AgentCoreLambdas
from custom_resources import AgentCoreResources


class AgentCoreStack(Stack):
    """Stack for Demo 04 — AgentCore Memory (4 strategies) + S3 Vectors extractor indexes.

    Both services use Lambda-backed Custom Resources (no native CloudFormation support).
    The Lambda bundles boto3>=1.43.72 via a pre-built layer.

    Account and region are resolved at deploy time from the active credentials.
    """

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # boto3 layer built by deploy.sh before cdk deploy
        boto3_layer = lambda_.LayerVersion(
            self, "Boto3Layer",
            code=lambda_.Code.from_asset("layers/boto3-layer"),
            compatible_runtimes=[lambda_.Runtime.PYTHON_3_12],
            compatible_architectures=[lambda_.Architecture.ARM_64],
            description="boto3>=1.43.72 — S3 Vectors and AgentCore Memory APIs",
        )

        lambdas = AgentCoreLambdas(self, "Lambdas", boto3_layer=boto3_layer)

        resources = AgentCoreResources(
            self, "AgentCore",
            cr_handler=lambdas.cr_handler,
            # No account ID hardcoded — CDK Fn.sub resolves ${AWS::AccountId} at deploy time.
            vector_bucket_name=cdk.Fn.sub("agent-memory-demo-vectors-${AWS::AccountId}"),
            memory_name="SelectiveMemoryDemo",
            embed_dims=1024,
        )

        CfnOutput(self, "VectorBucketName",
                  value=resources.vector_bucket_name,
                  description="Set as VECTOR_BUCKET in the demo .env file")

        CfnOutput(self, "AgentCoreMemoryName",
                  value=resources.agentcore_memory_name,
                  description="Set as AGENTCORE_MEMORY_NAME in the demo .env file")

        CfnOutput(self, "AgentCoreMemoryId",
                  value=resources.agentcore_memory_id,
                  description="The full AgentCore memory ID (includes the suffix)")
