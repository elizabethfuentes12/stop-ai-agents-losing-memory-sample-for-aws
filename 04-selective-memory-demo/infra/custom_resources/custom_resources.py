import aws_cdk as cdk
from aws_cdk import CustomResource, aws_lambda as lambda_
from aws_cdk import custom_resources as cr
from constructs import Construct

# Demo 04 extractor uses 4 S3V indexes (one per memory type).
EXTRACTOR_INDEXES = ["selective-facts", "selective-prefs", "selective-summary", "selective-episodes"]


class AgentCoreResources(Construct):
    """Provisions S3 Vectors indexes (extractor) and AgentCore Memory (Demo 04).

    Uses Lambda-backed Custom Resources because neither service has native
    CloudFormation support. The Lambda bundles boto3>=1.43.72.
    """

    def __init__(self, scope: Construct, construct_id: str,
                 cr_handler: lambda_.Function,
                 vector_bucket_name: str,
                 memory_name: str,
                 embed_dims: int = 1024,
                 **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        provider = cr.Provider(self, "Provider", on_event_handler=cr_handler)

        # S3 Vectors bucket (shared across all indexes)
        s3v_bucket = CustomResource(
            self, "S3VBucket",
            service_token=provider.service_token,
            properties={
                "ResourceType": "S3VectorBucket",
                "BucketName": vector_bucket_name,
            },
        )

        # One S3 Vectors index per memory type
        for idx_name in EXTRACTOR_INDEXES:
            logical_id = "".join(w.capitalize() for w in idx_name.replace("-", " ").split())
            idx_cr = CustomResource(
                self, f"S3VIdx{logical_id}",
                service_token=provider.service_token,
                properties={
                    "ResourceType": "S3VectorIndex",
                    "BucketName": vector_bucket_name,
                    "IndexName": idx_name,
                    "Dimensions": str(embed_dims),
                },
            )
            idx_cr.node.add_dependency(s3v_bucket)

        # AgentCore Memory with 4 built-in strategies
        agentcore = CustomResource(
            self, "AgentCoreMemory",
            service_token=provider.service_token,
            properties={
                "ResourceType": "AgentCoreMemory",
                "MemoryName": memory_name,
            },
        )

        self.vector_bucket_name = s3v_bucket.get_att_string("BucketName")
        self.agentcore_memory_id = agentcore.get_att_string("MemoryId")
        self.agentcore_memory_name = memory_name
