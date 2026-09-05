from aws_cdk import RemovalPolicy, aws_s3 as s3
from constructs import Construct


class SessionsBucket(Construct):
    """Private S3 bucket for Strands S3SessionManager.

    The bucket name is auto-generated (unique per account/region).
    No account ID is hardcoded, CDK emits it as a stack output.
    """

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.bucket = s3.Bucket(
            self, "Bucket",
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            versioned=False,
        )
