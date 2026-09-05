import aws_cdk as cdk
from aws_cdk import Stack, CfnOutput
from constructs import Construct
from buckets import SessionsBucket


class SessionsStack(Stack):
    """Stack for Demo 01, creates the S3 bucket used by S3SessionManager (Test 4).

    Account ID is resolved at deploy time from the current credentials.
    Run deploy.sh to deploy; the bucket name is written to ../.env automatically.
    """

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        sessions = SessionsBucket(self, "Sessions")

        CfnOutput(self, "SessionsBucketName",
                  value=sessions.bucket.bucket_name,
                  description="Set this as SESSIONS_BUCKET in the demo .env file")

        CfnOutput(self, "SessionsBucketArn",
                  value=sessions.bucket.bucket_arn)
