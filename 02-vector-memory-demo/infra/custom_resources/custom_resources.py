import aws_cdk as cdk
from aws_cdk import CustomResource, aws_lambda as lambda_
from aws_cdk import custom_resources as cr
from constructs import Construct


class VectorStoreResources(Construct):
    """Provisions S3 Vectors bucket+index and DynamoDB table+vector-index.

    Uses a Lambda-backed Custom Resource because both services were added to boto3
    after their native CloudFormation support — the Lambda bundles boto3>=1.43.72.
    """

    def __init__(self, scope: Construct, construct_id: str,
                 cr_handler: lambda_.Function,
                 vector_bucket_name: str,
                 vector_index_name: str,
                 dynamodb_table_name: str,
                 dynamodb_vector_index: str,
                 embed_dims: int = 1024,
                 **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        provider = cr.Provider(self, "Provider", on_event_handler=cr_handler)

        # 1. S3 Vectors bucket
        s3v_bucket = CustomResource(
            self, "S3VBucket",
            service_token=provider.service_token,
            properties={
                "ResourceType": "S3VectorBucket",
                "BucketName": vector_bucket_name,
            },
        )

        # 2. S3 Vectors index (depends on bucket)
        s3v_index = CustomResource(
            self, "S3VIndex",
            service_token=provider.service_token,
            properties={
                "ResourceType": "S3VectorIndex",
                "BucketName": vector_bucket_name,
                "IndexName": vector_index_name,
                "Dimensions": str(embed_dims),
                "DistanceMetric": "cosine",
            },
        )
        s3v_index.node.add_dependency(s3v_bucket)

        # 3. DynamoDB table with vector index
        ddb_table = CustomResource(
            self, "DDBTable",
            service_token=provider.service_token,
            properties={
                "ResourceType": "DynamoDBVectorTable",
                "TableName": dynamodb_table_name,
                "VectorIndexName": dynamodb_vector_index,
                "VectorAttribute": "embedding",
                "Dimensions": str(embed_dims),
            },
        )

        self.vector_bucket_name = s3v_bucket.get_att_string("BucketName")
        self.dynamodb_table_name = ddb_table.get_att_string("TableName")
