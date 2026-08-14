import aws_cdk as cdk
from infra import VectorStoreStack

app = cdk.App()

# env=None → account and region resolved from active AWS credentials at deploy time.
# No account IDs hardcoded anywhere in this app.
VectorStoreStack(app, "AgentMemoryVectorStoreStack")

app.synth()
