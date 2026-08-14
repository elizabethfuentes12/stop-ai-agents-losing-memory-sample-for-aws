"""Parse CDK outputs.json and write resource names to the demo's .env file."""
import json
import pathlib
import sys


def set_env_key(env_path: pathlib.Path, key: str, value: str) -> None:
    lines = env_path.read_text().splitlines() if env_path.exists() else []
    updated = False
    new_lines = []
    for line in lines:
        if line.startswith(f"{key}=") or line.startswith(f"# {key}="):
            new_lines.append(f"{key}={value}")
            updated = True
        else:
            new_lines.append(line)
    if not updated:
        new_lines.append(f"{key}={value}")
    env_path.write_text("\n".join(new_lines) + "\n")


outputs_path = pathlib.Path(sys.argv[1])
demo_root = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else pathlib.Path(__file__).parent.parent

outputs = json.loads(outputs_path.read_text())

for stack_outputs in outputs.values():
    env_file = demo_root / ".env"
    if "VectorBucketName" in stack_outputs:
        set_env_key(env_file, "VECTOR_BUCKET", stack_outputs["VectorBucketName"])
        print(f"  VECTOR_BUCKET={stack_outputs['VectorBucketName']}")
    if "VectorIndexName" in stack_outputs:
        set_env_key(env_file, "VECTOR_INDEX", stack_outputs["VectorIndexName"])
        print(f"  VECTOR_INDEX={stack_outputs['VectorIndexName']}")
    if "DynamoDBTableName" in stack_outputs:
        set_env_key(env_file, "DYNAMODB_TABLE", stack_outputs["DynamoDBTableName"])
        print(f"  DYNAMODB_TABLE={stack_outputs['DynamoDBTableName']}")

print(f"Done — {demo_root}/.env updated.")
