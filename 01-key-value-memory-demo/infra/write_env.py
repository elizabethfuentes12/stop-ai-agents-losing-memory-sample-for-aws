"""Parse CDK outputs.json and write the bucket name to the demo's .env file."""
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
    if "SessionsBucketName" in stack_outputs:
        env_file = demo_root / ".env"
        set_env_key(env_file, "SESSIONS_BUCKET", stack_outputs["SessionsBucketName"])
        print(f"  {env_file}: SESSIONS_BUCKET={stack_outputs['SessionsBucketName']}")

print("Done — .env updated.")
