"""Build a boto3 S3 client for the tytonai S3-compatible endpoint from .env config.

All download features share this client. Credentials/config come from environment
variables (loaded via python-dotenv) — see .env.example for the expected keys.
"""

from __future__ import annotations

import os

import boto3
from botocore.client import BaseClient
from botocore.config import Config
from dotenv import load_dotenv


def _endpoint_url(host: str, https: str) -> str:
    """Turn an endpoint host + an AWS_HTTPS flag into a full URL."""
    scheme = "https" if https.strip().upper() in {"YES", "TRUE", "1"} else "http"
    return f"{scheme}://{host}"


def make_s3_client(env: dict[str, str] | None = None) -> BaseClient:
    """Return a boto3 S3 client configured for the tytonai endpoint.

    Pass `env` explicitly to stay pure/testable; defaults to os.environ.
    Path-style addressing is forced when AWS_VIRTUAL_HOSTING is FALSE.
    """
    env = dict(os.environ if env is None else env)
    virtual = env.get("AWS_VIRTUAL_HOSTING", "FALSE").strip().upper() in {"YES", "TRUE", "1"}
    config = Config(
        signature_version="s3v4",
        s3={"addressing_style": "virtual" if virtual else "path"},
    )
    return boto3.client(
        "s3",
        endpoint_url=_endpoint_url(env["AWS_S3_ENDPOINT"], env.get("AWS_HTTPS", "YES")),
        region_name=env.get("AWS_REGION", "us-east-1"),
        aws_access_key_id=env.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=env.get("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=env.get("AWS_SESSION_TOKEN") or None,
        config=config,
    )


# ════════════════════════════════════════════════════════════════════════════
#  RUN — load .env, build the client, smoke-test connectivity (Shift+Enter).
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    load_dotenv()  # reads .env from the current working dir

    s3 = make_s3_client()
    print("endpoint:", s3.meta.endpoint_url)

    bucket = os.environ["S3_FILE_BUCKET"]            # the R&D bucket from .env
    resp = s3.list_objects_v2(Bucket=bucket, MaxKeys=5)
    for obj in resp.get("Contents", []):
        print(obj["Key"], obj["Size"])
