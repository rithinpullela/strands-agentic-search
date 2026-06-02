"""Tests for .env-driven configuration and MCP env passthrough.

These construct config dataclasses directly (their field factories read
``os.environ`` live) rather than reloading the module, so they don't re-run
``load_dotenv`` and stay independent of any developer ``.env`` on disk.
"""

from __future__ import annotations

from strands_agentic_search.config import LLMConfig, MCPConfig, ServerConfig

_AUTH_VARS = (
    "OPENSEARCH_USERNAME", "OPENSEARCH_PASSWORD", "OPENSEARCH_NO_AUTH",
    "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
    "AWS_REGION", "AWS_PROFILE", "AWS_IAM_ARN", "AWS_OPENSEARCH_SERVERLESS",
)


def _clear_auth(monkeypatch):
    for k in _AUTH_VARS:
        monkeypatch.delenv(k, raising=False)


def test_env_var_overrides_default(monkeypatch):
    monkeypatch.setenv("PORT", "9999")
    monkeypatch.setenv("BEDROCK_MODEL_ID", "test-model")
    assert ServerConfig().port == 9999
    assert LLMConfig().model_id == "test-model"


def test_server_env_defaults_to_no_auth(monkeypatch):
    _clear_auth(monkeypatch)
    env = MCPConfig().server_env
    assert env["OPENSEARCH_NO_AUTH"] == "true"
    assert env["OPENSEARCH_URL"] == "http://localhost:9200"


def test_server_env_forwards_basic_auth(monkeypatch):
    _clear_auth(monkeypatch)
    monkeypatch.setenv("OPENSEARCH_USERNAME", "admin")
    monkeypatch.setenv("OPENSEARCH_PASSWORD", "secret")
    env = MCPConfig().server_env
    assert env["OPENSEARCH_USERNAME"] == "admin"
    assert env["OPENSEARCH_PASSWORD"] == "secret"
    # basic auth present → do not force no-auth
    assert "OPENSEARCH_NO_AUTH" not in env


def test_server_env_forwards_aws_creds(monkeypatch):
    _clear_auth(monkeypatch)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "shh")
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    monkeypatch.setenv("AWS_OPENSEARCH_SERVERLESS", "true")
    env = MCPConfig().server_env
    assert env["AWS_ACCESS_KEY_ID"] == "AKIA"
    assert env["AWS_REGION"] == "us-west-2"
    assert env["AWS_OPENSEARCH_SERVERLESS"] == "true"
    # AWS auth present → do not force no-auth
    assert "OPENSEARCH_NO_AUTH" not in env


def test_explicit_no_auth_is_preserved(monkeypatch):
    _clear_auth(monkeypatch)
    monkeypatch.setenv("OPENSEARCH_NO_AUTH", "true")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "shh")
    env = MCPConfig().server_env
    assert env["OPENSEARCH_NO_AUTH"] == "true"
