"""GitHub Copilot authentication module using Pydantic models."""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
from loguru import logger
from pydantic import BaseModel, Field, RootModel, field_validator


def get_base_headers() -> Dict[str, str]:
    """Get base headers for GitHub API requests."""
    return {
        "Copilot-Integration-Id": "vscode-chat",
        "Editor-Version": "1.101.2",
    }


# Pydantic models for configuration
class AppEntry(BaseModel):
    """Application entry with OAuth token."""

    oauth_token: str


class AppsConfig(RootModel[Dict[str, AppEntry]]):
    """GitHub Copilot apps configuration."""

    @classmethod
    def from_dict(cls, data: Dict[str, Dict[str, str]]) -> "AppsConfig":
        """Create AppsConfig from raw dictionary data."""
        apps = {k: AppEntry(**v) for k, v in data.items()}
        return cls(root=apps)

    def get_first_oauth_token(self) -> str:
        """Get the first OAuth token from the configuration."""
        if not self.root:
            raise ValueError("No OAuth tokens found in configuration")
        first_entry = next(iter(self.root.values()))
        return first_entry.oauth_token


class TokenEndpoints(BaseModel):
    """Token endpoints from GitHub API response."""

    api: Optional[str] = None


class TokenResponse(BaseModel):
    """GitHub Copilot token response."""

    token: str
    expires_at: Optional[int] = None
    endpoints: Optional[TokenEndpoints] = None


class CopilotToken(BaseModel):
    """Copilot token with expiration tracking."""

    base_url_api: str
    token: str
    expires_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc) + timedelta(hours=1)
    )

    @field_validator("expires_at", mode="before")
    @classmethod
    def parse_expires_at(cls, v):
        """Parse expires_at from timestamp or datetime."""
        if isinstance(v, int):
            return datetime.fromtimestamp(v, tz=timezone.utc)
        if isinstance(v, datetime):
            if v.tzinfo is None:
                return v.replace(tzinfo=timezone.utc)
            return v
        return v

    def is_expired(self) -> bool:
        """Check if the token is expired."""
        return datetime.now(timezone.utc) >= self.expires_at

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}


# Constants
GITHUB_TOKEN_URL = "https://api.github.com/copilot_internal/v2/token"
CONFIG_PATH = Path.home() / ".config/github-copilot/apps.json"


async def get_oauth_token() -> str:
    """Get OAuth token from GitHub Copilot configuration file."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {CONFIG_PATH}. Please ensure GitHub Copilot is configured in VS Code."
        )

    try:
        with open(CONFIG_PATH, "r") as f:
            data = json.load(f)

        config = AppsConfig.from_dict(data)
        return config.get_first_oauth_token()

    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(f"Failed to parse config file: {e}")


async def fetch_copilot_token(oauth_token: str) -> CopilotToken:
    """Fetch Copilot token from GitHub API using OAuth token."""
    base_headers = get_base_headers()
    headers = {
        "Authorization": f"Bearer {oauth_token}",
        **base_headers,
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(GITHUB_TOKEN_URL, headers=headers)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            error_body = (
                await response.atext() if hasattr(response, "atext") else str(e)
            )
            raise ValueError(
                f"Failed to fetch Copilot token: {e.response.status_code} - {error_body}"
            )
        except httpx.RequestError as e:
            raise ValueError(f"Request error while fetching Copilot token: {e}")

    try:
        token_data = TokenResponse.model_validate(response.json())
    except Exception as e:
        raise ValueError(
            f"Failed to parse token response: {e}. Response: {response.text}"
        )

    # Create CopilotToken with expiration
    expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
    if token_data.expires_at:
        expires_at = datetime.fromtimestamp(token_data.expires_at, tz=timezone.utc)

    return CopilotToken(
        base_url_api=token_data.endpoints.api,
        token=token_data.token,
        expires_at=expires_at,
    )


async def test_token(token: str) -> bool:
    """Test a Copilot token with a simple API call."""
    test_url = "https://api.individual.githubcopilot.com/models"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        **base_headers,
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(test_url, headers=headers)
            return response.status_code == 200
        except Exception:
            return False


# Example usage and testing
if __name__ == "__main__":

    async def main():
        try:
            # Test OAuth token retrieval
            oauth_token = await get_oauth_token()
            print(f"✅ OAuth token retrieved (length: {len(oauth_token)})")

            # Test Copilot token fetching
            copilot_token = await fetch_copilot_token(oauth_token)
            print(f"✅ Copilot token retrieved, expires at: {copilot_token.expires_at}")
            print(f"✅ Token is expired: {copilot_token.is_expired()}")

            # Test token validation
            is_valid = await test_token(copilot_token.token)
            print(f"✅ Token validation: {'PASS' if is_valid else 'FAIL'}")

        except Exception as e:
            logger.exception(f"Error: {e}")

    asyncio.run(main())
