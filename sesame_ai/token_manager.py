# sesame_ai/token_manager.py

import os
import time
import logging

from .api import SesameAI

logger = logging.getLogger('sesame.token_manager')


class TokenManager:
    """Manages authentication: reads SESAME_REFRESH_TOKEN from .env, refreshes."""

    def __init__(self, api_client=None, token_file=None):
        self.api_client = api_client if api_client else SesameAI()
        self.token_file = token_file
        self._cached_id_token: str | None = None
        self._expires_at: float = 0.0

    def _read_dotenv(self):
        """Parse .env from the current working directory."""
        env_path = os.path.join(os.getcwd(), '.env')
        if not os.path.isfile(env_path):
            return {}
        result = {}
        try:
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if '=' not in line:
                        continue
                    key, _, value = line.partition('=')
                    key = key.strip()
                    value = value.strip()
                    if (value.startswith('"') and value.endswith('"')) or \
                       (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]
                    result[key] = value
        except OSError:
            pass
        return result

    def _get_config(self, key, default=None):
        value = os.environ.get(key)
        if value is not None:
            return value
        return self._read_dotenv().get(key, default)

    def get_valid_token(self):
        """Return a valid ID token, refreshing from SESAME_REFRESH_TOKEN."""
        if self._cached_id_token and time.time() < self._expires_at - 300:
            return self._cached_id_token

        refresh_token = self._get_config("SESAME_REFRESH_TOKEN")
        if not refresh_token:
            raise RuntimeError(
                "SESAME_REFRESH_TOKEN not set. "
                "Add it to your .env file or environment."
            )

        logger.info("Refreshing ID token using SESAME_REFRESH_TOKEN")
        resp = self.api_client.refresh_authentication_token(refresh_token)
        self._cached_id_token = resp.id_token
        self._expires_at = time.time() + int(resp.expires_in or 3600)
        return self._cached_id_token
