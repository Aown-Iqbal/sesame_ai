# sesame_ai/__init__.py

from .api import SesameAI
from .websocket import SesameWebSocket
from .exceptions import SesameAIError, AuthenticationError, APIError, InvalidTokenError, NetworkError
from .models import RefreshTokenResponse
from .token_manager import TokenManager

__version__ = "0.2.0"
__author__ = "ijub"
__license__ = "MIT"

__all__ = [
    'SesameAI',
    'SesameWebSocket',
    'TokenManager',
    'SesameAIError',
    'AuthenticationError',
    'APIError',
    'InvalidTokenError',
    'NetworkError',
    'RefreshTokenResponse',
]
