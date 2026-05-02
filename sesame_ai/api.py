# sesame_ai/api.py

import requests
from .config import get_headers, get_params, get_endpoint_url
from .models import RefreshTokenResponse
from .exceptions import APIError, InvalidTokenError, NetworkError


class SesameAI:
    """Firebase auth client for SesameAI — token refresh only."""

    def __init__(self, api_key=None):
        self.api_key = api_key

    def _make_auth_request(self, request_type, payload, is_form_data=False):
        headers = get_headers(request_type)
        if is_form_data:
            headers = {k: v for k, v in headers.items() if k != 'content-type'}
        params = get_params(request_type, self.api_key)
        url = get_endpoint_url(request_type)

        try:
            if is_form_data:
                response = requests.post(url, params=params, headers=headers, data=payload)
            else:
                response = requests.post(url, params=params, headers=headers, json=payload)

            response.raise_for_status()
            response_json = response.json()

            if 'error' in response_json:
                self._handle_api_error(response_json['error'])

            return response_json

        except requests.exceptions.HTTPError as e:
            detail = ""
            try:
                detail = e.response.text[:500]
            except Exception:
                pass
            raise NetworkError(f"{e}\nResponse: {detail}")
        except requests.exceptions.RequestException as e:
            raise NetworkError(f"Network error: {str(e)}")

    def _handle_api_error(self, error):
        error_code = error.get('code', 400)
        error_message = error.get('message', 'Unknown error')
        error_details = error.get('errors', [])

        if error_message in ('INVALID_ID_TOKEN', 'INVALID_REFRESH_TOKEN'):
            raise InvalidTokenError()

        raise APIError(error_code, error_message, error_details)

    def refresh_authentication_token(self, refresh_token):
        """Refresh an ID token using a refresh token."""
        payload = {
            'grant_type': 'refresh_token',
            'refresh_token': refresh_token,
        }
        response_json = self._make_auth_request('refresh', payload, is_form_data=True)
        return RefreshTokenResponse(response_json)
