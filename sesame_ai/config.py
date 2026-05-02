# sesame_ai/config.py

import json
import base64
from datetime import datetime

DEFAULT_API_KEY = "AIzaSyDtC7Uwb5pGAsdmrH2T4Gqdk5Mga07jYPM"

FIREBASE_TOKEN_URL = "https://securetoken.googleapis.com/v1/token"


def _get_firebase_client_header():
    client = {
        "version": 2,
        "heartbeats": [{
            "agent": "fire-core/0.11.1 fire-core-esm2017/0.11.1 fire-js/ fire-js-all-app/11.3.1 fire-auth/1.9.0 fire-auth-esm2017/1.9.0",
            "dates": [datetime.now().strftime("%Y-%m-%d")],
        }],
    }
    return base64.b64encode(
        json.dumps(client, separators=(",", ":")).encode()
    ).decode()


def get_headers(_request_type=None):
    return {
        'accept': '*/*',
        'accept-language': 'en-US,en;q=0.9',
        'content-type': 'application/json',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36',
        'x-firebase-client': _get_firebase_client_header(),
        'x-client-data': 'COKQywE=',
        'x-client-version': 'Chrome/JsCore/11.3.1/FirebaseCore-web',
        'x-firebase-gmpid': '1:1072000975600:web:75b0bf3a9bb8d92e767835',
    }


def get_params(_request_type=None, api_key=None):
    key = api_key if api_key else DEFAULT_API_KEY
    return {'key': key}


def get_endpoint_url(_request_type=None):
    return FIREBASE_TOKEN_URL
