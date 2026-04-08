"""
WebSocket smoke test (no audio required).

Usage (PowerShell):
  $env:SESAME_ID_TOKEN="..."
  python examples/ws_smoke_test.py --character Miles
"""

import os
import sys
import json
import base64
import argparse
import logging
import time

from sesame_ai import SesameWebSocket


def _decode_jwt_payload(jwt: str) -> dict:
    # JWT = header.payload.signature (base64url). We only need the payload.
    parts = jwt.split(".")
    if len(parts) != 3:
        raise ValueError("Token does not look like a JWT (expected 3 dot-separated parts)")
    payload_b64 = parts[1]
    payload_b64 += "=" * ((4 - len(payload_b64) % 4) % 4)
    return json.loads(base64.urlsafe_b64decode(payload_b64.encode("utf-8")).decode("utf-8"))


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="SesameAI WebSocket smoke test")
    parser.add_argument("--character", default="Miles", help="Character (e.g., Miles, Maya)")
    parser.add_argument("--character-param", default=None, help="Override connect URL character query param")
    parser.add_argument("--client-name", default="Consumer-Web-App", help="Override connect URL client_name")
    parser.add_argument("--timezone", default="America/Chicago", help="Timezone for connect URL usercontext")
    parser.add_argument("--call-settings-json", default=None, help="Raw JSON for call_connect content.settings")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger("sesame").setLevel(logging.DEBUG)
        logging.getLogger("websocket").setLevel(logging.INFO)

    token = os.environ.get("SESAME_ID_TOKEN")
    if not token:
        print("Missing SESAME_ID_TOKEN in environment.", file=sys.stderr)
        return 2

    try:
        payload = _decode_jwt_payload(token)
        provider = (payload.get("firebase", {}) or {}).get("sign_in_provider") or payload.get("provider_id")
        email = payload.get("email")
        logging.info("Token provider=%s email=%s", provider, email)
    except Exception as e:
        logging.warning("Could not decode token payload (%s). Continuing anyway.", e)

    call_settings = None
    if args.call_settings_json:
        call_settings = json.loads(args.call_settings_json)

    ws = SesameWebSocket(
        id_token=token,
        character=args.character,
        character_param=args.character_param,
        client_name=args.client_name,
        usercontext={"timezone": args.timezone},
        call_settings=call_settings,
    )
    logging.info("Trying character_param=%s call_settings=%s", ws.character_param, ws.call_settings)
    ok = ws.connect(blocking=True)
    if ok and ws.is_connected():
        logging.info("WebSocket connected (session_id=%s call_id=%s)", ws.session_id, ws.call_id)
        ws.disconnect()
        return 0

    logging.error("WebSocket did not connect (check logs above).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

