# sesame_ai/websocket.py

import json
import base64
import uuid
import ssl
import urllib.parse
import threading
import queue
import time
import logging
import websocket as websocket_module

logger = logging.getLogger('sesame.websocket')


class SesameWebSocket:
    """WebSocket client for real-time audio communication with SesameAI."""

    def __init__(
        self,
        id_token,
        character="Miles",
        client_name="Consumer-Web-App",
        usercontext=None,
        client_sample_rate=16000,
    ):
        self.id_token = id_token
        self.character = character
        self.client_name = client_name
        self.usercontext = usercontext or {"timezone": "America/Chicago"}
        self.client_sample_rate = int(client_sample_rate)

        self.ws = None
        self.session_id = None
        self.call_id = None

        self.server_sample_rate = 24000
        self.audio_codec = "none"

        self.reconnect = False
        self.is_private = False
        self.user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36"
        )

        self.audio_buffer = queue.Queue(maxsize=1000)
        self.last_error = None

        self.connected_event = threading.Event()

        self.on_connect_callback = None
        self.on_disconnect_callback = None
        self.on_raw_message_callback = None

    # -- public API --

    def connect(self, blocking=True):
        self.connected_event.clear()
        t = threading.Thread(target=self._connect_websocket, daemon=True)
        t.start()
        if blocking:
            return self.connected_event.wait(timeout=10)
        return True

    def disconnect(self):
        if not self.session_id or not self.call_id:
            return False
        message = {
            "type": "call_disconnect",
            "session_id": self.session_id,
            "call_id": self.call_id,
            "request_id": str(uuid.uuid4()),
            "content": {"reason": "user_request"},
        }
        self._send_message(message)
        return True

    def send_audio_data(self, raw_audio_bytes):
        if not self.session_id or not self.call_id:
            return False
        encoded = base64.b64encode(raw_audio_bytes).decode('utf-8')
        self._send_audio(encoded)
        return True

    def get_next_audio_chunk(self, timeout=None):
        try:
            return self.audio_buffer.get(timeout=timeout)
        except queue.Empty:
            return None

    def ping(self):
        """Send an application-level ping to keep the session alive."""
        if self.session_id is not None and self.call_id is not None:
            self._send_ping()

    def is_connected(self):
        return self.session_id is not None and self.call_id is not None

    def set_connect_callback(self, callback):
        self.on_connect_callback = callback

    def set_disconnect_callback(self, callback):
        self.on_disconnect_callback = callback

    def set_raw_message_callback(self, callback):
        self.on_raw_message_callback = callback

    # -- internal: connection --

    def _connect_websocket(self):
        headers = {
            'Origin': 'https://www.sesame.com',
            'User-Agent': self.user_agent,
        }
        params = {
            'id_token': self.id_token,
            'client_name': self.client_name,
            'usercontext': json.dumps(self.usercontext),
            'character': self.character,
        }
        base_url = 'wss://sesameai.app/agent-service-0/v1/connect'
        query_string = '&'.join(
            f"{key}={urllib.parse.quote(value)}" for key, value in params.items()
        )
        ws_url = f"{base_url}?{query_string}"

        self.ws = websocket_module.WebSocketApp(
            ws_url,
            header=headers,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self.ws.run_forever(
            sslopt={"cert_reqs": ssl.CERT_NONE},
            skip_utf8_validation=True,
        )

    # -- internal: WS event handlers --

    def _on_open(self, ws):
        logger.debug("WebSocket connection opened")

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)

            if self.on_raw_message_callback:
                try:
                    self.on_raw_message_callback(data)
                except Exception:
                    logger.debug("on_raw_message_callback raised", exc_info=True)

            msg_type = data.get('type')

            if msg_type == 'initialize':
                self.session_id = data.get('session_id')
                self._send_client_location_state()
                self._send_call_connect()
            elif msg_type == 'call_connect_response':
                self.session_id = data.get('session_id')
                self.call_id = data.get('call_id')
                content = data.get('content', {})
                self.server_sample_rate = content.get('sample_rate', self.server_sample_rate)
                self.audio_codec = content.get('audio_codec', 'none')
                self.connected_event.set()
                if self.on_connect_callback:
                    self.on_connect_callback()
            elif msg_type == 'audio':
                audio_data = data.get('content', {}).get('audio_data', '')
                if audio_data:
                    try:
                        audio_bytes = base64.b64decode(audio_data)
                        try:
                            self.audio_buffer.put_nowait(audio_bytes)
                        except queue.Full:
                            try:
                                self.audio_buffer.get_nowait()
                                self.audio_buffer.put_nowait(audio_bytes)
                            except queue.Empty:
                                pass
                    except Exception as e:
                        logger.error("Error processing audio: %s", e)
            elif msg_type == 'error':
                content = data.get("content")
                logger.error("Server error: %s", content)
                self.last_error = content
                self.connected_event.clear()
            elif msg_type == 'call_disconnect_response':
                self.call_id = None
                if self.on_disconnect_callback:
                    self.on_disconnect_callback()
            elif msg_type == 'ping_response':
                pass  # no-op

        except json.JSONDecodeError:
            logger.warning("Received non-JSON message")
        except Exception as e:
            logger.error("Error handling message: %s", e, exc_info=True)

    def _on_error(self, ws, error):
        logger.error("WebSocket error: %s", error)
        self.connected_event.clear()

    def _on_close(self, ws, close_status_code, close_msg):
        logger.debug("WebSocket closed: %s - %s", close_status_code, close_msg)
        self.connected_event.clear()
        if self.on_disconnect_callback:
            self.on_disconnect_callback()

    # -- internal: send helpers --

    def _send_data(self, message):
        try:
            return self._send_message(message)
        except Exception as e:
            logger.error("Error sending data: %s", e)
            return False

    def _send_message(self, message):
        if self.ws and self.ws.sock and self.ws.sock.connected:
            self.ws.send(json.dumps(message))
            return True
        else:
            logger.warning("WebSocket is not connected")
            return False

    def _send_ping(self):
        if not self.session_id:
            return
        message = {
            "type": "ping",
            "session_id": self.session_id,
            "call_id": self.call_id,
            "request_id": str(uuid.uuid4()),
            "content": "ping",
        }
        self._send_data(message)

    def _send_client_location_state(self):
        if not self.session_id:
            return
        message = {
            "type": "client_location_state",
            "session_id": self.session_id,
            "call_id": None,
            "content": {
                "latitude": 0,
                "longitude": 0,
                "address": "",
                "timezone": self.usercontext.get("timezone", "America/Chicago"),
            },
        }
        self._send_data(message)

    def _send_audio(self, data):
        if not self.session_id or not self.call_id:
            return
        message = {
            "type": "audio",
            "session_id": self.session_id,
            "call_id": self.call_id,
            "content": {"audio_data": data},
        }
        self._send_data(message)

    def _send_call_connect(self):
        if not self.session_id:
            return
        message = {
            "type": "call_connect",
            "session_id": self.session_id,
            "call_id": None,
            "request_id": str(uuid.uuid4()),
            "content": {
                "sample_rate": self.client_sample_rate,
                "audio_codec": "none",
                "reconnect": self.reconnect,
                "is_private": self.is_private,
                "client_name": self.client_name,
                "settings": {"character": self.character},
                "client_metadata": {
                    "language": "en-US",
                    "user_agent": self.user_agent,
                    "mobile_browser": False,
                    "media_devices": [
                        {
                            "deviceId": "default",
                            "kind": "audioinput",
                            "label": "Default - Microphone",
                            "groupId": "default",
                        },
                        {
                            "deviceId": "default",
                            "kind": "audiooutput",
                            "label": "Default - Speaker",
                            "groupId": "default",
                        },
                    ],
                },
            },
        }
        self._send_data(message)
