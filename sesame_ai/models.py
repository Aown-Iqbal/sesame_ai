# sesame_ai/models.py


class BaseResponse:
    """Base class for API responses."""

    def __init__(self, response_json):
        self.raw_response = response_json

    def __repr__(self):
        class_name = self.__class__.__name__
        attributes = ', '.join(
            f"{k}={v}" for k, v in self.__dict__.items()
            if k != 'raw_response' and not k.startswith('_')
        )
        return f"{class_name}({attributes})"


class RefreshTokenResponse(BaseResponse):
    """Response from the token refresh endpoint."""

    def __init__(self, response_json):
        super().__init__(response_json)
        self.access_token = response_json.get('access_token')
        self.expires_in = response_json.get('expires_in')
        self.token_type = response_json.get('token_type')
        self.refresh_token = response_json.get('refresh_token')
        self.id_token = response_json.get('id_token')
        self.user_id = response_json.get('user_id')
        self.project_id = response_json.get('project_id')
