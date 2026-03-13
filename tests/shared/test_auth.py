"""Tests for OAuth 2.0 shared code."""

from mcp.shared.auth import OAuthClientMetadata, OAuthMetadata, InvalidScopeError


def test_oauth():
    """Should not throw when parsing OAuth metadata."""
    OAuthMetadata.model_validate(
        {
            "issuer": "https://example.com",
            "authorization_endpoint": "https://example.com/oauth2/authorize",
            "token_endpoint": "https://example.com/oauth2/token",
            "scopes_supported": ["read", "write"],
            "response_types_supported": ["code", "token"],
            "token_endpoint_auth_methods_supported": ["client_secret_basic", "client_secret_post"],
        }
    )


def test_oidc():
    """Should not throw when parsing OIDC metadata."""
    OAuthMetadata.model_validate(
        {
            "issuer": "https://example.com",
            "authorization_endpoint": "https://example.com/oauth2/authorize",
            "token_endpoint": "https://example.com/oauth2/token",
            "end_session_endpoint": "https://example.com/logout",
            "id_token_signing_alg_values_supported": ["RS256"],
            "jwks_uri": "https://example.com/.well-known/jwks.json",
            "response_types_supported": ["code", "token"],
            "revocation_endpoint": "https://example.com/oauth2/revoke",
            "scopes_supported": ["openid", "read", "write"],
            "subject_types_supported": ["public"],
            "token_endpoint_auth_methods_supported": ["client_secret_basic", "client_secret_post"],
            "userinfo_endpoint": "https://example.com/oauth2/userInfo",
        }
    )


def test_oauth_with_jarm():
    """Should not throw when parsing OAuth metadata that includes JARM response modes."""
    OAuthMetadata.model_validate(
        {
            "issuer": "https://example.com",
            "authorization_endpoint": "https://example.com/oauth2/authorize",
            "token_endpoint": "https://example.com/oauth2/token",
            "scopes_supported": ["read", "write"],
            "response_types_supported": ["code", "token"],
            "response_modes_supported": [
                "query",
                "fragment",
                "form_post",
                "query.jwt",
                "fragment.jwt",
                "form_post.jwt",
                "jwt",
            ],
            "token_endpoint_auth_methods_supported": ["client_secret_basic", "client_secret_post"],
        }
    )


def test_validate_scope_none_registered_allows_any():
    """None registered scope should mean no restrictions - allow any scope."""
    metadata = OAuthClientMetadata(redirect_uris=["https://example.com/callback"], scope=None)
    result = metadata.validate_scope("read write")
    assert result == ["read", "write"]


def test_validate_scope_matching():
    """Should return requested scopes when they match registered scopes."""
    metadata = OAuthClientMetadata(redirect_uris=["https://example.com/callback"], scope="read write")
    result = metadata.validate_scope("read")
    assert result == ["read"]


def test_validate_scope_rejects_unregistered():
    """Should reject scopes not in registered scopes."""
    metadata = OAuthClientMetadata(redirect_uris=["https://example.com/callback"], scope="read")
    try:
        metadata.validate_scope("write")
        assert False, "Expected InvalidScopeError"
    except InvalidScopeError:
        pass


def test_validate_scope_empty_registered_rejects_all():
    """Empty string registered scope should reject all requested scopes."""
    metadata = OAuthClientMetadata(redirect_uris=["https://example.com/callback"], scope="")
    try:
        metadata.validate_scope("read")
        assert False, "Expected InvalidScopeError"
    except InvalidScopeError:
        pass
