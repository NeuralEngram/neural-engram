# accounts/security.py

"""
Password hashing and login tokens.

- Passwords: PBKDF2-HMAC-SHA256 with a random 16-byte salt (Python
  standard library; the same scheme Django uses by default). Stored as
  "pbkdf2_sha256$<iterations>$<salt hex>$<hash hex>".
- Tokens: signed JWT (HS256) carrying the username and role, valid for
  MAEM_TOKEN_HOURS (default 12). Signed with MAEM_JWT_SECRET, or — if
  that isn't set — a key derived from MAEM_API_KEY.
"""

import hashlib
import hmac
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

import jwt

DEFAULT_ITERATIONS = 200_000


def _iterations() -> int:
    raw = os.getenv("MAEM_PASSWORD_ITERATIONS", "").strip()
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_ITERATIONS
    return value if value >= 1_000 else DEFAULT_ITERATIONS


def hash_password(password: str) -> str:
    iterations = _iterations()
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, iterations, salt_hex, digest_hex = stored.split("$")
        if scheme != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"),
            bytes.fromhex(salt_hex), int(iterations))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(digest.hex(), digest_hex)


class TokenError(Exception):
    pass


def _secret() -> str:
    explicit = os.getenv("MAEM_JWT_SECRET", "").strip()
    if explicit:
        return explicit
    api_key = os.getenv("MAEM_API_KEY", "").strip()
    if not api_key:
        raise TokenError(
            "Neither MAEM_JWT_SECRET nor MAEM_API_KEY is set — "
            "cannot sign login tokens.")
    return hashlib.sha256(f"maem-jwt:{api_key}".encode()).hexdigest()


def _token_hours() -> float:
    try:
        return max(0.1, float(os.getenv("MAEM_TOKEN_HOURS", "12")))
    except ValueError:
        return 12.0


def create_token(username: str, role: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": username,
        "role": role,
        "iat": now,
        "exp": now + timedelta(hours=_token_hours()),
    }
    return jwt.encode(payload, _secret(), algorithm="HS256")


def decode_token(token: str) -> Dict[str, Any]:
    try:
        return jwt.decode(token, _secret(), algorithms=["HS256"])
    except jwt.ExpiredSignatureError as exc:
        raise TokenError("Session expired — please log in again.") from exc
    except jwt.PyJWTError as exc:
        raise TokenError("Invalid session token.") from exc
