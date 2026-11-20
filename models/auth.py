"""
JWT authentication + password hashing helpers for the Car Audit Tool.

Designed for offline use:
- No external auth provider. Users are stored locally in the SQLite database.
- Tokens are signed with an HMAC-SHA256 secret (configurable via env).
- The secret is auto-generated on first run and persisted to a local file
  (`database/jwt_secret.key`) so tokens stay valid across restarts.

A default admin user is auto-created on first startup unless disabled.
"""

from __future__ import annotations

import logging
import os
import secrets
import time
from typing import Any, Dict, Optional, Tuple

import bcrypt
import jwt


logger = logging.getLogger("car_audit_tool.auth")

# Algorithm used to sign tokens
ALGORITHM = "HS256"
DEFAULT_TOKEN_TTL_SEC = 8 * 3600          # 8 hours
DEFAULT_REFRESH_TTL_SEC = 7 * 24 * 3600    # 7 days


class AuthManager:
    """Owns the JWT secret and provides login / verify helpers."""

    def __init__(self, secret_path: str, secret_env: Optional[str] = None,
                 token_ttl: int = DEFAULT_TOKEN_TTL_SEC,
                 refresh_ttl: int = DEFAULT_REFRESH_TTL_SEC):
        self.token_ttl = token_ttl
        self.refresh_ttl = refresh_ttl
        self.secret = self._load_or_create_secret(secret_path, secret_env)

    # ----------------------------------------------------------------------
    # Secret management
    # ----------------------------------------------------------------------

    @staticmethod
    def _load_or_create_secret(secret_path: str, secret_env: Optional[str]) -> str:
        if secret_env:
            if len(secret_env) < 32:
                logger.warning("JWT_SECRET is shorter than 32 characters; use a longer random value.")
            return secret_env
        if os.path.isfile(secret_path):
            with open(secret_path, "r", encoding="utf-8") as f:
                value = f.read().strip()
                if value:
                    # Tighten permissions on an existing file that may have been
                    # created (or copied) with looser perms, so the secret is never
                    # world-readable even across upgrades.
                    try:
                        os.chmod(secret_path, 0o600)
                    except OSError as e:  # pragma: no cover - best effort
                        logger.warning("Could not chmod %s to 0o600: %s", secret_path, e)
                    return value
        os.makedirs(os.path.dirname(secret_path) or ".", exist_ok=True)
        value = secrets.token_urlsafe(48)
        # Create with owner-only permissions from the start (no world-readable window).
        fd = os.open(secret_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(value)
        logger.info("Generated new JWT secret at %s", secret_path)
        return value

    # ----------------------------------------------------------------------
    # Password hashing
    # ----------------------------------------------------------------------

    @staticmethod
    def hash_password(plain: str) -> str:
        salt = bcrypt.gensalt(rounds=12)
        return bcrypt.hashpw(plain.encode("utf-8"), salt).decode("utf-8")

    @staticmethod
    def verify_password(plain: str, hashed: str) -> bool:
        if not hashed:
            return False
        try:
            return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
        except (ValueError, TypeError):
            return False

    # ----------------------------------------------------------------------
    # Token issue / verify
    # ----------------------------------------------------------------------

    def issue_tokens(self, user: Dict[str, Any]) -> Tuple[str, str]:
        """Return (access_token, refresh_token) for the given user dict.

        ``token_version`` is stamped into every token so that bumping the user's
        ``token_version`` column (on password change, role change, delete or
        logout) immediately invalidates every previously issued token of either
        type. Callers must pass the current ``token_version`` from the DB row.
        """
        now = int(time.time())
        base_claims = {
            "sub": str(user["id"]),
            "username": user["username"],
            "role": user.get("role", "user"),
            "ver": int(user.get("token_version", 1) or 1),
        }
        access = jwt.encode(
            {**base_claims, "iat": now, "exp": now + self.token_ttl, "type": "access"},
            self.secret,
            algorithm=ALGORITHM,
        )
        refresh = jwt.encode(
            {**base_claims, "iat": now, "exp": now + self.refresh_ttl, "type": "refresh"},
            self.secret,
            algorithm=ALGORITHM,
        )
        # PyJWT >= 2 returns str, but older versions return bytes - normalize
        if isinstance(access, bytes):
            access = access.decode("utf-8")
        if isinstance(refresh, bytes):
            refresh = refresh.decode("utf-8")
        return access, refresh

    def verify_token(self, token: str, expected_type: str = "access",
                     expected_token_version: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Decode and validate a JWT. Returns the claims dict or None.

        If ``expected_token_version`` is supplied (the user's current
        ``token_version`` from the DB), the token is rejected unless its ``ver``
        claim matches - so a bumped version (password change, role change,
        logout) instantly invalidates older tokens.
        """
        try:
            claims = jwt.decode(token, self.secret, algorithms=[ALGORITHM])
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
        if claims.get("type") != expected_type:
            return None
        if expected_token_version is not None:
            try:
                if int(claims.get("ver", 0)) != int(expected_token_version):
                    return None
            except (TypeError, ValueError):
                return None
        return claims

    def refresh_access(self, refresh_token: str, user: Optional[Dict[str, Any]] = None
                       ) -> Optional[Tuple[str, str]]:
        """
        Validate a refresh token and issue a fresh access+refresh pair.

        Pass the *current* user row from the database so deleted users cannot
        refresh and role changes take effect; if omitted, token claims are used.
        When the user row is supplied, the refresh token is also rejected if its
        ``ver`` claim is stale (i.e. the user bumped token_version since it was
        issued - e.g. by changing their password or logging out).
        """
        expected_ver = int(user.get("token_version", 1) or 1) if user else None
        claims = self.verify_token(refresh_token, expected_type="refresh",
                                   expected_token_version=expected_ver)
        if not claims:
            return None
        if user is None:
            user = {
                "id": claims["sub"],
                "username": claims.get("username", ""),
                "role": claims.get("role", "user"),
                "token_version": claims.get("ver", 1),
            }
        return self.issue_tokens(user)

    def user_id_from_refresh(self, refresh_token: str) -> Optional[int]:
        claims = self.verify_token(refresh_token, expected_type="refresh")
        try:
            return int(claims["sub"]) if claims else None
        except (KeyError, ValueError, TypeError):
            return None


# ---------------------------------------------------------------------------
# Initial admin seeding
# ---------------------------------------------------------------------------

DEFAULT_ADMIN_USERNAME = "admin"
MIN_PASSWORD_LENGTH = 8
# bcrypt only uses the first 72 bytes of a password. Depending on the bcrypt
# version a longer one is silently truncated (4.x) or raises ValueError (5.x,
# which surfaced as an HTTP 500), so reject it up front with a clear message.
MAX_PASSWORD_BYTES = 72


def password_problem(password: str) -> Optional[str]:
    """Return a human-readable reason the password is unacceptable, or None."""
    if len(password) < MIN_PASSWORD_LENGTH:
        return f"Password must be at least {MIN_PASSWORD_LENGTH} characters"
    if len(password.encode("utf-8")) > MAX_PASSWORD_BYTES:
        return f"Password must be at most {MAX_PASSWORD_BYTES} bytes (about {MAX_PASSWORD_BYTES} plain characters)"
    return None


def seed_default_admin(db, username: str = DEFAULT_ADMIN_USERNAME,
                       password: Optional[str] = None) -> Optional[str]:
    """Create the first admin user if the users table is empty.

    If `password` is not supplied, a random one is generated and printed ONCE
    to the console. There is no hard-coded default password.

    Returns the generated password (None if nothing was generated/seeded).
    The db object must expose `create_user(...)` and `count_users()`.
    """
    try:
        count = db.count_users()
    except Exception:
        logger.exception("Could not count users; skipping seed.")
        return None
    if count > 0:
        return None

    generated = password is None or password == ""
    if generated:
        password = secrets.token_urlsafe(12)
    elif password_problem(password):
        logger.error("DEFAULT_ADMIN_PASSWORD rejected (%s); not seeding.", password_problem(password))
        return None

    try:
        db.create_user(username=username, password_hash=AuthManager.hash_password(password), role="admin")
    except Exception:
        logger.exception("Failed to seed initial admin user.")
        return None

    print("=" * 60)
    print(" INITIAL ADMIN USER (auto-created)")
    print("   username: " + username)
    if generated:
        print("   password: " + password)
        print(" This password is shown ONCE. Change it after first login.")
    else:
        print("   password: (taken from DEFAULT_ADMIN_PASSWORD)")
    print("=" * 60)
    return password if generated else None
