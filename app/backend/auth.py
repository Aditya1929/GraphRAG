"""
Clerk JWT verification + OpenAI API key encryption.

Flow:
  1. Frontend sends Authorization: Bearer <clerk_session_token>
  2. Backend fetches Clerk's JWKS and verifies the JWT
  3. Returns the clerk_id (sub claim) for the authenticated user

Encryption:
  User OpenAI keys are encrypted with Fernet (AES-128 CBC + HMAC) before
  being stored in Neon. The ENCRYPTION_KEY env var holds the Fernet key.
"""

import os
import httpx
from typing import Optional
from jose import jwt, JWTError
from cryptography.fernet import Fernet
from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from dotenv import load_dotenv

load_dotenv()

CLERK_PUBLISHABLE_KEY = os.getenv("CLERK_PUBLISHABLE_KEY", "")
CLERK_SECRET_KEY = os.getenv("CLERK_SECRET_KEY", "")
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", "")

_fernet: Optional[Fernet] = None


def _get_fernet() -> Fernet:
    global _fernet
    if _fernet is None:
        if not ENCRYPTION_KEY:
            raise RuntimeError("ENCRYPTION_KEY env var is not set")
        _fernet = Fernet(ENCRYPTION_KEY.encode())
    return _fernet


def encrypt_key(plaintext: str) -> str:
    return _get_fernet().encrypt(plaintext.encode()).decode()


def decrypt_key(ciphertext: str) -> str:
    return _get_fernet().decrypt(ciphertext.encode()).decode()


# ── Clerk JWKS verification ───────────────────────────────────────────────────

_jwks_cache: Optional[dict] = None

async def _fetch_jwks() -> dict:
    global _jwks_cache
    if _jwks_cache:
        return _jwks_cache

    # Clerk's JWKS endpoint is derived from the publishable key
    # Format: pk_live_<base64_domain> or pk_test_<base64_domain>
    # The JWKS URL is https://<frontend_api>/.well-known/jwks.json
    # In development, Clerk exposes it at the instance URL
    clerk_frontend_api = os.getenv("CLERK_FRONTEND_API", "")
    if not clerk_frontend_api:
        # Derive from publishable key if possible
        # pk_test_XXXX → decode the XXXX part
        try:
            import base64
            parts = CLERK_PUBLISHABLE_KEY.split("_")
            if len(parts) >= 3:
                encoded = parts[2]
                # Add padding if needed
                encoded += "=" * (4 - len(encoded) % 4)
                clerk_frontend_api = base64.b64decode(encoded).decode().rstrip("$")
        except Exception:
            pass

    if not clerk_frontend_api:
        raise HTTPException(status_code=500, detail="Clerk frontend API not configured")

    jwks_url = f"https://{clerk_frontend_api}/.well-known/jwks.json"
    async with httpx.AsyncClient() as client:
        resp = await client.get(jwks_url, timeout=10)
        resp.raise_for_status()
        _jwks_cache = resp.json()
    return _jwks_cache


async def verify_clerk_token(token: str) -> str:
    """Verify a Clerk session JWT and return the user's clerk_id."""
    try:
        jwks = await _fetch_jwks()
        # python-jose handles RS256 + JWKS key selection automatically
        payload = jwt.decode(
            token,
            jwks,
            algorithms=["RS256"],
            options={"verify_aud": False},
        )
        clerk_id: str = payload.get("sub", "")
        if not clerk_id:
            raise HTTPException(status_code=401, detail="Invalid token: missing sub claim")
        return clerk_id
    except JWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")


# ── FastAPI dependency ────────────────────────────────────────────────────────

security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security),
) -> str:
    """FastAPI dependency — returns the authenticated Clerk user ID."""
    return await verify_clerk_token(credentials.credentials)
