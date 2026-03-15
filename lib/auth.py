from __future__ import annotations
import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

_anon_client: Client | None = None


def get_anon_client() -> Client:
    global _anon_client
    if _anon_client is None:
        _anon_client = create_client(
            os.environ["SUPABASE_URL"],
            os.environ["SUPABASE_ANON_KEY"],
        )
    return _anon_client


def register(email: str, password: str) -> dict:
    """Returns {"user": ..., "session": ..., "error": str | None}"""
    try:
        result = get_anon_client().auth.sign_up({"email": email, "password": password})
        return {"user": result.user, "session": result.session, "error": None}
    except Exception as e:
        return {"user": None, "session": None, "error": str(e)}


def login(email: str, password: str) -> dict:
    """Returns {"user": ..., "session": ..., "error": str | None}"""
    try:
        result = get_anon_client().auth.sign_in_with_password(
            {"email": email, "password": password}
        )
        return {"user": result.user, "session": result.session, "error": None}
    except Exception as e:
        return {"user": None, "session": None, "error": str(e)}


def logout(access_token: str) -> None:
    try:
        get_anon_client().auth.sign_out()
    except Exception:
        pass


def get_user(access_token: str) -> dict | None:
    try:
        result = get_anon_client().auth.get_user(access_token)
        return result.user
    except Exception:
        return None


def get_authed_db_client(access_token: str) -> Client:
    """Creates a Supabase client with user JWT — subject to RLS (NOT service key)."""
    client = create_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_ANON_KEY"],
    )
    client.postgrest.auth(access_token)
    return client
