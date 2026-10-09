"""Small, standard auth: bcrypt + SQLite users, JWT in an HttpOnly cookie."""
from __future__ import annotations

import re
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

from .config import Settings

USERNAME_RE = re.compile(r"^[a-z0-9_.-]{3,32}$")
_DUMMY_HASH = bcrypt.hashpw(b"dummy-password", bcrypt.gensalt(rounds=4))


class AuthError(Exception):
    pass


class UserExists(AuthError):
    pass


class InvalidCredentials(AuthError):
    pass


class LockedOut(AuthError):
    def __init__(self, retry_after: int):
        super().__init__("Too many failed attempts")
        self.retry_after = retry_after


def validate_password(pw: str) -> None:
    if len(pw.encode()) > 72:
        raise AuthError("Password must be at most 72 bytes")
    if len(pw) < 8 or not re.search(r"[A-Za-z]", pw) or not re.search(r"\d", pw):
        raise AuthError("Password needs 8+ characters with letters and digits")


class AuthService:
    def __init__(self, settings: Settings, bcrypt_rounds: int = 12):
        self.s = settings
        self.rounds = bcrypt_rounds
        self._failures: dict[str, list[float]] = {}
        self._lock = threading.Lock()
        settings.users_db.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS users ("
                "id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL, "
                "password_hash BLOB NOT NULL, created_at TEXT NOT NULL)"
            )

    def _db(self) -> sqlite3.Connection:
        return sqlite3.connect(self.s.users_db)

    # -- users ---------------------------------------------------------------
    def register(self, username: str, password: str) -> str:
        username = username.strip().lower()
        if not USERNAME_RE.match(username):
            raise AuthError("Username: 3-32 chars, a-z 0-9 _ . -")
        validate_password(password)
        hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=self.rounds))
        try:
            with self._db() as db:
                db.execute(
                    "INSERT INTO users (username, password_hash, created_at) VALUES (?,?,?)",
                    (username, hashed, datetime.now(timezone.utc).isoformat()),
                )
        except sqlite3.IntegrityError:
            raise UserExists("Username already taken") from None
        return username

    def authenticate(self, username: str, password: str) -> str:
        username = username.strip().lower()
        self._check_lock(username)
        with self._db() as db:
            row = db.execute("SELECT password_hash FROM users WHERE username=?", (username,)).fetchone()
        # Always run bcrypt so response time doesn't reveal whether the user exists.
        hashed = row[0] if row else _DUMMY_HASH
        try:
            ok = bcrypt.checkpw(password.encode()[:72], hashed) and row is not None
        except ValueError:
            ok = False
        if not ok:
            self._fail(username)
            raise InvalidCredentials("Invalid username or password")
        with self._lock:
            self._failures.pop(username, None)
        return username

    # -- throttling ----------------------------------------------------------
    def _recent(self, username: str) -> list[float]:
        cutoff = time.monotonic() - self.s.lockout_seconds
        return [t for t in self._failures.get(username, []) if t > cutoff]

    def _check_lock(self, username: str) -> None:
        with self._lock:
            recent = self._recent(username)
            self._failures[username] = recent
            if len(recent) >= self.s.max_failed_logins:
                raise LockedOut(int(recent[0] + self.s.lockout_seconds - time.monotonic()) + 1)

    def _fail(self, username: str) -> None:
        with self._lock:
            self._failures.setdefault(username, []).append(time.monotonic())

    # -- tokens --------------------------------------------------------------
    def issue_token(self, username: str) -> str:
        now = datetime.now(timezone.utc)
        payload = {"sub": username, "iat": now, "exp": now + timedelta(minutes=self.s.token_ttl_minutes)}
        return jwt.encode(payload, self.s.secret_key, algorithm="HS256")

    def verify_token(self, token: str) -> str:
        try:
            data = jwt.decode(token, self.s.secret_key, algorithms=["HS256"], options={"require": ["exp", "sub"]})
        except jwt.PyJWTError:
            raise InvalidCredentials("Invalid or expired session") from None
        return data["sub"]
