import time

import jwt
import pytest

from creditrisk.auth import AuthError, AuthService, InvalidCredentials, LockedOut, UserExists
from tests.conftest import PASSWORD


@pytest.fixture()
def svc(settings):
    return AuthService(settings, bcrypt_rounds=4)


def test_register_and_authenticate(svc):
    assert svc.register("  Bob ", PASSWORD) == "bob"
    assert svc.authenticate("BOB", PASSWORD) == "bob"


def test_password_is_hashed_at_rest(svc, settings):
    import sqlite3

    svc.register("bob", PASSWORD)
    stored = sqlite3.connect(settings.users_db).execute("SELECT password_hash FROM users").fetchone()[0]
    assert PASSWORD.encode() not in stored and stored.startswith(b"$2")


@pytest.mark.parametrize("pw", ["short1", "alllettersonly", "12345678", "x" * 80 + "1"])
def test_weak_passwords_rejected(svc, pw):
    with pytest.raises(AuthError):
        svc.register("carol", pw)


@pytest.mark.parametrize("name", ["ab", "has space", "UPPER!", "x" * 40])
def test_bad_usernames_rejected(svc, name):
    with pytest.raises(AuthError):
        svc.register(name, PASSWORD)


def test_duplicate_user(svc):
    svc.register("dave", PASSWORD)
    with pytest.raises(UserExists):
        svc.register("DAVE", PASSWORD)


def test_wrong_password_and_unknown_user_give_same_error(svc):
    svc.register("erin", PASSWORD)
    with pytest.raises(InvalidCredentials) as a:
        svc.authenticate("erin", "Wrong-pass1")
    with pytest.raises(InvalidCredentials) as b:
        svc.authenticate("nobody", "Wrong-pass1")
    assert str(a.value) == str(b.value)


def test_lockout_after_repeated_failures(svc):
    svc.register("frank", PASSWORD)
    for _ in range(3):
        with pytest.raises(InvalidCredentials):
            svc.authenticate("frank", "bad-password1")
    with pytest.raises(LockedOut) as e:
        svc.authenticate("frank", PASSWORD)  # even the right password is refused while locked
    assert e.value.retry_after > 0


def test_token_roundtrip_expiry_and_tampering(svc, settings):
    tok = svc.issue_token("gina")
    assert svc.verify_token(tok) == "gina"

    expired = jwt.encode({"sub": "gina", "exp": int(time.time()) - 5}, settings.secret_key, algorithm="HS256")
    with pytest.raises(InvalidCredentials):
        svc.verify_token(expired)

    forged = jwt.encode({"sub": "admin", "exp": int(time.time()) + 600}, "other-key" * 5, algorithm="HS256")
    with pytest.raises(InvalidCredentials):
        svc.verify_token(forged)

    none_alg = jwt.encode({"sub": "admin", "exp": int(time.time()) + 600}, None, algorithm="none")
    with pytest.raises(InvalidCredentials):
        svc.verify_token(none_alg)
