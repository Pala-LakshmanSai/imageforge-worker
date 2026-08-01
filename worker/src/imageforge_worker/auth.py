from __future__ import annotations

import secrets
from dataclasses import dataclass

from fastapi import Request

from .config import BEARER_TOKEN_PATTERN, Credential
from .errors import WorkerError


@dataclass(frozen=True, slots=True)
class Principal:
    user_id: str
    display_name: str


class BearerAuthenticator:
    def __init__(self, credentials: tuple[Credential, ...]) -> None:
        self._credentials = credentials

    async def __call__(self, request: Request) -> Principal:
        header = request.headers.get("authorization", "")
        scheme, separator, candidate = header.partition(" ")
        structurally_valid = (
            bool(separator)
            and scheme.casefold() == "bearer"
            and bool(candidate)
            and len(candidate) <= 512
            and candidate.isascii()
            and BEARER_TOKEN_PATTERN.fullmatch(candidate) is not None
        )
        candidate_to_compare = candidate if structurally_valid else "\0invalid-credential\0"

        match: Credential | None = None
        # Evaluate every configured token so a failed lookup is not an early-exit dictionary probe.
        for credential in self._credentials:
            if secrets.compare_digest(candidate_to_compare, credential.token):
                match = credential
        if match is None or not structurally_valid:
            raise WorkerError(
                status_code=401,
                code="authentication_required",
                message="A valid worker bearer credential is required.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return Principal(user_id=match.user_id, display_name=match.display_name)
