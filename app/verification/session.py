"""
Verification sessions.

A session is created when a document scan yields a usable portrait. It holds
the portrait embedding (never the raw image), the randomized challenge list,
and per-challenge completion state. Sessions live in process memory with a
short TTL and are deleted on completion or expiry - no biometric material is
ever written to disk or logged.

For horizontal scaling, swap this store for Redis with the same interface;
the TTL and single-use semantics carry over directly.
"""

import threading
import time
import uuid
from dataclasses import dataclass, field

import numpy as np

SESSION_TTL_SECONDS = 600


@dataclass
class VerificationSession:
    token: str
    document_type: str
    portrait_embedding: np.ndarray
    engine: str
    challenges: list[str]
    created_at: float = field(default_factory=time.time)
    completed: dict[str, bool] = field(default_factory=dict)
    finished: bool = False

    @property
    def expired(self) -> bool:
        return time.time() - self.created_at > SESSION_TTL_SECONDS

    @property
    def liveness_passed(self) -> bool:
        return all(self.completed.get(c) for c in self.challenges)


class SessionStore:
    def __init__(self):
        self._sessions: dict[str, VerificationSession] = {}
        self._lock = threading.Lock()

    def create(self, document_type: str, portrait_embedding: np.ndarray,
               engine: str, challenges: list[str]) -> VerificationSession:
        session = VerificationSession(
            token=uuid.uuid4().hex,
            document_type=document_type,
            portrait_embedding=portrait_embedding,
            engine=engine,
            challenges=challenges,
        )
        with self._lock:
            self._prune()
            self._sessions[session.token] = session
        return session

    def get(self, token: str) -> VerificationSession | None:
        with self._lock:
            self._prune()
            session = self._sessions.get(token)
            if session and session.expired:
                del self._sessions[token]
                return None
            return session

    def delete(self, token: str) -> None:
        with self._lock:
            self._sessions.pop(token, None)

    def _prune(self) -> None:
        dead = [t for t, s in self._sessions.items() if s.expired or s.finished]
        for t in dead:
            del self._sessions[t]


store = SessionStore()
