"""Normalized game record shared by both platform adapters."""

from dataclasses import dataclass

# chess.com per-player result codes that count as a draw
DRAW_CODES = {"agreed", "repetition", "stalemate", "insufficient", "50move", "timevsinsufficient"}


@dataclass
class Game:
    platform: str            # "chesscom" | "lichess"
    color: str               # "white" | "black" (our player's side)
    time_class: str          # "bullet" | "blitz" | "rapid" | "classical" | "daily"
    result: str              # "win" | "draw" | "loss"
    termination: str         # raw reason: "checkmated", "resigned", "timeout", ...
    eco: str | None
    opening: str | None
    end_time: int            # epoch seconds
    player_rating: int | None
    opponent_rating: int | None
    moves: list[str]         # mainline SAN, empty when the source has none
