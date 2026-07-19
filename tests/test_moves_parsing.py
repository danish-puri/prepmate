"""Move retention in the adapters: chess.com PGN movetext and the lichess
moves string both land on Game.moves as clean mainline SAN."""

from backend.adapters import chesscom, lichess
from backend.adapters.chesscom import _san_moves

CLOCK_PGN = (
    '[Event "Live Chess"]\n'
    '[White "alice"]\n'
    '[Black "e4"]\n'
    '[ECOUrl "https://www.chess.com/openings/Kings-Gambit-Accepted"]\n'
    "\n"
    "1. e4 {[%clk 0:02:58.6]} 1... e5 {[%clk 0:02:57.1]} 2. f4 {[%clk 0:02:55]} "
    "2... exf4 3. Nf3 g5 4. Bc4 g4 5. O-O gxf3 6. Qxf3 Qf6 7. e5 Qxe5 8. Bxf7+ Kxf7 1-0"
)


def test_clock_comments_move_numbers_and_result_stripped():
    assert _san_moves(CLOCK_PGN) == [
        "e4", "e5", "f4", "exf4", "Nf3", "g5", "Bc4", "g4",
        "O-O", "gxf3", "Qxf3", "Qf6", "e5", "Qxe5", "Bxf7+", "Kxf7",
    ]


def test_headers_never_leak_into_moves():
    # [Black "e4"] sits above the blank line and must not be parsed as a move
    assert _san_moves('[White "d4"]\n[Black "e4"]\n\n1. c4 c5 0-1') == ["c4", "c5"]


def test_every_result_token_dropped():
    for result in ("1-0", "0-1", "1/2-1/2", "*"):
        assert _san_moves(f"1. e4 e5 {result}") == ["e4", "e5"]


def test_castling_promotion_and_mate_suffixes():
    assert _san_moves("1. O-O-O O-O 2. e8=Q+ axb1=N#") == ["O-O-O", "O-O", "e8=Q+", "axb1=N#"]


def test_disambiguated_piece_moves():
    # includes the double-disambiguation form (file and rank, e.g. three queens)
    assert _san_moves("1. Nbd7 R1e2 2. Qh4e1 Qh4xe1+") == ["Nbd7", "R1e2", "Qh4e1", "Qh4xe1+"]


def test_glued_move_numbers():
    assert _san_moves("1.e4 e5 2.Nf3 Nc6 10...Nf6") == ["e4", "e5", "Nf3", "Nc6", "Nf6"]


def test_movetext_without_headers_still_parses():
    assert _san_moves("1. d4 d5 2. c4") == ["d4", "d5", "c4"]


def test_chesscom_parse_game_keeps_moves():
    g = chesscom._parse_game({
        "rated": True, "rules": "chess", "time_class": "blitz", "end_time": 1,
        "white": {"username": "alice", "result": "win", "rating": 1900},
        "black": {"username": "bob", "result": "resigned", "rating": 1850},
        "pgn": '[ECO "C20"]\n\n1. e4 e5 2. Qh5 Nc6 3. Bc4 g6 1-0',
    }, "alice")
    assert g.moves == ["e4", "e5", "Qh5", "Nc6", "Bc4", "g6"]


LI_GAME = {
    "players": {"white": {"user": {"id": "alice"}, "rating": 1900},
                "black": {"user": {"id": "bob"}, "rating": 1850}},
    "status": "resign", "winner": "white", "speed": "blitz",
    "createdAt": 1750000000000, "moves": "e4 c5 Nf3",
}


def test_lichess_parse_game_keeps_moves():
    assert lichess._parse_game(LI_GAME, "alice").moves == ["e4", "c5", "Nf3"]


def test_lichess_missing_moves_field_is_empty_list():
    bare = {k: v for k, v in LI_GAME.items() if k != "moves"}
    assert lichess._parse_game(bare, "alice").moves == []
