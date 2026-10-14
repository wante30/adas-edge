import pytest

from competition_io.hud_v1 import CommandError, Error, Request, VECTORS


@pytest.mark.parametrize("command,word", VECTORS)
def test_handoff_vectors(command, word):
    assert command.encode() == word
    assert Request.decode(word) == command
    assert int.from_bytes(word.to_bytes(4, "little"), "little") == word


@pytest.mark.parametrize("word,error", [
    (0x1101, Error.ZERO_SEQUENCE),
    (0x0001F101, Error.RESERVED_BITS),
    (0x00010001, Error.BAD_COMBINATION),
    (0x00011301, Error.BAD_COMBINATION),
    (0x100000000, Error.BAD_ACCESS),
])
def test_invalid_raw_commands(word, error):
    with pytest.raises(CommandError) as exc:
        Request.decode(word)
    assert exc.value.code == error


def test_all_state_blink_combinations_roundtrip_without_bit_aliasing():
    for state in range(512):
        for blink in range(8):
            valid = all(not blink & (1 << i) or state & mask for i, mask in enumerate((2, 4, 8)))
            request = Request(65535, True, state, blink)
            if valid:
                assert Request.decode(request.encode()) == request
            else:
                with pytest.raises(CommandError):
                    request.encode()
