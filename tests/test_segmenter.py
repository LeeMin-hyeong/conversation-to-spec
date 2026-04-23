from app.segmenter import segment_conversation


def test_segment_by_lines_when_multiline_text():
    text = (
        "Need a web portal for room booking.\n"
        "Users should book a room.\n"
        "It should be fast on mobile.\n"
    )
    units = segment_conversation(text)
    assert [u.id for u in units] == ["U1", "U2", "U3"]
    assert units[1].text == "Users should book a room."


def test_sentence_fallback_when_single_line_text():
    text = "Need a booking tool. It should work on phones. Maybe payment later."
    units = segment_conversation(text)
    assert len(units) >= 2
    assert units[0].id == "U1"
