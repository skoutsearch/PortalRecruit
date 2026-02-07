from src.processing.play_tagger import tag_play


def test_tag_play_parses_mmss_clock():
    tags = tag_play("drives for a layup", " 0:02 ")
    assert "buzzer_beater_scenario" in tags
    assert "late_clock" in tags


def test_tag_play_parses_numeric_clock_string():
    tags = tag_play("drive to the rim", "4")
    assert "late_clock" in tags
