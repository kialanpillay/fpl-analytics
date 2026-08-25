from fpl_analytics.assets import attach_assets, badge_url, photo_code, photo_url, shirt_url


def test_photo_code_strips_extension_and_prefix():
    assert photo_code("223340.jpg") == "223340"
    assert photo_code("p223340.png") == "223340"
    assert photo_code("") is None
    assert photo_code(None) is None


def test_photo_url():
    assert (
        photo_url("223340.jpg")
        == "https://resources.premierleague.com/premierleague/photos/players/250x250/p223340.png"
    )
    assert (
        photo_url("223340.jpg", size="110x140")
        == "https://resources.premierleague.com/premierleague/photos/players/110x140/p223340.png"
    )


def test_shirt_and_badge_urls():
    assert (
        shirt_url(3)
        == "https://fantasy.premierleague.com/dist/img/shirts/standard/shirt_3-66.webp"
    )
    assert (
        shirt_url(3, gk=True)
        == "https://fantasy.premierleague.com/dist/img/shirts/standard/shirt_3-1-66.webp"
    )
    assert badge_url(3) == "https://resources.premierleague.com/premierleague/badges/70/t3.png"
    assert shirt_url(None) is None
    assert badge_url(0) is None


def test_attach_assets():
    out = attach_assets(
        {"photo": "223340.jpg", "team_code": 3, "position": "MID"},
    )
    assert out["photo_url"].endswith("p223340.png")
    assert out["shirt_url"].endswith("shirt_3-66.webp")
    assert out["badge_url"].endswith("t3.png")
    gk = attach_assets({"photo": "1.jpg", "team_code": 3, "position": "GKP"})
    assert gk["shirt_url"].endswith("shirt_3-1-66.webp")
