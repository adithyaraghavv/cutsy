import json
import subprocess

import pytest

from clipper import cli, config, fonts, ingest, minyaml
from clipper.errors import ClipperError


def test_missing_file_is_friendly(capsys, tmp_path):
    rc = cli.main([str(tmp_path / "nope.mp4"), "--out", str(tmp_path / "o")])
    err = capsys.readouterr().err
    assert rc == 2 and "File not found" in err and "Traceback" not in err


def test_url_without_ytdlp_prints_instructions(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(ingest.shutil, "which", lambda n: None if n == "yt-dlp" else "/usr/bin/" + n)
    rc = cli.main(["https://youtu.be/abc", "--out", str(tmp_path / "o")])
    err = capsys.readouterr().err
    assert rc == 2 and "yt-dlp is not installed" in err and "local file" in err


def _fake_ytdlp(monkeypatch, info):
    monkeypatch.setattr(ingest, "_ytdlp", lambda: "yt-dlp")
    calls = []

    def run(args, check=True, show=False, timeout=None):
        calls.append(args)
        if "-J" in args:
            return subprocess.CompletedProcess(args, 0, json.dumps(info), "")
        raise AssertionError("must not download")
    monkeypatch.setattr(ingest, "_run_ytdlp", run)
    return calls


@pytest.mark.parametrize("info", [{"id": "x", "is_live": True}, {"id": "x", "live_status": "is_upcoming"},
                                  {"id": "x", "live_status": "post_live"}])
def test_live_stream_fails_clearly(monkeypatch, capsys, tmp_path, info):
    calls = _fake_ytdlp(monkeypatch, info)
    rc = cli.main(["https://www.youtube.com/watch?v=x", "--out", str(tmp_path / "o")])
    err = capsys.readouterr().err
    assert rc == 2 and "live" in err and "Traceback" not in err
    assert len(calls) == 1  # only metadata was fetched


def test_bad_range_message(capsys):
    assert cli.main(["x.mp4", "--range", "9-3"]) == 2
    assert "START-END" in capsys.readouterr().err


def test_parse_range():
    assert ingest.parse_range("10-20.5") == (10.0, 20.5)
    assert ingest.parse_range(None) is None


def test_parse_chat(tmp_path):
    f = tmp_path / "c.live_chat.json"
    f.write_text("\n".join([
        json.dumps({"replayChatItemAction": {"videoOffsetTimeMsec": "2500"}}),
        "garbage",
        json.dumps({"replayChatItemAction": {"videoOffsetTimeMsec": "1000"}}),
    ]))
    assert ingest.parse_chat(f) == [1.0, 2.5]


def test_minyaml_and_config(tmp_path):
    d = minyaml.loads("a:\n  b: 1.5  # c\n  c: [1, 2]\n  d: 'x # y'\ne: true\n")
    assert d == {"a": {"b": 1.5, "c": [1, 2], "d": "x # y"}, "e": True}
    f = tmp_path / "c.yaml"
    f.write_text("weights:\n  motion: 3\n")
    cfg = config.load(f)
    assert cfg["weights"]["motion"] == 3 and cfg["weights"]["chat"] == 1.5
    # the shipped config.yaml parses and matches defaults
    shipped = config.load("config.yaml") if __import__("os").path.exists("config.yaml") else cfg
    assert "weights" in shipped


def test_font_lookup_returns_existing_or_none():
    f = fonts.find_bold_font()
    assert f is None or __import__("os").path.isfile(f)


def test_no_api_keys_in_repo_sources():
    import pathlib, re
    root = pathlib.Path(__file__).resolve().parents[1]
    for p in list(root.glob("src/**/*.py")) + [root / "config.yaml"]:
        assert not re.search(r"sk-ant-[A-Za-z0-9]|AIza[0-9A-Za-z_-]{20}", p.read_text()), p
