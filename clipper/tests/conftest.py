import subprocess

import pytest

# 60 s synthetic clip: quiet + static, except 30-45 s which has a loud 440 Hz tone and a moving square.
def _make(path, w, h):
    sq = 40
    vf = (
        f"color=c=0x303030:s={w}x{h}:r=25:d=60[bg];"
        f"color=c=red:s={sq}x{sq}:r=25:d=60[sq];"
        f"[bg][sq]overlay=x='if(between(t,30,45),mod(t*120,{w - sq}),-100)':y={h // 2}[v]"
    )
    af = "aevalsrc='0.6*sin(2*PI*440*t)*between(t,30,45)+0.004*sin(2*PI*90*t)':s=44100:d=60[a]"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-filter_complex", vf + ";" + af, "-map", "[v]", "-map", "[a]",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path)],
        check=True,
    )


@pytest.fixture(scope="session")
def landscape(tmp_path_factory):
    p = tmp_path_factory.mktemp("vid") / "land.mp4"
    _make(p, 320, 180)
    return p


@pytest.fixture(scope="session")
def portrait(tmp_path_factory):
    p = tmp_path_factory.mktemp("vid") / "port.mp4"
    _make(p, 180, 320)
    return p


@pytest.fixture
def cfg():
    from clipper import config
    return config.load(None)


def probe_json(path):
    import json
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", str(path)],
        capture_output=True, text=True, check=True).stdout
    return json.loads(out)
