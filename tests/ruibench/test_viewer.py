"""The viewer: every episode can be replayed from its log alone."""

import json
import re
from pathlib import Path

import pytest

from ruibench.core.config import RunConfig
from ruibench.core.types import CameraFrame
from ruibench.reports import viewer
from ruibench.runner import RunPlan, run_plan
from ruibench.runner.recorder import FrameRecorder, to_image

CFG = RunConfig(engine="mock", interface="I0_direct", clock="zero_latency")


def _page_data(html: str) -> dict:
    m = re.search(r'<script id="ruibench-data" type="application/json">(.*?)</script>', html, re.S)
    assert m, "the page carries its data"
    return json.loads(m.group(1).replace("<\\/", "</"))


@pytest.fixture(scope="module")
def mock_logs():
    results = run_plan(RunPlan(["DS4", "DI4", "CL2"], levels=("easy",), seeds=(0,)),
                       agent_factory=lambda i: i.reference_factory(i), config=CFG)
    return [json.loads(json.dumps(r.log.as_dict())) for r in results]


def test_logs_carry_what_the_viewer_needs(mock_logs):
    for log in mock_logs:
        assert log["instruction"]
        assert log["scene"]["bodies"] and log["scene"]["robots"]
        assert all("yaw" in o for o in log["trajectory"][0]["objects"].values())


def test_the_viewer_replays_mock_episodes(mock_logs, tmp_path):
    out = viewer.write(mock_logs, tmp_path / "v.html", title="test")
    data = _page_data(out.read_text())
    assert [e["task"] for e in data["episodes"]] == ["DS4", "DI4", "CL2"]
    for ep in data["episodes"]:
        n = len(ep["traj"]["t"])
        assert n > 10
        assert all(len(rows) == n for rows in ep["traj"]["objects"].values())
        assert all(len(rows) == n for rows in ep["traj"]["arms"].values())
        assert ep["scene"]["bodies"] and ep["scene"]["armIds"] == ["robot0/right"]
        assert all(b["color"].startswith("#") for b in ep["scene"]["bodies"])
        assert ep["duration"] >= ep["traj"]["t"][-1]
        assert not ep["resultsGrade"] and ep["engine"] == "mock"
        assert all(isinstance(t["cmds"], list) for t in ep["turns"])
    box, simon, _names = data["episodes"]
    assert box["taskMetrics"]["weighings"] >= 1                       # reported beside success
    assert simon["events"][0]["newInstruction"].startswith("Simon says")   # the call, verbatim
    held = [row[4] for rows in box["traj"]["objects"].values() for row in rows]
    assert 1 in held                                                  # a coin was carried


def test_the_page_needs_nothing_from_the_network(mock_logs, tmp_path):
    html = viewer.write(mock_logs[:1], tmp_path / "v.html").read_text()
    assert "https://" not in html
    assert "googleapis" in viewer.render(mock_logs[:1], web_fonts=True)
    fragment = viewer.render(mock_logs[:1], fragment=True)
    assert fragment.startswith("<title>") and "<html" not in fragment


class _Camera:
    capabilities = type("C", (), {"rendering": True})()

    def __init__(self):
        self.sim_time = 0.0

    def render(self, name):
        import numpy as np
        return CameraFrame(camera=name, width=64, height=48, captured_at=self.sim_time,
                           image=np.full((48, 64, 3), 120, dtype="uint8"))


def test_recorder_keeps_every_observation_and_frames_in_between(tmp_path):
    pytest.importorskip("PIL")
    engine = _Camera()
    rec = FrameRecorder(tmp_path / "frames", every=0.5, size=32)
    rec.start("CL3 easy/s0")
    rec.observation({"agentview": engine.render("agentview")})
    for _ in range(10):                    # one second in 0.1 s steps
        engine.sim_time = round(engine.sim_time + 0.1, 3)
        rec.tick(engine, ["agentview", "wrist"])
    kinds = [r["kind"] for r in rec.records]
    assert kinds[0] == "observation"
    assert kinds.count("between") == 4      # 0.5 s and 1.0 s, two cameras each
    for r in rec.records:
        assert (tmp_path / "frames" / r["path"]).exists()
    assert "/" not in rec.records[0]["path"].split("/")[0]            # one safe folder name
    assert to_image(None, 64) is None


def test_frames_are_linked_or_inlined(tmp_path, mock_logs):
    pytest.importorskip("PIL")
    from PIL import Image

    root = tmp_path / "run.frames"
    (root / "ep").mkdir(parents=True)
    Image.new("RGB", (8, 8), "red").save(root / "ep" / "0.jpg")
    log = dict(mock_logs[0])
    log["config"] = {**log["config"], "record_dir": str(root)}
    log["frames"] = [{"t": 0.0, "camera": "agentview", "kind": "observation", "path": "ep/0.jpg"}]
    linked = viewer.episode_data(log, tmp_path)["frames"]["agentview"][0][1]
    inlined = viewer.episode_data(log, tmp_path, embed=True)["frames"]["agentview"][0][1]
    assert linked == "run.frames/ep/0.jpg"
    assert inlined.startswith("data:image/jpeg;base64,")


def test_the_gallery_is_a_contact_sheet_of_mission_starts(tmp_path):
    from ruibench.reports.gallery import contact_sheet, image_name, write_manifest

    write_manifest(tmp_path, ["CL3", "DS1"], ["easy", "hard"], [0, 1, 2], 128)
    (tmp_path / image_name("CL3", "hard", 2, "agentview")).write_bytes(b"\xff\xd8\xff")
    page = contact_sheet(tmp_path).read_text()
    assert image_name("CL3", "hard", 2, "agentview") in page
    assert image_name("CL3", "hard", 1, "agentview") not in page      # not rendered: left out
    assert "https://" not in page


def get_task(task_id):
    from ruibench.tasks import get

    return get(task_id)


def _showcase_dir(tmp_path, logs):
    from ruibench.reports import showcase

    for log in logs:
        showcase.log_path(tmp_path, log["task_id"], "success").write_text(json.dumps(log) + "\n")
        showcase.video_path(tmp_path, log["task_id"], "success").write_bytes(b"\0")
    return tmp_path


def test_the_showcase_lists_every_task_on_the_board_by_suite(tmp_path):
    from ruibench.reports import showcase
    from ruibench.tasks import SUITES, main_board
    from ruibench.tasks.instance import implemented

    page = showcase.render(tmp_path)
    for _, suite, _ in SUITES.values():
        assert suite in page
    ids = re.findall(r'<span class="id">(\w+)</span>', page)
    assert len(ids) == len(main_board()) == 6 * len(SUITES) and "CL7" not in ids   # held-out templates stay out
    assert page.count("not built yet") == len(ids) - len(implemented())
    # the easy clip and the hard one (v13), neither recorded here
    assert page.count("not recorded yet") == 2 * len(implemented())
    # self-contained: nothing is fetched from elsewhere (the credits only link out)
    assert "<script" not in page and "<link" not in page and 'src="https://' not in page


def test_the_showcase_credits_the_scanned_and_modelled_things_it_shows(tmp_path):
    from ruibench.reports import showcase

    import html

    from ruibench.engines.isaac.assets import get

    page = showcase.render(tmp_path)
    credits = page.split('<footer class="credits">', 1)[1]
    goat = get("goat_white").attribution.split(", ")[0]            # WK2's goat, a Sketchfab model
    assert f'{html.escape(goat)}</a><span class="lic">CC BY 4.0' in credits
    assert "Google Scanned Objects (Google LLC): " in credits and "FIRE_ENGINE" in credits  # DI4's
    assert "YCB Object and Model Set: " in credits and "003_cracker_box" in credits
    assert "brown_photostudio_02" in credits                        # the room's light


def test_the_showcase_shows_what_the_agent_gets_then_a_success(tmp_path, mock_logs):
    from ruibench.reports import showcase

    simon = next(log for log in mock_logs if log["task_id"] == "DI4")
    box = next(log for log in mock_logs if log["task_id"] == "DS4")
    dolls = next(log for log in mock_logs if log["task_id"] == "CL2")
    data = _showcase_dir(tmp_path, [simon, box, dolls])
    page = showcase.render(data)
    di4 = page.split('<span class="id">DI4</span>', 1)[1].split("</article>", 1)[0]
    ds4 = page.split('<span class="id">DS4</span>', 1)[1].split("</article>", 1)[0]
    cl2 = page.split('<span class="id">CL2</span>', 1)[1].split("</article>", 1)[0]
    # the prompt as the agent reads it, the hint told with it, and the
    # instructions it is changed to
    assert "play Simon says" in di4
    assert "tap it from above with the closed gripper" in di4       # DI4's hint
    assert "The instruction changes as the episode goes on" in di4 and "<li>Simon says: touch the" in di4
    assert "the heavier side goes down" in ds4
    assert "A film is shown first" in cl2 and 'class="told"' not in cl2   # a film, no hint
    assert 'src="DI4-success.mp4"' in di4 and "easy" in di4
    # a "success" that did not succeed is not shown
    failed = {**box, "score": {**box["score"], "success": False, "failure": "execution_failure"}}
    showcase.log_path(data, "DS4", "success").write_text(json.dumps(failed) + "\n")
    assert 'src="DS4-success.mp4"' not in showcase.render(data)
    assert [t.id for t, _ in showcase.missing([get_task("DS4"), get_task("DI4")], data,
                                              ("success",))] == ["DS4"]
    # the hard clips are not recorded here at all
    assert [(t.id, r) for t, r in showcase.missing([get_task("DI4")], data)] == [("DI4", "hard")]


def test_the_first_frame_and_a_films_frames_come_from_the_recording():
    from ruibench.reports import showcase

    log = {"frames": [
        {"t": 0.0, "camera": "agentview", "kind": "observation", "path": "e/0-a.jpg"},
        {"t": 0.0, "camera": "wrist", "kind": "observation", "path": "e/0-w.jpg"},
        {"t": 2.0, "camera": "agentview", "kind": "observation", "path": "e/2-a.jpg"},
    ] + [{"t": float(k), "camera": "agentview", "kind": "demonstration", "demonstration": 0,
          "path": f"e/demo0-{k}.jpg"} for k in range(10)]}
    assert showcase.first_frames(log) == {"agentview": "e/0-a.jpg", "wrist": "e/0-w.jpg"}
    assert showcase.film_frames(log, 0) == ["e/demo0-0.jpg", "e/demo0-3.jpg", "e/demo0-6.jpg",
                                            "e/demo0-9.jpg"]


def test_a_clip_on_another_seed_says_which(tmp_path, mock_logs):
    from ruibench.reports import showcase

    assert showcase.seed_for(get_task("DS2"), "success") == 0
    box = dict(next(log for log in mock_logs if log["task_id"] == "DS4"))
    data = _showcase_dir(tmp_path, [box])
    showcase.log_path(data, "DS4", "success").write_text(json.dumps({**box, "seed": 3}) + "\n")
    assert "the scripted reference succeeds · seed 3" in showcase.render(data)


def test_a_demonstration_leads_the_video_and_stays_out_of_the_replay(tmp_path, mock_logs):
    from ruibench.reports.video import DEMO_GAP, _frames_by_camera

    log = dict(mock_logs[0])
    root = tmp_path / "run.frames"
    (root / "ep").mkdir(parents=True)
    for name in ("demo0-0.jpg", "demo0-9.jpg", "0.jpg"):
        (root / "ep" / name).write_bytes(b"\xff\xd8\xff")
    log["config"] = {**log["config"], "record_dir": str(root)}
    log["demonstrations"] = [{"caption": "watch", "seconds": 9.0, "frames": 2}]
    log["frames"] = [
        {"t": 0.0, "camera": "agentview", "kind": "demonstration", "path": "ep/demo0-0.jpg", "demonstration": 0},
        {"t": 9.0, "camera": "agentview", "kind": "demonstration", "path": "ep/demo0-9.jpg", "demonstration": 0},
        {"t": 0.0, "camera": "agentview", "kind": "observation", "path": "ep/0.jpg"},
    ]
    times = [t for t, _ in _frames_by_camera(log)["agentview"]]
    assert times == [0.0, 9.0, 9.0 + DEMO_GAP]
    replay = viewer.episode_data(log, tmp_path)["frames"]["agentview"]
    assert [row[0] for row in replay] == [0.0]


def test_a_video_belongs_to_one_run(mock_logs):
    log = mock_logs[0]
    data = _page_data(viewer.render([log, log], videos={viewer.episode_key(log): "v.mp4"}))
    assert [e.get("video") for e in data["episodes"]] == ["v.mp4", None]


def test_a_thinned_trajectory_keeps_the_end_state(mock_logs):
    full = viewer.episode_data(mock_logs[0], Path("."))["traj"]
    thin = viewer.episode_data(mock_logs[0], Path("."), traj_stride=7)["traj"]
    assert thin["t"][0] == full["t"][0] and thin["t"][-1] == full["t"][-1]
    assert len(thin["t"]) < len(full["t"]) / 5


def test_report_and_viewer_share_one_design(mock_logs, tmp_path):
    from ruibench.reports import write

    report = write(mock_logs, tmp_path / "r.html").read_text()
    page = viewer.write(mock_logs, tmp_path / "v.html").read_text()
    for html in (report, page):
        assert "--accent:" in html and 'prefers-color-scheme: dark' in html


def test_static_assets_ship_with_the_package():
    text = (Path(__file__).parents[2] / "pyproject.toml").read_text()
    assert '"reports/static/*"' in text
    for name in ("tokens.css", "viewer.css", "viewer.js"):
        assert (Path(viewer.__file__).parent / "static" / name).exists()


def test_stop_ends_only_our_processes():
    from ruibench.runner.isolated import is_stoppable

    assert is_stoppable(["/v/bin/python", "-u", "-m", "ruibench", "run", "CL3"])
    assert is_stoppable(["python", "-m", "ruibench.runner.isolated", "spec.json"])
    assert is_stoppable(["python", "-m", "ruibench.reports.gallery", "CL3", "out"])
    assert is_stoppable(["/v/bin/ruibench", "doctor"])
    assert not is_stoppable(["bash", "-c", "ruibench run CL3"])      # the shell that ran it
    assert not is_stoppable(["python", "-m", "ruibench", "stop"])
    assert not is_stoppable(["python", "-m", "pytest"])


def test_pages_can_be_served_for_a_remote_browser(tmp_path, mock_logs):
    import threading
    import urllib.request

    from ruibench.reports import serve

    viewer.write(mock_logs[:1], tmp_path / "x.viewer.html")
    server = serve.make_server(tmp_path, port=serve.free_port("127.0.0.1", 18765))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        body = urllib.request.urlopen(f"http://127.0.0.1:{port}/x.viewer.html", timeout=5).read()
        assert b"ruibench-data" in body
        (tmp_path / "clip.mp4").write_bytes(bytes(range(200)))
        part = urllib.request.urlopen(urllib.request.Request(
            f"http://127.0.0.1:{port}/clip.mp4", headers={"Range": "bytes=10-19"}), timeout=5)
        assert part.status == 206 and part.read() == bytes(range(10, 20))   # videos can seek
        assert part.headers["Content-Range"] == "bytes 10-19/200"
        assert [p.name for p in serve.pages(tmp_path)] == ["x.viewer.html"]
    finally:
        server.shutdown()
        server.server_close()


def test_agent_specs_build_the_same_agents_everywhere():
    from ruibench.core.errors import ConfigError
    from ruibench.policies import Misperceiving, build_agent
    from ruibench.policies.negative_controls import IgnoreChange
    from ruibench.tasks.instance import build

    inst = build("DI4", "hard", 0)
    assert type(build_agent("reference", inst)) is type(inst.reference_factory(inst))
    assert isinstance(build_agent("ignore_change", inst), IgnoreChange)
    stressed = build_agent("ignore_change+misperceive=0.02", inst)
    assert isinstance(stressed, Misperceiving) and isinstance(stressed.inner, IgnoreChange)
    assert "misperceive" in stressed.name
    with pytest.raises(ConfigError):
        build_agent("reference+teleport=1", inst)


def test_a_misperceiving_agent_fails_for_physical_reasons():
    """Every bit of information, a few centimetres of error in where things are:
    the sorting task must notice.  The error is fixed per object and seed, and
    the arm's own position stays exact."""
    from ruibench.policies import Misperceiving, build_agent
    from ruibench.tasks.instance import build

    ref = run_plan(RunPlan(["CL3"], levels=("hard",), seeds=(0,)),
                   agent_factory=lambda i: build_agent("reference", i), config=CFG)[0]
    stressed = run_plan(RunPlan(["CL3"], levels=("hard",), seeds=(0,)),
                        agent_factory=lambda i: build_agent("reference+misperceive=0.03", i),
                        config=CFG)[0]
    assert ref.success and not stressed.success
    assert stressed.score.failure.value == "execution_failure"

    inst = build("CL3", "hard", 0)
    a = Misperceiving(inst.reference_factory(inst), sigma=0.03, seed=0)
    b = Misperceiving(inst.reference_factory(inst), sigma=0.03, seed=0)
    world = ref.trajectory[0]
    seen = a.seen(world)
    assert seen.obj("cube_red_a").pose.pos != world.obj("cube_red_a").pose.pos
    assert seen.arm("robot0/right").eef_pose == world.arm("robot0/right").eef_pose
    assert a.offset("cube0") == b.offset("cube0")


def test_recorded_episodes_become_videos(tmp_path, mock_logs):
    pytest.importorskip("PIL")
    from PIL import Image

    from ruibench.reports import video

    if not video.have_ffmpeg():
        pytest.skip("ffmpeg is not installed")
    root = tmp_path / "run.frames"
    (root / "ep").mkdir(parents=True)
    frames = []
    for k, t in enumerate((0.0, 0.5, 1.0)):
        for cam, color in (("agentview", (200, 40, 40)), ("wrist", (40, 40, 200))):
            Image.new("RGB", (64, 64), color).save(root / "ep" / f"{k}-{cam}.jpg")
            frames.append({"t": t, "camera": cam, "kind": "between", "path": f"ep/{k}-{cam}.jpg"})
    log = dict(mock_logs[0])
    log["config"] = {**log["config"], "record_dir": str(root)}
    log["frames"] = frames
    written = video.write_videos([log, mock_logs[1]], tmp_path / "videos", size=64)
    assert len(written) == 1 and written[0].stat().st_size > 0      # the unrecorded one is skipped
    assert written[0].name == video.video_name(viewer.episode_key(log))

    found = viewer.find_videos([log], tmp_path / "videos", tmp_path)
    assert found == {viewer.episode_key(log): f"videos/{written[0].name}"}
    data = _page_data(viewer.render([log], videos=found))
    assert data["episodes"][0]["video"] == f"videos/{written[0].name}"


def test_a_picture_of_one_colour_is_flagged():
    """A camera pressed against something shows one colour; the gallery says so."""
    import numpy as np
    from PIL import Image

    from ruibench.reports.gallery import blank

    rng = np.random.default_rng(0)
    assert blank(Image.fromarray(np.full((48, 48, 3), 30, dtype=np.uint8)))
    scene = rng.integers(0, 255, (48, 48, 3), dtype=np.uint8)
    assert not blank(Image.fromarray(scene))
    mostly = np.full((48, 48, 3), 30, dtype=np.uint8)
    mostly[:8] = 200                                   # a sixth of it is something else
    assert not blank(Image.fromarray(mostly))


def test_the_showcase_says_what_each_level_sets():
    from types import SimpleNamespace

    from ruibench.reports.showcase import level_text

    knob = lambda field, unit, **levels: SimpleNamespace(field=field, unit=unit, levels=levels)
    assert level_text(knob("space_width", "m", easy=0.1, hard=0.056), "hard") == "space width 5.6 cm"
    assert level_text(knob("clearance", "m", easy=0.003), "easy") == "clearance 3 mm"
    assert level_text(knob("n", "layers", easy=1, hard=7), "easy") == "1 layer"
    assert level_text(knob("n", "layers", easy=1, hard=7), "hard") == "7 layers"
    assert level_text(knob("K", "", easy="latching"), "easy") == "latching"
