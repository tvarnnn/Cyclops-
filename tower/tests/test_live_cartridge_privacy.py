"""Two privacy properties that a static test cannot establish.

`test_scene_understanding_persists_nothing` AST-walks the wire path for
calls that could write. That is a strong guard and it is not the same
claim as "a running session wrote nothing": a write could arrive through
a library call the forbidden-name list does not know, or through a
dependency. So this file watches the FILESYSTEM across a real run.

And `keep_page_images` is a constructor argument on Document Memory's
engine. Its default is off and the engine's own docstring says it must
stay off, but a default is a promise about what happens when nobody
chooses. What matters for a web process is whether anything can choose,
and the answer must be no: there is no configuration path that turns it
on, because this platform has no redaction and a stored page image is an
unredacted photograph of what a wearer was reading.
"""

import base64
import io
import threading

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tests import document_fixtures as fx
from tests.test_scene_wire_e2e import (
    TWO_PEOPLE_A_CHAIR_AND_A_LAPTOP,
    StubSceneEngine,
)

_OPEN: list = []


@pytest.fixture(autouse=True)
def _close_clients():
    yield
    while _OPEN:
        _OPEN.pop().__exit__(None, None, None)


def _tree(root):
    return {path.relative_to(root) for path in root.rglob("*")}


class TestASceneSessionWritesNothing:
    def test_a_full_run_leaves_the_filesystem_untouched(self, monkeypatch, tmp_path):
        """Snapshot, run, snapshot. The claim the AST test cannot make.

        The working directory is moved under `tmp_path` for the duration,
        so a relative path opened anywhere on the wire path lands
        somewhere this test can see -- a write to `data/…` from a process
        started in the repository root would otherwise be invisible here
        and would be found much later, by a person wondering what
        `data/scene/` is.
        """
        from tower import cartridge_runtime
        from tower.main import create_app
        from tower.scene.live import SceneLive

        workdir = tmp_path / "cwd"
        workdir.mkdir()
        monkeypatch.chdir(workdir)
        monkeypatch.setenv("TOWER_SCENE_UNDERSTANDING", "true")
        monkeypatch.delenv("TOWER_WORLD_ROOT", raising=False)
        monkeypatch.delenv("TOWER_CAPTURE_ROOT", raising=False)
        monkeypatch.delenv("TOWER_DOCUMENT_ROOT", raising=False)
        monkeypatch.setenv("TOWER_DOCUMENT_ENABLED", "false")
        monkeypatch.setattr(
            cartridge_runtime,
            "_scene_session",
            lambda settings: SceneLive(
                lambda: StubSceneEngine(TWO_PEOPLE_A_CHAIR_AND_A_LAPTOP)
            ),
        )
        client = TestClient(create_app())
        client.__enter__()
        _OPEN.append(client)

        before = _tree(workdir)
        client.post("/scene/start")
        buffer = io.BytesIO()
        Image.new("RGB", (640, 360), (30, 60, 90)).save(buffer, format="JPEG")
        data = base64.b64encode(buffer.getvalue()).decode("ascii")
        with client.websocket_connect("/ws") as ws:
            for seq in range(6):
                ws.send_json(
                    {
                        "type": "frame",
                        "seq": seq,
                        "width": 640,
                        "height": 360,
                        "format": "jpeg",
                        "data": data,
                    }
                )
                assert ws.receive_json()["type"] == "frame_result"
            for _ in range(400):
                if client.get("/scene").json()["scene_available"]:
                    break
            ws.send_json(
                {
                    "type": "result_subscribe",
                    "cartridge": "scene_understanding",
                    "result_type": "live",
                }
            )
            assert ws.receive_json()["type"] == "result_subscribed"
            assert ws.receive_json()["payload"]["scene_available"] is True
        client.post("/scene/stop")

        assert _tree(workdir) == before

    def test_the_payload_declares_that_it_persists_nothing(
        self, monkeypatch, tmp_path
    ):
        """The property, as a value a client can switch on.

        A document saying "persists nothing" is advice. A field saying
        `persistence: "none"` is something a consumer can render and a
        test can assert.
        """
        from tower import cartridge_runtime
        from tower.main import create_app
        from tower.scene.live import SceneLive

        monkeypatch.setenv("TOWER_SCENE_UNDERSTANDING", "true")
        monkeypatch.delenv("TOWER_WORLD_ROOT", raising=False)
        monkeypatch.setattr(
            cartridge_runtime,
            "_scene_session",
            lambda settings: SceneLive(
                lambda: StubSceneEngine(TWO_PEOPLE_A_CHAIR_AND_A_LAPTOP)
            ),
        )
        client = TestClient(create_app())
        client.__enter__()
        _OPEN.append(client)

        payload = client.get("/scene").json()

        assert payload["persistence"] == "none"
        assert payload["identity"] == "anonymous-and-unpublished"


class TestAWebProcessCannotStorePageImages:
    def test_the_production_session_keeps_no_page_images(self, monkeypatch, tmp_path):
        """No configuration turns this on, and that is the point.

        `keep_page_images` exists on the engine for an operator running a
        replay by hand, who has chosen it and knows what is on their own
        disk. A web process reachable from a phone is a different
        situation and gets no such switch: this platform has no
        redaction, and 06-PRIVACY-DATA.md is explicit that a crop is not
        inherently safe -- a photographed page routinely contains a
        bystander, a screen, or a second document.
        """
        from tower.cartridge_runtime import build_live_cartridges
        from tower.config import get_settings

        monkeypatch.setenv("TOWER_DOCUMENT_ROOT", str(tmp_path))
        monkeypatch.setenv("TOWER_DOCUMENT_CAPTURE", "true")
        # Every environment variable this cartridge reads, set to
        # everything they accept. None of them is a way in.
        for name in (
            "TOWER_DOCUMENT_KEEP_PAGE_IMAGES",
            "TOWER_DOCUMENT_IMAGES",
            "TOWER_KEEP_PAGE_IMAGES",
        ):
            monkeypatch.setenv(name, "true")

        live = build_live_cartridges(get_settings())
        try:
            assert live.document is not None
            assert live.document.status()["keeps_page_images"] is False
        finally:
            live.shutdown()

    def test_a_recorded_document_declares_no_imagery_and_no_redaction(
        self, monkeypatch, tmp_path
    ):
        """`redaction: "none"` is the honest value, and it must be visible.

        Claiming anything else would be a false privacy assurance. The
        pairing that matters is `retains_raw_imagery: false` WITH
        `redaction: "none"`: nothing was kept, and if anything ever were,
        it would not have been redacted.
        """
        from tests.test_documents_wire_e2e import _write_one_document
        from tower.main import create_app

        _write_one_document(tmp_path)
        monkeypatch.setenv("TOWER_DOCUMENT_ROOT", str(tmp_path))
        monkeypatch.delenv("TOWER_WORLD_ROOT", raising=False)
        client = TestClient(create_app())
        client.__enter__()
        _OPEN.append(client)

        payload = client.get("/documents").json()
        document = payload["documents"][0]

        assert document["retains_raw_imagery"] is False
        assert document["redaction"] == "none"
        # `none-retained`, not a constant that said the same thing
        # whatever was on disk. The pairing that matters is
        # `retains_raw_imagery: false` WITH `redaction: "none"`: nothing
        # was kept, and if anything ever were it would not have been
        # redacted.
        assert document["imagery_treatment"] == "none-retained"
        assert document["imagery_served"] is False
        # And the state named in iOS's own vocabulary, so the mapping is
        # the Tower's decision rather than the phone's guess. Never
        # `redacted`: this platform performs no redaction at all.
        assert document["imagery_ios_state"] == "rawEphemeral"
        assert payload["imagery_treatment"] == "none-retained"
        assert not (tmp_path / "pages").exists()

    def test_no_route_serves_an_image(self, monkeypatch, tmp_path):
        """Even a store that HAS page images must not serve one.

        A replay run by hand can produce them. This asserts the web
        process still refuses: `image_relpath` may name a file, and there
        is no route that resolves it.
        """
        from tests.test_documents_wire_e2e import _write_one_document
        from tower.main import create_app

        _write_one_document(tmp_path)
        monkeypatch.setenv("TOWER_DOCUMENT_ROOT", str(tmp_path))
        monkeypatch.delenv("TOWER_WORLD_ROOT", raising=False)
        client = TestClient(create_app())
        client.__enter__()
        _OPEN.append(client)

        routes = {
            getattr(route, "path", "") for route in client.app.router.routes
        }

        for route in routes:
            assert "image" not in route
            assert "thumbnail" not in route
            assert "page" not in route or "pages" not in route


def test_a_document_session_writes_only_into_its_own_root(monkeypatch, tmp_path):
    """Nothing lands outside the directory an operator configured.

    A relative default resolved against the process working directory is
    how a memory of what a wearer read ends up somewhere nobody chose --
    `scripts/document_memory_session.py` defaults to `data/document_memory`
    for exactly that reason, and a web process must not inherit it.
    """
    from tower import cartridge_runtime
    from tower.document_memory.dwell import DwellPolicy
    from tower.document_memory.live import DocumentLive
    from tower.document_memory.ocr import FixedTextRecogniser
    from tower.main import create_app

    workdir = tmp_path / "cwd"
    workdir.mkdir()
    root = tmp_path / "docs"
    monkeypatch.chdir(workdir)
    monkeypatch.setenv("TOWER_DOCUMENT_ROOT", str(root))
    monkeypatch.setenv("TOWER_DOCUMENT_CAPTURE", "true")
    monkeypatch.delenv("TOWER_WORLD_ROOT", raising=False)
    monkeypatch.delenv("TOWER_CAPTURE_ROOT", raising=False)
    monkeypatch.setattr(
        cartridge_runtime,
        "_document_session",
        lambda settings: DocumentLive(
            settings.document_root,
            policy=DwellPolicy(min_frames=3, min_seconds=0.6),
            recogniser_factory=lambda: FixedTextRecogniser(
                pages=[fx.page_regions(fx.TRANSFORMER_PAPER)]
            ),
        ),
    )
    client = TestClient(create_app())
    client.__enter__()
    _OPEN.append(client)

    client.post("/documents-session/start")
    deadline = threading.Event()
    timer = threading.Timer(15.0, deadline.set)
    timer.start()
    try:
        while not deadline.is_set():
            if client.get("/documents-session").json()["session"]["state"] == "running":
                break
            deadline.wait(0.005)
    finally:
        timer.cancel()

    frames = list(fx.document_frames(fx.TRANSFORMER_PAPER, 10))
    with client.websocket_connect("/ws") as ws:
        for seq, raw in enumerate(frames):
            ws.send_json(
                {
                    "type": "frame",
                    "seq": seq,
                    "width": 640,
                    "height": 480,
                    "format": "jpeg",
                    "data": base64.b64encode(raw).decode("ascii"),
                }
            )
            assert ws.receive_json()["type"] == "frame_result"
            threading.Event().wait(0.07)
    client.post("/documents-session/stop")

    assert _tree(workdir) == set(), "something was written beside the process"
