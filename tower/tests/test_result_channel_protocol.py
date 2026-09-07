"""The wire contract: discovery, subscription, ordering, cursors, errors.

Everything a fresh iOS client would have to get right, pinned so that
changing it is a deliberate act rather than a side effect.
"""

import json

import pytest

from tests.result_channel_fixtures import (  # noqa: F401
    _close_result_channel_clients,
    build_world,
    drain,
    make_client,
    pump,
    subscribe,
)
from tower.results import registry
from tower.results.contracts import (
    ENVELOPE_CONTRACT,
    WORLD_BUILDER_STATUS_CONTRACT,
)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """One real world, built once for the whole module.

    Module-scoped because building it runs the real engine over rendered
    frames and costs seconds. Nothing here mutates it -- this channel only
    reads -- so sharing is safe, and a test that needs a world of its own
    builds one explicitly.
    """
    root = tmp_path_factory.mktemp("worlds")
    world_id, session_id = build_world(root)
    return root, world_id, session_id


# -- discovery ----------------------------------------------------------


def test_http_and_websocket_declare_byte_identical_capabilities(monkeypatch, built):
    """Two surfaces onto one function, not two functions that agree today.

    iOS caches the declaration and compares contract identifiers for
    equality. If the socket and the route could drift, a client that
    cached one and validated against the other would decide the Tower had
    changed contract when nothing had.
    """
    root, _, _ = built
    client = make_client(monkeypatch, root)
    over_http = client.get("/cartridges").json()
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "cartridges"})
        over_ws = drain(ws, expect="cartridges")

    assert json.dumps(over_http, sort_keys=True) == json.dumps(
        over_ws, sort_keys=True
    )


def test_an_unconfigured_tower_still_offers_the_contract(monkeypatch):
    """IOS-to-Tower.md 0.1's third state, which must not collapse into the first.

    "the Tower offers a contract this build implements, but is
     unreachable -> connect"

    is a different instruction to a person than "not built yet". A Tower
    with no world root still knows how to speak the contract.
    """
    client = make_client(monkeypatch, None)
    declaration = client.get("/cartridges").json()
    offer = declaration["cartridges"][0]

    assert offer["contract"] == WORLD_BUILDER_STATUS_CONTRACT
    assert offer["available"] is False
    assert "world root" in offer["unavailable_reason"]


def test_cartridges_without_a_contract_are_not_offered(monkeypatch, built):
    """Presence in `not_offered` must never read as an offer.

    `experimental_cv` moved out of `not_offered` on 2026-08-27, when the
    experiment-registry and provenance work its entry was waiting on
    landed. The two sets must stay disjoint, which is the invariant this
    test is really for.
    """
    root, _, _ = built
    # Off explicitly: unset means auto since 2026-09-07, and this test is
    # about the unavailable wording, not about a host that can run it.
    monkeypatch.setenv("TOWER_SCENE_UNDERSTANDING", "off")
    client = make_client(monkeypatch, root)
    declaration = client.get("/cartridges").json()

    offered = {entry["cartridge"] for entry in declaration["cartridges"]}
    silent = {entry["cartridge"] for entry in declaration["not_offered"]}

    # All four cartridges that have a wire contract now OFFER it, so
    # `not_offered` is empty. Empty is a claim rather than an accident:
    # a cartridge belongs in `not_offered` only while it can say nothing
    # at all, and each of these can now state its own limits as fields.
    #
    # Object Memory is in NEITHER set, and that is the deliberate gap --
    # see `registry.NOT_OFFERED`. It is pinned as unknown by
    # `test_result_channel_isolation.py::test_no_other_cartridge_can_be_
    # subscribed_to`.
    assert offered == {
        "world_builder",
        "experimental_cv",
        "scene_understanding",
        "document_memory",
    }
    assert silent == set()
    assert offered.isdisjoint(silent)
    for entry in declaration["not_offered"]:
        assert "contract" not in entry

    # The two that moved on 2026-08-27 must be offered AND unavailable on
    # this fixture, which configures a world root and nothing else. That
    # pairing is the whole point of the three-state design: iOS renders it
    # as "connect", not "update the app" and not "not built yet".
    by_name = {entry["cartridge"]: entry for entry in declaration["cartridges"]}
    for name, variable in (
        ("scene_understanding", "TOWER_SCENE_UNDERSTANDING"),
        ("document_memory", "TOWER_DOCUMENT_ROOT"),
    ):
        assert by_name[name]["available"] is False
        assert variable in by_name[name]["unavailable_reason"]
        assert by_name[name]["contract"]


def test_the_world_builder_offer_stays_at_index_zero(monkeypatch, built):
    """Two tests in this file index `cartridges[0]`, and a shipped client
    may too. Adding an offer must not renumber an existing one."""
    root, _, _ = built
    client = make_client(monkeypatch, root)
    declaration = client.get("/cartridges").json()
    assert declaration["cartridges"][0]["cartridge"] == "world_builder"


def test_the_declaration_says_results_are_snapshots(monkeypatch, built):
    """IOS-to-Tower.md 1.3: a UI that assumes deltas draws a partial world
    as a complete one. So the mode is declared, not implied."""
    root, _, _ = built
    client = make_client(monkeypatch, root)
    offer = client.get("/cartridges").json()["cartridges"][0]
    assert offer["snapshot_only"] is True


# -- subscription and envelope -----------------------------------------


def test_a_subscription_begins_with_a_complete_snapshot(monkeypatch, built):
    root, world_id, session_id = built
    client = make_client(monkeypatch, root)
    with client.websocket_connect("/ws") as ws:
        reply = subscribe(ws)
        assert reply["type"] == "result_subscribed"
        assert reply["contract"] == WORLD_BUILDER_STATUS_CONTRACT

        envelope = drain(ws, expect="cartridge_result")

    assert envelope["seq"] == 1
    assert envelope["snapshot"] is True
    assert envelope["envelope_contract"] == ENVELOPE_CONTRACT
    assert envelope["time_basis"] == "tower-receipt"
    assert envelope["payload"]["world"]["world_id"] == world_id
    assert envelope["payload"]["session"]["session_id"] == session_id


def test_the_envelope_is_json_serialisable_and_bounded(monkeypatch, built):
    """A payload of fixed arity, so coalescing bounds bytes and not just count.

    The channel holds one snapshot per subscription. That is only a memory
    bound if a snapshot cannot itself be huge -- and this payload carries
    counts and states, never arrays of events, poses or points.
    """
    root, _, _ = built
    client = make_client(monkeypatch, root)
    with client.websocket_connect("/ws") as ws:
        subscribe(ws)
        envelope = drain(ws, expect="cartridge_result")

    encoded = json.dumps(envelope)
    assert json.loads(encoded) == envelope
    assert len(encoded) < 8000, f"snapshot grew to {len(encoded)} bytes"

    def _no_unbounded_lists(node, path="payload"):
        if isinstance(node, list):
            assert len(node) <= 16, f"{path} is an unbounded list"
            for item in node:
                _no_unbounded_lists(item, path)
        elif isinstance(node, dict):
            for key, value in node.items():
                _no_unbounded_lists(value, f"{path}.{key}")

    _no_unbounded_lists(envelope["payload"])


def test_seq_is_dense_per_subscription(monkeypatch, built):
    """Ordering is a dense per-subscription counter assigned at send time.

    Dense, not gapped: because every result is a complete snapshot, a
    client never needs a gap to tell it something was dropped. `coalesced`
    carries that instead, so a gap would only ever mean corruption.
    """
    root, _, _ = built
    client = make_client(monkeypatch, root)
    with client.websocket_connect("/ws") as ws:
        subscribe(ws)
        seqs = [drain(ws, expect="cartridge_result")["seq"]]
        for _ in range(3):
            pump(client)
            seqs.append(drain(ws, expect="cartridge_result")["seq"])

    assert seqs == [1, 2, 3, 4]


def test_revision_is_stable_while_nothing_changes(monkeypatch, built):
    """IOS-to-Tower.md 1.2: distinguish new data from repeated data.

    A finished world does not change, so its revision must not either --
    even though elapsed-time figures are re-sent by the heartbeat.
    """
    root, _, _ = built
    client = make_client(monkeypatch, root)
    with client.websocket_connect("/ws") as ws:
        subscribe(ws)
        first = drain(ws, expect="cartridge_result")
        pump(client)
        second = drain(ws, expect="cartridge_result")

    assert second["revision"] == first["revision"]
    assert second["revision_changed"] is False


def test_revision_changes_when_the_world_does(monkeypatch, tmp_path):
    root = tmp_path / "worlds"
    world_id, session_id = build_world(root, frames=8)
    client = make_client(monkeypatch, root)
    with client.websocket_connect("/ws") as ws:
        subscribe(ws)
        first = drain(ws, expect="cartridge_result")

        # A genuinely new session in the same world.
        build_world(root, frames=8, name="Second")
        pump(client)
        later = drain(ws, expect="cartridge_result")

    assert later["revision"] != first["revision"]
    assert later["revision_changed"] is True


def test_a_heartbeat_is_marked_as_not_a_change(monkeypatch, built):
    """Elapsed time advances without anything having happened.

    `mapping_seconds` is excluded from the revision so a live figure does
    not make every poll look like news, and the heartbeat that refreshes
    it says `revision_changed: false` so a client can skip the redraw.
    """
    root, _, _ = built
    client = make_client(monkeypatch, root)
    with client.websocket_connect("/ws") as ws:
        subscribe(ws)
        drain(ws, expect="cartridge_result")
        pump(client)
        beat = drain(ws, expect="cartridge_result")

    assert beat["revision_changed"] is False
    assert beat["payload"]["progress"]["mapping_seconds"] is not None


def test_an_unchanged_world_publishes_nothing_before_the_heartbeat(
    monkeypatch, built
):
    """The bandwidth half of the same rule.

    With no change and no heartbeat due, a poll pass must produce NO
    message at all. IOS-to-Tower.md 4.8 asks the Tower to coalesce before
    publishing; republishing an identical snapshot at poll rate is the
    thing that asks for.
    """
    root, _, _ = built
    client = make_client(monkeypatch, root)
    with client.websocket_connect("/ws") as ws:
        subscribe(ws)
        drain(ws, expect="cartridge_result")

        pump(client, times=5, heartbeat=3600.0)

        # Nothing was published, so a ping is answered immediately and
        # arrives before any result would have.
        ws.send_json({"type": "ping"})
        assert ws.receive_json() == {"type": "pong"}


# -- reconnect and cursors ---------------------------------------------


def test_reconnect_always_yields_a_full_snapshot(monkeypatch, built):
    """The reconnect contract, and the reason a cursor cannot lose data.

    There is no delta stream to resume into, so there is no gap for a
    cursor to mis-handle. Every subscription starts complete, whatever
    the client claims to remember.
    """
    root, _, _ = built
    client = make_client(monkeypatch, root)
    with client.websocket_connect("/ws") as ws:
        subscribe(ws)
        first = drain(ws, expect="cartridge_result")

    with client.websocket_connect("/ws") as ws:
        reply = subscribe(ws, since_revision=first["revision"])
        again = drain(ws, expect="cartridge_result")

    assert reply["cursor_status"] == "matched"
    assert again["seq"] == 1
    assert again["payload"] == first["payload"]


@pytest.mark.parametrize(
    ("cursor", "expected"),
    [
        (None, "absent"),
        ("deadbeefdeadbeef", "stale"),
        ("", "unrecognised"),
        (17, "unrecognised"),
    ],
)
def test_cursor_classification(monkeypatch, built, cursor, expected):
    """A stale or nonsense cursor is REPORTED, never an error.

    The reply is a complete snapshot either way, so refusing would deny a
    client correct data over a field that cannot affect correctness. What
    the status buys is the difference between "nothing changed while I was
    away" and "my cached revision is worthless".
    """
    root, _, _ = built
    client = make_client(monkeypatch, root)
    with client.websocket_connect("/ws") as ws:
        overrides = {} if cursor is None else {"since_revision": cursor}
        reply = subscribe(ws, **overrides)
        envelope = drain(ws, expect="cartridge_result")

    assert reply["cursor_status"] == expected
    assert envelope["payload"]["world"] is not None


def test_a_duplicate_subscription_is_independent(monkeypatch, built):
    """Two subscriptions to the same target get their own ids and sequences.

    They must not share a slot: one client draining slowly cannot be
    allowed to consume another's snapshot. Read as a MULTISET of four
    messages rather than in a fixed order -- the acknowledgement is
    written by the receive loop and the envelope by the sender task, and
    nothing orders those two against each other.
    """
    root, _, _ = built
    client = make_client(monkeypatch, root)
    request = {
        "type": "result_subscribe",
        "cartridge": "world_builder",
        "result_type": "status",
    }
    with client.websocket_connect("/ws") as ws:
        ws.send_json(request)
        ws.send_json(request)
        messages = [ws.receive_json() for _ in range(4)]

    acks = [m for m in messages if m["type"] == "result_subscribed"]
    results = [m for m in messages if m["type"] == "cartridge_result"]

    assert len(acks) == 2 and len(results) == 2
    ids = {ack["subscription_id"] for ack in acks}
    assert len(ids) == 2, "each subscription must get its own id"
    assert {r["subscription_id"] for r in results} == ids
    assert [r["seq"] for r in results] == [1, 1], (
        "sequences are per subscription, so both start at 1"
    )


# -- errors -------------------------------------------------------------


def test_unknown_cartridge_is_refused_with_what_is_offered(monkeypatch, built):
    """A name this Tower has never heard of, and what it offers instead.

    The name changed on 2026-08-27. It used to be `scene_understanding`,
    which is now a real offer -- and a test that kept using it would have
    silently started asserting something else. `translator` is chosen
    because it is a genuinely unimplemented cartridge on the roadmap
    rather than a nonsense string: an unknown cartridge is a cartridge
    that does not exist, not a malformed request.
    """
    root, _, _ = built
    client = make_client(monkeypatch, root)
    with client.websocket_connect("/ws") as ws:
        ws.send_json(
            {
                "type": "result_subscribe",
                "cartridge": "translator",
                "result_type": "status",
            }
        )
        error = drain(ws, expect="result_error")

    assert error["reason"] == "unknown_cartridge"
    assert error["offered"] == [
        "document_memory",
        "experimental_cv",
        "scene_understanding",
        "world_builder",
    ]


def test_unknown_result_type_is_distinct_from_unknown_cartridge(monkeypatch, built):
    """Two different problems, two different reasons.

    "this Tower has no such cartridge" and "that cartridge does not
    publish that" call for different client responses.
    """
    root, _, _ = built
    client = make_client(monkeypatch, root)
    with client.websocket_connect("/ws") as ws:
        ws.send_json(
            {
                "type": "result_subscribe",
                "cartridge": "world_builder",
                "result_type": "trajectory_stream",
            }
        )
        error = drain(ws, expect="result_error")

    assert error["reason"] == "unknown_result_type"


def test_a_contract_mismatch_is_refused_not_served(monkeypatch, built):
    """Equality, and nothing else.

    A mismatch is not "older" or "newer" -- identifiers are opaque. It
    means the two sides are not talking about the same agreement, and
    serving a payload the client would decode under different rules is
    the one outcome worse than refusing.
    """
    root, _, _ = built
    client = make_client(monkeypatch, root)
    with client.websocket_connect("/ws") as ws:
        ws.send_json(
            {
                "type": "result_subscribe",
                "cartridge": "world_builder",
                "result_type": "status",
                "contract": "world_builder.status/1999-01-01",
            }
        )
        error = drain(ws, expect="result_error")

    assert error["reason"] == "contract_mismatch"
    assert error["offered_contract"] == WORLD_BUILDER_STATUS_CONTRACT


def test_subscribing_to_an_unavailable_cartridge_is_refused(monkeypatch):
    client = make_client(monkeypatch, None)
    with client.websocket_connect("/ws") as ws:
        # subscribe() returns the FIRST reply, which here is the refusal.
        error = subscribe(ws)

    assert error["type"] == "result_error"
    assert error["reason"] == "cartridge_unavailable"


@pytest.mark.parametrize(
    "request_body",
    [
        {"type": "result_subscribe"},
        {"type": "result_subscribe", "cartridge": "world_builder"},
        {"type": "result_subscribe", "cartridge": 5, "result_type": "status"},
        {
            "type": "result_subscribe",
            "cartridge": "world_builder",
            "result_type": "status",
            "world_id": 42,
        },
        {
            "type": "result_subscribe",
            "cartridge": "world_builder",
            "result_type": "status",
            "session_id": ["a"],
        },
        {"type": "result_unsubscribe"},
        {"type": "result_unsubscribe", "subscription_id": 3},
    ],
)
def test_malformed_requests_are_refused_and_the_socket_survives(
    monkeypatch, built, request_body
):
    """A hostile or confused request must not end a connection.

    The same socket carries frames. A subscribe with a list where a string
    belongs cannot be allowed to take the frame path down with it.
    """
    root, _, _ = built
    client = make_client(monkeypatch, root)
    with client.websocket_connect("/ws") as ws:
        ws.send_json(request_body)
        error = drain(ws, expect="result_error")
        assert error["reason"] in ("malformed_request", "unknown_subscription")

        ws.send_json({"type": "ping"})
        assert drain(ws, expect="pong")["type"] == "pong"


def test_an_unknown_message_type_is_answered_not_ignored(monkeypatch, built):
    """IOS-to-Tower.md 2.2: iOS never lets a request silently no-op.

    Before this channel, an unrecognised message produced a server-side
    log line no client could see, so "not implemented" and "lost in
    flight" were indistinguishable from the phone.
    """
    root, _, _ = built
    client = make_client(monkeypatch, root)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "please_do_something_new"})
        error = drain(ws, expect="protocol_error")

    assert error["reason"] == "unknown_message_type"
    assert error["message_type"] == "please_do_something_new"


def test_unsubscribing_twice_reports_the_second_as_unknown(monkeypatch, built):
    root, _, _ = built
    client = make_client(monkeypatch, root)
    with client.websocket_connect("/ws") as ws:
        reply = subscribe(ws)
        drain(ws, expect="cartridge_result")
        sub_id = reply["subscription_id"]

        ws.send_json({"type": "result_unsubscribe", "subscription_id": sub_id})
        assert drain(ws, expect="result_unsubscribed")["subscription_id"] == sub_id

        ws.send_json({"type": "result_unsubscribe", "subscription_id": sub_id})
        assert drain(ws, expect="result_error")["reason"] == "unknown_subscription"


def test_the_registry_refuses_every_unoffered_pair(monkeypatch, built):
    """Guards the lookup itself, not just its use over the wire."""
    root, _, _ = built
    assert registry.find_offer(root, "world_builder", "status") is not None
    assert registry.find_offer(root, "world_builder", "geometry") is None
    assert registry.find_offer(root, "translator", "status") is None
    assert registry.find_offer(None, "world_builder", "status")["available"] is False
    # Offered, but unavailable without a Lab to serve it -- the third
    # state, not the first.
    assert registry.find_offer(root, "experimental_cv", "status") is not None
    assert registry.find_offer(root, "experimental_cv", "metrics") is None
    assert (
        registry.find_offer(root, "experimental_cv", "status")["available"] is False
    )

    # A RESULT TYPE that does not exist on a cartridge that does. Both new
    # cartridges get this too, because each offers exactly one pair and
    # the wrong half of a pair must refuse rather than fall through to the
    # only offer the cartridge has.
    assert registry.find_offer(root, "scene_understanding", "status") is None
    assert registry.find_offer(root, "document_memory", "live") is None
    assert (
        registry.find_offer(root, "scene_understanding", "live", scene_enabled=True)
        is not None
    )
    assert (
        registry.find_offer(
            root, "document_memory", "status", document_root="/somewhere"
        )
        is not None
    )

    # Enabled is not the same as configured elsewhere. Turning one on must
    # not make the other available.
    scene_on = registry.declare(root, scene_enabled=True)["cartridges"]
    by_name = {entry["cartridge"]: entry for entry in scene_on}
    assert by_name["scene_understanding"]["available"] is True
    assert by_name["document_memory"]["available"] is False


# -- the document and the code must not drift --------------------------


def test_the_contract_document_matches_the_code():
    """A contract document that drifts from the code is worse than none.

    A fresh iOS client is told to implement from `CARTRIDGE-RESULTS.md`
    without reading Tower's Python. Every number and identifier it quotes
    is therefore load-bearing, and the only way to keep them honest is to
    fail a test when they diverge.
    """
    import pathlib

    from tower.results.publisher import (
        DEFAULT_HEARTBEAT_SECONDS,
        DEFAULT_POLL_SECONDS,
        MAX_SUBSCRIPTIONS_PER_CONNECTION,
        SEND_TIMEOUT_S,
    )
    from tower.results import contracts
    from tower.routes import results_ws

    document = pathlib.Path("docs/contracts/CARTRIDGE-RESULTS.md").read_text(
        encoding="utf-8"
    )

    for value in (
        contracts.ENVELOPE_CONTRACT,
        contracts.WORLD_BUILDER_STATUS_CONTRACT,
        contracts.CARTRIDGE_WORLD_BUILDER,
        contracts.RESULT_TYPE_STATUS,
        contracts.TIME_BASIS,
    ):
        assert value in document, f"the contract document never mentions {value!r}"

    for label, value in (
        ("subscriptions per connection", MAX_SUBSCRIPTIONS_PER_CONNECTION),
        ("send timeout", SEND_TIMEOUT_S),
        ("poll interval", DEFAULT_POLL_SECONDS),
        ("heartbeat", DEFAULT_HEARTBEAT_SECONDS),
    ):
        rendered = str(int(value) if float(value).is_integer() else value)
        assert rendered in document, (
            f"the document does not state the {label} ({rendered})"
        )

    # Every error reason a client switches on must be documented.
    reasons = [
        getattr(results_ws, name)
        for name in dir(results_ws)
        if name.startswith("ERR_")
    ]
    assert reasons
    for reason in reasons:
        assert reason in document, f"undocumented error reason {reason!r}"

    # And every message type on the wire.
    for message_type in (
        results_ws.MSG_CARTRIDGES,
        results_ws.MSG_SUBSCRIBE,
        results_ws.MSG_UNSUBSCRIBE,
        results_ws.MSG_SUBSCRIBED,
        results_ws.MSG_UNSUBSCRIBED,
        results_ws.MSG_ERROR,
    ):
        assert message_type in document, f"undocumented message {message_type!r}"


def test_every_payload_key_is_documented(monkeypatch, built):
    """A key on the wire that the document never names is a key a consumer
    has to guess at."""
    import pathlib

    root, _, _ = built
    client = make_client(monkeypatch, root)
    with client.websocket_connect("/ws") as ws:
        subscribe(ws)
        envelope = drain(ws, expect="cartridge_result")

    document = pathlib.Path("docs/contracts/CARTRIDGE-RESULTS.md").read_text(
        encoding="utf-8"
    )
    missing = []

    def _walk(node, path=""):
        if isinstance(node, dict):
            for key, value in node.items():
                if key not in document:
                    missing.append(f"{path}.{key}")
                _walk(value, f"{path}.{key}")

    _walk(envelope)
    assert missing == [], f"undocumented payload keys: {missing}"
