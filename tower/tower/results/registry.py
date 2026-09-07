"""The capability declaration: which cartridges the Tower can serve.

`IOS-to-Tower.md` section 7 makes this the single most valuable thing the
Tower can build first:

    "Without it, every cartridge is stuck at 'the Tower says nothing',
     and no other work on this list can be shown at all. It is also the
     smallest."

Section 0.1 names the three states iOS must be able to keep apart,
because they call for opposite user responses:

    the Tower says nothing about this cartridge   -> "not built yet"
    the Tower offers a contract this build
        does not implement                        -> "update the app"
    the Tower offers a contract this build
        implements, but is unreachable            -> "connect"

That third state is why `available` is separate from `contract`. A
World Builder with no world root configured still OFFERS the contract --
this build knows how to speak it -- it just has nothing to serve. Folding
that into "no contract" would tell a person to update an app that is
already correct.

This is NOT dynamic module discovery. `04-MODULE-SYSTEM.md` forbids that
before V1.0, and iOS says so itself: it "caches a declaration rather than
fetching a registry". The set below is a static, hand-maintained
declaration compiled into this build. Nothing here enumerates loaded
modules, and nothing here can grow at runtime.

TWO LISTS, BECAUSE THERE ARE TWO TRANSPORTS.

`cartridges` is what can be SUBSCRIBED to: each entry is a
`(cartridge, result_type)` pair the result socket will serve.
`http_contracts` is what can be FETCHED, and it exists because iOS
caches a declaration -- a contract discoverable only by making a call is
a contract a phone cannot plan around, and Document Memory's library was
exactly that.

Only Document Memory's is listed today. World Builder's geometry and
Object Memory's observations are the same shape and are not declared;
adding them would mean importing their identifiers here, and this module
must stay cartridge-blind (`test_the_result_channel_core_is_cartridge_
blind`). Their identifiers live in adapter modules rather than in
`contracts.py`, so those two lanes own that move.
"""

import logging
from dataclasses import dataclass

from tower.results.contracts import (
    CARTRIDGE_DOCUMENT_MEMORY,
    CARTRIDGE_EXPERIMENTAL_CV,
    CARTRIDGE_SCENE_UNDERSTANDING,
    CARTRIDGE_WORLD_BUILDER,
    DOCUMENT_MEMORY_LIBRARY_CONTRACT,
    DOCUMENT_MEMORY_STATUS_CONTRACT,
    ENVELOPE_CONTRACT,
    EXPERIMENTAL_CV_STATUS_CONTRACT,
    RESULT_TYPE_LIVE,
    RESULT_TYPE_STATUS,
    SCENE_LIVE_CONTRACT,
    WORLD_BUILDER_STATUS_CONTRACT,
)

# Cartridges that exist in this repository but offer NO wire contract,
# with the reason. Deliberately reported in a separate list from the
# offers, and deliberately reported at all.
#
# A client must not treat presence here as an offer of anything -- these
# are iOS's "the Tower says nothing about this cartridge" case, and the
# names are here so an operator can see the difference between "Tower
# does not know what document_memory is" and "Tower knows and is not
# serving it yet". iOS keys on `cartridges`; this list is for humans.
#
# Document Memory and Scene Understanding left this list on 2026-08-27.
# They did not leave it because they got better -- Document Memory's
# detector still fires on essentially nothing at the geometry the glasses
# deliver. They left it because their limits became things the PAYLOAD
# STATES, which is what an offer is for. `not_offered` is for a cartridge
# that can say nothing at all; a cartridge that can say "I have observed
# nothing, and here is precisely why" belongs in `cartridges`, available
# or not.
# As of this integration the tuple is EMPTY, and that is a claim, not
# an oversight: every cartridge with a wire contract in this build now
# has an OFFER, available or not. Document Memory and Scene
# Understanding left on 2026-08-27 and the Experimental CV Lab left with
# `experimental_cv.status/2026-08-27`.
#
# OBJECT MEMORY IS IN NEITHER LIST, deliberately. Its identifier exists
# in `contracts.py` and its control surface is live at
# `/cartridges/{cartridge}/session`, but declaring it here breaks a
# pinned iOS test, so the socket declaration waits for the iOS lane to
# take both halves at once. See the note at CARTRIDGE_OBJECT_MEMORY.
# That is a decision for a human and must not be closed by an
# integrator noticing the gap.
NOT_OFFERED: tuple[dict, ...] = ()

# Why an offer can be present and unavailable. Module-level constants
# rather than inline strings, so a test can assert the exact wording a
# person will be shown without copying it, and so `/cartridges` and a
# refusal on the socket cannot drift into two different explanations of
# one configuration.
SCENE_DISABLED_REASON = (
    "Scene Understanding is not enabled on this Tower "
    "(TOWER_SCENE_UNDERSTANDING is unset or off), so no session can be "
    "started and there is no live scene to read. This build implements "
    "the contract"
)

# Names TOWER_DOCUMENT_ROOT even though the root has had a managed
# default since 2026-09-07: iOS string-matches that variable name in a
# 404 body to tell "the cartridge is switched off" from "the route is
# gone", and the words that still explain the state are the ones that
# say which switch was thrown.
DOCUMENT_DISABLED_REASON = (
    "Document Memory is switched off on this Tower (TOWER_DOCUMENT_ENABLED "
    "is off), so no document root is configured (TOWER_DOCUMENT_ROOT is "
    "unset), there is nowhere to record a document and nothing to read "
    "back. This build implements the contract"
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CartridgeOffer:
    """One (cartridge, result_type) pair this build can serve."""

    cartridge: str
    result_type: str
    contract: str
    available: bool
    unavailable_reason: str | None = None
    # Whether results arrive as complete snapshots or as deltas that must
    # be merged. Stated rather than implied: IOS-to-Tower.md 1.3 warns
    # that "a UI that assumes incremental updates will draw a partial
    # world as a complete one".
    snapshot_only: bool = True

    def to_json_dict(self) -> dict:
        return {
            "cartridge": self.cartridge,
            "result_type": self.result_type,
            "contract": self.contract,
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
            "snapshot_only": self.snapshot_only,
        }


def declare(
    world_root: str | None,
    *,
    document_root: str | None = None,
    scene_enabled: bool = False,
    scene_unavailable_reason: str | None = None,
    cv_lab=None,
) -> dict:
    """The full capability declaration.

    Takes its inputs rather than reading configuration itself so the
    declaration is a pure function of them -- which is what lets one test
    assert that the HTTP route and the WebSocket message produce
    byte-identical output. Passing the app instead would have made the
    function impure again, which is why three cartridges' worth of state
    arrives as three arguments and not as one `app`.

    Every argument past `world_root` is keyword-only WITH A DEFAULT, and
    that is a decision about what an omission should mean rather than a
    convenience. A caller that has not been taught about a cartridge gets
    that cartridge declared and UNAVAILABLE -- never silently offered as
    working. So the failure mode of forgetting to thread a value through
    is a Tower that under-promises, which iOS renders as "connect", and
    not one that promises a channel it cannot serve. Use
    `declaration_inputs(app_state)` rather than assembling them by hand.

    `cv_lab` is DUCK-TYPED, never imported: anything with an
    `availability()` returning `(available, reason)`. This module is part
    of the result channel cartridge-blind core and
    `test_the_result_channel_core_is_cartridge_blind` keeps it that way;
    an import of the Lab here would bake one cartridge into the shared
    surface, and this time the surface is a WIRE CONTRACT.

    Availability is about whether a session COULD BE STARTED, never about
    current activity. A Scene Understanding that is enabled but stopped is
    `available: true` -- it can be started -- and its payload says
    `lifecycle.state: "stopped"`. Folding "not running right now" into
    "unavailable" would tell a person their Tower cannot do this when in
    fact nobody has pressed Start, and those two call for opposite
    responses.

    That used to read "about CONFIGURATION", which was the intent but not
    the whole truth, and the gap was reachable. Scene Understanding needs
    torch, which ships in the optional `[ml]` extra; on a host without it
    the session constructed anyway (the default device is `cpu`, and only
    a non-cpu device imported torch), so this function was handed
    `scene_enabled=True` and promised a cartridge that failed ~50 ms into
    Start. The caller now imports the dependency while constructing, so
    what arrives here is the same answer it always claimed to be.

    `scene_unavailable_reason` exists because the fix made a second defect
    common. `SCENE_DISABLED_REASON` names the environment variable, which
    is right when nobody switched the cartridge on and WRONG when they
    did and construction failed -- it sends an operator to check the one
    thing that is already correct. It is `or`-ed, so an omitted reason
    keeps the pinned configured-off wording exactly.
    """
    if world_root is None:
        world_available = False
        world_reason = (
            "no world root is configured on this Tower (TOWER_WORLD_ROOT "
            "is unset), so there is no persisted world state to read"
        )
    else:
        world_available = True
        world_reason = None

    cv_available, cv_reason = _cv_lab_availability(cv_lab)

    offers = (
        CartridgeOffer(
            cartridge=CARTRIDGE_WORLD_BUILDER,
            result_type=RESULT_TYPE_STATUS,
            contract=WORLD_BUILDER_STATUS_CONTRACT,
            available=world_available,
            unavailable_reason=world_reason,
        ),
        CartridgeOffer(
            cartridge=CARTRIDGE_SCENE_UNDERSTANDING,
            result_type=RESULT_TYPE_LIVE,
            contract=SCENE_LIVE_CONTRACT,
            available=bool(scene_enabled),
            # `or`, so an omitted reason keeps the configured-off wording
            # that `test_the_declaration_names_the_variable` pins. The
            # specific reason only ever REPLACES it, never appends, and a
            # caller that has not been taught to thread it through gets
            # exactly the previous behaviour.
            unavailable_reason=(
                None
                if scene_enabled
                else (scene_unavailable_reason or SCENE_DISABLED_REASON)
            ),
        ),
        CartridgeOffer(
            cartridge=CARTRIDGE_DOCUMENT_MEMORY,
            result_type=RESULT_TYPE_STATUS,
            contract=DOCUMENT_MEMORY_STATUS_CONTRACT,
            available=document_root is not None,
            unavailable_reason=(
                None if document_root is not None else DOCUMENT_DISABLED_REASON
            ),
        ),
        CartridgeOffer(
            cartridge=CARTRIDGE_EXPERIMENTAL_CV,
            result_type=RESULT_TYPE_STATUS,
            contract=EXPERIMENTAL_CV_STATUS_CONTRACT,
            available=cv_available,
            unavailable_reason=cv_reason,
        ),
    )

    return {
        "type": "cartridges",
        "envelope_contract": ENVELOPE_CONTRACT,
        "cartridges": [offer.to_json_dict() for offer in offers],
        "not_offered": [dict(entry) for entry in NOT_OFFERED],
        "http_contracts": [
            {
                "cartridge": CARTRIDGE_DOCUMENT_MEMORY,
                "contract": DOCUMENT_MEMORY_LIBRARY_CONTRACT,
                "entry_route": "/documents",
                "available": document_root is not None,
                "unavailable_reason": (
                    None
                    if document_root is not None
                    else DOCUMENT_DISABLED_REASON
                ),
                "why_not_a_subscription": (
                    "document text is bulk and is the most sensitive data "
                    "this platform holds. The result socket shares its "
                    "send lock with the frame path, and a listing is "
                    "pulled on demand rather than pushed"
                ),
            }
        ],
    }


def _cv_lab_availability(cv_lab) -> tuple[bool, str | None]:
    """Whether this build can serve the CV Lab contract, and why not.

    A Tower with no Lab still OFFERS the contract -- this build knows how
    to speak it -- and reports it unavailable, which is the third state in
    IOS-to-Tower.md 0.1 ("offered, implemented, unreachable -> connect")
    rather than the first ("the Tower says nothing -> not built yet").
    Those call for opposite instructions to a person, which is why they
    cannot be one state.

    Never raises. A declaration is how a client learns what is possible;
    it must not fail because a subsystem is unwell.
    """
    if cv_lab is None:
        return False, (
            "this Tower is running without a CV Lab module, so no "
            "experiment can be enumerated or started"
        )
    try:
        available, reason = cv_lab.availability()
    except Exception:
        logger.exception("[Tower][Results] could not read CV Lab availability")
        return False, "the CV Lab could not report whether it is available"
    return bool(available), reason


def find_offer(
    world_root: str | None,
    cartridge: str,
    result_type: str,
    *,
    document_root: str | None = None,
    scene_enabled: bool = False,
    scene_unavailable_reason: str | None = None,
    cv_lab=None,
):
    """The offer matching a subscribe request, or None.

    None covers both "no such cartridge" and "no such result type on a
    cartridge that exists". The caller distinguishes them for the error
    message; this returns one thing so there is one lookup path.

    `scene_unavailable_reason` is threaded through for the same reason it
    exists at all: the offer this returns carries `unavailable_reason`,
    and `result_subscribe` puts it on the wire. Without it that ONE
    surface went on saying "TOWER_SCENE_UNDERSTANDING is unset or off"
    while `/cartridges` over HTTP, the socket declaration and `/scene`
    had all been taught the truth -- which is exactly the four-surfaces-
    disagreeing drift the shared constant was introduced to prevent.
    """
    declaration = declare(
        world_root,
        document_root=document_root,
        scene_enabled=scene_enabled,
        scene_unavailable_reason=scene_unavailable_reason,
        cv_lab=cv_lab,
    )
    for entry in declaration["cartridges"]:
        if entry["cartridge"] == cartridge and entry["result_type"] == result_type:
            return entry
    return None


def known_cartridges(
    world_root: str | None,
    *,
    document_root: str | None = None,
    scene_enabled: bool = False,
    cv_lab=None,
) -> set:
    declaration = declare(
        world_root,
        document_root=document_root,
        scene_enabled=scene_enabled,
        cv_lab=cv_lab,
    )
    return {entry["cartridge"] for entry in declaration["cartridges"]}


def declaration_inputs(app_state) -> dict:
    """The configuration `declare` needs, read off one app state.

    Separate from `declare` so that function stays a pure function of its
    arguments -- which is what lets a test assert the HTTP route and the
    WebSocket message are byte-identical rather than merely equal today.

    One definition, used by every surface, for the same reason. Two call
    sites each reaching for their own subset of `app.state` is precisely
    how `/cartridges` over HTTP and `{"type": "cartridges"}` on the socket
    would come to disagree, and the disagreement would be invisible until
    a phone hit the one that was wrong. Three cartridges now depend on
    that, so this is the only supported way to build the arguments.

    `getattr` with a default throughout, because most tests in this
    repository construct an app state by hand and set only what they care
    about. A missing attribute means "not configured", which under-
    promises -- see the note on defaults in `declare`.
    """
    return {
        "world_root": getattr(app_state, "world_root", None),
        "document_root": getattr(app_state, "document_root", None),
        "scene_enabled": bool(getattr(app_state, "scene_enabled", False)),
        "scene_unavailable_reason": getattr(
            app_state, "scene_unavailable_reason", None
        ),
        "cv_lab": getattr(app_state, "cv_lab", None),
    }
