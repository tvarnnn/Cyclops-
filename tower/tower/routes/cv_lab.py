"""`GET /cv-lab` -- the Lab's status document, over HTTP.

The same object the socket serves, from the same function, for the same
reason `GET /cartridges` exists beside `{"type": "cartridges"}`: a
capability you can only learn by opening a socket cannot be checked
before you open one, and this is the surface an operator reaches with
`curl` from the machine they are actually sitting at. The Tower is
normally driven over Tailscale, where a server-side log line is invisible.

Read-only, and there is deliberately no HTTP surface for start, pause or
stop. A command needs the connection it is issued on to still be there
when the outcome arrives -- a start may take two minutes to arm -- and a
request/response route would have to either block for that or lie about
having finished. The socket already carries the state that answers it.

A test asserts this route and the socket's `cv_lab_status` reply carry
the same document. Two surfaces onto one function, never two functions
that agree today.
"""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from tower.cv_lab.contracts import CONTROL_CONTRACT, ERR_LAB_UNAVAILABLE, STATUS_CONTRACT

router = APIRouter()


@router.get("/cv-lab")
def cv_lab(request: Request):
    lab = getattr(request.app.state, "cv_lab", None)
    if lab is None:
        # 503, not 404. The route exists on every build of this Tower; a
        # 404 would say "this Tower has never heard of a CV Lab", which is
        # a different claim and the one iOS renders as "not built yet".
        return JSONResponse(
            status_code=503,
            content={
                "reason": ERR_LAB_UNAVAILABLE,
                "message": "this Tower runs no CV Lab",
                "control_contract": CONTROL_CONTRACT,
                "contract": STATUS_CONTRACT,
            },
        )
    return {
        "control_contract": CONTROL_CONTRACT,
        "contract": STATUS_CONTRACT,
        "status": lab.status(),
        # Beside the document, not in it: see `CVLab.process_stats`.
        "process": lab.process_stats(),
    }
