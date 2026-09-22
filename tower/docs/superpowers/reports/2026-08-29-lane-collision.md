# Two agents, one working tree

**Date:** 2026-08-29
**Outcome:** nothing lost. Everything recovered, proven, and preserved.
**Artifacts:** `rescue/canonical-worktree-snapshot-2026-08-29` (554dfb3),
tag `forensics/2026-08-29-dropped-wb-stash` (bf5731df). Both pushed.

## What happened

Three autonomous sessions were working on Glasses at once, each on its own
feature. Two of them ended up in the same working tree.

The World Builder session was working **in the canonical checkout**
`C:\Users\<you>\Projects\Glasses`, on
`world-builder/fragment-registration-v1`. Partway through, the CV Lab
session — in that same checkout — ran `git switch -c
feature/cv-lab-live-visualization-v1` and started editing
`tower/tower/cv_lab/`.

Git said nothing. A `git switch` to a **new** branch carries modified and
untracked files across with it, so the World Builder session's dirty files
followed HEAD onto the CV Lab branch. Two lanes' changes then sat in one
tree, and in two files — `tower/tower/config.py` and `tower/tower/main.py`
— in the *same file*.

Both agents noticed and both behaved well. World Builder moved to
`Glasses-worktrees/wb-registration` rather than yanking the checkout back,
and left its files copied rather than moved. CV Lab hand-picked only its
own hunks and put the combined working copies back afterwards. It also
stashed the World Builder hunks at one point and later dropped that stash.

**The damage that did not happen:** one `git add -A && git commit` in the
CV Lab session would have published World Builder's unfinished work onto
the CV Lab branch, where nobody would have looked for it.

## Why git's protection did not fire

Git *does* enforce one branch per worktree, in both directions:

    $ git worktree add ../wt lane-a
    fatal: 'lane-a' is already used by worktree at '.../canonical'

    $ git switch lane-b          # held by another worktree
    fatal: 'lane-b' is already used by worktree at '.../wt-b'

That is the only mechanical guarantee git offers here, and it is useless
against this failure: the second agent asked for a branch that **did not
exist yet**, so no worktree held it and nothing was refused. Two agents
sharing one working tree defeat the guarantee by never triggering it.

There is no `pre-checkout` hook. `post-checkout` runs after the switch and
its exit code does not undo it. `reference-transaction` *can* abort a HEAD
change — tested, it works — but it fires on every ref update in the repo
including fetches and rebases, and it leaves partial state behind (the new
branch is created; only the HEAD move is refused). Blocking the switch is
not worth that blast radius.

So the guard sits at the commit, which is where a near-miss becomes
published damage.

## What the cleanup found

Twelve entries were dirty in the canonical checkout, all World Builder:
eight modified tracked files and four untracked ones.

- **Ten were byte-identical** (line endings normalised; `core.autocrlf` is
  `true` here, so raw checksums are meaningless) to World Builder commit
  `fc6eeaf` — a *stale snapshot*, since that lane had moved on to
  `e847339` in its own worktree.
- **Two were hybrids**: `config.py` and `main.py` held CV Lab's committed
  version plus World Builder's uncommitted hunks. The delta was **+17/−0**
  and **+8/−0** lines, purely additive, and every added line was already
  present on the pushed World Builder branch.
- A **dropped stash**, `bf5731df` — *"On
  feature/cv-lab-live-visualization-v1: wb-registration-wip"* — was found
  dangling by `git fsck`. It held the same two files and nothing unique.
  It is now tagged and pushed rather than left collectable.

**No cross-lane contamination reached any commit.** All three feature
branches were audited file by file against `main`; none contains another
lane's work. The two agents' reports were accurate.

## What changed as a result

1. The canonical checkout is now on `main`, clean, and is reserved for
   integration.
2. All four non-main branches live in `Glasses-worktrees/`, one each.
3. `CLAUDE.md` gained a **lane isolation policy** — the rule that the
   canonical checkout is not a lane, and the one-liner that tells an agent
   which tree it is in.
4. `.githooks/pre-commit` refuses an *agent* commit made in the canonical
   checkout, and warns a human without blocking. It is tracked, so it
   reaches every worktree.

## The identity check, for anyone writing tooling

`--git-dir` equals `--git-common-dir` only in the main worktree:

```sh
test "$(git rev-parse --path-format=absolute --git-dir)" \
   = "$(git rev-parse --path-format=absolute --git-common-dir)" \
   && echo CANONICAL || echo WORKTREE
```

## What the guard does not stop

The branch switch itself; `--no-verify`; two agents inside one *linked*
worktree; uncommitted mixing (only publishing it is blocked); and merges in
the canonical checkout, which is deliberate so integration keeps working.
Those remain rules rather than mechanisms.

The merge exemption took two tries. Git has no `pre-merge-commit` hook
installed here, so a conflict-free merge auto-commits and never consults the
guard — but a *conflicted* merge is finished with `git commit`, which does.
The first version of the hook therefore refused the careful, hand-resolved
merge while letting the automatic one through, which is exactly backwards. An
independent reviewer of the finished state found it. The hook now steps aside
whenever `MERGE_HEAD`, `CHERRY_PICK_HEAD`, or `REVERT_HEAD` is present, and
the refusal was re-tested afterwards to confirm it still fires for an ordinary
commit.
