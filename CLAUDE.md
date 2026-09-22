# Glasses Monorepo Agent Rules

Default to your assigned subsystem.

`ios/` owns Swift/iOS/DAT/UI/runtime work.

`tower/` owns Python/Tower/CV/ML/storage work.

Cross-read the other subsystem only when needed for integration,
contract reconciliation, debugging, or compatibility analysis.

Do not modify the other subsystem unless the task explicitly authorizes
cross-subsystem changes.

Shared protocol truth lives under `docs/contracts/`.

Shared current-state handoffs live under `docs/agent-handoffs/`.

Never solve a cross-system mismatch by importing implementation details
across subsystem boundaries.
---

# Glasses filesystem / worktree policy

This section is not style advice. It was written after a cleanup pass found
~1 GB of Glasses artifacts sitting at the Windows drive root and in the user
home directory: mutation trees (`C:\wbmut*`), benchmark output
(`C:\wb-adv`, `C:\wb-stage0`), thirteen world-builder run roots
(`C:\wbr*`, `C:\wbrev*`), a Tower source copy (`C:\wb-src`), and a scratch
directory (`~/wbscratch`). Two causes produced all of it, and both are
still available to you if you ignore what follows. See
`tower/docs/superpowers/reports/2026-08-27-filesystem-cleanup.md`.

## The two ways this goes wrong

1. **A relative artifact root resolved from the wrong directory.** Session
   CLIs default `--root` to a *relative* path. Run one from `C:\` and the
   artifacts land at `C:\`. This used to be silent; `tower/artifact_paths.py`
   now refuses it at the CLI boundary.

2. **MAX_PATH pressure.** Windows' 260-character limit makes deep paths
   genuinely fail (`Filename too long`), and the tempting fix is a very
   short path like `C:\m3s`. The pressure is real; the drive root is still
   the wrong answer. Use a short directory *under* `Projects\`, or enable
   long paths, rather than escaping to the root.

## The rules

1. Never create Glasses project artifacts directly under `C:\`.
2. Never create Glasses project artifacts directly under `C:\Users\<you>\`.
3. The canonical repository is `C:\Users\<you>\Projects\Glasses`.
4. Persistent Glasses worktrees live under `C:\Users\<you>\Projects\`.
   Prefer `C:\Users\<you>\Projects\Glasses-worktrees\` for new work.
5. Disposable development artifacts belong under
   `C:\Users\<you>\Projects\Glasses-scratch\`.
6. That includes benchmark copies, mutation trees, profiling output,
   temporary scripts, experimental repository copies, and generated
   scratch data.
7. Every `git worktree add` must name its destination explicitly.
8. Never rely on the current working directory, or on a default path, when
   creating a worktree.
9. Record every temporary resource you create, in your handoff.
10. Never move or delete another active agent's worktree.
11. Manage git worktrees only through git-aware operations
    (`git worktree add` / `list` / `move`), never by moving directories.
12. Tests and benchmarks must not leave persistent directories outside
    approved project locations. Use `tmp_path` / `tempfile`, which stay
    inside the OS temp directory and are allowed.
13. An autonomous run should quarantine its own disposable artifacts into
    `Glasses-scratch\` when it finishes, if it can do so safely.
14. **Deletion requires explicit human approval.** Not "it is
    reproducible", not "it is on GitHub", not "it is large".
15. When cleaning up without explicit deletion permission, *move* artifacts
    to `Glasses-scratch\`. Never delete them.

## Writing a new CLI that takes a path

Route it through the guard, or the test suite will fail:

```python
from tower.artifact_paths import artifact_root_arg

parser.add_argument("--root", type=artifact_root_arg, default=str(DEFAULT_ROOT))
```

`tests/test_artifact_paths.py::test_every_cli_root_flag_routes_through_the_guard`
scans `scripts/` and fails on any `--root` that skips it.

---

# Glasses lane isolation policy

This section is not style advice. It was written after two autonomous agents
shared one working tree. World Builder was working *in the canonical checkout*
on `world-builder/fragment-registration-v1`. CV Lab then, in that same
checkout, ran `git switch -c feature/cv-lab-live-visualization-v1` and took
the checkout off the other agent's branch mid-flight. Git said nothing: the
first agent's modified and untracked files simply followed HEAD onto the new
branch. CV Lab then had to hand-pick its own hunks out of files that held two
lanes' changes -- `tower/tower/config.py` and `tower/tower/main.py` -- and
"restored combined working copies afterwards". Nothing was lost. One
`git add -A && git commit` would have lost it. See
`tower/docs/superpowers/reports/2026-08-29-lane-collision.md`.

## Why git did not save you

Git *does* enforce one branch per worktree. It refuses in both directions:

    $ git worktree add ../wt lane-a
    fatal: 'lane-a' is already used by worktree at '.../canonical'

    $ git switch lane-b      # lane-b is checked out in another worktree
    fatal: 'lane-b' is already used by worktree at '.../wt-b'

That protection is real, and it is the only mechanical guarantee available.
It does nothing here because the second agent asked for a *new* branch, which
no worktree held. Two agents in one working tree defeat the guarantee by
never triggering it.

There is also no `pre-checkout` hook. Git has `post-checkout`, which runs
after the damage; its non-zero exit does not undo the switch. So the branch
switch itself cannot be blocked cheaply. The commit can be, and that is where
the guard sits.

## The rules

1. **The canonical checkout is not a lane.** `C:\Users\<you>\Projects\Glasses`
   is shared by every agent and every human. Never do lane work there.
2. Every agent works in its own linked worktree under
   `C:\Users\<you>\Projects\Glasses-worktrees\`.
3. Before your first edit, know which tree you are in:

   ```sh
   test "$(git rev-parse --path-format=absolute --git-dir)" \
      = "$(git rev-parse --path-format=absolute --git-common-dir)" \
      && echo CANONICAL || echo WORKTREE
   ```

   `CANONICAL` means stop and create a worktree first.
4. Never run `git switch`, `git checkout -b`, or `git branch --set-upstream`
   in the canonical checkout. Never run `git stash` there -- the state you
   would be stashing may not be yours.
5. If the canonical checkout is dirty and the changes are not yours, they
   belong to another agent. Do not commit them, revert them, stash them, or
   "restore" them. Report them and leave them alone.
6. Stage explicit paths. `git add -A` and `git commit -a` stage whatever is in
   the tree, which is exactly how another lane's work gets published.
7. Never hand-pick your hunks out of a file that holds two lanes' changes. A
   file in that state means rule 1 was already broken; stop and say so.
8. Never pass `--no-verify`. It exists for the human at the keyboard.
9. Name your worktree path and branch in your handoff, per rule 9 of the
   filesystem policy above. Rules 10 and 11 there apply unchanged: never move
   or delete another agent's worktree, and use `git worktree` to do it.

## Installing the guard

`.githooks/pre-commit` refuses any agent commit made in the canonical
checkout. There are two ways to activate it and they are mutually exclusive,
because `core.hooksPath` *overrides* `.git/hooks` rather than adding to it.

**Installed on this machine, and live now:** a copy at `.git/hooks/pre-commit`.
`.git/hooks` lives in the **common** directory, so one file covers the
canonical checkout and every linked worktree at once, it works no matter
which branch a worktree is on, and it never appears in `git status`. This is
what protects the repository today, including on branches that predate this
policy.

```sh
cp .githooks/pre-commit .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit
```

**The tracked route, once this policy is on `main`:** point git at the
tracked copy instead, so the hook travels with the repository and a fresh
clone gets it.

```sh
git -C C:/Users/<you>/Projects/Glasses config core.hooksPath .githooks
```

`core.hooksPath` lives in the shared config, so one command covers every
worktree. The path is relative and git resolves it against each worktree's
own top level, so each lane runs its own tracked copy. **The catch:** a
worktree sitting on a branch that predates this commit has no
`.githooks/pre-commit`, and git skips a missing hook silently — so until
every live branch carries it, the `.git/hooks` copy above is the one doing
the work. Do not set `core.hooksPath` without checking that.

Verify whichever route is active:

```sh
git config --get core.hooksPath          # .githooks, or empty
ls .git/hooks/pre-commit                 # the always-on copy
```

The hook detects the canonical checkout by comparing `--git-dir` with
`--git-common-dir`; they are equal only in the main worktree. It refuses when
`CLAUDECODE` is set, and warns without blocking otherwise, so a human is never
locked out of their own repository.

Integration merges in the canonical checkout still work, deliberately and in
both forms. A conflict-free `git merge` auto-commits without consulting
`pre-commit` at all, and the hook explicitly steps aside while `MERGE_HEAD`,
`CHERRY_PICK_HEAD`, or `REVERT_HEAD` is present, so a hand-resolved merge can
be finished too. Without that second exemption the guard would have refused
the careful path and waved the automatic one through.
