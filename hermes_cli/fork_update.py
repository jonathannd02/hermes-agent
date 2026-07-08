"""Fork update helper: keep a small custom branch rebased on upstream/main.

This fork maintains its features as a short stack of commits on a custom
branch (e.g. ``custom-main``).  ``hermes update`` calls
:func:`maybe_rebase_fork_branch` before the vanilla update flow:

1. Fetch ``upstream/main``.
2. If upstream has new commits, rebase the custom branch onto it inside a
   temporary detached worktree (the live checkout is never touched).
3. Force-push the rebased branch to ``origin`` (``--force-with-lease``).
4. Return the branch name so the vanilla flow targets it instead of
   ``main``.  Vanilla then does ``git reset --hard origin/<branch>`` (its
   normal diverged-history path) and runs every post-update step —
   dependency sync, web UI build, skills sync, config migration, gateway
   restarts.

Everything here is deterministic git.  When a rebase conflict happens (the
feature stack is deliberately tiny, so this should be rare), the rebase is
aborted, the live checkout is left untouched, and the exact manual commands
are printed.  ``rerere`` is enabled so a conflict resolved once is replayed
automatically on later updates.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

UPSTREAM_REMOTE = "upstream"
UPSTREAM_REF = "upstream/main"


def _run(git_cmd, cwd, *argv, check=False):
    return subprocess.run(
        list(git_cmd) + list(argv),
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=check,
    )


def _current_branch(git_cmd, repo_root) -> str:
    return _run(
        git_cmd, repo_root, "rev-parse", "--abbrev-ref", "HEAD", check=True
    ).stdout.strip()


def _rev(git_cmd, repo_root, ref) -> str | None:
    result = _run(git_cmd, repo_root, "rev-parse", "--verify", "--quiet", ref)
    return result.stdout.strip() if result.returncode == 0 else None


def _count(git_cmd, repo_root, spec) -> int:
    result = _run(git_cmd, repo_root, "rev-list", "--count", spec)
    try:
        return int(result.stdout.strip())
    except ValueError:
        return -1


def _ensure_rerere(git_cmd, repo_root) -> None:
    """Enable rerere so a once-resolved conflict replays on later rebases."""
    _run(git_cmd, repo_root, "config", "rerere.enabled", "true")
    _run(git_cmd, repo_root, "config", "rerere.autoUpdate", "true")


def upstream_news_count(git_cmd, repo_root) -> int:
    """Fetch upstream/main and return how many commits HEAD is missing.

    Returns -1 when upstream can't be fetched/compared (no remote, offline).
    """
    if _run(git_cmd, repo_root, "remote", "get-url", UPSTREAM_REMOTE).returncode != 0:
        return -1
    if _run(git_cmd, repo_root, "fetch", UPSTREAM_REMOTE, "main", "--quiet").returncode != 0:
        return -1
    return _count(git_cmd, repo_root, f"HEAD..{UPSTREAM_REF}")


def maybe_rebase_fork_branch(git_cmd, repo_root, args) -> str | None:
    """Rebase the current custom branch onto upstream/main and push to origin.

    Returns the branch name the vanilla update flow should target, or None
    when the fork flow does not apply (on ``main``, detached HEAD, or an
    explicit ``--branch`` was given — vanilla semantics win in those cases).
    Exits with a clear message on rebase conflict or push failure; the live
    checkout is never modified here.
    """
    repo_root = Path(repo_root)
    if getattr(args, "branch", None):
        return None
    branch = _current_branch(git_cmd, repo_root)
    if branch in ("main", "HEAD"):
        return None

    _ensure_rerere(git_cmd, repo_root)

    news = upstream_news_count(git_cmd, repo_root)
    if news < 0:
        print("  ⚠ Could not reach upstream — updating from origin only.")
        return branch
    if news == 0:
        _push_branch_if_needed(git_cmd, repo_root, branch)
        return branch

    print(f"→ Upstream has {news} new commit(s) — rebasing '{branch}' onto {UPSTREAM_REF}...")
    branch_sha = _rev(git_cmd, repo_root, branch)
    if not branch_sha:
        print(f"✗ Could not resolve branch '{branch}'.")
        sys.exit(1)

    worktree = repo_root.parent / f".hermes-fork-rebase-{branch.replace('/', '-')}"
    _run(git_cmd, repo_root, "worktree", "remove", "--force", str(worktree))
    add = _run(git_cmd, repo_root, "worktree", "add", "--detach", str(worktree), branch_sha)
    if add.returncode != 0:
        print("✗ Could not create temporary rebase worktree:")
        print(f"  {add.stderr.strip().splitlines()[-1] if add.stderr.strip() else add.returncode}")
        sys.exit(1)

    try:
        rebase = _run(git_cmd, worktree, "rebase", UPSTREAM_REF)
        if rebase.returncode != 0:
            conflicted = _run(
                git_cmd, worktree, "diff", "--name-only", "--diff-filter=U"
            ).stdout.strip().splitlines()
            _run(git_cmd, worktree, "rebase", "--abort")
            print("✗ Rebase onto upstream/main hit conflicts — nothing was changed.")
            if conflicted:
                print("  Conflicting files:")
                for path in conflicted:
                    print(f"    • {path}")
            print()
            print("  Resolve once by hand (rerere will remember the resolution):")
            print(f"    cd {repo_root}")
            print(f"    git rebase {UPSTREAM_REF}")
            print("    # fix conflicts; git add -A; git rebase --continue")
            print(f"    git push --force-with-lease origin {branch}")
            print("    hermes update")
            sys.exit(1)

        new_sha = _rev(git_cmd, worktree, "HEAD")
    finally:
        _run(git_cmd, repo_root, "worktree", "remove", "--force", str(worktree))

    if not new_sha:
        print("✗ Rebase produced no result — aborting update.")
        sys.exit(1)

    push = _run(
        git_cmd, repo_root, "push", "--force-with-lease",
        "origin", f"{new_sha}:refs/heads/{branch}",
    )
    if push.returncode != 0:
        print("✗ Could not push rebased branch to origin:")
        if push.stderr.strip():
            print(f"  {push.stderr.strip().splitlines()[-1]}")
        print(f"  Push manually, then re-run: git push --force-with-lease origin {new_sha}:refs/heads/{branch}")
        sys.exit(1)

    rebased = _count(git_cmd, repo_root, f"{UPSTREAM_REF}..{new_sha}")
    print(f"  ✓ Rebased {rebased} fork commit(s) onto {UPSTREAM_REF} and pushed to origin/{branch}")
    return branch


def _push_branch_if_needed(git_cmd, repo_root, branch) -> None:
    """Keep origin/<branch> in sync with the local branch when it lags."""
    local = _rev(git_cmd, repo_root, branch)
    remote = _rev(git_cmd, repo_root, f"origin/{branch}")
    if local and local != remote:
        push = _run(
            git_cmd, repo_root, "push", "--force-with-lease",
            "origin", f"{local}:refs/heads/{branch}",
        )
        if push.returncode == 0:
            print(f"  ✓ Synced origin/{branch} to local branch")
