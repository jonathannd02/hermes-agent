#!/usr/bin/env bash
# fork-status — show this fork's delta vs upstream and flag features that
# upstream may have adopted (so they can be dropped from the stack).
#
# Usage: scripts/fork-status.sh   (run from anywhere inside the repo)
#
# Reading the output:
#  - "Fork commits": the whole stack this fork carries. Keep it short.
#  - "Adoption hints": fork-added symbols that now also exist in upstream
#    files — a strong sign upstream built (or merged) the same feature.
#    Verify with `git log upstream/main -S<symbol> --oneline | head` and, if
#    genuinely adopted, drop the fork commit:
#      git rebase -i upstream/main   # delete the obsolete commit line
#      git push --force-with-lease origin HEAD
#  - Commits that upstream cherry-picked verbatim disappear from the stack
#    automatically on the next `hermes update` rebase (git skips
#    patch-identical commits) — those need no action at all.

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

echo "→ Fetching upstream/main..."
git fetch upstream main --quiet

echo
echo "== Fork commits on top of upstream/main =="
git log --oneline upstream/main..HEAD

echo
echo "== Delta vs upstream (files) =="
git diff upstream/main...HEAD --stat | tail -n 15

echo
echo "== Collision forecast (upstream churn inside fork-modified files) =="
# Files BOTH sides changed since the fork's merge-base: each one is a
# potential rebase stop on the next `hermes update`. rerere replays
# resolutions it has already recorded; a NEW name here means the next
# rebase may pause for a one-time manual resolve.
merge_base=$(git merge-base HEAD upstream/main)
overlap=$(comm -12 \
  <(git diff --name-only "$merge_base"..upstream/main | sort -u) \
  <(git diff --name-only "$merge_base"..HEAD | sort -u))
if [ -z "$overlap" ]; then
  echo "  (clear — upstream churn does not touch fork-modified files)"
else
  echo "$overlap" | sed 's/^/  • /'
  echo "  → the next rebase may pause on these; resolve once, rerere remembers."
fi

echo
echo "== Adoption hints (fork-added symbols that now exist upstream) =="
# Scan only lines this fork ADDS to files that also exist upstream (features
# living in fork-only files can't collide by name until upstream adopts the
# concept — and then the modified-file hooks will light up here anyway).
# Distinctive = snake_case identifier of >= 12 chars from a def/class/register.
fork_only_files=$(git diff upstream/main...HEAD --name-only --diff-filter=A)
symbols=$(git diff upstream/main...HEAD --unified=0 --diff-filter=M -- '*.py' \
  | grep -E '^\+[^+]' \
  | grep -oE '(def|class) [A-Za-z_][A-Za-z0-9_]{11,}|name="[a-z_]{12,}"' \
  | sed -E 's/^(def|class) //; s/name="([a-z_]+)"/\1/' \
  | sort -u)

# Known intentional divergences — same name exists upstream by design.
# (send_message: upstream has the tool but deliberately doesn't register it.)
IGNORE="send_message"

hits=0
for sym in $symbols; do
  case " $IGNORE " in *" $sym "*) continue;; esac
  matches=$(git grep -l --fixed-strings "$sym" upstream/main -- '*.py' 2>/dev/null \
    | sed 's/^upstream\/main://' || true)
  real=""
  for m in $matches; do
    case "$fork_only_files" in *"$m"*) ;; *) real="$real $m";; esac
  done
  if [ -n "$real" ]; then
    hits=$((hits + 1))
    shown=$(echo "$real" | tr ' ' '\n' | grep -v '^$' | head -5 | tr '\n' ' ')
    total=$(echo "$real" | wc -w)
    extra=""; [ "$total" -gt 5 ] && extra="(+$((total - 5)) more)"
    [ "$hits" -le 20 ] && echo "  • '$sym' also upstream in: $shown$extra"
  fi
done
[ "$hits" -gt 20 ] && echo "  ... and $((hits - 20)) more"
[ "$hits" -eq 0 ] && echo "  (none — no fork-added symbol found in upstream code)"

echo
echo "Tip: verify a hint with: git log upstream/main -S<symbol> --oneline | head"
