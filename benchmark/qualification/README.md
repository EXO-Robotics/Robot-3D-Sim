# EXO Bench Foundation v1 evidence

`foundation-v1.spec.json` is the immutable qualification definition. The
canonical receipt is
`exo-h1-locomotion-foundation-v1.qualification.json`; its representative replay
and complete-archive checksum are under `evidence/`.

The evidence is intentionally committed after the implementation. The receipt
field `exo_bench.commit` identifies the preceding clean implementation commit
that was executed. This two-commit procedure avoids claiming that a receipt can
contain the hash of the same Git commit that contains the receipt.

Regeneration procedure:

1. Commit all implementation, manifest, course, reset-profile, lock, schema,
   test, workflow, and qualification-spec changes.
2. Confirm the worktree is clean.
3. Run:

   ```bash
   uv run --frozen exo-bench qualify \
     --output benchmark/qualification/exo-h1-locomotion-foundation-v1.qualification.json \
     --evidence-dir benchmark/qualification/evidence
   ```

4. Require `qualified: true`, verify the replay with
   `exo-bench replay --require-archive-receipt`, and commit only the generated
   receipt, `.exorun`, and `.sha256` evidence files.

If implementation inputs change, this receipt becomes historical evidence.
Create a new versioned specification and receipt; do not silently overwrite the
meaning of Foundation v1.
