# Agent instructions

## Versioning

After every change, always update the version numbers in both
`src/subtitle_engine/__init__.py` and `pyproject.toml` to the most appropriate
value for the change. Follow [Semantic Versioning](https://semver.org/):

- **Patch** (`x.y.Z`): bug fixes, refactors with no behavior change, test or
  doc-only updates.
- **Minor** (`x.Y.0`): new features, new CLI flags, new defaults that change
  output for existing users — anything additive but user-visible.
- **Major** (`X.0.0`): breaking changes — removed flags, removed public API,
  changed default behavior with no opt-out, renamed commands.

When in doubt, prefer the smaller bump. Both files must be updated together
and stay in sync. The new version should be reflected in any user-facing
docs (e.g. `README.md` examples) only when the example shows a concrete
version string. And give reasoning why did you choose that update.
