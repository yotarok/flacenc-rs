# Release Process Runbook

## Cut a release branch

```
git checkout -b rel_0_5_1
```

## Make a ChangeLog

Following prompt can be used to that with `codex`.

```
Update "[CHANGELOG.md](http://changelog.md/)" for release version 0.5.1 (flacenc-bin: 0.2.7) with the latest updates since the previous release. Please check the git history to describe the changes.
```

## Bump version numbers if necessary

Bump the version numbers in the following files:

- Cargo.toml
- flacenc-bin/Cargo.toml
  - Don't forget to bump the version of `flacenc` here.
- README.md

Quick check (with some false positives):

```
ack --ignore-dir=target --ignore-dir=.venv '0\.5\.0'
```

## Build Check

This actually an important step to

```
cargo build --release
cd flacenc-bin
cargo build --release
```

## Test

```
makers clippy
makers integration
```

Run fuzz test for few minutes.

```
cd fuzz
cargo +nightly fuzz run frame_encode
```

Check there's no document error.

```
cargo doc
```

## Run a remote benchmark

```
cd bench_runner
bash ./gce_run_job.sh
```

## Publish (flacenc)

Login and issue a new API key in crates.io <https://crates.io/me>.

```
cargo login
# Paste your API key
```

Make sure your git repo is clean.

### flacenc

```
cargo package
cd target/package/flacenc-0.5.1
cargo build && cargo test
```

```
cargo publish --dry-run
cargo publish
```

### flacenc-bin

```
cd flacenc-bin
cargo package
cd target/package/flacenc-bin-0.2.7
cargo build && cargo test
```

```
cargo publish --dry-run
cargo publish
```

### Tag the release branch

```
git tag flacenc-v0.5.1
git tag flacenc-bin-v0.2.7
```
