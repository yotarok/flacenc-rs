# flacenc-rs

[![Build Status](https://github.com/yotarok/flacenc-rs/workflows/Unittest/badge.svg)](https://github.com/yotarok/flacenc-rs/actions)
[![Crate](https://img.shields.io/crates/v/flacenc.svg)](https://crates.io/crates/flacenc)
[![Documentation](https://docs.rs/flacenc/badge.svg)](https://docs.rs/flacenc)

This crate provides some basic modules for building application-customized FLAC
(Free Lossless Audio Codec) encoder in your rust programs. The API provided by
this crate currently supports the following use cases:

1. Performing FLAC encoding of custom input sources.
1. Inspecting the encoded streams so you can analyze/ serialize parts of the
   encoder results.

It should also be noted that it is easy to hack and enhance this crate to adapt
to your use cases as this encoder is written in portable and (relatively) clean
codes.

If you are interested in a stand-alone FLAC encoder rather than a library for
embedding it, check out the CLI for this module, [`flacenc-bin`].

See the [auto-generated report] for the characteristics of the encoder compared
to [FLAC reference implementation](https://xiph.org/flac/download.html).

## Usage

Add the following line to your `Cargo.toml`:

```toml
flacenc = "0.3.1"
```

This crate is intended to be, and primarily developed with
[`portable_simd`](https://github.com/rust-lang/portable-simd), and the default
configuration above uses "fake" implementation of `portable_simd` for making it
possible to build within a stable toolchain. If you are okay with using a
nightly toolchain, use this crate with the SIMD features as follows:

```toml
flacenc = { version = "0.3.1", features = ["simd-nightly"] }
```

## Examples

See also the source code of `flacenc-bin` sub-crate as an example implementation
of FLAC encoder.

The simplest way to implement FLAC encoder given the recorded samples in
`&[i32]` is as follows:

```rust
use flacenc::component::BitRepr;
use flacenc::error::Verify;

let samples: &[i32] = &[0i32; 4096]; // replace this with real samples.

let (channels, bits_per_sample, sample_rate) = (2, 16, 44100);
let config = flacenc::config::Encoder::default().into_verified().expect(
  "Config data error."
);
let source = flacenc::source::MemSource::from_samples(
    samples, channels, bits_per_sample, sample_rate);
let flac_stream = flacenc::encode_with_fixed_block_size(
    &config, source, config.block_sizes[0]
).expect("Encode failed.");

// `Stream` imlpements `BitRepr` so you can obtain the encoded stream via
// `ByteSink` struct that implements `BitSink`.
let mut sink = flacenc::bitsink::ByteSink::new();
flac_stream.write(&mut sink);

// Then, e.g. you can write it to a file.
std::fs::write("/dev/null", sink.as_slice());

// or you can write only a specific frame.
let mut sink = flacenc::bitsink::ByteSink::new();
flac_stream.frame(0).unwrap().write(&mut sink);
```

`samples` here is an interleaved sequence, e.g. in the case of stereo inputs, it
is a sequence like `[left[0], right[0], left[1], right[1], ...]` where `left[t]`
and `right[t]` denote the `t`-th sample from the left and right channels,
respectively. All samples are assumed to be in the range of
`- 2.pow(bits_per_samples - 1) .. 2.pow(bits_per_samples - 1)`, i.e. if
`bits_per_samples == 16`, `samples[t]` must be `-32768 <= samples[t] <= 32767`.

### Customizing Encoder Behaviors

NOTE: Currently, `flacenc` is in its initial development stage
([major version zero](https://semver.org/#spec-item-4)). Therefore, the API may
change frequently. While in major-version-zero phase, we increment the minor
version ("Y" of the version "x.Y.z") when there's a breaking API change.

The current API provides several ways to control the encoding process. The
possible customization can be categorized into three groups:

1. Encoder algorithm customization by configuring [`config::Encoder`],
1. Input enhancement by implementing [`source::Source`] trait,
1. Add custom post-processing via structs in [`component`] submodule.

## Feature Flags

`flacenc` has a few Cargo features that changes the internal behaviors and APIs.

- `experimental`: Enables experimental coding algorithms that are typically
  slower. Due to its experimental nature, there's no documentation on how to
  activate each algorithm. You may need to explore `flacenc::config` module or
  source codes for better understanding.
- `log`: (This feature is enabled by default) Enables logging so an application
  program can handle the log by linking a log-handler crate (such as
  [`env_logger`] crate.) Logging is not done in a performance critical part of
  the code, so the computational cost due to this feature should be negligible.
- `simd-nightly`: Activates `portable-simd` feature of a rust nightly toolchain
  and use real SIMD processing instead of the fake one currently used by
  default.
- `mimalloc`: Enables [`mimalloc`](https://crates.io/crates/mimalloc) global
  allocator. This can lead a performance improvement in `par`-mode.
- `par`: (This feature is enabled by default) Enables multi-thread encoding in
  `encode_with_fixed_block_size` function if `config` argument is properly
  configured (when `par` is enabled the default configuration enables
  multi-thread mode.). If you want to disable multi-thread mode and make the
  dependency tree smaller, you may do that by `default-featuers = false`. `par`
  adds dependency to
  [`crossbeam-channel`](https://crates.io/crates/crossbeam-channel) crate.

In an example encoder `flacenc-bin`, all the features except for `simd-nightly`
are enabled by default. Further, `simd-nightly` is used in
[the benchmarking script](run_reporter.sh).

## Contributing

See [`CONTRIBUTING.md`] for details.

## License

Apache 2.0; see [`LICENSE`] for details.

## Disclaimer

This project is not an official Google project. It is not supported by Google
and Google specifically disclaims all warranties as to its quality,
merchantability, or fitness for a particular purpose.

This encoder is still unstable and sometimes the encoded file may contain
distortion, i.e. the encoder very rarely outputs broken signals. You can check
whether you encountered an encoder bug by running, e.g., the reference decoder.
The FLAC format contains MD5 digest of the input signal, and the reference
decoder checks if the digest of the decoded signal matches with the stored one.

[auto-generated report]: https://github.com/yotarok/flacenc-rs/blob/main/report/report.nightly.md
[`component`]: https://docs.rs/flacenc/latest/flacenc/component/index.html
[`config::encoder`]: https://docs.rs/flacenc/latest/flacenc/config/struct.Encoder.html
[`contributing.md`]: https://github.com/yotarok/flacenc-rs/blob/main/CONTRIBUTING.md
[`env_logger`]: https://crates.io/crates/env_logger
[`flacenc-bin`]: https://github.com/yotarok/flacenc-rs/blob/main/flacenc-bin/README.md
[`license`]: https://github.com/yotarok/flacenc-rs/blob/main/LICENSE
[`source::source`]: https://docs.rs/flacenc/latest/flacenc/source/trait.Source.html
