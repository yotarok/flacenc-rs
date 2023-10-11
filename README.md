# flacenc-rs

[![Build Status](https://github.com/yotarok/flacenc-rs/workflows/Unittest/badge.svg)](https://github.com/yotarok/flacenc-rs/actions)
[![Crate](https://img.shields.io/crates/v/flacenc.svg)](https://crates.io/crates/flacenc)
[![Documentation](https://docs.rs/flacenc/badge.svg)](https://docs.rs/flacenc)

This crate provides some basic modules for building application customized FLAC
(Free Lossless Audio Codec) encoder in rust programs.

See [the auto-generated report](report/report.md) for the characteristics of the
encoder compared to
[FLAC reference implementation](https://xiph.org/flac/download.html).

## Usage

Add the following line to your `Cargo.toml`:

```toml
flacenc = { version = "0.2.0", feature = ["fakesimd"] }
```

This crate is intended to be, and primarily developed with
[`portable_simd`](https://github.com/rust-lang/project-portable-simd), and the
`fakesimd` feature above is for emulating `portable_simd` in a stable toolchain.
If you are using a nightly toolchain, use this crate without `fakesimd` as
follows:

```toml
flacenc = "0.2.0"
```

## Examples

See also the source code of `flacenc-bin` sub-crate as a simple implementation
of FLAC encoder.

The simplest way to implement FLAC encoder given the recorded samples in
`&[i32]` is as follows:

```rust
use flacenc::component::BitRepr;

let samples: &[i32] = &[0i32; 4096]; // replace this with real samples.

let (channels, bits_per_sample, sample_rate) = (2, 16, 44100);
let config = flacenc::config::Encoder::default();
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
std::fs::write("/dev/null", sink.as_byte_slice());

// or you can write only a specific frame.
let mut sink = flacenc::bitsink::ByteSink::new();
flac_stream.frame(0).unwrap().write(&mut sink);
```

`samples` here is an interleaved sequence, e.g. in the case with stereo inputs,
it is a sequence like `[left_0, right_0, left_1, right_1, ...]` where
`{left|right}_N` denotes the `N`-th sample from the left or right channel. All
samples are assumed to be in the range of `- 2.pow(bits_per_samples - 1) ..
2.pow(bits_per_samples - 1)`, i.e. if `bits_per_samples == 16`, `samples[t]`
must be `-32768 <= samples[t] <= 32767`.

### Customizing Encoder Behaviors

NOTE: Currently, `flacenc` is in its initial development stage
([major version zero](https://semver.org/#spec-item-4)).

The current API provides several ways to control the encoding process. The
possible customization can be categorized into three groups:

1.  Encode behavior customization by configuring [`config::Encoder`],
2.  Input enhancement by implementing [`source::Source`] trait,
3.  Add custom post-processing via structs in [`component`] submodule.

[`config::Encoder`]: https://docs.rs/flacenc/latest/flacenc/config/struct.Encoder.html
[`source::Source`]: https://docs.rs/flacenc/latest/flacenc/source/trait.Source.html
[`component`]: https://docs.rs/flacenc/latest/flacenc/component/index.html

## Feature Flags

`flacenc` has a few Cargo features that changes the internal behaviors and APIs.

-   `experimental`: Enables experimental coding algorithms that are typically
    slower. Due to its experimental nature, there's no documentation on how to
    activate each algorithm. You may need to explore `flacenc::config` module or
    source codes.
-   `fakesimd`: Uses a naive array implementation for mimicking `portable_simd`
    feature that can only be used in a nightly toolchain. This feature is not
    enabled by default, but required to be activated for using this crate with a
    stable toolchain.
-   `mimalloc`: Enables [`mimalloc`](https://crates.io/crates/mimalloc) global
    allocator. This can lead a small performance improvement in `par`-mode.
-   `par`: (This feature is enabled by default) Enables multi-thread encoding in
    `encode_with_fixed_block_size` function if `config` argument is properly
    configured (when `par` is enabled the default configuration enables
    multi-thread mode.). If you want to disable multi-thread mode and make the
    dependency tree smaller, you may do that by `default-featuers = false`.
    `par` adds dependency to
    [`crossbeam-channel`](https://crates.io/crates/crossbeam-channel) crate.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for details.

## License

Apache 2.0; see [`LICENSE`](LICENSE.md) for details.

## Disclaimer

This project is not an official Google project. It is not supported by Google
and Google specifically disclaims all warranties as to its quality,
merchantability, or fitness for a particular purpose.

This encoder is still unstable and sometimes the encoded file may contain
distortion, i.e. the encoder is sometimes not "lossless". You can check whether
you encountered an encoder bug by running the reference decoder. The FLAC format
contains MD5 digest of the input signal, and the reference decoder checks if the
digest of the decoded signal matches with the stored one.
