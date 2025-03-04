# Example CLI for flacenc-rs

A CLI tool for encoding wav files into flac format, which compresses the input
waveform by ~60% without any degradation. This is an example application of
[flacenc](https://github.com/yotarok/flacenc-rs) library.

## Usage

To install (with using nightly rust; recommended):

```bash
cargo +nightly install flacenc-bin --features "simd-nightly"
```

Or, if you want to use stable channel:

```bash
cargo install flacenc-bin
```

Then, you can run encoding as follows:

```bash
flacenc --output output.flac input.wav
```

If you want to customize the encoder behavior, you can specify an additional
config file. To do so, first, you may generate the default config file by:

```bash
flacenc --output output.flac --dump-config config.toml input.wav
```

Then, edit `config.toml` and encode with the customized config, as:

```bash
flacenc --output output.flac --config config.toml input.wav
```

### (Experimental) decoder mode

This CLI tool can also decode FLAC files, as follows:

```bash
flacenc decode --output output.wav input.flac
```

## Limitation

The encoder CLI only supports WAV file with 8/ 16/ 24-bit PCM, currently.

The decoder CLI only supports raw-FLAC file (typically with ".flac" extension).
It currently does not support FLAC files with escape codes. However, at the
moment, escape codes are not widely used (e.g. the FLAC reference encoder
does not output FLAC files with escape codes.)

## Feature Flags

This binary crate has a few feature flags to enable additional features:

- `pprof`: If activated, the binary accept an additional command line argument
  `--pprof-output [FILE]`. If this flag is set, profiling data that can be
  processed by [`pprof`] are collected during the encoding process.
- `simd-nightly`: If activated, the dependency library is built with
  `simd-nightly` feature. This is a recommended setting; however, it is only
  available in a nightly toolchain.

## Contributing

See [`CONTRIBUTING.md`] for details.

## License

Apache 2.0; see [`LICENSE`] for details.

## Disclaimer

This encoder is still unstable and sometimes the encoded file may contain
distortion, i.e. the encoder very rarely outputs broken signals. You can check
whether you encountered an encoder bug by running, e.g., the reference decoder.
The FLAC format contains MD5 digest of the input signal, and the reference
decoder checks if the digest of the decoded signal matches with the stored one.

[`contributing.md`]: https://github.com/yotarok/flacenc-rs/blob/main/CONTRIBUTING.md
[`license`]: https://github.com/yotarok/flacenc-rs/blob/main/LICENSE
[`pprof`]: https://github.com/google/pprof
