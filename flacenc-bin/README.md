# Example CLI for flacenc-rs

A CLI tool for encoding wav files into flac format, which compresses the
input waveform by ~60% without any degradation. This is an example application
of [flacenc](https://github.com/yotarok/flacenc-rs) library.

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
config toml file. To do so, first, you may generate the default config file by:

```bash
flacenc --output output.flac --dump-config config.toml input.wav
```

Then, you can edit `config.toml` and do:

```bash
flacenc --output output.flac --config config.toml input.wav
```
