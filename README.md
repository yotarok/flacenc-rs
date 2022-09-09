# flacenc-rs

FLAC (Free Lossless Audio Codec) encoder for rustaceans.
See [the auto-generated report](report/report.md) for the characteristics of
the encoder compared to [FLAC reference implementation](https://xiph.org/flac/download.html)

This encoder currently requires nightly rust for compilation because of
`portable_simd` feature used.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for details.

## License

Apache 2.0; see [`LICENSE`](LICENSE) for details.

## Disclaimer

This project is not an official Google project. It is not supported by
Google and Google specifically disclaims all warranties as to its quality,
merchantability, or fitness for a particular purpose.

This encoder is still unstable and sometimes the encoded file may contain
distortion, i.e. the encoder is sometimes not "lossless". You can check whether
you encountered an encoder bug by running the reference decoder.  The FLAC
format contains MD5 digest of the input signal, and the reference decoder
checks if the digest of the decoded signal matches with the stored one.

