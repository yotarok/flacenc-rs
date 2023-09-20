# Example CLI for flacenc-rs

See [flacenc-rs](https://github.com/yotarok/flacenc-rs) for project overview.


## Usage

```bash
flacenc --output output.flac input.wav
```

If you want to customize encoder behavior, you can specify an additional config
toml file. To do so, first, you may generate the default config file by:

```bash
flacenc --output output.flac --dump-config config.toml input.wav
```

Then, you can edit `config.toml` and do:

```bash
flacenc --output output.flac --config config.toml input.wav
```

