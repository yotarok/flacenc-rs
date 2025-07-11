# ChangeLog

## [Unreleased]

### Breaking Changes

- API for creating `FrameHeader` is changed. Previously, some intermediate
  data-structures are exposed so the client code can even generate the invalid
  `FrameHeader`. Now, `SampleSizeSpec`, `SampleRateSpec`, and `BlockSizeSpec`
  are hidden, and internally generated. (#212, #219)
- `MetadataBlock` is also hidden now. Since `Stream` currently only handles
  `MetadataBlockData` and it's rather simpler than using a wrapping type
  `MetadataBlock`, this type is changed to be an internal type. (#218)

### Changed

- Performance improvement for LPC by changing algorithm. (#189)
- An abstraction layer to explicitly compute that unweighted auto-correlation
  (#190). Previously, it was done by a weight function that returns 1 always.
- "crc" dependency is upgraded, and now we use pre-computed tables. (#191)
- Removed dev-dependency to "bitvec" crate. (#193)
- Added more "#[inline]" to explicitly instruct inlining (#197)
- Experimental "flacdec-bin" is removed and fused into "flacenc-bin". (#209)
- Changed the default partition parameter for FixedLPC (#217)

### Added

- new decoder functionality (depending on "nom") is now publicized. (#208)
- added integration test that covers CLI `main` functions. (#214)

### Fixed

- Fixed QuickTime playback bug. (bug: #195, #198, follow-up fix in experimental
  decoder implementation: #201)
- Fixed a difference between flacenc and the reference encoder regarding the
  last-frame handling. (#203, #205, bug: #199)
- Documentation and code comment fix. (#188, #206, #207)
- Fixed Makefile bug that outputs intermediate results of "integration-nightly"
  to a wrong directory. (#204)
- Fixed Makefile bug that still refers "flacdec-bin" directory in some tasks.
  (#215)
- "component.rs" is split to multiple submodules. (#210, #211)
- Fixed encoder CLI error when dealing with WAV file with bps=8 or 24. (#216)

## 0.4.0 (flacenc-bin: 0.2.4) - 2024-03-05

### Breaking Changes

- `serde` traits are now an opt-out feature (enabled by default; #131)
- Some inaccurate constant definitions are fixed (#134)
  - All `MAX_` constants are now unified to represent inclusive ranges
- Renamed some constant names to be consistent with other names (#145)
  - `{MIN|MAX}_BLOCKSIZE` is now `{MIN|MAX}_BLOCK_SIZE`.
- Some constructors now return `Result<Self, VerifyError>` instead of `Self`
  - `component::Stream` (#157)
  - `source::FrameBuf` (#182)
- `component::MetadataBlockType` is now merged into `MetadataBlockData` (#162)
- `config::Encoder::block_sizes` is deprecated, use a scalar version
  `config::Encoder::block_size` instead (#186)

### Added

- API for constructing/ modifying components. Those changes are mainly for
  implementing decoding feature.
  - Constructor for `component::SubFrame`s (#156)
  - Constructor for `component::FrameHeader` (#160)
  - Field setters for `component::StreamInfo` (#159)
  - Accessors for `component::ChannelAssignment` (#163)
  - Constructor for `component::Frame` (#164)
  - Make `component::Stream::with_stream_info` visible (#165)
  - Add `component::MetadataBlockData::as_stream_info` (#166)
  - Change to derive Serialize/ Deserialize for `component::*` (#167)
- Decoder CLI (#139, #169)
  - This is currently only for a demonstration purpose. We do not publish a
    binary crate for this, and we don't try to maintain compatibility of this
    program at this moment.
- `MetadataBlock` can now handle opaque data. (#161)
- Input verification mechanism for software robustness:
  - Macros for data verification (#133)
  - Components are now verifiable (#135, #155)
  - Input verification (#136, #182)
  - `Verified` type for static checking of verification (#182)
- Configuration struct for FixedLPC coding (#140)

### Fixed

- Breakage due to toolchain change
  - API change of `std::simd` in nightly (#152)
  - Old `ahash` no longer builds with nightly (#179)
  - Lint rule changes (#154, #168, #170)
- Breakage due to dev-dependency change (#153)
  - This is fixed by just skipping tests in MSRV check, and by just checking it
    builds. This is not ideal, so we may change it again later.

### Refactored

- `quantize_parameter` is now not an associated item of `QuantizedParameter`
  (#126)
- `repeat` macro for cleaner loop-unrolling (#127)
- Bitcount for `Residual` is now pre-computed when it is constructed (#129)
- Introduced bitcount-estimator for FixedLPC (#143)
- Optimization of bitstream dump (#149)
- LPC
  - Recovery method when Levinson recursion failed is improved (#172)
  - Implemented generic precision LPC and use `f64` as a hard-coded default
    (where `f32` was used before; #174, #176)
  - Performance optimization (#175, #177, #178, #183)
- New signal generator for testing is introduced (#137)
- Code documentations (#150, #158, #171, #185)
- Special functions for 2-channel 16-bit inputs
  - upcasting (#146)
  - deinterleaving (#147)
- Development infrastructure improvements
  - Some benchmarks (#128)
  - Fuzz test (#138, #139, #141, #144, #180)
  - `Makefile.toml` instead of shell scripts (#130, #132)
  - CI for `flacdec-bin` and `fuzz` directories (#181)

## Hotfix (flacenc-bin: 0.2.3) - 2023-11-01

- flacenc-bin: fix `Cargo.toml` dependency specification.
  - This requires upload of the bin crate to crates.io, and an increment of the
    bin crate version.

## 0.3.1 - 2023-10-30

### Added

- Micro benchmarks for further optimization (#105)
- Mutable reference to `Source` is now also a `Source` (#106)
- [bin] fancier terminal output
  - iRTF and compression rate (#107)
  - Features used for building the library (#122)
- Some infrastructure for refactoring and optimization
  - `simd_map_and_reduce` (#114)
  - `slice_as_simd_mut`, `unaligned_map_and_update` (#117)
  - `SimdVec` for SIMD aligned vectors (#116)
- `log` feature (#121)
- Documentation (#123, #124)

### Fixed

- Document issues (#104)
- Test flakiness (#115)

### Changed

- Dependency change
  - `toml` is removed from library dependency (#112)
  - `md5` is replaced by `md-5` (#118)
  - `nalgebra` version is bumped to "0.32" (#120)
  - [build-dependencies] `built` is added (#122)
- Performance optimization
  - Misc. performance optimization (#108)
  - Components for fixed LPC is only constructed after optimizing the order
    (#109)
  - Performance optimization of LPC (#110, #114, #117)
  - Performance optimization of PRC (#111, #119)
  - Performance optimization of bitstream formatter (#113)

## 0.3.0 - 2023-10-20

### Breaking Changes

- Default feature change
  - flacenc-bin now uses mimalloc by default (#65)
  - "fakesimd" feature is now not-"simd-nightly" feature (#77)
  - and we finally give up having "simd-nightly" as a default (#87)
- Source API change (#70, #88)
  - This was introduced for more flexible buffer definition to make it possible
    to evict MD5 computation to another thread (#71)
  - Renamed `PackedBits` to `Bits` (#88)
    - The term "packed" is heavily overloaded and considered to be unnecessary
      in this case
    - Associated constants are now hidden (#99)
- BitSink API change
  - `ByteSink::as_byte_slice` and `ByteSink::into_bytes` are now generalized to
    non-`u8` storage and renamed to `as_slice` and `into_inner` (#97)
- Component API change
  - `Residual::rice_param` is renamed to `Residual::rice_parameter` (#101)
    - Given that the official document doesn't use this abbreviation and also it
      only saves 4 characters, it should be better to spell out.

### Added

- MSRV check in CI (#82)
- Fancier UI for flacenc-bin (#90)

### Fixed

- LPC order == 0 is no longer allowed (#68)
- "par" was not enabled in "flacenc-bin" (#83)
  - This issue is introduced after "0.2.0" release
- Fixed to address dependabot alerts #6 (#84, #85)
- Fixed an incorrect behavior when `Frame` is updated after bitstream is
  precomputed (#89)

### Changed

- Optimization
  - Bitstream (#63, #64, #66, #73, #79, #81)
    - Change `ByteSink` to be an alias of `MemSink<u8>` and introduced
      `MemSink<u64>` for faster bitstream computation (#79)
    - Changed to use precomputed buffer in `Frame` also for `count_bits` method
      (#81)
  - Input (#66, #71)
  - PRC (#66, #80)
  - LPC (#66)
  - Use unsafe transmute instead of copy from simd aligned value to a generic
    unaligned array (#76)
- Refactoring
  - LPC window config and computation is decoupled (#69)
  - `Crc::new` is now called in a static score (#72)
  - `reuse` macro for DRY (#74, #78)
  - `import_simd` macro for DRY (#86)
  - Stereo coding module (#75)
  - Fixed LPC module (#78)
- Document improvements (#62, #65, #67)

## 0.2.0 - 2023-10-09

### Breaking Changes

- Renamed `ByteVec` to `ByteSink`, `PreloadedSignal` to `MemSource` (#59)
- Refactored for release 0.2.0 (#59)
- Make CLI independent from `libsndfile` (#47)
  - As a result, CLI now only supports WAV file
- bitvec no longer implements `BitSink` trait (#22)
- Most of internal functions are now hidden in `pub(crate)` (#40, #59)

### Added

- `fakesimd` feature for stable toolchain (#32, #33, #37, #38)
- `pprof` feature for CLI (#21)
- `mimalloc` feature for faster multithread encoding (#57)
- `par` feature for multi-threading, and related improvements (#28, #42, #45,
  #52, #56)

### Fixed

- CLI tools are now built with "experimental" feature (#29)
- Rice parameter optimization wasn't complete (#55)

### Changed

- Generic performance improvements (#27, #31, #36, #43, #49, #50, #51, #53, #54)
- Output data structure replacing `bitvec` for better performance (#22, #24,
  #25, #26)
- Build flag change for faster CLI (#34, #39, #46, #57)

## 0.1.1 - 2023-09-20

### Fixed

- Instability when a rice parameter becomes too large (#13)

### Changed

- `PreloadedSignal::from_samples` is now a public function. (#8)
- CLI tool and library are now separated to different crates (#15)
- Experimental algorithms moved to the optional feature (#17)
