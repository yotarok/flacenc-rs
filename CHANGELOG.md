# ChangeLog

## Prerelease

### Fixed

- \[bin\] fix `Cargo.toml` depepndency specification.
  - This requires reupload of the bin crate to crates.io, and an increment of
    the bin crate version.

## 0.3.1 - 2023-10-30

### Added

- Micro benchmarks for further optimization (#105)
- Mutable reference to `Source` is now also a `Source` (#106)
- \[bin\] fancier terminal output
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
  - \[build-dependencies\] `built` is added (#122)
- Performance optimization
  - Misc. performance optimization (#108)
  - Components for fixed LPC is only constructed after optimizing the order
    (#109)
  - Performance optimization of LPC (#110, #114, #117)
  - Performance optimizatoin of PRC (#111, #119)
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
