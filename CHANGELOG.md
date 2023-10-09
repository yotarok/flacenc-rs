# ChangeLog

## 0.2.0 - 2023-10-09

### Breaking Changes

-   Renamed `ByteVec` to `ByteSink`, `PreloadedSignal` to `MemSource` (#59)
-   Refactored for release 0.2.0 (#59)
-   Make CLI independent from `libsndfile` (#47)
    -   As a result, CLI now only supports WAV file
-   bitvec no longer implements `BitSink` trait (#22)
-   Most of internal functions are now hidden in `pub(crate)` (#40, #59)

### Added

-   `fakesimd` feature for stable toolchain (#32, #33, #37, #38)
-   `pprof` feature for CLI (#21)
-   `mimalloc` feature for faster multithread encoding (#57)
-   `par` feature for multi-threading, and related improvements
    (#28, #42, #45, #52, #56)

### Fixed

-   CLI tools are now built with "experimental" feature (#29)
-   Rice parameter optimization wasn't complete (#55)

### Changed

-   Generic performance improvements
    (#27, #31, #36, #43, #49, #50, #51, #53, #54)
-   Output data structure replacing `bitvec` for better performance
    (#22, #24, #25, #26)
-   Build flag change for faster CLI (#34, #39, #46, #57)

## 0.1.1 - 2023-09-20

### Fixed

-   Instability when a rice parameter becomes too large (#13)

### Changed

-   `PreloadedSignal::from_samples` is now a public function. (#8)
-   CLI tool and library are now separated to different crates (#15)
-   Experimental algorithms moved to the optional feature (#17)
