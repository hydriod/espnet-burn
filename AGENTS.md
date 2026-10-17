# Copilot Instructions for espnet-burn

## Project Overview

**espnet-burn** is a Rust library that integrates the Burn deep learning framework with Open JTalk (Japanese phoneme/text processing) for speech synthesis applications. The project bridges Rust and C++ via CXX for FFI interoperability with the Open JTalk C/C++ library.

## Build, Test, and Lint Commands

### Prerequisites
- Rust 1.70+ with Cargo
- CMake 3.15+ (required for building Open JTalk C++ dependencies)
- C/C++ compiler toolchain (MSVC on Windows, GCC/Clang on Unix)

### Building
```bash
# Build the library in debug mode
cargo build

# Build optimized release binary
cargo build --release

# Build with specific features
cargo build --features "wgpu"
```

### Testing
```bash
# Run all tests
cargo test

# Run tests with output
cargo test -- --nocapture

# Run a specific test
cargo test test_name -- --exact

# Run tests in release mode
cargo test --release

# Generate coverage (requires cargo-tarpaulin)
cargo tarpaulin --out Html
```

### Linting and Formatting
```bash
# Check code style
cargo fmt --check

# Auto-format code
cargo fmt

# Run clippy linter
cargo clippy -- -D warnings

# Fix common issues automatically
cargo fix --allow-dirty
```

### Build Cache and Cleanup
```bash
# Remove build artifacts
cargo clean

# Check outdated dependencies
cargo outdated
```

## Architecture

### Module Structure
- **`text/`** - Text processing for phoneme handling
  - `phoneme_tokenizer.rs` - Converts Japanese text to phoneme sequences with prosody markers using Open JTalk
  - Uses regex-based feature extraction from MeCab labels
- **`openjtalk/`** - Placeholder for future OpenJTalk Rust wrapper bindings (currently empty)
- **`bin_mod/`** - Reserved for binary/CLI modules (currently empty)

### Key Integration Points

**Burn Integration**:
- Uses `burn::wgpu` backend for GPU-accelerated neural network operations
- Ready for future model inference integration

**CXX FFI Bridge**:
- `cxx = "1.0.199"` enables safe C++ interoperability
- `cxx-build` handles compile-time code generation for Rust-C++ bindings
- Currently used indirectly through the `open_jtalk` dependency

**Open JTalk Integration**:
- Git submodule at `./open_jtalk/` (r9y9/open_jtalk fork v1.10)
- Provides MeCab-based morphological analysis for Japanese text
- Generates label features for phoneme extraction and prosody analysis
- C/C++ CMake build managed separately; Rust crate binding expected at `../open_jtalk-rs/crates/open_jtalk` (currently missing - see Known Issues)

### Data Flow
```
Text Input
  ↓
text2mecab() → Morphological Analysis
  ↓
ManagedResource<Mecab> → Extract phoneme features
  ↓
NJD Processing → Accent phrase handling
  ↓
JpCommon Processing → Generate labeled features
  ↓
pyopenjtalk_g2p_prosody() → Phoneme + Prosody Markers
  ↓
Output: Vec<String> with phonemes, accents ([, ]), phrase breaks (#), etc.
```

### Prosody Markers
- `^` - Sentence start (sil)
- `$` - Sentence end (neutral)
- `?` - Question form
- `_` - Pause (pau)
- `[` - Pitch rise accent
- `]` - Pitch fall accent
- `#` - Accent phrase boundary
- Vowels: `A/I/U/E/O` (voiced), `a/i/u/e/o` (unvoiced, when `drop_unvoiced_vowel=true`)

## Key Conventions

### Code Organization
- Modules use declaration files (`mod.rs`) with clear sub-module organization
- Prefer descriptive function names over abbreviations (exception: common phonetics abbreviations like `g2p` for grapheme-to-phoneme)
- Functions working with Open JTalk data use prefix patterns: `text2*` for input processing, `*2features` for output extraction

### Error Handling
- Use `Result<T, String>` for fallible operations exposed via phoneme processing
- Return descriptive error messages for regex parsing failures and JTalk processing errors
- Handle edge cases explicitly: sil/pau tokens, empty inputs, mismatched accent phrase structures

### Testing Patterns
- Unit tests in `#[cfg(test)] mod tests` blocks within the same file
- Test names describe the behavior being verified (e.g., `it_works`)
- Integration tests test end-to-end phoneme extraction from actual Japanese text

### Dependencies Management
- Burn: Keep to stable releases; WGPU backend requires compatible GPU drivers
- CXX: Maintain exact version match with cxx-build to avoid codegen mismatches
- Open JTalk: Vendored as submodule; manual CMake build if modifying C++ code
- Regex: Use precompiled patterns for performance-critical paths (currently `Regex::new()` each call - optimization opportunity)

### Platform Considerations
- Windows: Uses MSVC compiler for C++ FFI; ensure Visual Studio build tools installed
- Linux/macOS: Tested with GCC and Clang in open_jtalk CI
- WGPU backend: Requires appropriate GPU drivers (D3D12 on Windows, Metal on macOS, Vulkan/GL on Linux)

## Known Issues & Limitations

**Build Blocker**: The dependency path in `espnet-burn/Cargo.toml` references `../open_jtalk-rs/crates/open_jtalk` which does not exist. Current workaround needed to locate or create the OpenJTalk Rust crate wrapper.

**Performance Optimization**: `phoneme_tokenizer.rs` recompiles regex patterns on each call; consider caching with `once_cell` or `lazy_static` for repeated text processing.

**Incomplete Modules**: `openjtalk/` and `bin_mod/` directories are placeholders; clear architectural intent needed for these future expansions.

## Useful References
- [Burn Framework Docs](https://burn.dev)
- [CXX Repository](https://github.com/dtolnay/cxx)
- [Open JTalk Source](https://github.com/r9y9/open_jtalk)
- Phoneme label format: HTS Full Context Label Format (used by Open JTalk MeCab)
