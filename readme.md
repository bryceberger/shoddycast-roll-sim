Rust nightly toolchain required [](https://rustup.rs/).

Show information with `cargo run --release -- --help`.

If you have a CPU that supports AVX512, you can enable some additional
implementations with the `"avx512"` feature, ex.
`cargo run --release --features avx512`

Untested on non-x86 CPUs. Some of the algorithms use Rust's portable SIMD. To
use those (and not compile the x86 code), disable the default features, ex.
`cargo run --release --no-default-features`
