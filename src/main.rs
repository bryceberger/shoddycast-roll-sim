#![feature(portable_simd)]
#![cfg_attr(
    feature = "avx512",
    allow(internal_features),
    feature(stdarch_x86_avx512, core_intrinsics)
)]

use clap::Parser;
use num_format::{Locale, ToFormattedString};

mod implementation;

/// Somem implementations depart from the original: instead of running 231 rolls
/// per iteration, they run 256. This is done solely to make implementation
/// easier and faster. The affected implementations have `(256 rolls)` after
/// their name.
#[derive(Parser)]
struct Args {
    /// Number of iterations to run.
    #[arg(short, long, default_value_t = 1_000_000_000)]
    num_iter: u64,

    /// Which algorithms to run. If omitted, will run all available.
    algs: Vec<Alg>,
}

#[derive(Clone, Copy, clap::ValueEnum)]
enum Alg {
    #[cfg(feature = "x86")]
    RandSimd,
    Wyrand,
    WyrandSimd,
    WyrandSimdMultipleRng,
    #[cfg(feature = "avx512")]
    Wyrand512,
}

impl Alg {
    fn name(self) -> &'static str {
        match self {
            #[cfg(feature = "x86")]
            Alg::RandSimd => "rand, simd",
            Alg::Wyrand => "wyrand (256 rolls)",
            Alg::WyrandSimd => "wyrand, simd",
            Alg::WyrandSimdMultipleRng => "wyrand, simd, multiple rng",
            #[cfg(feature = "avx512")]
            Alg::Wyrand512 => "wyrand, avx512 (256 rolls)",
        }
    }

    fn run(self, num_iter: u64) -> u32 {
        use implementation::*;
        match self {
            #[cfg(feature = "x86")]
            Alg::RandSimd => rand_simd(num_iter),
            Alg::Wyrand => wyrand(num_iter),
            Alg::WyrandSimd => wyrand_simd(num_iter),
            Alg::WyrandSimdMultipleRng => wyrand_simd_multiple_rng(num_iter),
            #[cfg(feature = "avx512")]
            Alg::Wyrand512 => wyrand_avx512(num_iter),
        }
    }
}

fn main() {
    let args = Args::parse();

    println!(
        "number of iterations: {}",
        args.num_iter.to_formatted_string(&Locale::en)
    );

    let algs = if !args.algs.is_empty() {
        args.algs.as_slice()
    } else {
        &[
            #[cfg(feature = "x86")]
            Alg::RandSimd,
            Alg::Wyrand,
            Alg::WyrandSimd,
            Alg::WyrandSimdMultipleRng,
            #[cfg(feature = "avx512")]
            Alg::Wyrand512,
        ]
    };

    let name_width = algs.iter().map(|a| a.name().len()).max().unwrap();
    println!(
        "{:name_width$} | {:8} | {:8} | {:15} | {}",
        "name", "time (us)", "time", "it/s", "number"
    );
    println!(
        "{}",
        "-".repeat(name_width + 3 + 8 + 3 + 8 + 3 + 15 + 3 + 7)
    );

    for alg in algs {
        run_alg(name_width, args.num_iter, *alg);
    }
}

fn run_alg(name_width: usize, num_iter: u64, alg: Alg) {
    let start = std::time::Instant::now();
    let max = alg.run(num_iter);
    let elapsed = start.elapsed();

    let it_s = ((num_iter as f64 / elapsed.as_micros() as f64) * 1_000_000.) as u64;
    println!(
        "{:name_width$} | {:>9} | {elapsed:8.2?} | {:>15} | {max}",
        alg.name(),
        elapsed.as_micros(),
        it_s.to_formatted_string(&Locale::en)
    );
}
