#![feature(portable_simd)]

use clap::Parser;
use num_format::{Locale, ToFormattedString};

mod implementation;

/// These implementations depart from the original: instead of running 231 rolls
/// per iteration, they run 256.
///
/// This is done solely to make implementation easier and faster.
#[derive(Parser)]
struct Args {
    /// Number of iterations to run.
    num_iter: u64,

    /// Which algorithms to run. If omitted, will run all.
    algs: Vec<Alg>,
}

#[derive(Clone, Copy, clap::ValueEnum)]
enum Alg {
    RandSimd,
    WyRand,
    WyRandSimd,
}

impl Alg {
    fn name(self) -> &'static str {
        match self {
            Alg::RandSimd => "rand, simd",
            Alg::WyRand => "wyrand",
            Alg::WyRandSimd => "wyrand, simd",
        }
    }

    fn run(self, num_iter: u64) -> u32 {
        match self {
            Alg::RandSimd => implementation::rand_simd(num_iter),
            Alg::WyRand => implementation::wyrand(num_iter),
            Alg::WyRandSimd => implementation::wyrand_simd(num_iter),
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
        &[Alg::RandSimd, Alg::WyRand, Alg::WyRandSimd]
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
