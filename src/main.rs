#![feature(portable_simd)]

use std::simd::cmp::SimdPartialOrd;

use num_format::{Locale, ToFormattedString};
use rand::prelude::*;
use rayon::prelude::*;

fn main() {
    let num_iter: u64 = std::env::args()
        .nth(1)
        .and_then(|x| x.parse().ok())
        .unwrap_or(1_000_000);
    let start = std::time::Instant::now();

    let max = par_iter_simd(num_iter);

    let elapsed = start.elapsed();
    let it_s = ((num_iter as f64 / elapsed.as_micros() as f64) * 1_000_000.) as u64;
    println!(
        "{} iterations in {:.3?} ({} it/s)",
        num_iter.to_formatted_string(&Locale::en),
        elapsed,
        it_s.to_formatted_string(&Locale::en)
    );
    println!("max: {}", max);
}

#[allow(dead_code)]
fn par_iter(num_iter: u64) -> u8 {
    // ~24,000,000 it/s
    (0..num_iter)
        .into_par_iter()
        .map_init(thread_rng, |rng, _| {
            (0..231)
                .map(|_| if rng.random::<u8>() % 4 == 0 { 0 } else { 1 })
                .sum::<u8>()
        })
        .reduce(|| 0, |x, y| x.max(y))
}

fn par_iter_simd(num_iter: u64) -> u32 {
    // simd = Single Instruction Multiple Data
    // e.g. `a + b` is actually `a[0] + b[0], a[1] + b[1], ...`

    // u8 = type of each element (unsigned, 8 bits. 8 bits is smallest available)
    // x8 = number of elements per register (u8x8 = 8 u8's crammed into a single value)

    // max value of u8 is 255. 25% of time will be less than 64
    // there's probably a better way to order the simd, but this works well
    // enough and should be somewhat portable

    // on my cpu (Ryzen 6900HS),
    // using a target of 100,000,000 iterations (somewhat slower with full 1B):
    // u8x8:  ~  50,000,000 it/s
    // u8x16: ~ 100,000,000 it/s
    // u8x32: ~ 140,000,000 it/s
    (0..num_iter)
        .into_par_iter()
        .map_init(thread_rng, |rng, _| {
            // if you change the lane width, also change this number
            // total amount of rolls per simulation will be `num_simd * lane_width`
            // e.g., 8 simd * 32 width = 256 rolls / sim
            //
            // this will affect the final scores (more rolls / sim -> higher numbers)
            let num_simd = 8;
            (0..num_simd)
                .map(|_| {
                    let rng: std::simd::u8x32 = rng.random();
                    rng.simd_lt(std::simd::u8x32::splat(64))
                        .to_bitmask()
                        .count_ones()
                })
                .sum()
        })
        .reduce(|| 0, |x, y| x.max(y))
}
