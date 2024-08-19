#![feature(portable_simd)]
#![feature(stdarch_x86_avx512)]

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

    let max = par_iter_3_manual(num_iter);

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

#[expect(dead_code)]
fn par_iter_1(num_iter: u64) -> u8 {
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

#[expect(dead_code)]
fn par_iter_2(num_iter: u64) -> u32 {
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

fn par_iter_3_manual(num_iter: u64) -> u32 {
    // ~ 500,000,000 it/s
    (0..num_iter)
        .into_par_iter()
        .map_init(thread_rng, |rng, _| unsafe {
            use core::arch::x86_64::*;
            let a: __m256i = rng.random();
            let b: __m256i = rng.random();
            let r = _mm256_and_si256(a, b);

            _mm256_extract_epi64::<0>(r).count_ones()
                + _mm256_extract_epi64::<1>(r).count_ones()
                + _mm256_extract_epi64::<2>(r).count_ones()
                + _mm256_extract_epi64::<3>(r).count_ones()
        })
        .reduce(|| 0, |x, y| x.max(y))
}

#[expect(dead_code)]
fn par_iter_3_avx512(num_iter: u64) -> i64 {
    // ~ 500,000,000 it/s
    // not really any faster than extracting and adding scalar
    // probably means there's a better way to do this
    (0..num_iter)
        .into_par_iter()
        .map_init(thread_rng, |rng, _| unsafe {
            use core::arch::x86_64::*;
            let a: __m256i = rng.random();
            let b: __m256i = rng.random();
            let r = _mm256_and_si256(a, b);

            let r = _mm256_popcnt_epi32(r);

            let b = _mm256_castsi256_si128(r);
            let t = _mm256_extracti128_si256::<1>(r);
            let r = _mm_add_epi64(b, t);

            let high = _mm_unpackhi_epi64(r, r);
            _mm_cvtsi128_si64(_mm_add_epi64(high, r))
        })
        .reduce(|| 0, |x, y| x.max(y))
}
