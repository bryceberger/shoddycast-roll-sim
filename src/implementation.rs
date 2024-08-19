use rand::prelude::*;
use rayon::prelude::*;

pub fn rand_simd(num_iter: u64) -> u32 {
    // ~ 0.5 MM it/s
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

pub fn wyrand(num_iter: u64) -> u32 {
    // ~ 1.2 MM it/s
    (0..num_iter)
        .into_par_iter()
        .map_init(
            || wyrand::WyRand::new(random()),
            |rng, _| {
                (0..4)
                    .map(|_| {
                        let a = rng.rand();
                        let b = rng.rand();
                        (a & b).count_ones()
                    })
                    .sum()
            },
        )
        .reduce(|| 0, |x, y| x.max(y))
}

pub fn wyrand_simd(num_iter: u64) -> u32 {
    // ~ 1.4 MM it/s
    (0..num_iter)
        .into_par_iter()
        .map_init(
            || wyrand::WyRand::new(random()),
            |rng, _| {
                let a =
                    std::simd::u64x4::from_array([rng.rand(), rng.rand(), rng.rand(), rng.rand()]);
                let b =
                    std::simd::u64x4::from_array([rng.rand(), rng.rand(), rng.rand(), rng.rand()]);
                let r = a & b;
                r[0].count_ones() + r[1].count_ones() + r[2].count_ones() + r[3].count_ones()
            },
        )
        .reduce(|| 0, |x, y| x.max(y))
}
