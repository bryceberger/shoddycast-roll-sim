use rand::prelude::*;
use rayon::prelude::*;
use wyrand::WyRand;

fn base<T, R>(
    num_iter: u64,
    init: impl Fn() -> T + Sync + Send,
    op: impl Fn(&mut T) -> R + Sync + Send,
) -> R
where
    R: Send + Default + std::cmp::Ord,
{
    (0..num_iter)
        .into_par_iter()
        .map_init(init, |val, _| op(val))
        .reduce(Default::default, |x, y| x.max(y))
}

fn wyrand_init() -> WyRand {
    WyRand::new(random())
}

#[cfg(feature = "x86")]
pub fn rand_simd(num_iter: u64) -> u32 {
    base(num_iter, thread_rng, |rng| unsafe {
        use core::arch::x86_64::*;
        let a: __m256i = rng.random();
        let b: __m256i = rng.random();
        let r = _mm256_and_si256(a, b);

        _mm256_extract_epi64::<0>(r).count_ones()
            + _mm256_extract_epi64::<1>(r).count_ones()
            + _mm256_extract_epi64::<2>(r).count_ones()
            + _mm256_extract_epi64::<3>(r).count_ones()
    })
}

pub fn wyrand(num_iter: u64) -> u32 {
    base(num_iter, wyrand_init, |rng| {
        (0..4)
            .map(|_| {
                let a = rng.rand();
                let b = rng.rand();
                (a & b).count_ones()
            })
            .sum()
    })
}

pub fn wyrand_simd(num_iter: u64) -> u32 {
    base(num_iter, wyrand_init, |rng| {
        let a = std::simd::u64x4::from_array([rng.rand(), rng.rand(), rng.rand(), rng.rand()]);
        let b = std::simd::u64x4::from_array([rng.rand(), rng.rand(), rng.rand(), rng.rand()]);
        let r = a & b;
        r[0].count_ones() + r[1].count_ones() + r[2].count_ones() + r[3].count_ones()
    })
}

pub fn wyrand_simd_multiple_rng(num_iter: u64) -> u32 {
    base(
        num_iter / 4,
        || std::array::from_fn(|_| wyrand_init()),
        |[r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, ra, rb, rc, rd, re, rf]| {
            type T = std::simd::u64x4;
            let a0: T = [r0.rand(), r1.rand(), r2.rand(), r3.rand()].into();
            let b0: T = [r4.rand(), r5.rand(), r6.rand(), r7.rand()].into();

            let a1: T = [r8.rand(), r9.rand(), ra.rand(), rb.rand()].into();
            let b1: T = [rc.rand(), rd.rand(), re.rand(), rf.rand()].into();

            let a2: T = [r0.rand(), r1.rand(), r2.rand(), r3.rand()].into();
            let b2: T = [r4.rand(), r5.rand(), r6.rand(), r7.rand()].into();

            let a3: T = [r8.rand(), r9.rand(), ra.rand(), rb.rand()].into();
            let b3: T = [rc.rand(), rd.rand(), re.rand(), rf.rand()].into();

            let r0 = a0 & b0;
            let r1 = a1 & b1;
            let r2 = a2 & b2;
            let r3 = a3 & b3;

            let popcnt = |r: T| {
                r[0].count_ones() + r[1].count_ones() + r[2].count_ones() + r[3].count_ones()
            };
            let r0 = popcnt(r0);
            let r1 = popcnt(r1);
            let r2 = popcnt(r2);
            let r3 = popcnt(r3);
            r0.max(r1).max(r2).max(r3)
        },
    )
}

#[cfg(feature = "avx512")]
pub fn wyrand_avx512(num_iter: u64) -> u32 {
    base(
        num_iter,
        || std::array::from_fn(|_| wyrand_init()),
        |[r0, r1, r2, r3, r4, r5, r6, r7]| unsafe {
            use core::arch::x86_64::*;
            let rand: __m512i = std::simd::u64x8::from_array([
                r0.rand(),
                r1.rand(),
                r2.rand(),
                r3.rand(),
                r4.rand(),
                r5.rand(),
                r6.rand(),
                r7.rand(),
            ])
            .into();

            let low = _mm512_castsi512_si256(rand);
            let high = _mm512_extracti64x4_epi64::<1>(rand);
            let r = _mm256_popcnt_epi64(_mm256_and_si256(low, high));
            core::intrinsics::simd::simd_reduce_add_unordered::<_, i64>(r) as _
        },
    )
}
