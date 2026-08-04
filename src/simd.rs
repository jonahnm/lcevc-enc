//! Explicit x86 SIMD helpers for the hottest full-frame passes: the
//! sample<->s16 fixed-point conversions and squared-difference (SSE)
//! accumulation. The LCEVC pipeline is i16-based, so everything stays in
//! 16-bit lanes; AVX2 and SSE2 paths with a scalar fallback.

/// Convert samples to the internal i16 fixed-point representation.
pub fn samples_to_s16(data: &[u16], out: &mut [i16], depth: u8) {
    debug_assert_eq!(data.len(), out.len());
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            unsafe { samples_to_s16_avx2(data, out, depth) };
            return;
        }
        if std::is_x86_feature_detected!("sse2") {
            unsafe { samples_to_s16_sse2(data, out, depth) };
            return;
        }
    }
    for (i, &v) in data.iter().enumerate() {
        out[i] = crate::frame::sample_to_s16(v, depth);
    }
}

/// Convert i16 fixed-point back to samples (rounding + offset + clamp).
pub fn s16_to_samples(data: &[i16], out: &mut [u16], depth: u8) {
    debug_assert_eq!(data.len(), out.len());
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            unsafe { s16_to_samples_avx2(data, out, depth) };
            return;
        }
        if std::is_x86_feature_detected!("sse2") {
            unsafe { s16_to_samples_sse2(data, out, depth) };
            return;
        }
    }
    for (i, &v) in data.iter().enumerate() {
        out[i] = crate::frame::s16_to_sample(v, depth);
    }
}

/// Sum of squared u16 differences: `sum (a[i]-b[i])^2`.
pub fn sse_diff_u16(a: &[u16], b: &[u16]) -> u64 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            return unsafe { sse_diff_u16_avx2(a, b) };
        }
        if std::is_x86_feature_detected!("sse2") {
            return unsafe { sse_diff_u16_sse2(a, b) };
        }
    }
    let mut sse: u64 = 0;
    for i in 0..a.len() {
        let d = a[i] as i64 - b[i] as i64;
        sse += (d * d) as u64;
    }
    sse
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn samples_to_s16_avx2(data: &[u16], out: &mut [i16], depth: u8) {
    use std::arch::x86_64::*;
    let mut i = 0;
    while i + 16 <= data.len() {
        let v = _mm256_loadu_si256(data.as_ptr().add(i) as *const __m256i);
        let s = match 15 - depth as i32 {
            5 => _mm256_slli_epi16::<5>(v),
            _ => _mm256_slli_epi16::<7>(v),
        };
        let r = _mm256_sub_epi16(s, _mm256_set1_epi16(0x4000));
        _mm256_storeu_si256(out.as_mut_ptr().add(i) as *mut __m256i, r);
        i += 16;
    }
    for j in i..data.len() {
        out[j] = crate::frame::sample_to_s16(data[j], depth);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn samples_to_s16_sse2(data: &[u16], out: &mut [i16], depth: u8) {
    use std::arch::x86_64::*;
    let mut i = 0;
    while i + 8 <= data.len() {
        let v = _mm_loadu_si128(data.as_ptr().add(i) as *const __m128i);
        let s = match 15 - depth as i32 {
            5 => _mm_slli_epi16::<5>(v),
            _ => _mm_slli_epi16::<7>(v),
        };
        let r = _mm_sub_epi16(s, _mm_set1_epi16(0x4000));
        _mm_storeu_si128(out.as_mut_ptr().add(i) as *mut __m128i, r);
        i += 8;
    }
    for j in i..data.len() {
        out[j] = crate::frame::sample_to_s16(data[j], depth);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn s16_to_samples_avx2(data: &[i16], out: &mut [u16], depth: u8) {
    use std::arch::x86_64::*;
    let round = 1 << (14 - depth as i32);
    let off = 1 << (depth as i32 - 1);
    let maxv = (1 << depth as i32) - 1;
    let rv = _mm256_set1_epi16(round);
    let ov = _mm256_set1_epi16(off);
    let mx = _mm256_set1_epi16(maxv);
    let zr = _mm256_setzero_si256();
    let mut i = 0;
    while i + 16 <= data.len() {
        let v = _mm256_loadu_si256(data.as_ptr().add(i) as *const __m256i);
        let v = _mm256_add_epi16(v, rv);
        let s = match 15 - depth as i32 {
            5 => _mm256_srai_epi16::<5>(v),
            _ => _mm256_srai_epi16::<7>(v),
        };
        let s = _mm256_add_epi16(s, ov);
        let s = _mm256_max_epi16(s, zr);
        let s = _mm256_min_epi16(s, mx);
        _mm256_storeu_si256(out.as_mut_ptr().add(i) as *mut __m256i, s);
        i += 16;
    }
    for j in i..data.len() {
        out[j] = crate::frame::s16_to_sample(data[j], depth);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn s16_to_samples_sse2(data: &[i16], out: &mut [u16], depth: u8) {
    use std::arch::x86_64::*;
    let round = 1 << (14 - depth as i32);
    let off = 1 << (depth as i32 - 1);
    let maxv = (1 << depth as i32) - 1;
    let rv = _mm_set1_epi16(round);
    let ov = _mm_set1_epi16(off);
    let mx = _mm_set1_epi16(maxv);
    let zr = _mm_setzero_si128();
    let mut i = 0;
    while i + 8 <= data.len() {
        let v = _mm_loadu_si128(data.as_ptr().add(i) as *const __m128i);
        let v = _mm_add_epi16(v, rv);
        let s = match 15 - depth as i32 {
            5 => _mm_srai_epi16::<5>(v),
            _ => _mm_srai_epi16::<7>(v),
        };
        let s = _mm_add_epi16(s, ov);
        let s = _mm_max_epi16(s, zr);
        let s = _mm_min_epi16(s, mx);
        _mm_storeu_si128(out.as_mut_ptr().add(i) as *mut __m128i, s);
        i += 8;
    }
    for j in i..data.len() {
        out[j] = crate::frame::s16_to_sample(data[j], depth);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sse_diff_u16_avx2(a: &[u16], b: &[u16]) -> u64 {
    use std::arch::x86_64::*;
    let mut acc = _mm256_setzero_si256();
    let mut i = 0;
    while i + 16 <= a.len() {
        let va = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
        let vb = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
        let d = _mm256_sub_epi16(va, vb);
        let sq = _mm256_madd_epi16(d, d);
        acc = _mm256_add_epi32(acc, sq);
        i += 16;
    }
    let mut lanes = [0u32; 8];
    _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, acc);
    let mut sse: u64 = lanes.iter().map(|&l| l as u64).sum();
    for j in i..a.len() {
        let d = a[j] as i64 - b[j] as i64;
        sse += (d * d) as u64;
    }
    sse
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn sse_diff_u16_sse2(a: &[u16], b: &[u16]) -> u64 {
    use std::arch::x86_64::*;
    let mut acc = _mm_setzero_si128();
    let mut i = 0;
    while i + 8 <= a.len() {
        let va = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
        let vb = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);
        let d = _mm_sub_epi16(va, vb);
        let sq = _mm_madd_epi16(d, d);
        acc = _mm_add_epi32(acc, sq);
        i += 8;
    }
    let mut lanes = [0u32; 4];
    _mm_storeu_si128(lanes.as_mut_ptr() as *mut __m128i, acc);
    let mut sse: u64 = lanes.iter().map(|&l| l as u64).sum();
    for j in i..a.len() {
        let d = a[j] as i64 - b[j] as i64;
        sse += (d * d) as u64;
    }
    sse
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_samples_to_s16(data: &[u16], depth: u8) -> Vec<i16> {
        data.iter().map(|&v| crate::frame::sample_to_s16(v, depth)).collect()
    }

    fn scalar_s16_to_samples(data: &[i16], depth: u8) -> Vec<u16> {
        data.iter().map(|&v| crate::frame::s16_to_sample(v, depth)).collect()
    }

    #[test]
    fn conversions_match_scalar() {
        for depth in [8u8, 10u8] {
            let data: Vec<u16> = (0..300u16).map(|i| (i * 7) % (1u16 << depth)).collect();
            let mut out = vec![0i16; data.len()];
            samples_to_s16(&data, &mut out, depth);
            assert_eq!(out, scalar_samples_to_s16(&data, depth), "to_s16 depth {depth}");

            let mut back = vec![0u16; data.len()];
            s16_to_samples(&out, &mut back, depth);
            assert_eq!(back, scalar_s16_to_samples(&out, depth), "s16_to depth {depth}");
        }
    }

    #[test]
    fn sse_diff_matches_scalar() {
        let a: Vec<u16> = (0..300u16).map(|i| (i * 13) % 1024).collect();
        let b: Vec<u16> = (0..300u16).map(|i| (i * 29) % 1024).collect();
        let scalar: u64 = a.iter().zip(&b).map(|(&x, &y)| {
            let d = x as i64 - y as i64;
            (d * d) as u64
        }).sum();
        assert_eq!(sse_diff_u16(&a, &b), scalar);
    }
}
