//! Upscaling, bit-exact with the reference decoder's pixel processing
//! (`upscale_scalar.c` + `upscale_macros/scalar_{1d,2d}_loop.h`).
//!
//! The upscaler operates on the signed fixed-point planes. Both the 1D
//! (horizontal 2:1) and 2D (both axes 2:1) variants process the source in
//! blocks of 4 samples with edge clamping, convolve with the 4-tap kernel
//! (two phases), round `(acc + 0x2000) >> 14`, and optionally apply the
//! predicted-average (PA) modification.

use crate::config::ScalingMode;
use crate::frame::{sat_s15, sat_sample, Plane, PlaneS16};

const KERNEL_SHIFT: i32 = 14;

#[inline]
fn shift_round_s15(acc: i32) -> i16 {
    sat_s15((acc + 0x2000) >> KERNEL_SHIFT)
}

/// Horizontal convolution of a block of 4 source samples with the kernel.
/// `pel` holds 8 loaded source samples (already edge-clamped), centered on
/// the block. Returns the 8 upsampled outputs.
fn convolve_horizontal(pel: &[i16; 8], kernel: &[i16; 4]) -> [i16; 8] {
    let k = |i: usize| kernel[i] as i32;
    let p = |i: usize| pel[i] as i32;
    let mut out = [0i16; 8];
    for i in 0..4 {
        let odd = p(i) * k(3) + p(i + 1) * k(2) + p(i + 2) * k(1) + p(i + 3) * k(0);
        let even = p(i + 1) * k(0) + p(i + 2) * k(1) + p(i + 3) * k(2) + p(i + 4) * k(3);
        out[2 * i] = shift_round_s15(odd);
        out[2 * i + 1] = shift_round_s15(even);
    }
    out
}

/// Load the 8 source samples for a block, edge-clamped.
fn load_pel(src: &PlaneS16, src_x: i32, src_width: i32, y: usize) -> [i16; 8] {
    let mut pel = [0i16; 8];
    for i in 0..8 {
        let sx = (src_x - 2 + i as i32).clamp(0, src_width - 1) as usize;
        pel[i] = src.get(sx, y);
    }
    pel
}


#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn upscale_block_2d_avx2(
    s0: &[i16; 8], s1: &[i16; 8], s2: &[i16; 8], s3: &[i16; 8], s4: &[i16; 8],
    kernel: &[i16; 4],
) -> ([i16; 8], [i16; 8]) {
    use std::arch::x86_64::*;
    let (k0, k1, k2, k3) = (kernel[0], kernel[1], kernel[2], kernel[3]);

    // ---- Vertical pass (per-lane 4-tap via interleaved pairs) ----
    let mut v0 = [0i16; 8];
    let mut v1 = [0i16; 8];
    for phase in 0..2 {
        let (sa, sb, sc, sd, k_ab, k_cd) = if phase == 0 {
            (s0, s1, s2, s3, (k3, k2), (k1, k0))
        } else {
            (s1, s2, s3, s4, (k0, k1), (k2, k3))
        };
        let (lo, hi) = if phase == 0 {
            let il = _mm256_unpacklo_epi16(
                _mm256_loadu_si256(sa.as_ptr() as *const __m256i),
                _mm256_loadu_si256(sb.as_ptr() as *const __m256i),
            );
            let ih = _mm256_unpackhi_epi16(
                _mm256_loadu_si256(sa.as_ptr() as *const __m256i),
                _mm256_loadu_si256(sb.as_ptr() as *const __m256i),
            );
            let jl = _mm256_unpacklo_epi16(
                _mm256_loadu_si256(sc.as_ptr() as *const __m256i),
                _mm256_loadu_si256(sd.as_ptr() as *const __m256i),
            );
            let jh = _mm256_unpackhi_epi16(
                _mm256_loadu_si256(sc.as_ptr() as *const __m256i),
                _mm256_loadu_si256(sd.as_ptr() as *const __m256i),
            );
            let kab = _mm256_set1_epi16(k_ab.0);
            let kab2 = _mm256_set1_epi16(k_ab.1);
            let kcd = _mm256_set1_epi16(k_cd.0);
            let kcd2 = _mm256_set1_epi16(k_cd.1);
            let lo = _mm256_add_epi32(
                _mm256_madd_epi16(il, _mm256_unpacklo_epi16(kab, kab2)),
                _mm256_madd_epi16(jl, _mm256_unpacklo_epi16(kcd, kcd2)),
            );
            let hi = _mm256_add_epi32(
                _mm256_madd_epi16(ih, _mm256_unpacklo_epi16(kab, kab2)),
                _mm256_madd_epi16(jh, _mm256_unpacklo_epi16(kcd, kcd2)),
            );
            (lo, hi)
        } else {
            // phase 1: (s1,s2) and (s3,s4)
            let il = _mm256_unpacklo_epi16(
                _mm256_loadu_si256(sa.as_ptr() as *const __m256i),
                _mm256_loadu_si256(sb.as_ptr() as *const __m256i),
            );
            let ih = _mm256_unpackhi_epi16(
                _mm256_loadu_si256(sa.as_ptr() as *const __m256i),
                _mm256_loadu_si256(sb.as_ptr() as *const __m256i),
            );
            let jl = _mm256_unpacklo_epi16(
                _mm256_loadu_si256(sc.as_ptr() as *const __m256i),
                _mm256_loadu_si256(sd.as_ptr() as *const __m256i),
            );
            let jh = _mm256_unpackhi_epi16(
                _mm256_loadu_si256(sc.as_ptr() as *const __m256i),
                _mm256_loadu_si256(sd.as_ptr() as *const __m256i),
            );
            let kab = _mm256_set1_epi16(k_ab.0);
            let kab2 = _mm256_set1_epi16(k_ab.1);
            let kcd = _mm256_set1_epi16(k_cd.0);
            let kcd2 = _mm256_set1_epi16(k_cd.1);
            let lo = _mm256_add_epi32(
                _mm256_madd_epi16(il, _mm256_unpacklo_epi16(kab, kab2)),
                _mm256_madd_epi16(jl, _mm256_unpacklo_epi16(kcd, kcd2)),
            );
            let hi = _mm256_add_epi32(
                _mm256_madd_epi16(ih, _mm256_unpacklo_epi16(kab, kab2)),
                _mm256_madd_epi16(jh, _mm256_unpacklo_epi16(kcd, kcd2)),
            );
            (lo, hi)
        };
        // round + shift, then pack (saturating) and store the 8 lanes
        let rnd = _mm256_set1_epi32(0x2000);
        let lo = _mm256_srai_epi32::<14>(_mm256_add_epi32(lo, rnd));
        let hi = _mm256_srai_epi32::<14>(_mm256_add_epi32(hi, rnd));
        let p = _mm256_packs_epi32(lo, hi);
        let out = _mm256_castsi256_si128(p);
        if phase == 0 {
            _mm_storeu_si128(v0.as_mut_ptr() as *mut __m128i, out);
        } else {
            _mm_storeu_si128(v1.as_mut_ptr() as *mut __m128i, out);
        }
    }

    // ---- Horizontal pass (convolve_horizontal on v0/v1) ----
    let hcon = |v: &[i16; 8]| -> [i16; 8] {
        let vv = _mm256_loadu_si256(v.as_ptr() as *const __m256i);
        // vs1..vs4: the v shifted by 1..4 lanes (2 bytes per lane)
        let vs1 = _mm256_srli_si256::<2>(vv);
        let vs2 = _mm256_srli_si256::<4>(vv);
        let vs3 = _mm256_srli_si256::<6>(vv);
        let vs4 = _mm256_srli_si256::<8>(vv);
        let k32 = _mm256_unpacklo_epi16(_mm256_set1_epi16(k3), _mm256_set1_epi16(k2));
        let k10 = _mm256_unpacklo_epi16(_mm256_set1_epi16(k1), _mm256_set1_epi16(k0));
        let k01 = _mm256_unpacklo_epi16(_mm256_set1_epi16(k0), _mm256_set1_epi16(k1));
        let k23 = _mm256_unpacklo_epi16(_mm256_set1_epi16(k2), _mm256_set1_epi16(k3));
        // odd: out[2i] = v[i]*k3 + v[i+1]*k2 + v[i+2]*k1 + v[i+3]*k0
        let odd = _mm256_add_epi32(
            _mm256_madd_epi16(_mm256_unpacklo_epi16(vv, vs1), k32),
            _mm256_madd_epi16(_mm256_unpacklo_epi16(vs2, vs3), k10),
        );
        // even: out[2i+1] = v[i+1]*k0 + v[i+2]*k1 + v[i+3]*k2 + v[i+4]*k3
        let even = _mm256_add_epi32(
            _mm256_madd_epi16(_mm256_unpacklo_epi16(vs1, vs2), k01),
            _mm256_madd_epi16(_mm256_unpacklo_epi16(vs3, vs4), k23),
        );
        let rnd = _mm256_set1_epi32(0x2000);
        let odd = _mm256_srai_epi32::<14>(_mm256_add_epi32(odd, rnd));
        let even = _mm256_srai_epi32::<14>(_mm256_add_epi32(even, rnd));
        let p = _mm256_packs_epi32(odd, even);
        // p low-128 = (odd0..3, even0..3); interleave with itself shifted
        // by 4 lanes to get (odd0,even0,odd1,even1,...)
        let plo = _mm256_castsi256_si128(p);
        let ps = _mm_srli_si128::<8>(plo);
        let out = _mm_unpacklo_epi16(plo, ps);
        let mut row = [0i16; 8];
        _mm_storeu_si128(row.as_mut_ptr() as *mut __m128i, out);
        row
    };
    let row0 = hcon(&v0);
    let row1 = hcon(&v1);
    (row0, row1)
}

/// 2:1 horizontal upscale.
pub fn upscale_1d(src: &PlaneS16, kernel: &[i16; 4], apply_pa: bool) -> PlaneS16 {
    let src_width = src.width as i32;
    let src_height = src.height;
    let dst_width = (src_width * 2) as usize;
    let mut dst = PlaneS16::new(dst_width, src_height);

    let right_edge_src_x = if src_width >= 4 { src_width - 4 } else { 0 };

    for y in 0..src_height {
        let mut src_x = 0;
        while src_x <= src_width {
            let right_edge = src_x >= right_edge_src_x;
            let block_src_x = if right_edge { right_edge_src_x } else { src_x };
            let dst_x = (block_src_x * 2) as usize;

            let pel = load_pel(src, block_src_x, src_width, y);
            let mut row = convolve_horizontal(&pel, kernel);

            if apply_pa {
                let mut base = [0i16; 4];
                for i in 0..4 {
                    let bx = (block_src_x + i as i32).clamp(0, src_width - 1) as usize;
                    base[i] = src.get(bx, y);
                }
                for i in 0..4 {
                    let p = 2 * i;
                    let mean2 = ((row[p] as i32 + row[p + 1] as i32 + 1) >> 1) as i16;
                    let adjust = (base[i] as i32) - (mean2 as i32);
                    row[p] = (row[p] as i32 + adjust).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                    row[p + 1] =
                        (row[p + 1] as i32 + adjust).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                }
            }

            for i in 0..8 {
                dst.set(dst_x + i, y, row[i]);
            }

            src_x += 4;
        }
    }
    dst
}

/// 2:1 vertical + horizontal upscale.
pub fn upscale_2d(src: &PlaneS16, kernel: &[i16; 4], apply_pa: bool) -> PlaneS16 {
    let src_width = src.width as i32;
    let src_height = src.height as i32;
    let dst_width = (src_width * 2) as usize;
    let dst_height = (src_height * 2) as usize;
    let mut dst = PlaneS16::new(dst_width, dst_height);

    let right_edge_src_x = if src_width >= 4 { src_width - 4 } else { 0 };

    for src_y in 0..src_height {
        let y0 = (src_y - 2).clamp(0, src_height - 1) as usize;
        let y1 = (src_y - 1).clamp(0, src_height - 1) as usize;
        let y2 = src_y as usize;
        let y3 = (src_y + 1).clamp(0, src_height - 1) as usize;
        let y4 = (src_y + 2).clamp(0, src_height - 1) as usize;

        let dst_y0 = ((src_y * 2) + 0).clamp(0, dst_height as i32 - 1) as usize;
        let dst_y1 = ((src_y * 2) + 1).clamp(0, dst_height as i32 - 1) as usize;

        let k = |i: usize| kernel[i] as i32;

        let mut src_x = 0;
        while src_x <= src_width {
            let right_edge = src_x >= right_edge_src_x;
            let block_src_x = if right_edge { right_edge_src_x } else { src_x };
            let dst_x = (block_src_x * 2) as usize;

            // Interior blocks (no edge clamping needed) use the SIMD path.
            let interior = block_src_x >= 2 && block_src_x + 5 < src_width;
            #[cfg(target_arch = "x86_64")]
            let simd = interior && std::is_x86_feature_detected!("avx2");
            #[cfg(not(target_arch = "x86_64"))]
            let simd = false;
            #[cfg(target_arch = "x86_64")]
            let (mut row0, mut row1) = if simd {
                let mut s0 = [0i16; 8];
                let mut s1 = [0i16; 8];
                let mut s2 = [0i16; 8];
                let mut s3 = [0i16; 8];
                let mut s4 = [0i16; 8];
                let bx = block_src_x as usize;
                let w = src_width as usize;
                for i in 0..8 {
                    s0[i] = src.get(bx - 2 + i, y0);
                    s1[i] = src.get(bx - 2 + i, y1);
                    s2[i] = src.get(bx - 2 + i, y2);
                    s3[i] = src.get(bx - 2 + i, y3);
                    s4[i] = src.get(bx - 2 + i, y4);
                }
                let _ = w;
                unsafe { upscale_block_2d_avx2(&s0, &s1, &s2, &s3, &s4, kernel) }
            } else {
                let mut s0 = [0i16; 8];
                let mut s1 = [0i16; 8];
                let mut s2 = [0i16; 8];
                let mut s3 = [0i16; 8];
                let mut s4 = [0i16; 8];
                for i in 0..8 {
                    let sx = (block_src_x - 2 + i as i32).clamp(0, src_width - 1) as usize;
                    s0[i] = src.get(sx, y0);
                    s1[i] = src.get(sx, y1);
                    s2[i] = src.get(sx, y2);
                    s3[i] = src.get(sx, y3);
                    s4[i] = src.get(sx, y4);
                }

                // Vertical pass.
                let mut v0 = [0i16; 8];
                let mut v1 = [0i16; 8];
                for i in 0..8 {
                    let a0 = s0[i] as i32 * k(3) + s1[i] as i32 * k(2) + s2[i] as i32 * k(1) + s3[i] as i32 * k(0);
                    let a1 = s1[i] as i32 * k(0) + s2[i] as i32 * k(1) + s3[i] as i32 * k(2) + s4[i] as i32 * k(3);
                    v0[i] = shift_round_s15(a0);
                    v1[i] = shift_round_s15(a1);
                }

                // Horizontal pass on both intermediate rows.
                let row0 = convolve_horizontal(&v0, kernel);
                let row1 = convolve_horizontal(&v1, kernel);
                (row0, row1)
            };
            #[cfg(not(target_arch = "x86_64"))]
            let (mut row0, mut row1) = {
                let mut s0 = [0i16; 8];
                let mut s1 = [0i16; 8];
                let mut s2 = [0i16; 8];
                let mut s3 = [0i16; 8];
                let mut s4 = [0i16; 8];
                for i in 0..8 {
                    let sx = (block_src_x - 2 + i as i32).clamp(0, src_width - 1) as usize;
                    s0[i] = src.get(sx, y0);
                    s1[i] = src.get(sx, y1);
                    s2[i] = src.get(sx, y2);
                    s3[i] = src.get(sx, y3);
                    s4[i] = src.get(sx, y4);
                }
                let mut v0 = [0i16; 8];
                let mut v1 = [0i16; 8];
                for i in 0..8 {
                    let a0 = s0[i] as i32 * k(3) + s1[i] as i32 * k(2) + s2[i] as i32 * k(1) + s3[i] as i32 * k(0);
                    let a1 = s1[i] as i32 * k(0) + s2[i] as i32 * k(1) + s3[i] as i32 * k(2) + s4[i] as i32 * k(3);
                    v0[i] = shift_round_s15(a0);
                    v1[i] = shift_round_s15(a1);
                }
                let row0 = convolve_horizontal(&v0, kernel);
                let row1 = convolve_horizontal(&v1, kernel);
                (row0, row1)
            };

            let mut row0 = row0;
            let mut row1 = row1;

            if apply_pa {
                let mut base = [0i16; 4];
                for i in 0..4 {
                    let bx = (block_src_x + i as i32).clamp(0, src_width - 1) as usize;
                    base[i] = src.get(bx, y2);
                }
                for i in 0..4 {
                    let p = 2 * i;
                    let mean4 =
                        ((row0[p] as i32 + row0[p + 1] as i32 + row1[p] as i32 + row1[p + 1] as i32 + 2)
                            >> 2) as i16;
                    let adjust = base[i] as i32 - mean4 as i32;
                    row0[p] = (row0[p] as i32 + adjust).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                    row0[p + 1] =
                        (row0[p + 1] as i32 + adjust).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                    row1[p] = (row1[p] as i32 + adjust).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                    row1[p + 1] =
                        (row1[p + 1] as i32 + adjust).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                }
            }

            for i in 0..8 {
                dst.set(dst_x + i, dst_y0, row0[i]);
                dst.set(dst_x + i, dst_y1, row1[i]);
            }

            src_x += 4;
        }
    }
    dst
}

/// Upscale a plane with the given scaling mode.
pub fn upscale_plane(
    src: &PlaneS16,
    mode: ScalingMode,
    kernel: &[i16; 4],
    apply_pa: bool,
) -> PlaneS16 {
    match mode {
        ScalingMode::Scale0D => src.clone(),
        ScalingMode::Scale1D => upscale_1d(src, kernel, apply_pa),
        ScalingMode::Scale2D => upscale_2d(src, kernel, apply_pa),
    }
}

/// Downscale an 8-bit plane by 2:1 in one or both dimensions using a 2x2
/// box average (encoder-side choice; not part of the bitstream format).
pub fn downscale_plane(src: &Plane, mode: ScalingMode, depth: u8) -> Plane {
    if mode == ScalingMode::Scale0D {
        return src.clone();
    }
    let dst_w = (src.width + 1) / 2;
    let dst_h = match mode {
        ScalingMode::Scale2D => (src.height + 1) / 2,
        _ => src.height,
    };
    let mut dst = Plane::new(dst_w, dst_h);
    for y in 0..dst_h {
        for x in 0..dst_w {
            let sxx = (x * 2).min(src.width - 1);
            let syy = (y * 2).min(src.height - 1);
            let mut acc = src.get(sxx, syy) as i32;
            let mut n = 1;
            if x * 2 + 1 < src.width {
                acc += src.get(sxx + 1, syy) as i32;
                n += 1;
            }
            if y * 2 + 1 < src.height {
                acc += src.get(sxx, syy + 1) as i32;
                n += 1;
                if x * 2 + 1 < src.width {
                    acc += src.get(sxx + 1, syy + 1) as i32;
                    n += 1;
                }
            }
            dst.set(x, y, sat_sample((acc + n / 2) / n, depth));
        }
    }
    dst
}

#[cfg(all(test, target_arch = "x86_64"))]
mod simd_tests {
    use super::*;

    #[test]
    fn upscale_block_avx2_matches_scalar() {
        let kernel: [i16; 4] = [-2360, 15855, 4165, -1276];
        for seed in 0..50u64 {
            let mut rng = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let gen = |rng: &mut u64| -> i16 {
                *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((*rng >> 33) % 65536) as i16
            };
            let s0 = [gen(&mut rng), gen(&mut rng), gen(&mut rng), gen(&mut rng), gen(&mut rng), gen(&mut rng), gen(&mut rng), gen(&mut rng)];
            let s1 = [gen(&mut rng), gen(&mut rng), gen(&mut rng), gen(&mut rng), gen(&mut rng), gen(&mut rng), gen(&mut rng), gen(&mut rng)];
            let s2 = [gen(&mut rng), gen(&mut rng), gen(&mut rng), gen(&mut rng), gen(&mut rng), gen(&mut rng), gen(&mut rng), gen(&mut rng)];
            let s3 = [gen(&mut rng), gen(&mut rng), gen(&mut rng), gen(&mut rng), gen(&mut rng), gen(&mut rng), gen(&mut rng), gen(&mut rng)];
            let s4 = [gen(&mut rng), gen(&mut rng), gen(&mut rng), gen(&mut rng), gen(&mut rng), gen(&mut rng), gen(&mut rng), gen(&mut rng)];
            // scalar vertical + horizontal
            let mut v0 = [0i16; 8];
            let mut v1 = [0i16; 8];
            for i in 0..8 {
                let a0 = s0[i] as i32 * kernel[3] as i32 + s1[i] as i32 * kernel[2] as i32
                    + s2[i] as i32 * kernel[1] as i32 + s3[i] as i32 * kernel[0] as i32;
                let a1 = s1[i] as i32 * kernel[0] as i32 + s2[i] as i32 * kernel[1] as i32
                    + s3[i] as i32 * kernel[2] as i32 + s4[i] as i32 * kernel[3] as i32;
                v0[i] = shift_round_s15(a0);
                v1[i] = shift_round_s15(a1);
            }
            let expected0 = convolve_horizontal(&v0, &kernel);
            let expected1 = convolve_horizontal(&v1, &kernel);
            let (got0, got1) = unsafe { upscale_block_2d_avx2(&s0, &s1, &s2, &s3, &s4, &kernel) };
            assert_eq!(got0, expected0, "row0 seed {seed}");
            assert_eq!(got1, expected1, "row1 seed {seed}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::Plane;

    #[test]
    fn bilinear_1d_matches_hand_calc() {
        // Source: [0, 64, 128, 192] at fixed point: (v-128)*128.
        let mut src = PlaneS16::new(4, 1);
        let vals = [0u8, 64, 128, 192];
        for (i, &v) in vals.iter().enumerate() {
            src.set(i, 0, crate::frame::sample_to_s16(v as u16, 8));
        }
        let dst = upscale_1d(&src, &[0, 12288, 4096, 0], false);
        // Fixed point: -16384, -8192, 0, 8192. Pel block (right edge, width 4):
        // [-16384,-16384,-16384,-8192,0,8192,8192,8192]
        // out[2i] = (pel[i+1]*4096 + pel[i+2]*12288 + 0x2000) >> 14
        // out[2i+1] = (pel[i+2]*12288 + pel[i+3]*4096 + 0x2000) >> 14
        let exp = [-16384i16, -14336, -10240, -6144, -2048, 2048, 6144, 8192];
        for i in 0..8 {
            assert_eq!(dst.get(i, 0), exp[i], "out {i}");
        }
    }

    #[test]
    fn nearest_kernel_copies() {
        // Nearest kernel: dst(2i) = pel[i+2], dst(2i+1) = pel[i+1].
        let mut src = PlaneS16::new(4, 1);
        let vals = [10i16, 20, 30, 40];
        for (i, &v) in vals.iter().enumerate() {
            src.set(i, 0, v);
        }
        let dst = upscale_1d(&src, &[0, 16384, 0, 0], false);
        // pel = [10,10,10,20,30,40,40,40]
        // out[2i] = pel[i+2] (k2 at position i+2), out[2i+1] = pel[i+2]
        let exp = [10i16, 10, 20, 20, 30, 30, 40, 40];
        for i in 0..8 {
            assert_eq!(dst.get(i, 0), exp[i], "out {i}");
        }
    }

    #[test]
    fn downscale_2x2_average() {
        let mut src = Plane::new(4, 4);
        for y in 0..4 {
            for x in 0..4 {
                src.set(x, y, if (x / 2 + y / 2) % 2 == 0 { 100 } else { 200 });
            }
        }
        let dst = downscale_plane(&src, ScalingMode::Scale2D, 8);
        assert_eq!(dst.width, 2);
        assert_eq!(dst.height, 2);
        assert_eq!(dst.get(0, 0), 100);
        assert_eq!(dst.get(1, 0), 200);
        assert_eq!(dst.get(0, 1), 200);
        assert_eq!(dst.get(1, 1), 100);
    }
}
#[cfg(test)]
mod upscale_debug_tests {
    use crate::config::ScalingMode;
    use crate::frame::Plane;
    use crate::upscale::{downscale_plane, upscale_plane};
    use crate::frame::PlaneS16;
    use crate::frame::sample_to_s16;

    #[test]
    fn debug_upscale_1d_smooth() {
        // Base like the real one: 640 wide, row 0 = smooth gradient.
        let mut base = Plane::new(640, 1);
        for x in 0..640 {
            base.set(x, 0, (80 + x / 32) as u16);
        }
        let s16 = base.to_s16(8);
        let dst = upscale_plane(&s16, ScalingMode::Scale1D, &[0, 12288, 4096, 0], false);
        let mut bad = Vec::new();
        for x in 0..1280 {
            let v = dst.get(x, 0);
            let expected = (80 + (x / 2) / 32) as i16; // approx
            let up = sample_to_s16(if x % 64 < 32 { (80 + x/64) as u16 } else { (80 + x/64 + 1) as u16 }, 8);
            let _ = up;
            let _ = expected;
            if v == 0 || v == sample_to_s16(128, 8) || v == sample_to_s16(0, 8) {
                bad.push((x, v));
            }
        }
        println!("bad positions: {bad:?}");
        assert!(bad.is_empty(), "garbage found");
    }
}
