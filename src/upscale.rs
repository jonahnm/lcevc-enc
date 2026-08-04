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
