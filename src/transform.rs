//! Transform module.
//!
//! The decoder (lcevcdec `transform.c`) applies fixed integer inverse
//! transforms (the spec's "directional decomposition transform"). The
//! encoder uses the exact algebraic inverses so that
//! `forward(inverse(c)) == c` in exact arithmetic.
//!
//! Residual layout: for a 2x2 transform the residual index is row-major
//! (`x + 2*y`). For a 4x4 transform the residual buffer is stored in a
//! 2x2 grid of 2x2 sub-blocks (the "layer ordering" of the decoder's
//! deblocking filter):
//!
//! ```text
//! [ 0  1  4  5 ]
//! [ 2  3  6  7 ]
//! [ 8  9 12 13 ]
//! [10 11 14 15 ]
//! ```
//!
//! i.e. `index = (x & 1) | ((y & 1) << 1) | ((x >> 1) << 2) | ((y >> 1) << 3)`.

/// Map a sample position inside a transform block to the residual buffer index.
pub fn residual_index(tu_size: usize, x: usize, y: usize) -> usize {
    match tu_size {
        2 => x + 2 * y,
        _ => (x & 1) | ((y & 1) << 1) | ((x >> 1) << 2) | ((y >> 1) << 3),
    }
}

/// Inverse of residual_index.
pub fn residual_position(tu_size: usize, index: usize) -> (usize, usize) {
    match tu_size {
        2 => (index & 1, index >> 1),
        _ => ((index & 1) | ((index >> 2) & 1) << 1, ((index >> 1) & 1) | ((index >> 3) & 1) << 1),
    }
}

// ---------------------------------------------------------------------------
// Decoder-mirror inverse transforms (bit-exact with lcevcdec transform.c).
// ---------------------------------------------------------------------------

/// 2x2 transform, scaling mode 0 (none) or 2 (2D).
pub fn inverse_dd_2d(c: &[i16; 4]) -> [i16; 4] {
    [
        sat16(c[0] as i32 + c[1] as i32 + c[2] as i32 + c[3] as i32),
        sat16(c[0] as i32 - c[1] as i32 + c[2] as i32 - c[3] as i32),
        sat16(c[0] as i32 + c[1] as i32 - c[2] as i32 - c[3] as i32),
        sat16(c[0] as i32 - c[1] as i32 - c[2] as i32 + c[3] as i32),
    ]
}

/// 2x2 transform, scaling mode 1 (1D horizontal).
pub fn inverse_dd_1d(c: &[i16; 4]) -> [i16; 4] {
    [
        sat16(c[0] as i32 + c[1] as i32 + c[2] as i32),
        sat16(c[0] as i32 - c[1] as i32 - c[2] as i32),
        sat16(c[3] as i32 + c[1] as i32 - c[2] as i32),
        sat16(c[3] as i32 - c[1] as i32 + c[2] as i32),
    ]
}

fn h4_group(c: &[i16; 16], off: usize) -> [i32; 4] {
    // Per-group (c[4i..4i+3]) stage-1 values: [a, h, v, d].
    let g = |k: usize| c[off + k] as i32;
    [
        g(0) + g(1) + g(2) + g(3),
        g(0) - g(1) + g(2) - g(3),
        g(0) + g(1) - g(2) - g(3),
        g(0) - g(1) - g(2) + g(3),
    ]
}

/// 4x4 transform, scaling mode 2 (2D). Stage 1 groups coefficients
/// c[4i..4i+3] into (a,h,v,d)_i; stage 2 applies the 4-point Hadamard to
/// each of the four component vectors.
pub fn inverse_dds_2d(c: &[i16; 16]) -> [i16; 16] {
    let g0 = h4_group(c, 0);
    let g1 = h4_group(c, 4);
    let g2 = h4_group(c, 8);
    let g3 = h4_group(c, 12);
    let mut r = [0i16; 16];
    for comp in 0..4 {
        let q = [g0[comp], g1[comp], g2[comp], g3[comp]];
        let base = comp * 4;
        r[base] = sat16(q[0] + q[1] + q[2] + q[3]);
        r[base + 1] = sat16(q[0] - q[1] + q[2] - q[3]);
        r[base + 2] = sat16(q[0] + q[1] - q[2] - q[3]);
        r[base + 3] = sat16(q[0] - q[1] - q[2] + q[3]);
    }
    r
}

/// 4x4 transform, scaling mode 1 (1D horizontal). The second pass is the
/// DD1D stage-2 applied to each component vector:
///   r0 = x0+x1+x3, r1 = x0-x1-x3, r2 = x1+x2-x3, r3 = x2-x1+x3.
pub fn inverse_dds_1d(c: &[i16; 16]) -> [i16; 16] {
    let g0 = h4_group(c, 0);
    let g1 = h4_group(c, 4);
    let g2 = h4_group(c, 8);
    let g3 = h4_group(c, 12);
    let mut r = [0i16; 16];
    for comp in 0..4 {
        let q = [g0[comp], g1[comp], g2[comp], g3[comp]];
        let base = comp * 4;
        r[base] = sat16(q[0] + q[1] + q[3]);
        r[base + 1] = sat16(q[0] - q[1] - q[3]);
        r[base + 2] = sat16(q[1] + q[2] - q[3]);
        r[base + 3] = sat16(q[2] - q[1] + q[3]);
    }
    r
}

// ---------------------------------------------------------------------------
// Encoder forward transforms (exact inverses, scaled integer numerators).
//
// Each forward transform returns `(num, denom)` such that the exact
// coefficient is `num / denom`.
// ---------------------------------------------------------------------------

/// Forward 2x2 (2D): C = H4^-1 R = H4 R / 4.
pub fn forward_dd_2d(r: &[i16; 4]) -> ([i32; 4], i32) {
    let g = |k: usize| r[k] as i32;
    let c = [
        g(0) + g(1) + g(2) + g(3),
        g(0) - g(1) + g(2) - g(3),
        g(0) + g(1) - g(2) - g(3),
        g(0) - g(1) - g(2) + g(3),
    ];
    (c, 4)
}

/// Forward 2x2 (1D): C = H1D^-1 R. The decoder's inverseDD1D matrix is
/// [[1,1,1,0],[1,-1,-1,0],[0,1,-1,1],[0,-1,1,1]], whose exact inverse is
/// c0=(r0+r1)/2, c1=(r0-r1+r2-r3)/4, c2=(r0-r1-r2+r3)/4, c3=(r2+r3)/2.
/// All numerators are scaled by 4 (denominator 4).
pub fn forward_dd_1d(r: &[i16; 4]) -> ([i32; 4], i32) {
    let g = |k: usize| r[k] as i32;
    let c = [
        2 * (g(0) + g(1)),
        g(0) - g(1) + g(2) - g(3),
        g(0) - g(1) - g(2) + g(3),
        2 * (g(2) + g(3)),
    ];
    (c, 4)
}

/// Forward 4x4 (2D): both stages are H4 (scaled by 1/4 each), total /16.
/// Stage 2 runs on the residual quadrants; stage 1 groups the components
/// back into coefficient groups.
pub fn forward_dds_2d(r: &[i16; 16]) -> ([i32; 16], i32) {
    // Stage 2 (invert the decoder's second pass): per residual quadrant q,
    // the four stage-1 components are H4 r_q / 4.
    let q = |off: usize| {
        let g = |k: usize| r[off + k] as i32;
        [
            g(0) + g(1) + g(2) + g(3),
            g(0) - g(1) + g(2) - g(3),
            g(0) + g(1) - g(2) - g(3),
            g(0) - g(1) - g(2) + g(3),
        ]
    };
    let quads = [q(0), q(4), q(8), q(12)];
    // Stage 1 (invert the decoder's first pass): c[4i..4i+3] =
    // H4 (comp0_i, comp1_i, comp2_i, comp3_i) / 4.
    let mut c = [0i32; 16];
    for i in 0..4 {
        let (a, h, v, d) = (quads[0][i], quads[1][i], quads[2][i], quads[3][i]);
        c[4 * i] = a + h + v + d;
        c[4 * i + 1] = a - h + v - d;
        c[4 * i + 2] = a + h - v - d;
        c[4 * i + 3] = a - h - v + d;
    }
    (c, 16)
}

/// Forward 4x4 (1D): stage 2 is the DD1D^-1 (with the x2/x3 permutation
/// folded in), stage 1 is H4, giving a total denominator of 16.
pub fn forward_dds_1d(r: &[i16; 16]) -> ([i32; 16], i32) {
    // Stage 2: invert the decoder's second pass for a component vector q:
    //   r0 = x0+x1+x3, r1 = x0-x1-x3, r2 = x1+x2-x3, r3 = x2-x1+x3
    // with (x0,x1,x2,x3) = (a0,a1,a3,a2). DD1D^-1 gives
    //   x0 = (r0+r1)/2, x1 = (r0-r1+r2-r3)/4,
    //   x2 = (r0-r1-r2+r3)/4, x3 = (r2+r3)/2
    // and the group-ordered components are (x0, x1, x3, x2), so
    // (4*comp0, 4*comp1, 4*comp2, 4*comp3) =
    //   (2(r0+r1), r0-r1+r2-r3, 2(r2+r3), r0-r1-r2+r3).
    let stage = |off: usize| {
        let g = |k: usize| r[off + k] as i32;
        [
            2 * (g(0) + g(1)),          // 4*comp0
            g(0) - g(1) + g(2) - g(3),  // 4*comp1
            2 * (g(2) + g(3)),          // 4*comp2
            g(0) - g(1) - g(2) + g(3),  // 4*comp3
        ]
    };
    let q0 = stage(0);
    let q1 = stage(4);
    let q2 = stage(8);
    let q3 = stage(12);
    // Stage 1: c[4i..4i+3] = H4 (comp0_i, comp1_i, comp2_i, comp3_i) / 4.
    // With comp = qj[i]/4 the numerator scale is 4*4 = 16.
    let mut c = [0i32; 16];
    for i in 0..4 {
        let (a, h, v, d) = (q0[i], q1[i], q2[i], q3[i]);
        c[4 * i] = a + h + v + d;
        c[4 * i + 1] = a - h + v - d;
        c[4 * i + 2] = a + h - v - d;
        c[4 * i + 3] = a - h - v + d;
    }
    (c, 16)
}

/// Select the forward transform for a transform type and scaling mode.
pub enum ForwardTransform {
    Dd1D,
    Dd2D,
    Dds1D,
    Dds2D,
}

impl ForwardTransform {
    pub fn new(transform_type: u8, scaling_1d: bool) -> Self {
        match (transform_type, scaling_1d) {
            (0, true) => ForwardTransform::Dd1D,
            (0, false) => ForwardTransform::Dd2D,
            (1, true) => ForwardTransform::Dds1D,
            _ => ForwardTransform::Dds2D,
        }
    }

    /// Apply the forward transform; returns (numerators, denominator).
    pub fn apply(&self, r: &[i16]) -> (Vec<i32>, i32) {
        match self {
            ForwardTransform::Dd1D => {
                let (c, d) = forward_dd_1d(r.try_into().unwrap());
                (c.to_vec(), d)
            }
            ForwardTransform::Dd2D => {
                let (c, d) = forward_dd_2d(r.try_into().unwrap());
                (c.to_vec(), d)
            }
            ForwardTransform::Dds1D => {
                let (c, d) = forward_dds_1d(r.try_into().unwrap());
                (c.to_vec(), d)
            }
            ForwardTransform::Dds2D => {
                let (c, d) = forward_dds_2d(r.try_into().unwrap());
                (c.to_vec(), d)
            }
        }
    }

    /// Apply the decoder-mirror inverse transform (used in the reconstruction
    /// loop and by tests).
    pub fn inverse(&self, c: &[i16]) -> Vec<i16> {
        match self {
            ForwardTransform::Dd1D => inverse_dd_1d(c.try_into().unwrap()).to_vec(),
            ForwardTransform::Dd2D => inverse_dd_2d(c.try_into().unwrap()).to_vec(),
            ForwardTransform::Dds1D => inverse_dds_1d(c.try_into().unwrap()).to_vec(),
            ForwardTransform::Dds2D => inverse_dds_2d(c.try_into().unwrap()).to_vec(),
        }
    }
}

#[inline]
pub fn sat16(v: i32) -> i16 {
    v.clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip_forward_inverse(transform: &ForwardTransform, r: &[i16]) {
        let (num, denom) = transform.apply(r);
        let n = r.len();
        // Exact inverse: c = num/denom; reconstruct r_hat = inverse(round(c))
        // can't round-trip exactly because of quantization, but verify the
        // algebraic identity: denom * inverse(c) == H * num for the linear
        // map. Instead verify: inverse(round(num/denom)) with the inverse
        // applied to the scaled coefficients must satisfy
        //   denom * inverse(num) == M * num  ==  denom^2 * r
        // for the orthogonal (2D) case; for 1D use direct verification.
        let c_scaled: Vec<i16> = num
            .iter()
            .map(|&x| {
                // round(num / denom)
                let q = if x >= 0 { (x + denom / 2) / denom } else { (x - denom / 2) / denom };
                q.clamp(i16::MIN as i32, i16::MAX as i32) as i16
            })
            .collect();
        let r_hat = transform.inverse(&c_scaled);
        let _ = n;
        // Reconstructed residual must be close (within rounding of the
        // quantization). For exactness check, verify the linear map:
        // inverse(num) * denom == num applied through decoder matrix == r * denom^2
        // Only verified implicitly via the DD case below.
        let _ = r_hat;
    }

    #[test]
    fn dd2d_forward_inverse_exact() {
        // Forward: num = H^T r; decoder inverse: H num = H H^T r = 4 r.
        for r in [[3i16, -7, 12, 4], [1, 1, 1, 1], [0, 0, 0, 0], [-8, 1, 0, -2]] {
            let (num, d) = forward_dd_2d(&r);
            assert_eq!(d, 4);
            let num16: [i16; 4] = num.map(|x| x as i16);
            let got = inverse_dd_2d(&num16);
            for i in 0..4 {
                assert_eq!(got[i], r[i] * 4, "residual {i}");
            }
        }
    }

    #[test]
    fn dd1d_forward_inverse_exact() {
        for r in [[1i16, 2, 3, 4], [5, -3, 2, 9], [0, 0, 0, 0], [-8, 1, 0, -2]] {
            let (num, d) = forward_dd_1d(&r);
            assert_eq!(d, 4);
            let num16: [i16; 4] = num.map(|x| x as i16);
            let got = inverse_dd_1d(&num16);
            for i in 0..4 {
                assert_eq!(got[i], r[i] * 4, "residual {i}");
            }
        }
    }

    #[test]
    fn dds2d_forward_inverse_exact() {
        let r: [i16; 16] = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        ];
        let (num, d) = forward_dds_2d(&r);
        assert_eq!(d, 16);
        let num16: [i16; 16] = num.map(|x| x as i16);
        let got = inverse_dds_2d(&num16);
        for i in 0..16 {
            assert_eq!(got[i], r[i] * 16, "residual {i}");
        }
    }

    #[test]
    fn dds1d_forward_inverse_exact() {
        let r: [i16; 16] = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        ];
        let (num, d) = forward_dds_1d(&r);
        assert_eq!(d, 16);
        let num16: [i16; 16] = num.map(|x| x as i16);
        let got = inverse_dds_1d(&num16);
        for i in 0..16 {
            assert_eq!(got[i], r[i] * 16, "residual {i}");
        }
    }

    #[test]
    fn residual_layout_roundtrip() {
        for size in [2usize, 4] {
            for y in 0..size {
                for x in 0..size {
                    let i = residual_index(size, x, y);
                    assert!(i < size * size);
                    assert_eq!(residual_position(size, i), (x, y));
                }
            }
        }
    }

    #[test]
    fn dds_1d_matrix_matches_decoder_mapping() {
        // Spot-check residuals from a single coefficient against the decoder's
        // inverseDDS1D formula on paper:
        // c = [8,0,...]: group 0 = [8,0,0,0]: a0 = 8, h0 = v0 = d0 = 8,
        // groups 1..3 zero: a = [8,0,0,0], h = v = d = [8,0,0,0].
        // residuals[0..3] (a-quadrant) = a0+a1+a3, a0-a1-a3, a1+a2-a3,
        // a2-a1+a3 = [8, 8, 0, 0].
        let mut c = [0i16; 16];
        c[0] = 8;
        let r = inverse_dds_1d(&c);
        assert_eq!(&r[0..4], &[8, 8, 0, 0]);
        // h-quadrant = [8, 8, 0, 0] too (h0 = 8).
        assert_eq!(&r[4..8], &[8, 8, 0, 0]);
    }
}

