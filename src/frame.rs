//! Picture plane representations used by the encoder.
//!
//! The LCEVC decoding pipeline operates on signed 16-bit "fixed point"
//! planes: a `depth`-bit sample v is represented as
//! `(v << (15 - depth)) - 0x4000` (i.e. `(v - 2^(depth-1)) * 2^(14-depth)`).
//! Residuals from the transform are added directly in this domain, and the
//! final conversion back to samples is
//! `((v + 2^(13-depth)) >> (15-depth)) + 2^(depth-1)` (saturated).

/// A signed 16-bit fixed-point plane (row-major).
#[derive(Clone, Debug)]
pub struct PlaneS16 {
    pub width: usize,
    pub height: usize,
    pub data: Vec<i16>,
}

impl PlaneS16 {
    pub fn new(width: usize, height: usize) -> Self {
        PlaneS16 {
            width,
            height,
            data: vec![0; width * height],
        }
    }

    pub fn zeros(width: usize, height: usize) -> Self {
        Self::new(width, height)
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> i16 {
        self.data[y * self.width + x]
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, v: i16) {
        self.data[y * self.width + x] = v;
    }
}

/// Convert a sample to the internal signed fixed-point representation.
#[inline]
pub fn sample_to_s16(v: u16, depth: u8) -> i16 {
    (((v as i32) << (15 - depth as i32)) - 0x4000) as i16
}

/// Convert a fixed-point value back to a sample of the given depth.
#[inline]
pub fn s16_to_sample(v: i16, depth: u8) -> u16 {
    sat_sample(((v as i32 + (1 << (14 - depth as i32))) >> (15 - depth as i32)) + (1 << (depth as i32 - 1)), depth)
}

#[inline]
pub fn sat_sample(v: i32, depth: u8) -> u16 {
    v.clamp(0, (1 << depth) - 1) as u16
}

#[inline]
pub fn sat_s15(v: i32) -> i16 {
    v.clamp(-(1 << 14), (1 << 14) - 1) as i16
}

/// An unsigned sample plane (source/base/output); sample depth is carried
/// by the configuration (8 or 10 bits).
#[derive(Clone, Debug)]
pub struct Plane {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u16>,
}

impl Plane {
    pub fn new(width: usize, height: usize) -> Self {
        Plane {
            width,
            height,
            data: vec![0; width * height],
        }
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> u16 {
        self.data[y * self.width + x]
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, v: u16) {
        self.data[y * self.width + x] = v;
    }

    /// Convert to the internal signed fixed-point representation.
    pub fn to_s16(&self, depth: u8) -> PlaneS16 {
        PlaneS16 {
            width: self.width,
            height: self.height,
            data: self.data.iter().map(|&v| sample_to_s16(v, depth)).collect(),
        }
    }
}

impl PlaneS16 {
    /// Convert back to samples of the given depth.
    pub fn to_plane(&self, depth: u8) -> Plane {
        Plane {
            width: self.width,
            height: self.height,
            data: self.data.iter().map(|&v| s16_to_sample(v, depth)).collect(),
        }
    }
}

/// One full picture: three planes.
#[derive(Clone, Debug)]
pub struct Picture {
    pub width: usize,
    pub height: usize,
    pub planes: Vec<Plane>,
}

impl Picture {
    pub fn new(width: usize, height: usize, chroma: crate::config::ChromaFormat) -> Self {
        let (hshift, vshift) = chroma.shift();
        let nplanes = chroma.num_planes();
        let mut planes = Vec::with_capacity(nplanes);
        for p in 0..nplanes {
            let (pw, ph) = if p == 0 {
                (width, height)
            } else {
                ((width + (1 << hshift) - 1) >> hshift, (height + (1 << vshift) - 1) >> vshift)
            };
            planes.push(Plane::new(pw, ph));
        }
        Picture { width, height, planes }
    }
}
