//! LCEVC stream configuration: the global/sequence parameters plus defaults
//! and validation rules mirrored from the reference decoder
//! (`LdeGlobalConfig` and friends).

use crate::bitstream::write_multibyte;
use crate::bitstream::BitWriter;

/// Resolution table of the spec (Table 20). Index = resolution_type.
/// Index 0 is unused; 63 is custom. Matches `kResolutions` in the decoder.
pub const RESOLUTIONS: [(u16, u16); 64] = {
    const fn r(w: u16, h: u16) -> (u16, u16) {
        (w, h)
    }
    let mut t = [(0u16, 0u16); 64];
    t[1] = r(360, 200);
    t[2] = r(400, 240);
    t[3] = r(480, 320);
    t[4] = r(640, 360);
    t[5] = r(640, 480);
    t[6] = r(768, 480);
    t[7] = r(800, 600);
    t[8] = r(852, 480);
    t[9] = r(854, 480);
    t[10] = r(856, 480);
    t[11] = r(960, 540);
    t[12] = r(960, 640);
    t[13] = r(1024, 576);
    t[14] = r(1024, 600);
    t[15] = r(1024, 768);
    t[16] = r(1152, 864);
    t[17] = r(1280, 720);
    t[18] = r(1280, 800);
    t[19] = r(1280, 1024);
    t[20] = r(1360, 768);
    t[21] = r(1366, 768);
    t[22] = r(1400, 1050);
    t[23] = r(1440, 900);
    t[24] = r(1600, 1200);
    t[25] = r(1680, 1050);
    t[26] = r(1920, 1080);
    t[27] = r(1920, 1200);
    t[28] = r(2048, 1080);
    t[29] = r(2048, 1152);
    t[30] = r(2048, 1536);
    t[31] = r(2160, 1440);
    t[32] = r(2560, 1440);
    t[33] = r(2560, 1600);
    t[34] = r(2560, 2048);
    t[35] = r(3200, 1800);
    t[36] = r(3200, 2048);
    t[37] = r(3200, 2400);
    t[38] = r(3440, 1440);
    t[39] = r(3840, 1600);
    t[40] = r(3840, 2160);
    t[41] = r(3840, 2400);
    t[42] = r(4096, 2160);
    t[43] = r(4096, 3072);
    t[44] = r(5120, 2880);
    t[45] = r(5120, 3200);
    t[46] = r(5120, 4096);
    t[47] = r(6400, 4096);
    t[48] = r(6400, 4800);
    t[49] = r(7680, 4320);
    t[50] = r(7680, 4800);
    t
};

pub const RESOLUTION_CUSTOM: u8 = 63;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TransformType {
    /// 2x2 directional decomposition transform (4 layers).
    Dd,
    /// 4x4 directional decomposition transform (16 layers).
    Dds,
}

impl TransformType {
    pub fn layers(self) -> usize {
        match self {
            TransformType::Dd => 4,
            TransformType::Dds => 16,
        }
    }
    pub fn tu_size(self) -> usize {
        match self {
            TransformType::Dd => 2,
            TransformType::Dds => 4,
        }
    }
    pub fn to_bit(self) -> u8 {
        match self {
            TransformType::Dd => 0,
            TransformType::Dds => 1,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum UpsampleType {
    Nearest,
    Linear,
    Cubic,
    ModifiedCubic,
}

impl UpsampleType {
    pub fn to_bit(self) -> u8 {
        match self {
            UpsampleType::Nearest => 0,
            UpsampleType::Linear => 1,
            UpsampleType::Cubic => 2,
            UpsampleType::ModifiedCubic => 3,
        }
    }
    /// Kernel taps: {k0, k1, k2, k3} as in the decoder's `kKernels`.
    pub fn kernel(self) -> [i16; 4] {
        match self {
            UpsampleType::Nearest => [0, 16384, 0, 0],
            UpsampleType::Linear => [0, 12288, 4096, 0],
            UpsampleType::Cubic => [-1382, 14285, 3942, -461],
            UpsampleType::ModifiedCubic => [-2360, 15855, 4165, -1276],
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ScalingMode {
    /// No scaling.
    Scale0D,
    /// 2:1 horizontally only.
    Scale1D,
    /// 2:1 in both dimensions.
    Scale2D,
}

impl ScalingMode {
    pub fn to_bit(self) -> u8 {
        self as u8
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ChromaFormat {
    Monochrome,
    C420,
    C422,
    C444,
}

impl ChromaFormat {
    pub fn to_bit(self) -> u8 {
        self as u8
    }
    pub fn num_planes(self) -> usize {
        match self {
            ChromaFormat::Monochrome => 1,
            _ => 3,
        }
    }
    /// Luma sample shifts to chroma dimensions.
    pub fn shift(self) -> (u32, u32) {
        match self {
            ChromaFormat::Monochrome => (0, 0),
            ChromaFormat::C420 => (1, 1),
            ChromaFormat::C422 => (1, 0),
            ChromaFormat::C444 => (0, 0),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TileDimensions {
    None,
    T512x256,
    T1024x512,
    Custom,
}

impl TileDimensions {
    pub fn to_bit(self) -> u8 {
        match self {
            TileDimensions::None => 0,
            TileDimensions::T512x256 => 1,
            TileDimensions::T1024x512 => 2,
            TileDimensions::Custom => 3,
        }
    }
    pub fn default_size(self) -> Option<(u16, u16)> {
        match self {
            TileDimensions::T512x256 => Some((512, 256)),
            TileDimensions::T1024x512 => Some((1024, 512)),
            _ => None,
        }
    }
}

/// Quantization-matrix mode (picture config, 3 bits).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum QuantMatrixMode {
    UsePrevious,
    UseDefault,
    CustomBoth,
    CustomLoq0,
    CustomLoq1,
    CustomBothUnique,
}

/// Colour metadata carried from the source into the VVC base VUI and the
/// MP4 `colr` box. The name fields are the ffmpeg option names (e.g.
/// "bt2020", "smpte2084", "bt2020nc"); the numeric fields are the
/// CICP values used by the MP4 nclx box.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct ColourInfo {
    pub primaries_name: String,
    pub transfer_name: String,
    pub matrix_name: String,
    pub primaries: u16,
    pub transfer: u16,
    pub matrix: u16,
    pub full_range: bool,
}

/// The full encoder configuration, held constant for the stream.
#[derive(Clone, Debug)]
pub struct LcevcConfig {
    pub width: u16,
    pub height: u16,

    pub profile: u8,
    pub level: u8,
    pub sublevel: u8,

    pub chroma: ChromaFormat,
    pub base_depth: u8,       // bits per sample, base layer (8, 10, 12, 14)
    pub enhancement_depth: u8, // bits per sample, enhanced output
    pub transform: TransformType,
    pub upsampler: UpsampleType,
    pub scaling_l1: ScalingMode, // base -> L1 (scalingModes[LOQ1] in decoder)
    pub scaling_l2: ScalingMode, // L1 -> L2 (scalingModes[LOQ0] in decoder)
    pub tile_dimensions: TileDimensions,
    pub custom_tile_size: Option<(u16, u16)>,
    pub per_tile_entropy_compression: bool, // compression_type_entropy_enabled_per_tile_flag
    pub tile_size_compression: u8,          // compression_type_size_per_tile (0..2)
    pub predicted_average: bool,
    pub temporal_enabled: bool,
    pub temporal_tile_intra_signalling: bool,
    pub temporal_step_width_modifier: u8,
    pub chroma_step_width_multiplier: u8,
    pub user_data: u8, // 0 = disabled, 1 = 2 bits, 2 = 6 bits
    pub colour: Option<ColourInfo>,
    pub loq1_use_enhanced_depth: bool,
    pub level1_filtering_signalled: bool,
    pub level1_filtering_first_coefficient: u8, // signalled value (0..15)
    pub level1_filtering_second_coefficient: u8,
}

impl Default for LcevcConfig {
    fn default() -> Self {
        LcevcConfig {
            width: 1920,
            height: 1080,
            profile: 0,
            level: 3,
            sublevel: 0,
            chroma: ChromaFormat::C420,
            base_depth: 8,
            enhancement_depth: 8,
            transform: TransformType::Dds,
            upsampler: UpsampleType::Linear,
            scaling_l1: ScalingMode::Scale2D,
            scaling_l2: ScalingMode::Scale2D,
            tile_dimensions: TileDimensions::None,
            custom_tile_size: None,
            per_tile_entropy_compression: false,
            tile_size_compression: 0,
            predicted_average: true,
            temporal_enabled: false,
            temporal_tile_intra_signalling: false,
            temporal_step_width_modifier: 48,
            chroma_step_width_multiplier: 64,
            user_data: 0,
            colour: None,
            loq1_use_enhanced_depth: false,
            level1_filtering_signalled: false,
            level1_filtering_first_coefficient: 0,
            level1_filtering_second_coefficient: 0,
        }
    }
}

impl LcevcConfig {
    pub fn num_planes(&self) -> usize {
        self.chroma.num_planes()
    }

    pub fn num_layers(&self) -> usize {
        self.transform.layers()
    }

    pub fn tu_size(&self) -> usize {
        self.transform.tu_size()
    }

    pub fn is_tiled(&self) -> bool {
        self.tile_dimensions != TileDimensions::None
    }

    /// Validate the configuration against the decoder's constraints.
    pub fn validate(&self) -> Result<(), String> {
        if self.width == 0 || self.height == 0 {
            return Err("zero resolution".into());
        }
        if self.tile_dimensions != TileDimensions::None {
            if self.tile_dimensions == TileDimensions::Custom {
                let (tw, th) = self
                    .custom_tile_size
                    .ok_or("custom tile size required for TileDimensions::Custom")?;
                if tw == 0 || th == 0 {
                    return Err("invalid custom tile size".into());
                }
            }
        }
        // Resolution alignment check (mirrors validateResolution in the
        // decoder): width/height must be whole transforms after scaling and
        // chroma subsampling.
        let transform_alignment = self.tu_size() as u32;
        let hori = transform_alignment
            * if self.scaling_l2 != ScalingMode::Scale0D { 2 } else { 1 }
            * if self.chroma != ChromaFormat::Monochrome && self.chroma != ChromaFormat::C444 {
                2
            } else {
                1
            };
        let vert = transform_alignment
            * if self.scaling_l2 == ScalingMode::Scale2D { 2 } else { 1 }
            * if self.chroma == ChromaFormat::C420 { 2 } else { 1 };
        if self.width as u32 % hori != 0 || self.height as u32 % vert != 0 {
            return Err(format!(
                "resolution {}x{} not aligned for this configuration (needs {}x{})",
                self.width, self.height, hori, vert
            ));
        }
        // Tiled picture: tile dims must be divisible by the transform size for
        // every plane (mirrors calculateTileCounts).
        if self.is_tiled() {
            let (tw0, th0) = match self.tile_dimensions {
                TileDimensions::Custom => self.custom_tile_size.unwrap(),
                _ => self.tile_dimensions.default_size().unwrap(),
            };
            let (hshift, vshift) = self.chroma.shift();
            for p in 0..self.num_planes() {
                let (tw, th) = if p == 0 {
                    (tw0, th0)
                } else {
                    (
                        (tw0 + (1 << hshift) - 1) >> hshift,
                        (th0 + (1 << vshift) - 1) >> vshift,
                    )
                };
                let ts = self.tu_size() as u16;
                if tw % ts != 0 || th % ts != 0 {
                    return Err(format!(
                        "tile dimensions {tw}x{th} (plane {p}) not divisible by transform size {ts}"
                    ));
                }
            }
        }
        Ok(())
    }

    /// Resolution type code for the configured picture size.
    pub fn resolution_type(&self) -> u8 {
        for (i, &(w, h)) in RESOLUTIONS.iter().enumerate() {
            if i > 0 && w == self.width && h == self.height {
                return i as u8;
            }
        }
        RESOLUTION_CUSTOM
    }

    /// Dimensions of LOQ0 (L2), LOQ1 (L1) and LOQ2 (base) in luma samples.
    /// Mirrors `ldePlaneDimensionsFromConfig` with loqIdx scaling.
    pub fn loq_dimensions(&self) -> [(u16, u16); 3] {
        let mut dims = [(0u16, 0u16); 3];
        let (mut w, mut h) = (self.width, self.height);
        dims[0] = (w, h);
        if self.scaling_l2 != ScalingMode::Scale0D {
            w = (w + 1) >> 1;
            if self.scaling_l2 == ScalingMode::Scale2D {
                h = (h + 1) >> 1;
            }
        }
        dims[1] = (w, h);
        if self.scaling_l1 != ScalingMode::Scale0D {
            w = (w + 1) >> 1;
            if self.scaling_l1 == ScalingMode::Scale2D {
                h = (h + 1) >> 1;
            }
        }
        dims[2] = (w, h);
        dims
    }

    /// Plane dimensions (luma or chroma) for a given LOQ.
    pub fn plane_dimensions(&self, loq: usize, plane: usize) -> (u16, u16) {
        let (mut w, mut h) = self.loq_dimensions()[loq];
        if plane > 0 {
            let (hshift, vshift) = self.chroma.shift();
            w = (w + (1 << hshift) - 1) >> hshift;
            h = (h + (1 << vshift) - 1) >> vshift;
        }
        (w, h)
    }

    /// Tile dimensions per plane (mirrors calculateTileDimensions: chroma
    /// tiles are the chroma-sized tiles).
    pub fn tile_dimensions_plane(&self, plane: usize) -> (u16, u16) {
        if !self.is_tiled() {
            let (w, h) = self.plane_dimensions(0, plane);
            return (w, h);
        }
        let (tw0, th0) = match self.tile_dimensions {
            TileDimensions::Custom => self.custom_tile_size.unwrap(),
            _ => self.tile_dimensions.default_size().unwrap(),
        };
        if plane == 0 {
            (tw0, th0)
        } else {
            let (hshift, vshift) = self.chroma.shift();
            (
                (tw0 + (1 << hshift) - 1) >> hshift,
                (th0 + (1 << vshift) - 1) >> vshift,
            )
        }
    }

    /// Number of tiles for a plane at a given LOQ
    /// (mirrors calculateTileCounts: ceil(loqWidth / tileWidth) * ...).
    pub fn num_tiles(&self, loq: usize, plane: usize) -> usize {
        let (pw, ph) = self.plane_dimensions(loq, plane);
        let (tw, th) = self.tile_dimensions_plane(plane);
        let across = (pw + tw - 1) / tw;
        let down = (ph + th - 1) / th;
        (across * down) as usize
    }

    /// Sample depth of the enhanced output (8 or 10 bits).
    pub fn sample_depth(&self) -> u8 {
        self.enhancement_depth
    }

    /// Derived step widths for each LOQ (per picture, set later).
    pub fn is_8bit(&self) -> bool {
        self.base_depth == 8 && self.enhancement_depth == 8
    }

    /// Write the sequence configuration payload (block type 0).
    pub fn write_sequence_config(&self) -> Vec<u8> {
        let mut w = BitWriter::new();
        w.write_bits(self.profile as u64, 4);
        w.write_bits(self.level as u64, 4);
        w.write_bits(self.sublevel as u64, 2);
        w.write_bit(false); // conformance_window_flag
        w.write_bits(0, 5); // reserved
        if self.profile == 15 || self.level == 15 {
            // extended_profile_idc (3), extended_level_idc (7), reserved (1)
            w.write_bits(0, 3);
            w.write_bits(0, 7);
            w.write_bit(false);
        }
        w.finish()
    }

    /// Write the global configuration payload (block type 1).
    /// Layout matches parseBlockGlobalConfig in the reference decoder.
    pub fn write_global_config(&self) -> Vec<u8> {
        let mut w = BitWriter::new();

        // Byte 1.
        let plane_mode_flag = self.num_planes() == 3;
        w.write_bit(plane_mode_flag);
        w.write_bits(self.resolution_type() as u64, 6);
        w.write_bits(self.transform.to_bit() as u64, 1);

        // Byte 2.
        w.write_bits(self.chroma.to_bit() as u64, 2);
        w.write_bits((self.base_depth / 2 - 4) as u64, 2);
        w.write_bits((self.enhancement_depth / 2 - 4) as u64, 2);
        w.write_bit(self.temporal_step_width_modifier != 48);
        w.write_bit(self.predicted_average);

        // Byte 3.
        w.write_bit(self.temporal_tile_intra_signalling);
        w.write_bit(self.temporal_enabled);
        w.write_bits(self.upsampler.to_bit() as u64, 3);
        w.write_bit(self.level1_filtering_signalled);
        w.write_bits(self.scaling_l1.to_bit() as u64, 2);

        // Byte 4.
        w.write_bits(self.scaling_l2.to_bit() as u64, 2);
        w.write_bits(self.tile_dimensions.to_bit() as u64, 2);
        w.write_bits(self.user_data as u64, 2);
        w.write_bit(self.loq1_use_enhanced_depth);
        w.write_bit(self.chroma_step_width_multiplier != 64);

        // Plane type byte (when plane_mode_flag == 1):
        // [plane_type:4][reserved:4], plane_type 1 = YUV (3 planes).
        if plane_mode_flag {
            w.write_bits(1, 4);
            w.write_bits(0, 4);
        }

        // temporal_step_width_modifier.
        if self.temporal_step_width_modifier != 48 {
            w.write_byte(self.temporal_step_width_modifier);
        }

        // level1 filtering coefficients.
        if self.level1_filtering_signalled {
            w.write_bits(self.level1_filtering_first_coefficient as u64, 4);
            w.write_bits(self.level1_filtering_second_coefficient as u64, 4);
        }

        // Tiling data.
        if self.is_tiled() {
            match self.tile_dimensions {
                TileDimensions::Custom => {
                    let (tw, th) = self.custom_tile_size.unwrap();
                    w.write_u16(tw);
                    w.write_u16(th);
                }
                _ => {}
            }
            w.write_bits(0, 5); // reserved
            w.write_bit(self.per_tile_entropy_compression);
            w.write_bits(self.tile_size_compression as u64, 2);
        }

        // Custom resolution.
        if self.resolution_type() == RESOLUTION_CUSTOM {
            w.write_u16(self.width);
            w.write_u16(self.height);
        }

        // chroma_step_width_multiplier.
        if self.chroma_step_width_multiplier != 64 {
            w.write_byte(self.chroma_step_width_multiplier);
        }

        w.finish()
    }
}

/// Derive a level_idc from the output sample rate (Table A.1).
pub fn level_for_sample_rate(width: u32, height: u32, fps: u32) -> u8 {
    let samples = width as u64 * height as u64 * fps as u64;
    if samples <= 29_410_000 {
        1
    } else if samples <= 124_560_000 {
        2
    } else if samples <= 527_650_000 {
        3
    } else {
        4
    }
}

/// Convenience: write a multibyte into a byte vector (used for chunk sizes).
pub fn multibyte_bytes(value: u64) -> Vec<u8> {
    let mut w = BitWriter::new();
    write_multibyte(&mut w, value);
    w.finish()
}
