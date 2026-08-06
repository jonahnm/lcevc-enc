//! Dequantization mirror: bit-exact reimplementation of the decoder's
//! `dequant.c` (step width derivation, deadzone, offsets) used by the
//! encoder's reconstruction loop and its quantizer.


pub const QMIN_STEP_WIDTH: i32 = 1;
pub const QMAX_STEP_WIDTH: i32 = 32767;

pub const KA: i32 = 39;
pub const KB: i32 = 126484;
pub const KC: i32 = 5242;
pub const KD: i32 = 99614;

const SW_DIVISOR_NO_DQ_OFFSET: i64 = 2147483648; // 1 << 31
const SW_DIVISOR: i64 = 32768;
const QM_SCALE_MAX: i64 = 196608; // 3 << 16
const DEADZONE_SW_LIMIT: i32 = 12249;
const FP_ONE_OVER_255: u16 = 257; // floor(1/255 * 65536)

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TemporalSignal {
    Inter = 0,
    Intra = 1,
}

/// Default quantization matrices (decoder's `kQuantMatrixDefault*`).
pub fn default_quant_matrix(transform_dds: bool, scaling_1d: bool, loq: usize) -> Vec<u8> {
    const DD_1D: [[u8; 4]; 2] = [[0, 2, 0, 0], [0, 3, 0, 32]];
    const DD_2D: [[u8; 4]; 2] = [[32, 3, 0, 32], [0, 3, 0, 32]];
    const DDS_1D: [[u8; 16]; 2] = [
        [13, 26, 19, 32, 52, 1, 78, 9, 13, 26, 19, 32, 150, 91, 91, 19],
        [0, 0, 0, 2, 52, 1, 78, 9, 26, 72, 0, 3, 150, 91, 91, 19],
    ];
    const DDS_2D: [[u8; 16]; 2] = [
        [13, 26, 19, 32, 52, 1, 78, 9, 26, 72, 0, 3, 150, 91, 91, 19],
        [0, 0, 0, 2, 52, 1, 78, 9, 26, 72, 0, 3, 150, 91, 91, 19],
    ];
    if !transform_dds {
        return if scaling_1d { DD_1D[loq].to_vec() } else { DD_2D[loq].to_vec() };
    }
    if scaling_1d {
        DDS_1D[loq].to_vec()
    } else {
        DDS_2D[loq].to_vec()
    }
}

/// Content-adaptive quant matrix: rescale the default per-layer matrix by
/// the ratio of the observed coefficient energy to its geometric mean,
/// `qm_l = default_l * (E_l / gm)^beta`. Layers with above-average energy get
/// a coarser step (fewer bits), flat layers stay fine. Signalled in the
/// picture config (quant_matrix_mode = CustomBothUnique).
pub fn content_quant_matrix(energy: &[f64], default: &[u8], beta: f64) -> Vec<u8> {
    let n = default.len().min(energy.len());
    if n == 0 {
        return default.to_vec();
    }
    let mut gm = 0.0f64;
    for l in 0..n {
        gm += energy[l].max(1e-9).ln();
    }
    gm = (gm / n as f64).exp().max(1e-9);
    let mut out = Vec::with_capacity(n);
    for l in 0..n {
        let ratio = (energy[l].max(1e-9) / gm).powf(beta);
        let v = (default[l] as f64 * ratio).round().clamp(1.0, 255.0) as u8;
        out.push(v);
    }
    out
}

/// Natural log of `step_width` as a U12.4 fixed point value returned as f64,
/// mirroring `calculateFixedPointU12_4Ln`.
fn fixed_point_u12_4_ln(step_width: i32) -> f64 {
    let log_sw = (step_width as f64).ln();
    let integer_log = log_sw.floor();
    let fractional = ((log_sw - integer_log) * 4096.0).floor() / 4096.0;
    integer_log + fractional
}

/// Modified temporal step width, mirroring `calculateFixedPointTemporalSW`.
fn fixed_point_temporal_sw(temporal_sw_modifier: u32, temporal_sw_unmodified: i32) -> i32 {
    let step_width_modifier = (temporal_sw_modifier * FP_ONE_OVER_255 as u32).min(1 << 15);
    let multiplier = (1 << 16) - step_width_modifier;
    let floored = (multiplier * temporal_sw_unmodified as u32) >> 16;
    floored.clamp(QMIN_STEP_WIDTH as u32, QMAX_STEP_WIDTH as u32) as i32
}

fn calculate_dequant_offset_actual(
    layer_sw: i32,
    master_sw: i32,
    dequant_offset: i32,
    const_offset_mode: bool,
) -> i32 {
    if dequant_offset == -1 || dequant_offset == 0 {
        return 0;
    }
    let log_layer = (-KC as f64 * fixed_point_u12_4_ln(layer_sw)) as i32;
    let log_master = (KC as f64 * fixed_point_u12_4_ln(master_sw)) as i32;
    let mut actual: i64 = if const_offset_mode {
        (dequant_offset as i64) << 9
    } else {
        (dequant_offset as i64) << 11
    };
    actual = (log_layer as i64 + actual + log_master as i64) * layer_sw as i64;
    (actual >> 16) as i32
}

fn calculate_step_width_modifier(layer_sw: i32, dequant_offset_actual: i32, dequant_offset: i32,
                                 const_offset_mode: bool) -> i32 {
    if dequant_offset == -1 {
        let log_by_layer = (KD as f64 - KC as f64 * fixed_point_u12_4_ln(layer_sw)) as i64;
        let pow = log_by_layer * layer_sw as i64 * layer_sw as i64;
        return (pow / SW_DIVISOR_NO_DQ_OFFSET) as i32;
    }
    if const_offset_mode {
        return 0;
    }
    ((dequant_offset_actual as i64 * layer_sw as i64) / SW_DIVISOR) as i32
}

fn calculate_deadzone_width(master_sw: i32, layer_sw: i32) -> i32 {
    if master_sw <= 16 {
        return master_sw >> 1;
    }
    if layer_sw > DEADZONE_SW_LIMIT {
        return i32::MAX;
    }
    let scaled = (1 << 16) - (((KA * layer_sw) + KB) >> 1);
    ((scaled as i64 * layer_sw as i64) >> 16) as i32
}

/// Per-layer dequant parameters for one temporal signal and one plane/LOQ.
#[derive(Clone, Debug)]
pub struct DequantLayer {
    /// Final step width (layerSW after modifier, clamped) used in
    /// `d = q * step_width + offset`.
    pub step_width: i16,
    /// Applied offset (deadzone), signed.
    pub offset: i16,
    /// The deadzone width used for this layer (positive value).
    pub deadzone: i32,
}

/// Full dequant table for a plane/LOQ: [temporal][layer].
#[derive(Clone, Debug)]
pub struct DequantTable {
    pub layers: Vec<Vec<DequantLayer>>, // [temporal][layer]
}

impl DequantTable {
    /// Compute the dequant table, mirroring `calculateDequant` in the
    /// reference decoder. `quant_matrix` is the per-LOQ matrix.
    pub fn compute(
        loq_sw: i32,
        quant_matrix: &[u8],
        temporal_enabled: bool,
        loq_is_zero: bool,
        temporal_refresh: bool,
        temporal_step_width_modifier: u8,
        dequant_offset: i32,
        dequant_offset_mode: bool,
    ) -> DequantTable {
        let mut layers = Vec::new();
        if std::env::var("LCEVC_DUMP_DQ").is_ok() && loq_sw > 100 {
            eprintln!("MIRROR_DQ sw={loq_sw}:");
        }
        for temporal in [TemporalSignal::Inter, TemporalSignal::Intra] {
            let mut temporal_sw = loq_sw.clamp(QMIN_STEP_WIDTH, QMAX_STEP_WIDTH);
            if temporal == TemporalSignal::Inter && loq_is_zero && temporal_enabled
                && !temporal_refresh
            {
                temporal_sw = fixed_point_temporal_sw(
                    temporal_step_width_modifier as u32,
                    temporal_sw,
                );
            }
            let mut row = Vec::new();
            for &qm in quant_matrix {
                let mut layer_qm: i64 = qm as i64 * temporal_sw as i64;
                layer_qm += 1 << 16;
                layer_qm = layer_qm.clamp(0, QM_SCALE_MAX);
                layer_qm *= temporal_sw as i64;
                layer_qm >>= 16;

                let mut layer_sw = layer_qm.clamp(QMIN_STEP_WIDTH as i64, QMAX_STEP_WIDTH as i64) as i32;
                let offset_actual = calculate_dequant_offset_actual(
                    layer_sw, temporal_sw, dequant_offset, dequant_offset_mode);
                let sw_modifier = calculate_step_width_modifier(
                    layer_sw, offset_actual, dequant_offset, dequant_offset_mode);
                layer_sw = (layer_sw + sw_modifier).clamp(QMIN_STEP_WIDTH, QMAX_STEP_WIDTH);

                let deadzone = calculate_deadzone_width(temporal_sw, layer_sw);
                // The reference decoder applies the offset as a raw i16
                // truncation of the (possibly saturated) deadzone value.
                let offset: i16 = if dequant_offset == -1 || !dequant_offset_mode {
                    (-deadzone) as i16
                } else {
                    let applied = offset_actual as i64 - deadzone as i64;
                    applied.clamp(i16::MIN as i64, i16::MAX as i64) as i16
                };

                if std::env::var("LCEVC_DUMP_DQ").is_ok() && loq_sw > 100 {
                    eprintln!(
                        "  t={} l={} sw={} off={}",
                        temporal as usize,
                        row.len(),
                        layer_sw,
                        if dequant_offset == -1 || !dequant_offset_mode {
                            (-calculate_deadzone_width(temporal_sw, layer_sw)) as i16
                        } else {
                            offset
                        }
                    );
                }
                row.push(DequantLayer {
                    step_width: layer_sw as i16,
                    offset,
                    deadzone,
                });
            }
            layers.push(row);
        }
        DequantTable { layers }
    }
}

/// Compute the LOQ step width for a plane (mirrors `calculateLOQStepWidth`).
pub fn loq_step_width(step_width: i32, plane: usize, loq_is_zero: bool,
                      chroma_multiplier: u8) -> i32 {
    if plane > 0 && loq_is_zero {
        ((step_width * chroma_multiplier as i32) >> 6).clamp(QMIN_STEP_WIDTH, QMAX_STEP_WIDTH)
    } else {
        step_width.clamp(QMIN_STEP_WIDTH, QMAX_STEP_WIDTH)
    }
}

/// Default step widths: the encoder signals both; when not signalled the
/// decoder uses QMAX for LOQ1.
pub const DEFAULT_STEP_WIDTH_L1: i32 = 32767;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u12_4_ln_matches() {
        // ln(100) ~= 4.605170: integer part 4, fractional in 1/4096 steps.
        let v = fixed_point_u12_4_ln(100);
        let int = v.floor() as i32;
        let frac = ((v - v.floor()) * 4096.0).floor() as i32;
        assert_eq!(int, 4);
        assert_eq!(frac, 2478); // floor((ln(100)-4)*4096) = floor(2478.78) = 2478
    }

    #[test]
    fn temporal_sw_modifier() {
        // modifier 48 -> 48*257 = 12336, multiplier = 53200,
        // 53200 * 1000 >> 16 = 811 -> clamped [1, 32767].
        let sw = fixed_point_temporal_sw(48, 1000);
        assert_eq!(sw, 811);
        // Modifier 0 -> unchanged.
        assert_eq!(fixed_point_temporal_sw(0, 1000), 1000);
    }

    #[test]
    fn dequant_defaults() {
        // LOQ0, plane 0, 4:2:0, DDS 2D default matrix, step 256, no offset.
        let qm = default_quant_matrix(true, false, 0);
        let table = DequantTable::compute(256, &qm, false, true, false, 48, -1, false);
        // Layer 0: qm=13: layerQM = clamp(13*256 + 65536, 0, 196608) = 68864;
        // layerSW = 68864*256 >> 16 = 269.
        // swModifier = (99614 - 5242*lnU12_4(269)) * 269^2 / 2^31
        let layer = &table.layers[0][0];
        assert_eq!(layer.step_width, 271);
        // Offset must be -deadzone.
        assert_eq!(layer.offset, -(layer.deadzone as i16));
        // Deadzone: masterSW=256 > 16, layerSW=271:
        // scaled = 65536 - ((39*271 + 126484) >> 1) = 65536 - 68526 = -2990
        // deadzone = (-2990 * 271) >> 16 = -810290 >> 16 = -13
        assert_eq!(layer.deadzone, -13);
        assert_eq!(layer.offset, 13);
    }

    #[test]
    fn chroma_step_width_multiplier() {
        assert_eq!(loq_step_width(256, 1, true, 64), 256);
        assert_eq!(loq_step_width(256, 1, true, 32), 128);
        assert_eq!(loq_step_width(256, 0, true, 32), 256);
    }
}
