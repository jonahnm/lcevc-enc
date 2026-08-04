//! The LCEVC frame encoder: reference-pyramid construction, residual
//! coding with exact decoder-mirror reconstruction, temporal prediction and
//! chunk generation.
//!
//! Pipeline per frame (mirroring the decoder's pipeline in reverse):
//!
//! 1. The source picture is downscaled to the LOQ1 (L1) and LOQ2 (base)
//!    resolutions using the configured scaling modes.
//! 2. The base picture is encoded and decoded by the base codec.
//! 3. L1: decoded base -> upscale -> residual vs L1 target -> transform,
//!    quantize, entropy-encode, and reconstruct (dequant + inverse
//!    transform + optional L1 filter) -> combined intermediate picture.
//! 4. L2: combined intermediate -> upscale -> residual vs source ->
//!    transform, quantize, entropy-encode, reconstruct (with optional
//!    temporal prediction) -> output picture.
//! 5. Per-layer (value, zero-run) event lists become coefficient chunks;
//!    per-plane temporal signals become the temporal chunk.

use crate::config::{LcevcConfig, ScalingMode};
use crate::dequant::{self, DequantTable, TemporalSignal};
use crate::entropy::rle::{write_coefficient_chunk, CoeffEvent, TemporalRun};
use crate::frame::{Plane, PlaneS16, Picture};
use crate::transform::{ForwardTransform, residual_index};
use crate::upscale::{downscale_plane, upscale_plane};

/// Block size for temporal/block-major TU ordering (32x32).
const BLOCK_SHIFT: u32 = 5;
const BLOCK_SIZE: usize = 32;

/// One encoded coefficient chunk (a layer of one plane/LOQ/tile).
pub struct Chunk {
    pub entropy_enabled: bool,
    pub rle_only: bool,
    pub data: Vec<u8>,
}

/// Per-frame encoding results.
pub struct EncodedFrame {
    pub idr: bool,
    pub picture_config: Vec<u8>,
    pub encoded_data: Vec<u8>, // payload_encoded_data or _tiled
    /// residual[plane][loq 0=L1,1=L2][layer] -> tiles
    pub residual_chunks: Vec<Vec<Vec<Vec<Chunk>>>>,
    /// temporal[plane] -> tiles
    pub temporal_chunks: Vec<Vec<Chunk>>,
    pub temporal_signalling_present: bool,
    pub temporal_refresh: bool,
    pub output: Picture,
    /// Decoded base picture (LOQ2 resolution) used for this frame.
    pub base_picture: Picture,
    /// Total encoded enhancement bytes for this frame (payloads only).
    pub byte_count: usize,
}

/// Accumulated statistics.
#[derive(Default, Clone, Debug)]
pub struct EncoderStats {
    pub frames: u32,
    pub l1_events: u64,
    pub l2_events: u64,
    pub l1_chunks: u64,
    pub l2_chunks: u64,
    pub bytes: u64,
}

pub struct Encoder {
    pub config: LcevcConfig,
    pub step_width_l1: u32,
    pub step_width_l2: u32,
    pub quant_matrix: [Vec<u8>; 2], // per LOQ (0 = L2, 1 = L1)
    pub stats: EncoderStats,

    // Persistent state.
    frame_index: u32,
    temporal_buffers: Vec<PlaneS16>, // per plane at LOQ0 resolution

    // Rate-control state: the last chosen step widths (per picture).
    rc_prev_sw1: u32,
    rc_prev_sw2: u32,

    /// Content-adaptive quant-matrix exponent (0 = keep the default).
    pub qm_beta: f64,
}

// ---------------------------------------------------------------------------
// TU iteration (mirrors transform_unit.c)
// ---------------------------------------------------------------------------

pub struct TuState {
    pub width: usize,
    pub height: usize,
    pub x_offset: usize,
    pub y_offset: usize,
    pub tu_width_shift: u32,
    pub num_across: usize,
    pub tu_total: usize,
    block: TuBlock,
}

struct TuBlock {
    tu_per_block_dims_shift: u32,
    tu_per_block_dims: usize,
    tu_per_block_shift: u32,
    tu_per_block: usize,
    tu_per_block_row_right_edge: usize,
    tu_per_block_bottom_edge: usize,
    tu_per_row: usize,
    whole_blocks_per_row: usize,
    whole_blocks_per_col: usize,
}

impl TuState {
    pub fn new(width: usize, height: usize, x_offset: usize, y_offset: usize, tu_width_shift: u32) -> TuState {
        let num_across = width >> tu_width_shift;
        let tu_total = num_across * (height >> tu_width_shift);
        let dims_shift = if tu_width_shift == 1 { 4 } else { 3 };
        let tu_per_block_dims = 1usize << dims_shift;
        TuState {
            width,
            height,
            x_offset,
            y_offset,
            tu_width_shift,
            num_across,
            tu_total,
            block: TuBlock {
                tu_per_block_dims_shift: dims_shift,
                tu_per_block_dims,
                tu_per_block_shift: dims_shift << 1,
                tu_per_block: 1usize << (dims_shift << 1),
                tu_per_block_row_right_edge: (width & (BLOCK_SIZE - 1)) >> tu_width_shift,
                tu_per_block_bottom_edge: ((height & (BLOCK_SIZE - 1)) >> tu_width_shift) << dims_shift,
                tu_per_row: num_across << dims_shift,
                whole_blocks_per_row: width >> BLOCK_SHIFT,
                whole_blocks_per_col: height >> BLOCK_SHIFT,
            },
        }
    }

    /// Surface raster TU position (used when temporal is disabled and the
    /// picture is not tiled).
    pub fn surface_position(&self, tu_index: usize) -> (usize, usize) {
        let x = ((tu_index % self.num_across) << self.tu_width_shift) + self.x_offset;
        let y = ((tu_index / self.num_across) << self.tu_width_shift) + self.y_offset;
        (x, y)
    }

    /// 32x32 block-major TU position.
    pub fn block_position(&self, tu_index: usize) -> (usize, usize) {
        let b = &self.block;
        let block_row_index = tu_index / b.tu_per_row;
        let row_tu_index = tu_index - block_row_index * b.tu_per_row;

        let (block_col_index, block_tu_index) = if block_row_index >= b.whole_blocks_per_col {
            (row_tu_index / b.tu_per_block_bottom_edge, row_tu_index % b.tu_per_block_bottom_edge)
        } else {
            (row_tu_index >> b.tu_per_block_shift, row_tu_index & (b.tu_per_block - 1))
        };

        let (tu_y_coord, tu_x_coord) = if block_col_index >= b.whole_blocks_per_row {
            (
                block_tu_index / b.tu_per_block_row_right_edge,
                block_tu_index % b.tu_per_block_row_right_edge,
            )
        } else {
            (
                block_tu_index >> b.tu_per_block_dims_shift,
                block_tu_index & (b.tu_per_block_dims - 1),
            )
        };

        let tu_x = tu_x_coord + (block_col_index << b.tu_per_block_dims_shift);
        let tu_y = tu_y_coord + (block_row_index << b.tu_per_block_dims_shift);
        (
            (tu_x << self.tu_width_shift) + self.x_offset,
            (tu_y << self.tu_width_shift) + self.y_offset,
        )
    }
}

// ---------------------------------------------------------------------------
// Quantization
// ---------------------------------------------------------------------------

/// Deadzone quantizer for the exact rational coefficient `num / denom`,
/// using the decoder's step width and deadzone so that `q` is the
/// distortion-minimizing integer for the decoder's reconstruction.
fn quantize(num: i32, denom: i32, step_width: i32, deadzone: i32) -> i16 {
    if num == 0 {
        return 0;
    }
    let abs = (num as i64).unsigned_abs();
    let denom = denom as u64;
    let sw = step_width as u64;
    let dz = deadzone as i64;
    // q = round((|c| + dz) / sw) = floor((2*|num| + denom*(2*dz + sw)) / (2*denom*sw))
    let signed_dz = 2 * dz + step_width as i64;
    let numerator = 2 * abs + denom * signed_dz.unsigned_abs();
    let denominator = 2 * denom * sw;
    let mut q = (numerator / denominator) as i32;
    if num < 0 {
        q = -q;
    }
    q.clamp(-8192, 8191) as i16
}

// ---------------------------------------------------------------------------
// TU processing
// ---------------------------------------------------------------------------

/// Push one quantized coefficient into a layer's event list, accumulating
/// zero runs between nonzero values. `run` tracks the zeros seen so far
/// (passed in/out across TU boundaries).
///
/// Event semantics (as consumed by the decoder): `(value, zeros)` means the
/// value occupies the current TU and the `zeros` following TUs are zero. So a
/// run of `run` leading zeros before the first nonzero coefficient is
/// signalled as `(0, run - 1)`.
fn push_coeff(events: &mut Vec<CoeffEvent>, run: &mut u32, c: i16) {
    if c == 0 {
        *run += 1;
        return;
    }
    if let Some(last) = events.last_mut() {
        last.zero_run += *run;
    } else if *run > 0 {
        // Leading zeros before the first nonzero coefficient.
        events.push(CoeffEvent { value: 0, zero_run: *run - 1 });
    }
    *run = 0;
    events.push(CoeffEvent { value: c, zero_run: 0 });
}

/// Flush the accumulated trailing zero run into the event list so that the
/// stream covers every TU of the plane. Without this the decoder would
/// consume padding bits as coefficients for the final TUs.
fn flush_run(events: &mut Vec<CoeffEvent>, run: u32) {
    if run == 0 {
        return;
    }
    if let Some(last) = events.last_mut() {
        last.zero_run += run;
    } else if run > 0 {
        events.push(CoeffEvent { value: 0, zero_run: run - 1 });
    }
}

/// Analyze one TU without committing: compute the residual vs the given
/// prediction (which already includes the temporal prediction for inter
/// blocks), transform, quantize, reconstruct. Returns the per-layer
/// quantized coefficients and the reconstructed residual block (in the
/// interleaved layout).
#[allow(clippy::too_many_arguments)]
fn encode_tu(
    target: &PlaneS16,
    pred: &PlaneS16,
    table: &DequantTable,
    forward: &ForwardTransform,
    tu_size: usize,
    x: usize,
    y: usize,
    signal: TemporalSignal,
    deblock: bool,
    deblock_corner: u8,
    deblock_side: u8,
    energy: &mut [f64],
) -> (Vec<i16>, Vec<i16>) {
    // Build the residual block in the interleaved layout.
    let n = tu_size * tu_size;
    let mut residual = vec![0i16; n];
    for yy in 0..tu_size {
        for xx in 0..tu_size {
            let idx = residual_index(tu_size, xx, yy);
            residual[idx] = (target.get(x + xx, y + yy) as i32 - pred.get(x + xx, y + yy) as i32)
                .clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        }
    }
    encode_tu_residual(&residual, table, forward, signal, deblock, deblock_corner, deblock_side, energy)
}

/// Quantize and reconstruct an already-built residual block.
#[allow(clippy::too_many_arguments)]
fn encode_tu_residual(
    residual: &[i16],
    table: &DequantTable,
    forward: &ForwardTransform,
    signal: TemporalSignal,
    deblock: bool,
    deblock_corner: u8,
    deblock_side: u8,
    energy: &mut [f64],
) -> (Vec<i16>, Vec<i16>) {
    let tu_size = (residual.len() as f64).sqrt() as usize;
    let _ = (deblock, deblock_corner, deblock_side, tu_size);

    let (nums, denom) = forward.apply(residual);
    let num_layers = nums.len();
    for l in 0..num_layers {
        if let Some(e) = energy.get_mut(l) {
            let c = nums[l] as f64 / denom as f64;
            *e += c * c;
        }
    }

    // Quantize.
    let mut coeffs = vec![0i16; num_layers];
    for l in 0..num_layers {
        let layer = &table.layers[signal as usize][l];
        coeffs[l] = quantize(nums[l], denom, layer.step_width as i32, -(layer.offset as i32));
    }

    // RDOQ: refine each level with a rate-distortion cost. The transform is
    // Hadamard-based (orthogonal up to a constant absorbed in the lambda), so
    // the pixel SSE of a level change is (dequant(q) - orig)^2 times a
    // constant. Bits are estimated from the coefficient magnitude (1 LSB byte
    // for |q| <= 32, else LSB+MSB); zero also saves the run entropy.
    if table.layers[signal as usize][0].step_width > 0 {
        let mut rdq = coeffs.clone();
        for l in 0..num_layers {
            let layer = &table.layers[signal as usize][l];
            let sw = layer.step_width as i32;
            let off = layer.offset as i32;
            let q0 = coeffs[l] as i32;
            let orig = nums[l] as f64 / denom as f64;
            // Lambda ~ sw^2 tuned empirically (0.125 absorbs the transform
            // orthogonality constant); a bit is worth ~ sw/2 of error.
            let lambda = (sw as f64 * sw as f64) * 0.0625;
            let mut best = q0;
            let mut best_cost = f64::INFINITY;
            let candidates: [i32; 3] = [q0 - 1, q0, q0 + 1];
            for &c in &candidates {
                let dq = if c > 0 {
                    c * sw + off
                } else if c < 0 {
                    c * sw - off
                } else {
                    0
                };
                let err = dq as f64 - orig;
                let bits = if c == 0 {
                    0.0
                } else {
                    (if c.unsigned_abs() <= 32 { 8.0 } else { 16.0 }) + 1.0
                };
                let cost = err * err + lambda * bits;
                if cost < best_cost {
                    best_cost = cost;
                    best = c;
                }
            }
            // Also consider full zeroing for small levels (cheap to check).
            if q0.unsigned_abs() <= 2 {
                let cost = orig * orig;
                if cost < best_cost {
                    best = 0;
                }
            }
            rdq[l] = best.clamp(-8192, 8191) as i16;
        }
        coeffs = rdq;
    }

    // Reconstruct: dequant (mirror) then inverse transform.
    let mut dq = vec![0i16; num_layers];
    for l in 0..num_layers {
        let layer = &table.layers[signal as usize][l];
        let c = coeffs[l] as i32;
        let v = if c > 0 {
            c * layer.step_width as i32 + layer.offset as i32
        } else if c < 0 {
            c * layer.step_width as i32 - layer.offset as i32
        } else {
            0
        };
        dq[l] = v.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
    }
    let mut rec_residual = forward.inverse(&dq);

    // Optional L1 filter (deblock).
    if deblock && tu_size == 4 {
        let alpha = 16 - deblock_corner as i32;
        let beta = 16 - deblock_side as i32;
        let mut filtered = [0i16; 16];
        for i in 0..16 {
            filtered[i] = rec_residual[i];
        }
        // Corners and sides per the decoder's deblockResiduals.
        for &(idx, k) in &[
            (0usize, alpha), (5, alpha), (10, alpha), (15, alpha),
            (1, beta), (4, beta), (7, beta), (13, beta),
            (2, beta), (8, beta), (11, beta), (14, beta),
        ] {
            filtered[idx] = ((rec_residual[idx] as i32 * k) >> 4)
                .clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        }
        rec_residual = filtered.to_vec();
    }

    (coeffs, rec_residual)
}

/// Add a reconstructed residual block to a plane (with saturation).
fn commit_residual(plane: &mut PlaneS16, x: usize, y: usize, tu_size: usize, residual: &[i16]) {
    for yy in 0..tu_size {
        for xx in 0..tu_size {
            let idx = residual_index(tu_size, xx, yy);
            let v = plane.get(x + xx, y + yy) as i32 + residual[idx] as i32;
            plane.set(x + xx, y + yy, v.clamp(i16::MIN as i32, i16::MAX as i32) as i16);
        }
    }
}

/// SSE of a reconstructed block (prediction + residual) vs the target.
fn block_sse(pred: &PlaneS16, x: usize, y: usize, tu_size: usize, residual: &[i16],
             target: &PlaneS16) -> u64 {
    let mut sse = 0u64;
    for yy in 0..tu_size {
        for xx in 0..tu_size {
            let idx = residual_index(tu_size, xx, yy);
            let v = pred.get(x + xx, y + yy) as i32 + residual[idx] as i32;
            let d = (v - target.get(x + xx, y + yy) as i32) as i64;
            sse += (d * d) as u64;
        }
    }
    sse
}

/// Add a residual block to a plane (temporal buffer inter update).
fn add_block(plane: &mut PlaneS16, x: usize, y: usize, tu_size: usize, residual: &[i16]) {
    for yy in 0..tu_size {
        for xx in 0..tu_size {
            let idx = residual_index(tu_size, xx, yy);
            let v = plane.get(x + xx, y + yy) as i32 + residual[idx] as i32;
            plane.set(x + xx, y + yy, v.clamp(i16::MIN as i32, i16::MAX as i32) as i16);
        }
    }
}

/// Overwrite a block of a plane (temporal buffer intra update).
fn set_block(plane: &mut PlaneS16, x: usize, y: usize, tu_size: usize, residual: &[i16]) {
    for yy in 0..tu_size {
        for xx in 0..tu_size {
            let idx = residual_index(tu_size, xx, yy);
            plane.set(x + xx, y + yy, residual[idx]);
        }
    }
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

impl Encoder {
    pub fn new(config: LcevcConfig, step_width_l1: u32, step_width_l2: u32) -> Encoder {
        let qm_scaling_1d = config.scaling_l2 == ScalingMode::Scale1D;
        let qm0 = dequant::default_quant_matrix(config.transform.layers() == 16, qm_scaling_1d, 0);
        let qm1 = dequant::default_quant_matrix(config.transform.layers() == 16, qm_scaling_1d, 1);
        let temporal_buffers = (0..config.num_planes())
            .map(|p| {
                let (w, h) = config.plane_dimensions(0, p);
                PlaneS16::new(w as usize, h as usize)
            })
            .collect();
        Encoder {
            config,
            step_width_l1,
            step_width_l2,
            quant_matrix: [qm0, qm1],
            stats: EncoderStats::default(),
            frame_index: 0,
            temporal_buffers,
            rc_prev_sw1: step_width_l1,
            rc_prev_sw2: step_width_l2,
            qm_beta: 0.3,
        }
    }

    pub fn config(&self) -> &LcevcConfig {
        &self.config
    }

    /// Encode one frame, running the base codec round trip internally.
    pub fn encode_frame(&mut self, source: &Picture) -> Result<EncodedFrame, String> {
        let cfg = &self.config;
        cfg.validate()?;

        // Reference pyramid: LOQ0 (source), LOQ1 (L1 target), LOQ2 (base).
        let mut base_targets: Vec<Plane> = Vec::with_capacity(cfg.num_planes());
        for p in 0..cfg.num_planes() {
            let l1_target = downscale_plane(&source.planes[p], cfg.scaling_l2, cfg.sample_depth());
            let base_target = downscale_plane(&l1_target, cfg.scaling_l1, cfg.sample_depth());
            base_targets.push(base_target);
        }

        // Base codec round trip.
        let base_picture = crate::base::encode_decode_base(cfg, &base_targets)?;
        self.encode_frame_with_base(source, &base_picture)
    }

    /// Rate-controlled encode: binary search the L1 step width (L2 = L1/4,
    /// clamped) for the finest step width whose payload fits `target_bytes`.
    /// Payload size is monotone in the step width (coarser = smaller).
    pub fn encode_frame_rc(
        &mut self,
        source: &Picture,
        base_picture: &Picture,
        target_bytes: usize,
    ) -> Result<(EncodedFrame, u32, u32), String> {
        let tb_backup = self.temporal_buffers.clone();
        let try_sw = |enc: &mut Self, sw1: u32, sw2: u32|
            -> Result<(EncodedFrame, usize), String> {
            let idx = enc.frame_index;
            let stats = enc.stats.clone();
            enc.temporal_buffers = tb_backup.clone();
            let frame = enc.encode_frame_inner(source, base_picture, sw1, sw2)?;
            let size = frame.picture_config.len() + frame.encoded_data.len();
            enc.frame_index = idx;
            enc.stats = stats;
            Ok((frame, size))
        };

        let mut lo = 16u32; // finest possible sw1
        let mut hi = 16384u32; // coarsest possible sw1
        let mut best: Option<(EncodedFrame, u32, u32, usize)> = None;
        for _ in 0..8 {
            let sw1 = (lo + hi) / 2;
            let sw2 = (sw1 / 4).max(1);
            let (frame, size) = try_sw(self, sw1, sw2)?;
            if size <= target_bytes {
                // fits: remember it, try finer
                best = Some((frame, sw1, sw2, size));
                hi = sw1;
                if sw1 == lo {
                    break;
                }
            } else {
                // too big: go coarser
                lo = sw1;
                if sw1 + 1 >= hi {
                    break;
                }
            }
        }

        let (frame, sw1, sw2, _) = best.unwrap();
        self.rc_prev_sw1 = sw1;
        self.rc_prev_sw2 = sw2;
        Ok((frame, sw1, sw2))
    }

    /// Encode one frame with a pre-decoded base picture and explicit
    /// L1/L2 step widths (rate-controlled path).
    pub fn encode_frame_with_base(
        &mut self,
        source: &Picture,
        base_picture: &Picture,
    ) -> Result<EncodedFrame, String> {
        let sw1 = self.step_width_l1;
        let sw2 = self.step_width_l2;
        self.encode_frame_inner(source, base_picture, sw1, sw2)
    }

    fn encode_frame_inner(
        &mut self,
        source: &Picture,
        base_picture: &Picture,
        sw_l1: u32,
        sw_l2: u32,
    ) -> Result<EncodedFrame, String> {
        let cfg = &self.config;
        cfg.validate()?;

        let idr = self.frame_index == 0;
        let temporal_refresh = idr || !cfg.temporal_enabled;
        let temporal_signalling = cfg.temporal_enabled && !temporal_refresh;

        let kernel = cfg.upsampler.kernel();
        let apply_pa = cfg.predicted_average;
        let tu_size = cfg.tu_size();
        let num_layers = cfg.num_layers();
        let nplanes = cfg.num_planes();

        // Reference pyramid: LOQ0 (source), LOQ1 (L1 target).
        let mut l1_targets: Vec<Plane> = Vec::with_capacity(nplanes);
        for p in 0..nplanes {
            let l1_target = downscale_plane(&source.planes[p], cfg.scaling_l2, cfg.sample_depth());
            l1_targets.push(l1_target);
        }

        // Per-plane state.
        let mut residual_chunks: Vec<Vec<Vec<Vec<Chunk>>>> = Vec::new();
        let mut temporal_chunks: Vec<Vec<Chunk>> = Vec::new();
        let mut output = Picture::new(cfg.width as usize, cfg.height as usize, cfg.chroma);
        // Base-only prediction (no enhancement) used to decide whether the
        // residual actually improves the reconstruction.
        let mut base_only_planes: Vec<Plane> = Vec::with_capacity(nplanes);
        let mut base_only_sse: u64 = 0;

        let tu_order = cfg.temporal_enabled || cfg.is_tiled();
        let mut frame_energy = [[0.0f64; 16]; 2];
        // Snapshot the temporal buffer: on a dropped frame the decoder leaves
        // it untouched, so the encoder must too.
        let tb_start = if cfg.temporal_enabled { self.temporal_buffers.clone() } else { Vec::new() };

        for plane in 0..nplanes {
            let base_plane = &base_picture.planes[plane];
            let base_s16 = base_plane.to_s16(cfg.sample_depth());
            let l1_pred = upscale_plane(&base_s16, cfg.scaling_l1, &kernel, apply_pa);
            let l1_target_s16 = l1_targets[plane].to_s16(cfg.sample_depth());
            let mut l1_recon = l1_pred.clone();

            // ---- LOQ1 (L1) ----
            let sw1 = dequant::loq_step_width(
                sw_l1 as i32, plane, false, cfg.chroma_step_width_multiplier);
            let table1 = DequantTable::compute(
                sw1, &self.quant_matrix[1], cfg.temporal_enabled, false, temporal_refresh,
                cfg.temporal_step_width_modifier, -1, false);
            let forward1 = ForwardTransform::new(cfg.transform.to_bit(), cfg.scaling_l1 == ScalingMode::Scale1D);
            let deblock = cfg.level1_filtering_signalled;

            let n_l1_tiles = cfg.num_tiles(1, plane);
            let mut l1_events: Vec<Vec<Vec<CoeffEvent>>> =
                vec![vec![Vec::new(); num_layers]; n_l1_tiles];
            let mut l1_runs: Vec<Vec<u32>> = vec![vec![0; num_layers]; n_l1_tiles];
            let mut energy_l1 = [0.0f64; 16];
            process_loq_tiles(
                cfg, 1, plane, |tu_state, i, _| {
                    tu_state.surface_or_block_position(i, tu_order)
                },
                &mut |x, y, tile| {
                    let (coeffs, residual) = encode_tu(
                        &l1_target_s16, &l1_pred, &table1, &forward1, tu_size, x, y,
                        TemporalSignal::Inter, deblock,
                        cfg.level1_filtering_first_coefficient,
                        cfg.level1_filtering_second_coefficient,
                        &mut energy_l1,
                    );
                    commit_residual(&mut l1_recon, x, y, tu_size, &residual);
                    for l in 0..num_layers {
                        push_coeff(&mut l1_events[tile][l], &mut l1_runs[tile][l], coeffs[l]);
                    }
                    if plane == 0 && x == 0 && y == 0 {
                    }
                },
            )?;

            // ---- LOQ0 (L2) ----
            let l2_pred = upscale_plane(&l1_recon, cfg.scaling_l2, &kernel, apply_pa);
            let src_s16 = source.planes[plane].to_s16(cfg.sample_depth());
            let mut l2_recon = l2_pred.clone();
            {
                // Base-only prediction: the pure base upscaled (no residual).
                // SSE measured in the sample domain to match recon_sse.
                let base_only_plane = upscale_plane(&l1_pred, cfg.scaling_l2, &kernel, apply_pa)
                    .to_plane(cfg.sample_depth());
                let mut sse = 0u64;
                for (a, b) in source.planes[plane].data.iter().zip(base_only_plane.data.iter()) {
                    let d = (*a as i64 - *b as i64).unsigned_abs();
                    sse += d * d;
                }
                base_only_sse += sse;
                base_only_planes.push(base_only_plane);
            }

            let temporal_enabled = cfg.temporal_enabled;
            let sw2 = dequant::loq_step_width(
                sw_l2 as i32, plane, true, cfg.chroma_step_width_multiplier);
            let table_inter = DequantTable::compute(
                sw2, &self.quant_matrix[0], temporal_enabled, true, temporal_refresh,
                cfg.temporal_step_width_modifier, -1, false);
            let table_intra = DequantTable::compute(
                sw2, &self.quant_matrix[0], false, true, false,
                cfg.temporal_step_width_modifier, -1, false);
            let forward0 = ForwardTransform::new(cfg.transform.to_bit(), cfg.scaling_l2 == ScalingMode::Scale1D);

            let mut tb = std::mem::replace(&mut self.temporal_buffers[plane], PlaneS16::new(1, 1));
            let mut energy_l2 = [0.0f64; 16];
            let n_l2_tiles = cfg.num_tiles(0, plane);
            let mut l2_events: Vec<Vec<Vec<CoeffEvent>>> =
                vec![vec![Vec::new(); num_layers]; n_l2_tiles];
            let mut l2_runs: Vec<Vec<u32>> = vec![vec![0; num_layers]; n_l2_tiles];
            let mut temporal_runs: Vec<Vec<TemporalRun>> = Vec::with_capacity(n_l2_tiles);
            for _ in 0..n_l2_tiles {
                temporal_runs.push(Vec::new());
            }

            process_loq_tiles(
                cfg, 0, plane, |tu_state, i, _| {
                    tu_state.surface_or_block_position(i, tu_order)
                },
                &mut |x, y, tile| {
                    let (coeffs, signal) = if temporal_enabled && !temporal_refresh {
                        // Inter trial: prediction includes the temporal buffer.
                        // Build the residual locally (pred + tb per sample) —
                        // cloning the whole plane per TU would be O(TUs x W x H).
                        let mut residual = vec![0i16; tu_size * tu_size];
                        for yy in 0..tu_size {
                            for xx in 0..tu_size {
                                let idx = residual_index(tu_size, xx, yy);
                                let v = l2_pred.get(x + xx, y + yy) as i32
                                    + tb.get(x + xx, y + yy) as i32;
                                residual[idx] = (src_s16.get(x + xx, y + yy) as i32 - v)
                                    .clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                            }
                        }
                        let (c_inter, r_inter) = encode_tu_residual(
                            &residual, &table_inter, &forward0,
                            TemporalSignal::Inter, false, 0, 0, &mut energy_l2);
                        let mut sse_inter = 0i64;
                        for yy in 0..tu_size {
                            for xx in 0..tu_size {
                                let v = (l2_pred.get(x + xx, y + yy) as i32
                                    + tb.get(x + xx, y + yy) as i32)
                                    .clamp(i16::MIN as i32, i16::MAX as i32) as i64;
                                let d = src_s16.get(x + xx, y + yy) as i64
                                    - (v + r_inter[residual_index(tu_size, xx, yy)] as i64);
                                sse_inter += d * d;
                            }
                        }
                        let (c_intra, r_intra) = encode_tu(
                            &src_s16, &l2_pred, &table_intra, &forward0, tu_size, x, y,
                            TemporalSignal::Intra, false, 0, 0, &mut energy_l2);
                        let sse_intra = block_sse(&l2_pred, x, y, tu_size, &r_intra, &src_s16);
                        if sse_inter <= sse_intra as i64 {
                            add_block(&mut tb, x, y, tu_size, &r_inter);
                            set_recon_from_tb(&mut l2_recon, &l2_pred, &tb, x, y, tu_size);
                            (c_inter, TemporalSignal::Inter)
                        } else {
                            set_block(&mut tb, x, y, tu_size, &r_intra);
                            set_recon_from_tb(&mut l2_recon, &l2_pred, &tb, x, y, tu_size);
                            (c_intra, TemporalSignal::Intra)
                        }
                    } else {
                        // No temporal prediction (disabled or refresh frame):
                        // the decoder always "sets" the temporal buffer.
                        let (coeffs, residual) = encode_tu(
                            &src_s16, &l2_pred, &table_inter, &forward0, tu_size, x, y,
                            TemporalSignal::Inter, false, 0, 0, &mut energy_l2);
                        commit_residual(&mut l2_recon, x, y, tu_size, &residual);
                        if temporal_enabled {
                            set_block(&mut tb, x, y, tu_size, &residual);
                        }
                        (coeffs, TemporalSignal::Intra)
                    };
                    for l in 0..num_layers {
                                        push_coeff(&mut l2_events[tile][l], &mut l2_runs[tile][l], coeffs[l]);
                    }
                    if temporal_enabled && !temporal_refresh {
                        let runs = &mut temporal_runs[tile];
                        if let Some(last) = runs.last_mut() {
                            if last.signal == signal as u8 {
                                last.count += 1;
                                return;
                            }
                        }
                        runs.push(TemporalRun { signal: signal as u8, count: 1 });
                    }
                },
            )?;

            self.temporal_buffers[plane] = tb;
            for l in 0..num_layers {
                frame_energy[1][l] += energy_l1[l];
                frame_energy[0][l] += energy_l2[l];
            }
            // Flush trailing zero runs so every TU of every layer is covered
            // (the decoder otherwise reads padding as coefficients).
            for tile in 0..n_l1_tiles {
                for l in 0..num_layers {
                    flush_run(&mut l1_events[tile][l], l1_runs[tile][l]);
                }
            }
            for tile in 0..n_l1_tiles {
                for l in 0..num_layers {
                    flush_run(&mut l1_events[tile][l], l1_runs[tile][l]);
                }
            }
            for tile in 0..n_l2_tiles {
                for l in 0..num_layers {
                    flush_run(&mut l2_events[tile][l], l2_runs[tile][l]);
                }
            }
            if plane == 0 {
            }

            // Assemble the per-plane chunk arrays.
            let mut plane_residual = Vec::with_capacity(2);
            for (loq_idx, tile_events) in [&l1_events, &l2_events].iter().enumerate() {
                let tile_events: &Vec<Vec<Vec<CoeffEvent>>> = tile_events;
                let mut loq_chunks = Vec::with_capacity(num_layers);
                for layer in 0..num_layers {
                    let mut tiles = Vec::with_capacity(tile_events.len());
                    for tile_ev in tile_events {
                        let events = &tile_ev[layer];
                        if events.is_empty() {
                            tiles.push(Chunk { entropy_enabled: false, rle_only: false, data: Vec::new() });
                        } else {
                            let cd = write_coefficient_chunk(events);
                            if plane == 0 && loq_idx == 1 && layer == 1 {
                            }
                            tiles.push(Chunk {
                                entropy_enabled: true,
                                rle_only: false,
                                data: cd,
                            });
                            if loq_idx == 0 {
                                self.stats.l1_chunks += 1;
                            } else {
                                self.stats.l2_chunks += 1;
                            }
                        }
                    }
                    loq_chunks.push(tiles);
                }
                plane_residual.push(loq_chunks);
            }
            residual_chunks.push(plane_residual);

            // Temporal chunk per tile (LOQ0 tile count).
            if temporal_signalling {
                let n_tiles = cfg.num_tiles(0, plane);
                let mut tiles = Vec::with_capacity(n_tiles);
                for tile in 0..n_tiles {
                    let runs = &temporal_runs[tile];
                    let data = crate::entropy::rle::write_temporal_chunk(runs);
                    tiles.push(Chunk { entropy_enabled: true, rle_only: false, data });
                }
                temporal_chunks.push(tiles);
            } else {
                let n_tiles = cfg.num_tiles(0, plane);
                temporal_chunks.push(
                    (0..n_tiles).map(|_| Chunk { entropy_enabled: false, rle_only: false, data: Vec::new() }).collect(),
                );
            }

            output.planes[plane] = l2_recon.to_plane(cfg.sample_depth());
        }

        // Adaptive enhancement: if the residual does not improve the
        // reconstruction over the base-only prediction, drop it entirely
        // (zero chunks). The decoder then outputs the base as-is and the
        // LCEVC overhead is just the per-frame headers. This only applies
        // when the base is at the output resolution (no upscaling needed).
        if !base_only_planes.is_empty() {
            let mut recon_sse: u64 = 0;
            for p in 0..source.planes.len() {
                for (a, b) in source.planes[p].data.iter().zip(output.planes[p].data.iter()) {
                    let d = (*a as i64 - *b as i64).unsigned_abs();
                    recon_sse += d * d;
                }
            }
            if recon_sse >= base_only_sse {
                for plane_chunks in &mut residual_chunks {
                    for loq in plane_chunks {
                        for layer in loq {
                            for chunk in layer {
                                chunk.entropy_enabled = false;
                                chunk.rle_only = false;
                                chunk.data.clear();
                            }
                        }
                    }
                }
                for plane_t in &mut temporal_chunks {
                    for chunk in plane_t {
                        chunk.entropy_enabled = false;
                        chunk.rle_only = false;
                        chunk.data.clear();
                    }
                }
                output.planes = base_only_planes;
                if cfg.temporal_enabled {
                    self.temporal_buffers = tb_start;
                }
            }
        }

        self.stats.frames += 1;
        self.frame_index += 1;

        // The picture config must signal the matrix the frame was actually
        // quantized with (the previous frame's adaptation), then update the
        // matrix for the NEXT frame.
        let picture_config = crate::payload::write_picture_config(
            cfg, sw_l1, sw_l2, temporal_refresh, &self.quant_matrix);
        let qm_scaling_1d = cfg.scaling_l2 == ScalingMode::Scale1D;
        let dds = num_layers == 16;
        let def0 = dequant::default_quant_matrix(dds, qm_scaling_1d, 0);
        let def1 = dequant::default_quant_matrix(dds, qm_scaling_1d, 1);
        self.quant_matrix[0] = dequant::content_quant_matrix(&frame_energy[0], &def0, self.qm_beta);
        self.quant_matrix[1] = dequant::content_quant_matrix(&frame_energy[1], &def1, self.qm_beta);
        let encoded_data = crate::payload::write_encoded_data(
            cfg, &residual_chunks, &temporal_chunks, temporal_signalling);

        let byte_count = picture_config.len() + encoded_data.len();
        self.stats.bytes += byte_count as u64;

        Ok(EncodedFrame {
            idr,
            picture_config,
            encoded_data,
            residual_chunks,
            temporal_chunks,
            temporal_signalling_present: temporal_signalling,
            temporal_refresh,
            output,
            base_picture: base_picture.clone(),
            byte_count,
        })
    }

    /// Reset state (temporal buffers etc.) — used before a new sequence.
    pub fn reset(&mut self) {
        self.frame_index = 0;
        for tb in &mut self.temporal_buffers {
            for v in &mut tb.data {
                *v = 0;
            }
        }
        self.stats = EncoderStats::default();
    }
}

impl TuState {
    fn surface_or_block_position(&self, tu_index: usize, block_order: bool) -> (usize, usize) {
        if block_order {
            self.block_position(tu_index)
        } else {
            self.surface_position(tu_index)
        }
    }
}

/// Iterate over the tiles of a LOQ for one plane, invoking `op` for every TU.
fn process_loq_tiles<F>(
    cfg: &LcevcConfig,
    loq: usize,
    plane: usize,
    position: impl Fn(&TuState, usize, usize) -> (usize, usize),
    op: &mut F,
) -> Result<(), String>
where
    F: FnMut(usize, usize, usize),
{
    let (pw, ph) = cfg.plane_dimensions(loq, plane);
    let (tw, th) = cfg.tile_dimensions_plane(plane);
    let tiles_across = (pw + tw - 1) / tw;
    let n_tiles = tiles_across * ((ph + th - 1) / th);
    let shift = cfg.tu_size().trailing_zeros();

    for tile in 0..n_tiles {
        let tile_x = (tile % tiles_across) * tw;
        let tile_y = (tile / tiles_across) * th;
        let tile_w = tw.min(pw - tile_x);
        let tile_h = th.min(ph - tile_y);
        let tu_state = TuState::new(tile_w as usize, tile_h as usize, tile_x as usize, tile_y as usize, shift);
        for i in 0..tu_state.tu_total {
            let (x, y) = position(&tu_state, i, tile as usize);
            op(x, y, tile as usize);
        }
    }
    Ok(())
}

/// Write the reconstruction for a block as prediction + temporal buffer
/// (the decoder's `applyAddTemporal` output).
fn set_recon_from_tb(recon: &mut PlaneS16, pred: &PlaneS16, tb: &PlaneS16, x: usize, y: usize,
                     tu_size: usize) {
    for yy in 0..tu_size {
        for xx in 0..tu_size {
            let v = pred.get(x + xx, y + yy) as i32 + tb.get(x + xx, y + yy) as i32;
            recon.set(x + xx, y + yy, v.clamp(i16::MIN as i32, i16::MAX as i32) as i16);
        }
    }
}
