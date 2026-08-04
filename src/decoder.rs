//! Mirror decoder: decodes an LCEVC enhancement stream (plus base pictures)
//! using this crate's bit-exact reimplementation of the reference decoder.
//! Used for self-verification and as an oracle when debugging the format.

use crate::config::{ChromaFormat, LcevcConfig};
use crate::dequant::{self, DequantTable, TemporalSignal};
use crate::entropy::huffman::BitReader;
use crate::encoder::TuState;
use crate::entropy::rle::{decode_coefficient_chunk, decode_temporal_chunk, CoeffEvent};
use crate::frame::{PlaneS16, Picture};
use crate::transform::{ForwardTransform, residual_index};
use crate::upscale::upscale_plane;

/// Parse the NAL-unit framing (start code, header, blocks) and return the
/// process blocks as (payload_type, payload) tuples.
pub fn parse_nal(nal: &[u8]) -> Result<(bool, Vec<(u8, Vec<u8>)>), String> {
    // Skip the start code.
    if nal.len() < 5 {
        return Err("NAL too short".into());
    }
    let pos;
    if nal[0] == 0 && nal[1] == 0 && nal[2] == 1 {
        pos = 3;
    } else if nal[0] == 0 && nal[1] == 0 && nal[2] == 0 && nal[3] == 1 {
        pos = 4;
    } else {
        return Err("missing start code".into());
    }
    let mut pos = pos;
    let b0 = nal[pos];
    let b1 = nal[pos + 1];
    if (b0 & 0xC1) != 0x41 || b1 != 0xFF {
        return Err("bad NAL header".into());
    }
    let nal_type = (b0 & 0x3E) >> 1;
    let idr = nal_type == 29;
    pos += 2;

    // Un-escape (remove 0x03 after two 0x00) and drop the RBSP stop byte.
    let mut rbsp = Vec::new();
    let mut zeros = 0u32;
    while pos < nal.len() - 1 {
        let b = nal[pos];
        pos += 1;
        if zeros == 2 && b == 3 {
            zeros = 0;
            continue;
        }
        if b == 0 {
            zeros += 1;
        } else {
            zeros = 0;
        }
        rbsp.push(b);
    }

    // Parse blocks.
    let mut blocks = Vec::new();
    let mut off = 0usize;
    while off < rbsp.len() {
        let header = rbsp[off];
        off += 1;
        let size_type = (header >> 5) & 7;
        let payload_type = header & 0x1F;
        let mut size = size_type as usize;
        if size_type == 7 {
            size = 0;
            loop {
                let b = rbsp[off];
                off += 1;
                size = (size << 7) | (b & 0x7F) as usize;
                if b & 0x80 == 0 {
                    break;
                }
            }
        }
        if off + size > rbsp.len() {
            return Err("block overruns NAL".into());
        }
        blocks.push((payload_type, rbsp[off..off + size].to_vec()));
        off += size;
    }
    Ok((idr, blocks))
}

/// Parse the global configuration block.
fn parse_global_config(data: &[u8], cfg: &mut LcevcConfig) -> Result<(), String> {
    let mut r = BitReader::new(data);
    let plane_mode = r.read_bit().ok_or("eof")?;
    let res_type = r.read_bits(6).ok_or("eof")? as u8;
    let transform = r.read_bit().ok_or("eof")?;
    cfg.transform = if transform {
        crate::config::TransformType::Dds
    } else {
        crate::config::TransformType::Dd
    };
    cfg.chroma = match r.read_bits(2).ok_or("eof")? {
        0 => ChromaFormat::Monochrome,
        1 => ChromaFormat::C420,
        2 => ChromaFormat::C422,
        _ => ChromaFormat::C444,
    };
    cfg.base_depth = 8 + 2 * r.read_bits(2).ok_or("eof")? as u8;
    cfg.enhancement_depth = 8 + 2 * r.read_bits(2).ok_or("eof")? as u8;
    let tswm = r.read_bit().ok_or("eof")?;
    cfg.predicted_average = r.read_bit().ok_or("eof")?;
    cfg.temporal_tile_intra_signalling = r.read_bit().ok_or("eof")?;
    cfg.temporal_enabled = r.read_bit().ok_or("eof")?;
    cfg.upsampler = match r.read_bits(3).ok_or("eof")? {
        0 => crate::config::UpsampleType::Nearest,
        1 => crate::config::UpsampleType::Linear,
        2 => crate::config::UpsampleType::Cubic,
        _ => crate::config::UpsampleType::ModifiedCubic,
    };
    cfg.level1_filtering_signalled = r.read_bit().ok_or("eof")?;
    cfg.scaling_l1 = match r.read_bits(2).ok_or("eof")? {
        0 => crate::config::ScalingMode::Scale0D,
        1 => crate::config::ScalingMode::Scale1D,
        _ => crate::config::ScalingMode::Scale2D,
    };
    cfg.scaling_l2 = match r.read_bits(2).ok_or("eof")? {
        0 => crate::config::ScalingMode::Scale0D,
        1 => crate::config::ScalingMode::Scale1D,
        _ => crate::config::ScalingMode::Scale2D,
    };
    cfg.tile_dimensions = match r.read_bits(2).ok_or("eof")? {
        0 => crate::config::TileDimensions::None,
        1 => crate::config::TileDimensions::T512x256,
        2 => crate::config::TileDimensions::T1024x512,
        _ => crate::config::TileDimensions::Custom,
    };
    let user_data = r.read_bits(2).ok_or("eof")? as u8;
    cfg.user_data = user_data;
    cfg.loq1_use_enhanced_depth = r.read_bit().ok_or("eof")?;
    let cswf = r.read_bit().ok_or("eof")?;
    if plane_mode {
        let plane_type = r.read_bits(8).ok_or("eof")?;
        let _ = plane_type;
    }
    if tswm {
        cfg.temporal_step_width_modifier = r.read_bits(8).ok_or("eof")? as u8;
    }
    if cswf {
        cfg.chroma_step_width_multiplier = r.read_bits(8).ok_or("eof")? as u8;
    }
    // Resolution.
    let res_table = crate::config::RESOLUTIONS;
    if res_type < 51 && res_type > 0 {
        cfg.width = res_table[res_type as usize].0;
        cfg.height = res_table[res_type as usize].1;
    } else if res_type == 63 {
        // custom resolution follows; parse from the raw bit stream.
        let mut w = 0u16;
        let mut h = 0u16;
        for _ in 0..16 {
            w = (w << 1) | r.read_bit().ok_or("eof")? as u16;
        }
        for _ in 0..16 {
            h = (h << 1) | r.read_bit().ok_or("eof")? as u16;
        }
        cfg.width = w;
        cfg.height = h;
    } else {
        return Err(format!("unsupported resolution type {res_type}"));
    }
    Ok(())
}

/// Picture configuration extracted from the picture config block.
pub struct PictureConfig {
    pub entropy_enabled: bool,
    pub temporal_refresh: bool,
    pub temporal_signalling_present: bool,
    pub step_width_l1: u32,
    pub step_width_l2: u32,
    pub quant_matrix: Option<[Vec<u8>; 2]>,
}

pub fn parse_picture_config(data: &[u8], cfg: &LcevcConfig) -> Result<PictureConfig, String> {
    let mut r = BitReader::new(data);
    let no_enhancement = r.read_bit().ok_or("eof")?;
    let mut pc = PictureConfig {
        entropy_enabled: !no_enhancement,
        temporal_refresh: false,
        temporal_signalling_present: false,
        step_width_l1: dequant::DEFAULT_STEP_WIDTH_L1 as u32,
        step_width_l2: 0,
        quant_matrix: None,
    };
    if !no_enhancement {
        let qm_mode = r.read_bits(3).ok_or("eof")?;
        let n_layers = if cfg.transform.layers() == 16 { 16 } else { 4 };
        let dequant_offset_signalled = r.read_bit().ok_or("eof")?;
        let _ = dequant_offset_signalled;
        let _pic_type = r.read_bit().ok_or("eof")?;
        pc.temporal_refresh = r.read_bit().ok_or("eof")?;
        let stepw_l1_enabled = r.read_bit().ok_or("eof")?;
        pc.step_width_l2 = r.read_bits(15).ok_or("eof")? as u32;
        let _dither = r.read_bit().ok_or("eof")?;
        if stepw_l1_enabled {
            pc.step_width_l1 = r.read_bits(15).ok_or("eof")? as u32;
            let _l1_filter = r.read_bit().ok_or("eof")?;
        }
        // Custom quant matrices: mode 2/3/5 carry LOQ0, 4/5 carry LOQ1.
        if qm_mode == 2 || qm_mode == 3 || qm_mode == 5 {
            let mut qm0 = Vec::with_capacity(n_layers);
            for _ in 0..n_layers {
                qm0.push(r.read_bits(8).ok_or("eof")? as u8);
            }
            pc.quant_matrix = Some([qm0, Vec::new()]);
        }
        if qm_mode == 4 || qm_mode == 5 {
            let mut qm1 = Vec::with_capacity(n_layers);
            for _ in 0..n_layers {
                qm1.push(r.read_bits(8).ok_or("eof")? as u8);
            }
            let mut q = pc.quant_matrix.take().unwrap_or_else(|| {
                let mut v = Vec::with_capacity(n_layers);
                v.resize(n_layers, 0);
                [v, Vec::new()]
            });
            q[1] = qm1;
            pc.quant_matrix = Some(q);
        }
    }
    pc.temporal_signalling_present = cfg.temporal_enabled && !pc.temporal_refresh;
    Ok(pc)
}

/// One decoded chunk's (value, run) stream.
struct DecodedChunk {
    events: Vec<CoeffEvent>,
}

/// Decode the residual chunk payload (three Huffman tables + data).
fn decode_residual_chunk(data: &[u8], total_tu: usize, rle_only: bool) -> DecodedChunk {
    let events = decode_coefficient_chunk(data, rle_only, total_tu);
    DecodedChunk { events }
}

/// Run the mirror decoder over one frame.
/// `blocks` is the parsed NAL content; `base` the decoded base picture.
pub fn decode_frame(
    cfg_in: &LcevcConfig,
    blocks: &[(u8, Vec<u8>)],
    base: &Picture,
    temporal_buffers: &mut [PlaneS16],
) -> Result<Picture, String> {
    let mut cfg = cfg_in.clone();
    let mut picture_config: Option<PictureConfig> = None;
    let mut chunks: Vec<Vec<Vec<Vec<u8>>>> = Vec::new(); // [plane][loq 0=L1,1=L2][layer]
    let mut chunk_flags: Vec<Vec<Vec<(bool, bool)>>> = Vec::new();
    let mut temporal_chunks: Vec<Vec<u8>> = Vec::new();
    let mut temporal_flags: Vec<(bool, bool)> = Vec::new();

    for (payload_type, payload) in blocks {
        match *payload_type {
            0 => {} // sequence config
            1 => parse_global_config(payload, &mut cfg)?,
            2 => picture_config = Some(parse_picture_config(payload, &cfg)?),
            3 | 4 => {
                // encoded data
                let nplanes = cfg.num_planes();
                let num_layers = cfg.num_layers();
                let mut r = BitReader::new(payload);
                let mut per_plane: Vec<Vec<Vec<Vec<u8>>>> = Vec::new();
                let mut per_plane_flags: Vec<Vec<Vec<(bool, bool)>>> = Vec::new();
                let mut t_chunks: Vec<Vec<u8>> = Vec::new();
                let mut t_flags: Vec<(bool, bool)> = Vec::new();
                let _ = (&mut per_plane, &mut per_plane_flags);
                let pc = picture_config.as_ref().unwrap();

                if *payload_type == 3 {
                    // flags: 2 bits per chunk
                    let mut flags: Vec<Vec<Vec<(bool, bool)>>> = Vec::new();
                    for _plane in 0..nplanes {
                        let mut loqs = Vec::new();
                        for _loq in 0..2 {
                            let mut layers = Vec::new();
                            for _l in 0..num_layers {
                                let e = r.read_bit().ok_or("eof")?;
                                let rle = r.read_bit().ok_or("eof")?;
                                layers.push((e, rle));
                            }
                            loqs.push(layers);
                        }
                        flags.push(loqs);
                    }
                    if pc.temporal_signalling_present {
                        for _plane in 0..nplanes {
                            let e = r.read_bit().ok_or("eof")?;
                            let rle = r.read_bit().ok_or("eof")?;
                            t_flags.push((e, rle));
                        }
                    }
                    // byte alignment
                    while r.bits_remaining() % 8 != 0 {
                        r.read_bit().ok_or("eof")?;
                    }
                    // chunk data
                    let bytes = read_remaining(&mut r);
                    let mut off = 0usize;
                    let mut datas: Vec<Vec<Vec<Vec<u8>>>> = Vec::new();
                    for plane in 0..nplanes {
                        let mut loqs: Vec<Vec<Vec<u8>>> = Vec::new();
                        for loq in 0..2 {
                            let mut layers: Vec<Vec<u8>> = Vec::new();
                            for l in 0..num_layers {
                                let (e, _) = flags[plane][loq][l];
                                if e {
                                    let (size, n) = read_multibyte(&bytes[off..])?;
                                    off += n;
                                    layers.push(bytes[off..off + size].to_vec());
                                    off += size;
                                } else {
                                    layers.push(Vec::new());
                                }
                            }
                            loqs.push(layers);
                        }
                        datas.push(loqs);
                        // The temporal chunk for this plane follows its
                        // residual chunks in the stream.
                        if pc.temporal_signalling_present {
                            let (e, _) = t_flags[plane];
                            if e {
                                let (size, n) = read_multibyte(&bytes[off..])?;
                                off += n;
                                t_chunks.push(bytes[off..off + size].to_vec());
                                off += size;
                            } else {
                                t_chunks.push(Vec::new());
                            }
                        }
                    }
                    per_plane = datas;
                    per_plane_flags = flags;
                } else {
                    return Err("tiled mirror decode not implemented".into());
                }
                chunks = per_plane;
                chunk_flags = per_plane_flags;
                temporal_chunks = t_chunks;
                temporal_flags = t_flags;
            }
            _ => {}
        }
    }

    let pc = picture_config.ok_or("no picture config")?;
    let kernel = cfg.upsampler.kernel();
    let apply_pa = cfg.predicted_average;
    let tu_size = cfg.tu_size();
    let num_layers = cfg.num_layers();
    let nplanes = cfg.num_planes();
    let dds = num_layers == 16;

    let qm_scaling_1d = cfg.scaling_l2 == crate::config::ScalingMode::Scale1D;
    let (qm0, qm1) = match &pc.quant_matrix {
        Some(q) if q[0].len() == num_layers && q[1].len() == num_layers => {
            (q[0].clone(), q[1].clone())
        }
        _ => (
            dequant::default_quant_matrix(dds, qm_scaling_1d, 0),
            dequant::default_quant_matrix(dds, qm_scaling_1d, 1),
        ),
    };

    let mut output = Picture::new(cfg.width as usize, cfg.height as usize, cfg.chroma);

    for plane in 0..nplanes {
        let base_s16 = base.planes[plane].to_s16(cfg.sample_depth());
        let l1_pred = upscale_plane(&base_s16, cfg.scaling_l1, &kernel, apply_pa);
        let mut l1_recon = l1_pred.clone();

        // LOQ1
        {
            let sw = dequant::loq_step_width(
                pc.step_width_l1 as i32, plane, false, cfg.chroma_step_width_multiplier);
            let table = DequantTable::compute(
                sw, &qm1, cfg.temporal_enabled, false, pc.temporal_refresh,
                cfg.temporal_step_width_modifier, -1, false);
            let forward = ForwardTransform::new(cfg.transform.to_bit(), cfg.scaling_l1 == crate::config::ScalingMode::Scale1D);
            let deblock = cfg.level1_filtering_signalled;
            apply_chunks(
                &mut l1_recon, &l1_pred, &table, &forward, tu_size,
                &chunks[plane][0], &chunk_flags[plane][0], &cfg, 1, plane,
                TemporalSignal::Inter, deblock, false, None, &mut PlaneS16::new(1, 1),
            )?;
        }

        // LOQ0
        let l2_pred = upscale_plane(&l1_recon, cfg.scaling_l2, &kernel, apply_pa);
        let mut l2_recon = l2_pred.clone();
        {
            let sw = dequant::loq_step_width(
                pc.step_width_l2 as i32, plane, true, cfg.chroma_step_width_multiplier);
            let table = DequantTable::compute(
                sw, &qm0, cfg.temporal_enabled, true, pc.temporal_refresh,
                cfg.temporal_step_width_modifier, -1, false);
            let forward = ForwardTransform::new(cfg.transform.to_bit(), cfg.scaling_l2 == crate::config::ScalingMode::Scale1D);
            let temporal_on = cfg.temporal_enabled && !pc.temporal_refresh;
            let mut tb = std::mem::replace(&mut temporal_buffers[plane],
                                           PlaneS16::new(l2_pred.width, l2_pred.height));
            let temporal_data = if temporal_on {
                Some((temporal_chunks[plane].clone(), temporal_flags[plane]))
            } else {
                None
            };
            apply_chunks(
                &mut l2_recon, &l2_pred, &table, &forward, tu_size,
                &chunks[plane][1], &chunk_flags[plane][1], &cfg, 0, plane,
                TemporalSignal::Inter, false, temporal_on, temporal_data.as_ref(),
                &mut tb,
            )?;
            temporal_buffers[plane] = tb;
        }

        output.planes[plane] = l2_recon.to_plane(cfg.sample_depth());
    }

    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn apply_chunks(
    recon: &mut PlaneS16,
    pred: &PlaneS16,
    table: &DequantTable,
    forward: &ForwardTransform,
    tu_size: usize,
    layer_datas: &[Vec<u8>],
    layer_flags: &[(bool, bool)],
    cfg: &LcevcConfig,
    loq: usize,
    plane: usize,
    _signal: TemporalSignal,
    deblock: bool,
    temporal_enabled: bool,
    temporal_data: Option<&(Vec<u8>, (bool, bool))>,
    tb: &mut PlaneS16,
) -> Result<(), String> {
    let num_layers = layer_datas.len();
    let (pw, ph) = cfg.plane_dimensions(loq, plane);
    let shift = tu_size.trailing_zeros();
    let num_across = (pw as usize) >> shift;
    let total_tu = num_across * ((ph as usize) >> shift);

    let mut decoded: Vec<DecodedChunk> = Vec::new();
    for l in 0..num_layers {
        let (enabled, rle_only) = layer_flags[l];
        if enabled {
            decoded.push(decode_residual_chunk(&layer_datas[l], total_tu, rle_only));
        } else {
            decoded.push(DecodedChunk { events: Vec::new() });
        }
    }

    if loq == 0 && plane == 0 {
        let mut covered = 0u64;
        let mut hits = Vec::new();
        for (i, ev) in decoded[1].events.iter().enumerate() {
            let prev = covered;
            covered += 1 + ev.zero_run as u64;
            if prev < 200 && covered >= 200 { hits.push((i, prev, ev.value)); }
        }
    }
    // Temporal signal runs (LOQ0 only).
    let mut temporal_runs: Vec<(u8, u32)> = Vec::new();
    if temporal_enabled {
        if let Some((data, (enabled, rle_only))) = temporal_data {
            if *enabled && !rle_only {
                let runs = decode_temporal_chunk(data, total_tu as u32)
                    .ok_or("temporal decode failed")?;
                temporal_runs = runs.iter().map(|r| (r.signal, r.count)).collect();
                if loq == 0 && plane == 0 {
                }
            }
        }
    }

    // Walk TUs, consuming events.
    let mut pos: Vec<usize> = vec![0; num_layers];
    let mut zeros_left: Vec<u32> = vec![0; num_layers];
    let mut temporal_pos = 0usize;
    let mut temporal_remaining = 0u32;
    let mut temporal_signal: TemporalSignal = TemporalSignal::Intra;
    let block_order = cfg.temporal_enabled || cfg.is_tiled();
    for i in 0..total_tu {
        // Consume the temporal signal for this TU before reconstructing it.
        if temporal_enabled {
            if temporal_remaining == 0 && temporal_pos < temporal_runs.len() {
                let (signal, count) = temporal_runs[temporal_pos];
                temporal_pos += 1;
                temporal_signal = if signal == 1 { TemporalSignal::Intra } else { TemporalSignal::Inter };
                temporal_remaining = count;
            }
            if temporal_remaining > 0 {
                temporal_remaining -= 1;
            }
        }
        let (x, y) = if block_order {
            let st = TuState::new(pw as usize, ph as usize, 0, 0, shift);
            st.block_position(i)
        } else {
            (((i % num_across) << shift) as usize, ((i / num_across) << shift) as usize)
        };
        let mut dq = vec![0i16; num_layers];
        let mut nonzero = false;

        for l in 0..num_layers {
            let mut c: i16 = 0;
            if zeros_left[l] > 0 {
                zeros_left[l] -= 1;
            } else {
                // Consume the next event.
                if pos[l] < decoded[l].events.len() {
                    let ev = decoded[l].events[pos[l]];
                    pos[l] += 1;
                    c = ev.value;
                    zeros_left[l] = ev.zero_run;
                }
            }
            if c != 0 {
                nonzero = true;
                let signal_idx = temporal_signal as usize;
                let layer = &table.layers[signal_idx][l];
                let v = if c > 0 {
                    c as i32 * layer.step_width as i32 + layer.offset as i32
                } else {
                    c as i32 * layer.step_width as i32 - layer.offset as i32
                };
                dq[l] = v.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
            }
        }
        let update_tb = temporal_enabled || (loq == 0 && cfg.temporal_enabled);
        // The temporal buffer must be applied to the output even when the
        // residual coefficients are all zero (the decoder's applyAddTemporal).
        if nonzero || (loq == 0 && cfg.temporal_enabled) {
            let residual = if nonzero { forward.inverse(&dq) } else { [0i16; 16] };
            if update_tb {
                // Update the temporal buffer and the reconstruction:
                // inter: tb += residual; output = pred + tb
                // intra: tb = residual; output = pred + tb
                for yy in 0..tu_size {
                    for xx in 0..tu_size {
                        let idx = residual_index(tu_size, xx, yy);
                        let r = residual[idx] as i32;
                        let new_tb = if temporal_signal == TemporalSignal::Inter {
                            tb.get(x + xx, y + yy) as i32 + r
                        } else {
                            r
                        };
                        let v = pred.get(x + xx, y + yy) as i32 + new_tb;
                        recon.set(x + xx, y + yy, v.clamp(i16::MIN as i32, i16::MAX as i32) as i16);
                        tb.set(x + xx, y + yy, new_tb.clamp(i16::MIN as i32, i16::MAX as i32) as i16);
                    }
                }
            } else {
                for yy in 0..tu_size {
                    for xx in 0..tu_size {
                        let idx = residual_index(tu_size, xx, yy);
                        let v = recon.get(x + xx, y + yy) as i32 + residual[idx] as i32;
                        recon.set(x + xx, y + yy, v.clamp(i16::MIN as i32, i16::MAX as i32) as i16);
                    }
                }
            }
        }
        if plane == 0 && loq == 0 && i == 0 {
        }
        let _ = deblock;
    }
    Ok(())
}

fn read_remaining(r: &mut BitReader) -> Vec<u8> {
    let mut out = Vec::new();
    while r.bits_remaining() >= 8 {
        out.push(r.read_bits(8).unwrap() as u8);
    }
    out
}

fn read_multibyte(data: &[u8]) -> Result<(usize, usize), String> {
    let mut acc = 0usize;
    let mut n = 0;
    loop {
        if n >= data.len() {
            return Err("multibyte overrun".into());
        }
        let b = data[n];
        n += 1;
        acc = (acc << 7) | (b & 0x7F) as usize;
        if b & 0x80 == 0 {
            break;
        }
    }
    Ok((acc, n))
}
