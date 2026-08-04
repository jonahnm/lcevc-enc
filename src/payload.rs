//! Per-picture payload writers: picture configuration and encoded (tiled)
//! data, matching the reference decoder's parsing order exactly.

use crate::bitstream::{write_multibyte, BitWriter};
use crate::config::LcevcConfig;
use crate::encoder::Chunk;

/// Write the picture configuration payload (block type 2).
///
/// `quant_matrix` carries the per-LOQ matrices; when non-empty the config
/// signals quant_matrix_mode = CustomBothUnique (5) with both matrices in the
/// stream, so the decoder needs no cross-frame state.
pub fn write_picture_config(
    cfg: &LcevcConfig,
    step_width_l1: u32,
    step_width_l2: u32,
    temporal_refresh: bool,
    quant_matrix: &[Vec<u8>; 2],
) -> Vec<u8> {
    let _ = cfg;
    let mut w = BitWriter::new();

    // Byte 1: no_enhancement=0, quant_matrix_mode=5 (CustomBothUnique),
    // dequant_offset_signalled=0, picture_type=0 (frame),
    // temporal_refresh, step_width_level1_enabled=1.
    w.write_bit(false); // no_enhancement_bit_flag
    w.write_bits(5, 3); // quant_matrix_mode = CustomBothUnique
    w.write_bit(false); // dequant_offset_signalled_flag
    w.write_bit(false); // picture_type_bit_flag (frame)
    w.write_bit(temporal_refresh);
    w.write_bit(true); // step_width_level1_enabled_flag

    // u16: step_width_level2 (15 bits) + dithering_control_flag (0).
    let sw2 = step_width_l2.min(0x7FFF) as u16;
    w.write_bits(sw2 as u64, 15);
    w.write_bit(false); // dithering_control_flag

    // step_width_level1 (15 bits) + level1_filtering_enabled_flag (0).
    let sw1 = step_width_l1.min(0x7FFF) as u16;
    w.write_bits(sw1 as u64, 15);
    w.write_bit(false); // level1_filtering_enabled_flag

    // Custom quant matrices (LOQ0 first, then LOQ1), one byte per layer.
    for qm in [&quant_matrix[0], &quant_matrix[1]] {
        for &v in qm {
            w.write_bits(v as u64, 8);
        }
    }

    w.finish()
}

/// Build the encoded-data payload for a frame.
///
/// `residual_chunks[plane][loq 0=L1,1=L2][layer]` is the per-tile chunk list
/// (one chunk per tile for that plane/LOQ; a single chunk when not tiled).
/// `temporal_chunks[plane]` holds one chunk per LOQ0 tile.
pub fn write_encoded_data(
    cfg: &LcevcConfig,
    residual_chunks: &[Vec<Vec<Vec<Chunk>>>],
    temporal_chunks: &[Vec<Chunk>],
    temporal_signalling_present: bool,
) -> Vec<u8> {
    if cfg.is_tiled() {
        write_encoded_data_tiled(cfg, residual_chunks, temporal_chunks, temporal_signalling_present)
    } else {
        write_encoded_data_plain(cfg, residual_chunks, temporal_chunks, temporal_signalling_present)
    }
}

fn write_encoded_data_plain(
    cfg: &LcevcConfig,
    residual_chunks: &[Vec<Vec<Vec<Chunk>>>],
    temporal_chunks: &[Vec<Chunk>],
    temporal_signalling_present: bool,
) -> Vec<u8> {
    let nplanes = cfg.num_planes();
    let num_layers = cfg.num_layers();

    let mut w = BitWriter::new();

    // Flags: 2 bits per chunk, in parse order (plane, LOQ1 then LOQ0, layer).
    for plane in 0..nplanes {
        for loq in 0..2 {
            for layer in 0..num_layers {
                let chunk = &residual_chunks[plane][loq][layer][0];
                w.write_bit(chunk.entropy_enabled);
                w.write_bit(chunk.rle_only);
            }
        }
        if temporal_signalling_present {
            let chunk = &temporal_chunks[plane][0];
            w.write_bit(chunk.entropy_enabled);
            w.write_bit(chunk.rle_only);
        }
    }
    w.byte_alignment();

    // Chunk data: multibyte size + bytes, in the same order.
    let mut out = w.finish();
    for plane in 0..nplanes {
        for loq in 0..2 {
            for layer in 0..num_layers {
                let chunk = &residual_chunks[plane][loq][layer][0];
                if chunk.entropy_enabled {
                    write_multibyte_into(&mut out, chunk.data.len() as u64);
                    out.extend_from_slice(&chunk.data);
                }
            }
        }
        if temporal_signalling_present {
            let chunk = &temporal_chunks[plane][0];
            if chunk.entropy_enabled {
                write_multibyte_into(&mut out, chunk.data.len() as u64);
                out.extend_from_slice(&chunk.data);
            }
        }
    }
    out
}

fn write_encoded_data_tiled(
    cfg: &LcevcConfig,
    residual_chunks: &[Vec<Vec<Vec<Chunk>>>],
    temporal_chunks: &[Vec<Chunk>],
    temporal_signalling_present: bool,
) -> Vec<u8> {
    let nplanes = cfg.num_planes();
    let num_layers = cfg.num_layers();

    let mut w = BitWriter::new();

    // RLE-only flags: 1 bit per (plane, LOQ1 then LOQ0, layer), plus one per
    // plane for the temporal chunk. Broadcast to all tiles of the layer.
    for plane in 0..nplanes {
        for loq in 0..2 {
            for layer in 0..num_layers {
                w.write_bit(residual_chunks[plane][loq][layer][0].rle_only);
            }
        }
        if temporal_signalling_present {
            w.write_bit(temporal_chunks[plane][0].rle_only);
        }
    }
    w.byte_alignment();

    // Per-tile entropy flags: raw bits or RLE-compressed.
    let mut flag_w = BitWriter::new();
    let mut flags: Vec<bool> = Vec::new();
    for plane in 0..nplanes {
        for loq in 0..2 {
            for layer in 0..num_layers {
                for chunk in &residual_chunks[plane][loq][layer] {
                    flag_w.write_bit(chunk.entropy_enabled);
                    flags.push(chunk.entropy_enabled);
                }
            }
        }
        if temporal_signalling_present {
            for chunk in &temporal_chunks[plane] {
                flag_w.write_bit(chunk.entropy_enabled);
                flags.push(chunk.entropy_enabled);
            }
        }
    }
    let mut out = w.finish();

    if cfg.per_tile_entropy_compression {
        // RLE-compressed flags: [initial symbol byte][multibyte runs...],
        // alternating symbols (mirrors tiledRLEDecoder).
        let first = flags.first().copied().unwrap_or(false);
        out.push(first as u8);
        let mut runs: Vec<u64> = Vec::new();
        let mut cur = first;
        let mut count = 0u64;
        for &f in &flags {
            if f == cur {
                count += 1;
            } else {
                runs.push(count);
                cur = f;
                count = 1;
            }
        }
        if count > 0 {
            runs.push(count);
        }
        for &r in &runs {
            let mut wb = BitWriter::new();
            write_multibyte(&mut wb, r);
            out.extend(wb.finish());
        }
    } else {
        out.extend(flag_w.finish());
    }

    // Chunk data with optional size compression.
    for plane in 0..nplanes {
        for loq in 0..2 {
            for layer in 0..num_layers {
                write_layer_tile_data(cfg, &mut out, &residual_chunks[plane][loq][layer]);
            }
        }
        if temporal_signalling_present {
            write_layer_tile_data(cfg, &mut out, &temporal_chunks[plane]);
        }
    }
    out
}

/// Write one layer's tile data: either per-tile multibyte sizes inline, or a
/// compressed size stream (with optional delta coding) followed by the tile
/// data, matching `parseEncodedDataTiledEntropyChunk`/`TemporalChunk`.
fn write_layer_tile_data(cfg: &LcevcConfig, out: &mut Vec<u8>, chunks: &[Chunk]) {
    let enabled: Vec<&Chunk> = chunks.iter().filter(|c| c.entropy_enabled).collect();

    if cfg.tile_size_compression != 0 {
        // Compressed sizes stream (2 Huffman table headers + symbols).
        let sizes: Vec<u16> = enabled.iter().map(|c| c.data.len() as u16).collect();
        let signed = cfg.tile_size_compression == 2;
        if !signed {
            out.extend(crate::entropy::rle::write_sizes(&sizes, false));
        } else {
            // Delta coding: sizes[i] += sizes[i-1] on the decoder side.
            let mut deltas: Vec<u16> = Vec::with_capacity(sizes.len());
            let mut prev = 0i32;
            for &s in &sizes {
                deltas.push((s as i32 - prev) as u16);
                prev = s as i32;
            }
            out.extend(crate::entropy::rle::write_sizes(&deltas, true));
        }
    }

    // Tile data.
    for chunk in &enabled {
        if cfg.tile_size_compression == 0 {
            write_multibyte_into(out, chunk.data.len() as u64);
        }
        out.extend_from_slice(&chunk.data);
    }
}

fn write_multibyte_into(out: &mut Vec<u8>, value: u64) {
    let mut w = BitWriter::new();
    write_multibyte(&mut w, value);
    out.extend(w.finish());
}
