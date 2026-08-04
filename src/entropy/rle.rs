//! RLE (run-length) symbol streams, bit-compatible with the reference
//! decoder (`entropy.c` + `huffman.c`).
//!
//! ## Coefficient streams
//!
//! Three contexts/states exist: LSB, MSB and ZeroRun. Each byte belongs to
//! the context it is written in and is entropy-coded with that context's
//! Huffman table.
//!
//! * LSB byte: `[run:1][data:6][msb:1]`. `msb` set means another (MSB) byte
//!   follows carrying the more significant bits; when `msb` is set bit 7 is
//!   part of the data (7 bits). `run` set means a zero-run count follows.
//!   LSB-only value: `data - 32` (range [-32, 31]).
//!   LSB+MSB value: `(((msb & 0x7f) << 8 | (lsb & 0xfe)) - 0x4000) >> 1`
//!   (range [-8192, 8191]).
//! * MSB byte: `[run:1][data:7]` (bits 7..14 of the value).
//! * ZeroRun byte: `[run:1][count:7]`, chunks written most-significant-first;
//!   `run` set means another (less significant) chunk follows.
//!
//! The first byte of a stream is always a residual in the LSB context.
//!
//! ## Size streams
//!
//! LSB/MSB bytes without the value bias and without runs:
//! unsigned value: LSB `[data:7][msb:1]`, MSB `[data:7]`; 7 or 14 bits.
//! signed (deltas): 7-bit values sign-extended from bit 6; 14-bit values
//! sign-extended from bit 13.
//!
//! ## Temporal streams
//!
//! Two contexts (signal 0 and signal 1). The first byte after the tables is
//! a raw byte whose bit 0 gives the initial signal; then runs of a signal
//! are encoded as counts (7-bit chunks, most-significant-first, `run` bit =
//! continuation), each run's count using the current signal's table. The
//! signal toggles after every run.

use crate::bitstream::BitWriter;
use crate::entropy::huffman::{HuffmanTable, MAX_SYMBOLS};

pub const LSB: usize = 0;
pub const MSB: usize = 1;
pub const RL: usize = 2;

/// One event in a coefficient stream: a (possibly zero) value followed by a
/// run of zero coefficients.
#[derive(Clone, Copy, Debug)]
pub struct CoeffEvent {
    pub value: i16,
    pub zero_run: u32,
}

/// Encode a coefficient value into an LSB (and possibly MSB) byte.
/// Returns (lsb_byte, Option<msb_byte>).
fn encode_value_bytes(value: i16) -> (u8, Option<u8>) {
    if (-32..=31).contains(&value) {
        let data6 = (value + 32) as u8 & 0x3F;
        (data6 << 1, None)
    } else {
        // combined = 2v + 0x4000 in [0, 0x7FFE]; LSB byte carries bits 1..7
        // (7 data bits), MSB byte carries bits 8..14.
        let combined = (value as i32 + 0x2000) << 1;
        let data7 = (combined >> 1) & 0x7F;
        let lsb = ((data7 as u8) << 1) | 1;
        let msb = ((combined >> 8) & 0x7F) as u8;
        (lsb, Some(msb))
    }
}

/// Generate the RLE byte stream for a sequence of coefficient events.
/// Returns per-context symbol lists with frequencies, and the interleaved
/// `(context, byte)` sequence in exactly the order the decoder consumes it.
pub fn rle_encode_events(
    events: &[CoeffEvent],
) -> ([Vec<u8>; 3], [u64; 3], Vec<(usize, u8)>) {
    let mut symbols: [Vec<u8>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut freqs = [0u64; 3];
    let mut sequence: Vec<(usize, u8)> = Vec::new();

    macro_rules! push {
        ($ctx:expr, $b:expr) => {{
            symbols[$ctx].push($b);
            freqs[$ctx] += 1;
            sequence.push(($ctx, $b));
        }};
    }

    for (i, ev) in events.iter().enumerate() {
        let v = ev.value;
        let z = ev.zero_run;
        debug_assert!(i == 0 || v != 0, "zero values only valid as the first event");

        let (lsb, msb_opt) = encode_value_bytes(v);
        if msb_opt.is_none() {
            // LSB-only: run flag in bit 7.
            let byte = lsb | if z > 0 { 0x80 } else { 0 };
            push!(LSB, byte);
            if z > 0 {
                write_zero_run(&mut symbols[RL], &mut sequence, &mut freqs, z);
            }
        } else {
            let msb = msb_opt.unwrap();
            // LSB with MSB following: bit 7 is data, run flag is on the MSB.
            push!(LSB, lsb);
            push!(MSB, msb | if z > 0 { 0x80 } else { 0 });
            if z > 0 {
                write_zero_run(&mut symbols[RL], &mut sequence, &mut freqs, z);
            }
        }
    }

    (symbols, freqs, sequence)
}

/// Append the zero-run byte sequence (most significant 7-bit chunk first).
fn write_zero_run(out: &mut Vec<u8>, sequence: &mut Vec<(usize, u8)>, freqs: &mut [u64; 3], mut z: u32) {
    let mut chunks = Vec::new();
    loop {
        chunks.push((z & 0x7F) as u8);
        z >>= 7;
        if z == 0 {
            break;
        }
    }
    for i in (0..chunks.len()).rev() {
        let cont = if i > 0 { 0x80 } else { 0 };
        let b = chunks[i] | cont;
        out.push(b);
        freqs[RL] += 1;
        sequence.push((RL, b));
    }
}

/// Build per-context Huffman tables and write the full chunk payload
/// (three table headers followed by the entropy-coded byte stream, with the
/// symbols interleaved in decoder order).
pub fn write_coefficient_chunk(events: &[CoeffEvent]) -> Vec<u8> {
    let (symbols, freqs, sequence) = rle_encode_events(events);

    let tables: [HuffmanTable; 3] = std::array::from_fn(|ctx| {
        let mut full = [0u64; MAX_SYMBOLS];
        for &s in &symbols[ctx] {
            full[s as usize] += 1;
        }
        HuffmanTable::from_frequencies(&full)
    });
    let _ = freqs;

    let mut w = BitWriter::new();
    for ctx in 0..3 {
        tables[ctx].write_header(&mut w);
    }
    for (ctx, b) in &sequence {
        tables[*ctx].write_symbol(&mut w, *b);
    }
    w.finish()
}

/// Write an RLE-only chunk (no Huffman tables). The bytes are the raw
/// interleaved RLE stream.
pub fn write_coefficient_chunk_rle_only(events: &[CoeffEvent]) -> Vec<u8> {
    let (_, _, sequence) = rle_encode_events(events);
    sequence.into_iter().map(|(_, b)| b).collect()
}

// ---------------------------------------------------------------------------
// Size streams
// ---------------------------------------------------------------------------

/// Encode sizes (unsigned), returning the payload (2 headers + data).
/// The LSB/MSB symbols are interleaved per size in decoder order.
pub fn write_sizes(sizes: &[u16], signed_delta: bool) -> Vec<u8> {
    let mut symbols: [Vec<u8>; 2] = [Vec::new(), Vec::new()];
    let mut sequence: Vec<(usize, u8)> = Vec::new();
    for &size in sizes {
        let mut lsb_bytes = Vec::new();
        let mut msb_bytes = Vec::new();
        encode_size_symbols(size, signed_delta, &mut lsb_bytes, &mut msb_bytes);
        symbols[LSB].extend(lsb_bytes.iter().copied());
        symbols[MSB].extend(msb_bytes.iter().copied());
        for &b in &lsb_bytes {
            sequence.push((LSB, b));
        }
        for &b in &msb_bytes {
            sequence.push((MSB, b));
        }
    }

    let tables: [HuffmanTable; 2] = std::array::from_fn(|ctx| {
        let mut full = [0u64; MAX_SYMBOLS];
        for &s in &symbols[ctx] {
            full[s as usize] += 1;
        }
        HuffmanTable::from_frequencies(&full)
    });

    let mut w = BitWriter::new();
    for ctx in 0..2 {
        tables[ctx].write_header(&mut w);
    }
    for (ctx, b) in &sequence {
        tables[*ctx].write_symbol(&mut w, *b);
    }
    w.finish()
}

fn encode_size_symbols(size: u16, signed_delta: bool, lsb_out: &mut Vec<u8>, msb_out: &mut Vec<u8>) {
    let value: i32 = if signed_delta {
        // 7-bit or 14-bit two's complement range.
        (size as i16) as i32
    } else {
        size as i32
    };
    if !signed_delta {
        if value < 128 {
            lsb_out.push(((value & 0x7F) << 1) as u8);
        } else {
            lsb_out.push((((value & 0x7F) << 1) | 1) as u8);
            msb_out.push(((value >> 7) & 0x7F) as u8);
        }
    } else {
        // Signed: value in [-64, 63] single byte, else 14-bit. In the 14-bit
        // case the sign is carried in bit 7 of the MSB byte (the reference
        // decoder sign-extends `val` from bit 14).
        if (-64..=63).contains(&value) {
            lsb_out.push(((value & 0x7F) << 1) as u8);
        } else {
            let mag = (value >> 7) & 0x7F;
            lsb_out.push((((value & 0x7F) << 1) | 1) as u8);
            msb_out.push((mag | if value < 0 { 0x80 } else { 0 }) as u8);
        }
    }
}

// ---------------------------------------------------------------------------
// Temporal streams
// ---------------------------------------------------------------------------

/// A run of temporal signal: (signal, count of TUs).
#[derive(Clone, Copy, Debug)]
pub struct TemporalRun {
    pub signal: u8, // 0 = inter, 1 = intra
    pub count: u32,
}

/// Write a temporal chunk: two table headers (signal 0, signal 1), the raw
/// initial byte and the run counts. Runs must alternate signal and the
/// count bytes for run i use the table of run i's signal.
pub fn write_temporal_chunk(runs: &[TemporalRun]) -> Vec<u8> {
    debug_assert!(!runs.is_empty());
    let initial = runs[0].signal & 1;
    for (i, r) in runs.iter().enumerate() {
        debug_assert_eq!((r.signal & 1), (initial ^ (i as u8 & 1)));
    }

    // Count symbols per run, and per-signal frequency tables.
    let mut run_symbols: Vec<Vec<u8>> = Vec::with_capacity(runs.len());
    let mut freqs: [Vec<u64>; 2] = [vec![0u64; MAX_SYMBOLS], vec![0u64; MAX_SYMBOLS]];
    for r in runs {
        let mut bytes = Vec::new();
        write_count(&mut bytes, r.count);
        for &b in &bytes {
            freqs[r.signal as usize][b as usize] += 1;
        }
        run_symbols.push(bytes);
    }

    let tables = [
        HuffmanTable::from_frequencies(&freqs[0][..].try_into().unwrap()),
        HuffmanTable::from_frequencies(&freqs[1][..].try_into().unwrap()),
    ];

    let mut w = BitWriter::new();
    tables[0].write_header(&mut w);
    tables[1].write_header(&mut w);
    w.write_byte(initial);
    for (i, r) in runs.iter().enumerate() {
        for &s in &run_symbols[i] {
            tables[r.signal as usize].write_symbol(&mut w, s);
        }
    }
    w.finish()
}

/// Append the count bytes (most significant 7-bit chunk first, continuation
/// bit set except on the last).
fn write_count(out: &mut Vec<u8>, mut count: u32) {
    debug_assert!(count > 0);
    let mut chunks = Vec::new();
    loop {
        chunks.push((count & 0x7F) as u8);
        count >>= 7;
        if count == 0 {
            break;
        }
    }
    for i in (0..chunks.len()).rev() {
        let cont = if i > 0 { 0x80 } else { 0 };
        out.push(chunks[i] | cont);
    }
}

// ---------------------------------------------------------------------------
// Decoder side (for tests and self-verification)
// ---------------------------------------------------------------------------

use crate::entropy::huffman::{BitReader, HuffmanDecoderTable};

/// Decode a coefficient chunk back into (value, zero_run) events.
/// Decoding stops cleanly at end of data (partial trailing events are
/// dropped, matching how the reference decoder finishes its TU loop).
pub fn decode_coefficient_chunk(data: &[u8], rle_only: bool, total_positions: usize) -> Vec<CoeffEvent> {
    let mut r = BitReader::new(data);
    let mut tables: [Option<HuffmanDecoderTable>; 3] = [None, None, None];
    if !rle_only {
        tables[LSB] = HuffmanDecoderTable::parse(&mut r);
        tables[MSB] = HuffmanDecoderTable::parse(&mut r);
        tables[RL] = HuffmanDecoderTable::parse(&mut r);
        if tables.iter().any(|t| t.is_none()) {
            return Vec::new();
        }
    }

    let mut byte_pos = 0usize;
    let mut read_symbol = |state: usize| -> Option<u8> {
        if rle_only {
            if byte_pos >= data.len() {
                return None;
            }
            let b = data[byte_pos];
            byte_pos += 1;
            Some(b)
        } else {
            tables[state].as_ref().unwrap().decode(&mut r)
        }
    };

    let mut events = Vec::new();
    let mut covered = 0usize;
    let mut state = LSB;
    let mut pending_value: i16 = 0;
    let mut have_pending = false;
    let mut zero_run: u32 = 0;

    loop {
        let sym = match read_symbol(state) {
            Some(s) => s,
            None => break,
        };
        match state {
            LSB => {
                let msb_flag = sym & 0x01;
                if msb_flag != 0 {
                    pending_value = (sym & 0xFE) as i16;
                    have_pending = true;
                    state = MSB;
                } else {
                    let value = (((sym >> 1) & 0x3F) as i32 - 32) as i16;
                    if sym & 0x80 != 0 {
                        pending_value = value;
                        have_pending = true;
                        zero_run = 0;
                        state = RL;
                    } else {
                        events.push(CoeffEvent { value, zero_run: 0 });
                        have_pending = false;
                        covered += 1;
                    }
                }
            }
            MSB => {
                let msb_bits = (sym & 0x7F) as i32;
                let combined = (msb_bits << 8) | (pending_value as i32 & 0xFF);
                pending_value = ((combined - 0x4000) >> 1) as i16;
                have_pending = true;
                if sym & 0x80 != 0 {
                    zero_run = 0;
                    state = RL;
                } else {
                    events.push(CoeffEvent { value: pending_value, zero_run: 0 });
                    have_pending = false;
                    state = LSB;
                    covered += 1;
                }
            }
            RL => {
                zero_run = (zero_run << 7) | (sym & 0x7F) as u32;
                if sym & 0x80 == 0 {
                    let run = zero_run;
                    events.push(CoeffEvent { value: pending_value, zero_run: run });
                    have_pending = false;
                    zero_run = 0;
                    state = LSB;
                    covered += 1 + run as usize;
                }
            }
            _ => unreachable!(),
        }
        if covered >= total_positions {
            break;
        }
    }
    let _ = have_pending;
    events
}

/// Decode a size stream into sizes.
pub fn decode_sizes(data: &[u8], num_sizes: usize, signed_delta: bool) -> Option<Vec<i16>> {
    let mut r = BitReader::new(data);
    let t_lsb = HuffmanDecoderTable::parse(&mut r)?;
    let t_msb = HuffmanDecoderTable::parse(&mut r)?;
    let mut out = Vec::with_capacity(num_sizes);
    for _ in 0..num_sizes {
        let lsb = t_lsb.decode(&mut r)?;
        let (value, has_msb) = if lsb & 0x01 != 0 {
            let msb = t_msb.decode(&mut r)?;
            ((((msb as i32) << 7) | ((lsb >> 1) as i32)) as i16, true)
        } else {
            (((lsb >> 1) as i16), false)
        };
        let signed_val = if signed_delta {
            if has_msb {
                // 14-bit sign extension (sign carried in bit 14 via the MSB
                // byte's bit 7, matching the reference decoder).
                let raw = value as i32;
                let mag = raw & 0x3FFF;
                if raw & 0x4000 != 0 { (mag - 0x4000) as i16 } else { mag as i16 }
            } else {
                let v = (value as i8) as i32;
                let v = v & 0x7F;
                if v & 0x40 != 0 { (v | !0x7F) as i16 } else { v as i16 }
            }
        } else {
            value
        };
        out.push(signed_val);
    }
    Some(out)
}

/// Decode a temporal chunk into runs, stopping once `total_tu` TUs have been
/// covered (the reference decoder stops when its TU loop is exhausted).
pub fn decode_temporal_chunk(data: &[u8], total_tu: u32) -> Option<Vec<TemporalRun>> {
    let mut r = BitReader::new(data);
    let t0 = HuffmanDecoderTable::parse(&mut r)?;
    let t1 = HuffmanDecoderTable::parse(&mut r)?;
    let raw = r.read_bits(8)? as u8;
    let mut signal = raw & 0x01;
    let mut runs = Vec::new();
    let mut covered = 0u32;
    loop {
        // Read a count with the current signal's table.
        let table = if signal == 0 { &t0 } else { &t1 };
        let mut count = 0u32;
        loop {
            let Some(sym) = table.decode(&mut r) else { return Some(runs); };
            count = (count << 7) | (sym & 0x7F) as u32;
            if sym & 0x80 == 0 {
                break;
            }
        }
        if count == 0 {
            break;
        }
        runs.push(TemporalRun { signal, count });
        covered += count;
        signal ^= 1;
        if covered >= total_tu {
            break;
        }
    }
    Some(runs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_huffman_stream() {
        use crate::entropy::huffman::{HuffmanTable, BitReader, HuffmanDecoderTable};
        // Symbol sequence from the combined runs case.
        let seq = [(0usize, 202u8), (2, 255), (2, 127), (0, 188), (2, 129), (2, 128), (2, 0),
                   (0, 201), (1, 192), (2, 134), (2, 141), (2, 32), (0, 57), (1, 63)];
        let mut syms: [Vec<u8>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        for (c, b) in &seq { syms[*c].push(*b); }
        let tables: [HuffmanTable; 3] = std::array::from_fn(|ctx| {
            let mut full = [0u64; 256];
            for &s in &syms[ctx] { full[s as usize] += 1; }
            HuffmanTable::from_frequencies(&full)
        });
        let mut w = crate::bitstream::BitWriter::new();
        for ctx in 0..3 { tables[ctx].write_header(&mut w); }
        for (c, b) in &seq { tables[*c].write_symbol(&mut w, *b); }
        let bytes = w.finish();
        println!("huffman bytes: {bytes:?}");
        let mut r = BitReader::new(&bytes);
        let dt = [
            HuffmanDecoderTable::parse(&mut r).unwrap(),
            HuffmanDecoderTable::parse(&mut r).unwrap(),
            HuffmanDecoderTable::parse(&mut r).unwrap(),
        ];
        let mut decoded = Vec::new();
        for (c, _) in &seq {
            let sym = dt[*c].decode(&mut r);
            let eb = syms[*c][0];
            println!("ctx {c}: expected {eb} got {sym:?}");
            decoded.push(sym);
        }
    }

    #[test]
    fn debug_size_deltas() {
        let deltas = [253i16, -214, 97, -98];
        let data = write_sizes(&deltas.iter().map(|&d| d as u16).collect::<Vec<_>>(), true);
        println!("size data: {data:?}");
        let decoded = decode_sizes(&data, deltas.len(), true).unwrap();
        println!("decoded: {decoded:?}");
        assert_eq!(decoded, deltas.to_vec());
    }

    #[test]
    fn debug_combined_runs() {
        let events = vec![
            CoeffEvent { value: 5, zero_run: 16383 },
            CoeffEvent { value: -2, zero_run: 16384 },
            CoeffEvent { value: 100, zero_run: 100000 },
            CoeffEvent { value: -100, zero_run: 0 },
        ];
        let data = write_coefficient_chunk_rle_only(&events);
        println!("data: {data:?}");
        let decoded = decode_coefficient_chunk(&data, true, 10000000);
        println!("decoded: {decoded:?}");
        let (_, _, seq) = rle_encode_events(&events);
        println!("seq: {seq:?}");
    }

    #[test]
    fn debug_large_runs() {
        for run in [127u32, 128, 16383, 16384, 20000, 100000, 1000000] {
            let events = vec![CoeffEvent { value: 5, zero_run: run }];
            let data = write_coefficient_chunk_rle_only(&events);
            let decoded = decode_coefficient_chunk(&data, true, 10000000);
            let ok = decoded.len() == 1 && decoded[0].value == 5 && decoded[0].zero_run == run;
            println!("run {run}: data={data:?} decoded={decoded:?} ok={ok}");
            assert!(ok, "run {run} failed");
        }
    }

    #[test]
    fn coefficient_chunk_roundtrip_large_runs() {
        // Runs beyond 14 bits (3+ bytes) and leading-zero runs.
        let events = vec![
            CoeffEvent { value: 5, zero_run: 16383 },
            CoeffEvent { value: -2, zero_run: 16384 },
            CoeffEvent { value: 100, zero_run: 100000 },
            CoeffEvent { value: -100, zero_run: 0 },
        ];
        let total: usize = events.iter().map(|e| 1 + e.zero_run as usize).sum();
        let data = write_coefficient_chunk(&events);
        let decoded = decode_coefficient_chunk(&data, false, total);
        assert_eq!(decoded.len(), events.len());
        for (a, b) in events.iter().zip(decoded.iter()) {
            assert_eq!(a.value, b.value);
            assert_eq!(a.zero_run, b.zero_run);
        }
        let data2 = write_coefficient_chunk_rle_only(&events);
        let decoded2 = decode_coefficient_chunk(&data2, true, total);
        assert_eq!(decoded2.len(), events.len());
        for (a, b) in events.iter().zip(decoded2.iter()) {
            assert_eq!(a.value, b.value);
            assert_eq!(a.zero_run, b.zero_run);
        }
    }

    #[test]
    fn coefficient_chunk_roundtrip() {
        // Note: the first event must be a value (possibly zero) followed by
        // its zero run.
        let events = vec![
            CoeffEvent { value: 5, zero_run: 3 },
            CoeffEvent { value: -2, zero_run: 0 },
            CoeffEvent { value: 100, zero_run: 1 },
            CoeffEvent { value: -100, zero_run: 10 },
        ];
        let data = write_coefficient_chunk(&events);
        let decoded = decode_coefficient_chunk(&data, false, 18);
        assert_eq!(decoded.len(), events.len());
        for (a, b) in events.iter().zip(decoded.iter()) {
            assert_eq!(a.value, b.value);
            assert_eq!(a.zero_run, b.zero_run);
        }
    }

    #[test]
    fn coefficient_chunk_roundtrip_rle_only() {
        let events = vec![
            CoeffEvent { value: 5, zero_run: 3 },
            CoeffEvent { value: -2, zero_run: 0 },
            CoeffEvent { value: 100, zero_run: 1 },
            CoeffEvent { value: -100, zero_run: 10 },
        ];
        let data = write_coefficient_chunk_rle_only(&events);
        let decoded = decode_coefficient_chunk(&data, true, 18);
        assert_eq!(decoded.len(), events.len());
        for (a, b) in events.iter().zip(decoded.iter()) {
            assert_eq!(a.value, b.value);
            assert_eq!(a.zero_run, b.zero_run);
        }
    }

    #[test]
    fn sizes_roundtrip_unsigned() {
        let sizes = [5u16, 127, 128, 129, 16383, 1, 1000];
        let data = write_sizes(&sizes, false);
        let decoded = decode_sizes(&data, sizes.len(), false).unwrap();
        for (a, b) in sizes.iter().zip(decoded.iter()) {
            assert_eq!(*a as i16, *b);
        }
    }

    #[test]
    fn sizes_roundtrip_signed_deltas() {
        let deltas = [5i16, -5, 63, -64, 1000, -1000, 8191, -8192];
        let data = write_sizes(&deltas.iter().map(|&d| d as u16).collect::<Vec<_>>(), true);
        let decoded = decode_sizes(&data, deltas.len(), true).unwrap();
        for (a, b) in deltas.iter().zip(decoded.iter()) {
            assert_eq!(*a, *b);
        }
    }

    #[test]
    fn debug_temporal_single_run() {
        let runs = vec![TemporalRun { signal: 0, count: 57600 }];
        let data = write_temporal_chunk(&runs);
        println!("temporal bytes: {data:?}");
        let decoded = decode_temporal_chunk(&data, 57600).unwrap();
        println!("decoded: {decoded:?}");
        assert_eq!(decoded.len(), runs.len());
        assert_eq!(decoded[0].signal, runs[0].signal);
        assert_eq!(decoded[0].count, runs[0].count);
    }

    #[test]
    fn temporal_roundtrip() {
        let runs = vec![
            TemporalRun { signal: 0, count: 3 },
            TemporalRun { signal: 1, count: 2 },
            TemporalRun { signal: 0, count: 5 },
            TemporalRun { signal: 1, count: 1 },
        ];
        let data = write_temporal_chunk(&runs);
        let decoded = decode_temporal_chunk(&data, 11).unwrap();
        assert_eq!(decoded.len(), runs.len());
        for (a, b) in runs.iter().zip(decoded.iter()) {
            assert_eq!(a.signal, b.signal);
            assert_eq!(a.count, b.count);
        }
    }
}

