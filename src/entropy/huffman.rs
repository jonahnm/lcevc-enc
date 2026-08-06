//! Huffman ("Prefix Coding") tables and coding, bit-compatible with the
//! reference decoder (`huffman.c`).
//!
//! Stream header layout (bitstream version "AlignWithSpec", v2):
//!
//! ```text
//! min_code_length  u(5)
//! max_code_length  u(5)
//! if min == 31 && max == 31:  empty table (no further bits)
//! else if min == 0 && max == 0:  single symbol u(8)
//! else:
//!     length_bits = ceil(log2(max - min + 1))   // v2 table
//!     presence_bitmap u(1)
//!     if presence_bitmap == 1:
//!         for symbol in 0..256:
//!             present u(1)
//!             if present: (len - min) u(length_bits)
//!     else:
//!         symbol_count u(5)
//!         for i in 0..symbol_count:
//!             symbol u(8), (len - min) u(length_bits)
//! ```
//!
//! Codes are canonical: symbols sorted by (length asc, symbol desc), codes
//! assigned MSB-first starting from the longest length.

use crate::bitstream::BitWriter;

pub const MAX_SYMBOLS: usize = 256;

/// ceil(log2(x + 1)) for the v2 bit-width table, for x in 0..=31.
pub fn length_bits_for_range(range: u32) -> u32 {
    debug_assert!(range <= 31);
    let mut x = range + 1;
    let mut bits = 0u32;
    while x > 1 {
        x = (x + 1) >> 1;
        bits += 1;
    }
    bits
}

/// Build Huffman code lengths from symbol frequencies.
/// Returns `None` if there are no symbols (all frequencies zero), or
/// `Some((lengths, min_len, max_len))` otherwise.
pub fn build_code_lengths(frequencies: &[u64; MAX_SYMBOLS]) -> Option<([u8; MAX_SYMBOLS], u8, u8)> {
    // Standard Huffman via a simple priority selection (binary heap-free:
    // 256 symbols, quadratic scan is fine).
    struct Node {
        freq: u64,
        // Leaf: symbol = Some(s); internal: children.
        symbol: Option<usize>,
        left: Option<Box<Node>>,
        right: Option<Box<Node>>,
    }

    let mut nodes: Vec<Node> = frequencies
        .iter()
        .enumerate()
        .filter(|(_, &f)| f > 0)
        .map(|(s, &f)| Node { freq: f, symbol: Some(s), left: None, right: None })
        .collect();

    if nodes.is_empty() {
        return None;
    }
    if nodes.len() == 1 {
        let mut lengths = [0u8; MAX_SYMBOLS];
        lengths[nodes[0].symbol.unwrap()] = 1;
        return Some((lengths, 1, 1));
    }

    while nodes.len() > 1 {
        // Find two smallest (freq, then index for determinism).
        let mut i1 = 0;
        let mut i2 = 1;
        if nodes[i1].freq > nodes[i2].freq {
            std::mem::swap(&mut i1, &mut i2);
        }
        for i in 2..nodes.len() {
            if nodes[i].freq < nodes[i1].freq
                || (nodes[i].freq == nodes[i1].freq && nodes[i].symbol < nodes[i1].symbol)
            {
                i2 = i1;
                i1 = i;
            } else if nodes[i].freq < nodes[i2].freq
                || (nodes[i].freq == nodes[i2].freq && nodes[i].symbol < nodes[i2].symbol)
            {
                i2 = i;
            }
        }
        // Remove the larger index first so the smaller index stays valid.
        let (right, left) = if i1 > i2 {
            (nodes.remove(i1), nodes.remove(i2))
        } else {
            (nodes.remove(i2), nodes.remove(i1))
        };
        let freq = left.freq + right.freq;
        nodes.push(Node { freq, symbol: None, left: Some(Box::new(left)), right: Some(Box::new(right)) });
    }

    let root = nodes.remove(0);
    let mut lengths = [0u8; MAX_SYMBOLS];
    fn walk(node: &Node, depth: u32, lengths: &mut [u8; MAX_SYMBOLS]) {
        if let Some(s) = node.symbol {
            lengths[s] = depth as u8;
        } else {
            walk(node.left.as_ref().unwrap(), depth + 1, lengths);
            walk(node.right.as_ref().unwrap(), depth + 1, lengths);
        }
    }
    walk(&root, 0, &mut lengths);

    let min_len = lengths.iter().copied().filter(|&l| l > 0).min().unwrap();
    let max_len = lengths.iter().copied().max().unwrap();
    Some((lengths, min_len, max_len))
}

/// A single Huffman table ready for encoding.
pub struct HuffmanTable {
    /// code[length][index within that length group] -> symbol.
    /// Codes are canonical; the encoder needs symbol -> (code, length).
    pub code: [u32; MAX_SYMBOLS],
    pub length: [u8; MAX_SYMBOLS],
    pub min_len: u8,
    pub max_len: u8,
    /// True when the table has exactly one symbol (signalled via 0,0 header).
    pub single_symbol: Option<u8>,
    /// True when the table is empty (31,31 header).
    pub empty: bool,
}

impl HuffmanTable {
    /// Build from frequencies.
    pub fn from_frequencies(frequencies: &[u64; MAX_SYMBOLS]) -> HuffmanTable {
        let Some((lengths, min_len, max_len)) = build_code_lengths(frequencies) else {
            return HuffmanTable {
                code: [0; MAX_SYMBOLS],
                length: [0; MAX_SYMBOLS],
                min_len: 31,
                max_len: 31,
                single_symbol: None,
                empty: true,
            };
        };

        let nonzero: Vec<usize> = (0..MAX_SYMBOLS).filter(|&s| lengths[s] > 0).collect();
        if nonzero.len() == 1 {
            return HuffmanTable {
                code: [0; MAX_SYMBOLS],
                length: [0; MAX_SYMBOLS],
                min_len: 0,
                max_len: 0,
                single_symbol: Some(nonzero[0] as u8),
                empty: false,
            };
        }

        // Canonical code assignment (mirrors generateCodes):
        // entries sorted by (length asc, symbol desc).
        let mut entries: Vec<(u8, u8)> = nonzero
            .iter()
            .map(|&s| (lengths[s], s as u8))
            .collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)));

        let mut code = [0u32; MAX_SYMBOLS];
        let mut curr_length = max_len;
        let mut curr_code = 0u32;
        for idx in (0..entries.len()).rev() {
            let (bits, symbol) = entries[idx];
            if bits < curr_length {
                curr_code >>= curr_length - bits;
                curr_length = bits;
            }
            code[symbol as usize] = curr_code;
            curr_code += 1;
        }

        HuffmanTable {
            code,
            length: lengths,
            min_len,
            max_len,
            single_symbol: None,
            empty: false,
        }
    }

    /// Write the table header to the bit writer.
    pub fn write_header(&self, w: &mut BitWriter) {
        w.write_bits(self.min_len as u64, 5);
        w.write_bits(self.max_len as u64, 5);
        if self.empty {
            return;
        }
        if let Some(s) = self.single_symbol {
            w.write_byte(s);
            return;
        }
        let length_bits = length_bits_for_range(self.max_len as u32 - self.min_len as u32);
        let nonzero: Vec<u8> = (0..MAX_SYMBOLS)
            .filter(|&s| self.length[s] > 0)
            .map(|s| s as u8)
            .collect();
        if nonzero.len() <= 31 {
            w.write_bit(false); // presence bitmap = 0
            w.write_bits(nonzero.len() as u64, 5);
            for &s in &nonzero {
                w.write_byte(s);
                w.write_bits((self.length[s as usize] - self.min_len) as u64, length_bits);
            }
        } else {
            w.write_bit(true); // presence bitmap = 1
            for s in 0..256u16 {
                if self.length[s as usize] > 0 {
                    w.write_bit(true);
                    w.write_bits((self.length[s as usize] - self.min_len) as u64, length_bits);
                } else {
                    w.write_bit(false);
                }
            }
        }
    }

    /// Encode one symbol into the bit writer.
    #[inline]
    pub fn write_symbol(&self, w: &mut BitWriter, symbol: u8) {
        if let Some(s) = self.single_symbol {
            debug_assert_eq!(s, symbol);
            return;
        }
        debug_assert!(!self.empty);
        let len = self.length[symbol as usize] as u32;
        w.write_bits(self.code[symbol as usize] as u64, len);
    }
}

/// Bit reader (MSB-first), for the decoder side used in tests and the
/// self-verification path.
pub struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        BitReader { data, pos: 0 }
    }

    pub fn read_bit(&mut self) -> Option<bool> {
        if self.pos / 8 >= self.data.len() {
            return None;
        }
        let b = (self.data[self.pos / 8] >> (7 - (self.pos % 8))) & 1;
        self.pos += 1;
        Some(b == 1)
    }

    pub fn read_bits(&mut self, n: u32) -> Option<u64> {
        let mut v = 0u64;
        for _ in 0..n {
            v = (v << 1) | self.read_bit()? as u64;
        }
        Some(v)
    }

    pub fn bits_remaining(&self) -> usize {
        self.data.len() * 8 - self.pos
    }
}

/// Parsed Huffman table (decoder side).
#[derive(Debug)]
pub struct HuffmanDecoderTable {
    /// Canonical entries sorted by (length asc, symbol desc) with assigned
    /// codes.
    pub entries: Vec<(u8, u8, u32)>, // (length, symbol, code)
    pub min_len: u8,
    pub max_len: u8,
    pub single_symbol: Option<u8>,
    pub empty: bool,
}

impl HuffmanDecoderTable {
    /// Parse a table header from the reader.
    pub fn parse(r: &mut BitReader) -> Option<HuffmanDecoderTable> {
        if std::env::var("LCEVC_DUMP_HT").is_ok() {
            let mut save = BitReader { data: r.data, pos: r.pos };
            let raw: Vec<u8> = (0..10).map(|_| save.read_bit().unwrap() as u8).collect();
            eprintln!("  raw10: {raw:?}");
        }
        let min_len = r.read_bits(5)? as u8;
        let max_len = r.read_bits(5)? as u8;
        if std::env::var("LCEVC_DUMP_HT").is_ok() {
            eprintln!("  HT min={min_len} max={max_len}");
        }
        if max_len < min_len {
            return None;
        }
        if min_len == 31 && max_len == 31 {
            return Some(HuffmanDecoderTable {
                entries: Vec::new(),
                min_len,
                max_len,
                single_symbol: None,
                empty: true,
            });
        }
        if min_len == 0 && max_len == 0 {
            let symbol = r.read_bits(8)? as u8;
            return Some(HuffmanDecoderTable {
                entries: Vec::new(),
                min_len,
                max_len,
                single_symbol: Some(symbol),
                empty: false,
            });
        }
        let length_bits = length_bits_for_range(max_len as u32 - min_len as u32);
        let presence = r.read_bit()?;
        if std::env::var("LCEVC_DUMP_HT").is_ok() {
            eprintln!("HT min={min_len} max={max_len} lb={length_bits} presence={presence}");
        }
        let mut lengths = [0u8; MAX_SYMBOLS];
        if presence {
            for i in 0..256u16 {
                if r.read_bit()? {
                    let mut extra = 0u64;
                    if length_bits > 0 {
                        extra = r.read_bits(length_bits)?;
                    }
                    let l = min_len + extra as u8;
                    if l > max_len {
                        return None;
                    }
                    if l > 0 {
                        lengths[i as usize] = l;
                    }
                }
            }
        } else {
            let count = r.read_bits(5)? as usize;
            if count == 0 {
                return None;
            }
            for _ in 0..count {
                let symbol = r.read_bits(8)? as usize;
                let mut extra = 0u64;
                if length_bits > 0 {
                    extra = r.read_bits(length_bits)?;
                }
                let l = min_len + extra as u8;
                if l > max_len || l == 0 {
                    return None;
                }
                if std::env::var("LCEVC_DUMP_HT").is_ok() {
                    eprintln!("  sym={symbol} len={l}");
                }
                lengths[symbol] = l;
            }
        }

        let nonzero: Vec<usize> = (0..MAX_SYMBOLS).filter(|&s| lengths[s] > 0).collect();
        if nonzero.is_empty() {
            return None;
        }

        // Canonical assignment (mirrors generateCodes).
        let mut entries: Vec<(u8, u8, u32)> = nonzero
            .iter()
            .map(|&s| (lengths[s], s as u8, 0u32))
            .collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)));
        let mut curr_length = max_len;
        let mut curr_code = 0u32;
        for idx in (0..entries.len()).rev() {
            let bits = entries[idx].0;
            if bits < curr_length {
                curr_code >>= curr_length - bits;
                curr_length = bits;
            }
            entries[idx].2 = curr_code;
            curr_code += 1;
        }

        if std::env::var("LCEVC_DUMP_HT").is_ok() {
            eprintln!("  codes:");
            for (l, sym, code) in entries.iter().take(12) {
                eprintln!("    sym={sym} len={l} code={code:04x}");
            }
        }
        Some(HuffmanDecoderTable {
            entries,
            min_len,
            max_len,
            single_symbol: None,
            empty: false,
        })
    }

    /// Decode the next symbol.
    pub fn decode(&self, r: &mut BitReader) -> Option<u8> {
        if let Some(s) = self.single_symbol {
            return Some(s);
        }
        if self.empty {
            return None;
        }
        // Read bits one at a time; match against entries by (length, code).
        let mut code = 0u32;
        let mut len = 0u32;
        loop {
            let bit = r.read_bit()?;
            code = (code << 1) | bit as u32;
            len += 1;
            // Search entries with this exact length.
            for &(l, symbol, c) in &self.entries {
                if l as u32 == len && c == code {
                    return Some(symbol);
                }
            }
            if len > self.max_len as u32 {
                return None;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_and_roundtrip() {
        // Build a frequency distribution and round-trip a symbol stream.
        let mut freqs = [0u64; 256];
        let symbols = [10u8, 200, 10, 5, 200, 10, 10, 5, 5, 5, 1, 1, 1, 1, 1, 0, 0, 0, 7, 7, 7, 7, 7, 7, 7, 7];
        for &s in &symbols {
            freqs[s as usize] += 1;
        }
        let table = HuffmanTable::from_frequencies(&freqs);
        assert!(!table.empty);
        assert!(table.single_symbol.is_none());

        let mut w = BitWriter::new();
        table.write_header(&mut w);
        for &s in &symbols {
            table.write_symbol(&mut w, s);
        }
        let bytes = w.finish();

        let mut r = BitReader::new(&bytes);
        let dt = HuffmanDecoderTable::parse(&mut r).unwrap();
        assert_eq!(dt.min_len, table.min_len);
        assert_eq!(dt.max_len, table.max_len);
        for &s in &symbols {
            assert_eq!(dt.decode(&mut r), Some(s));
        }
    }

    #[test]
    fn single_symbol_case() {
        let mut freqs = [0u64; 256];
        freqs[42] = 100;
        let table = HuffmanTable::from_frequencies(&freqs);
        assert_eq!(table.single_symbol, Some(42));
        assert_eq!(table.min_len, 0);
        assert_eq!(table.max_len, 0);

        let mut w = BitWriter::new();
        table.write_header(&mut w);
        for _ in 0..10 {
            table.write_symbol(&mut w, 42);
        }
        let bytes = w.finish();
        assert_eq!(bytes.len(), 3); // header only: 5+5+8 bits -> 3 bytes

        let mut r = BitReader::new(&bytes);
        let dt = HuffmanDecoderTable::parse(&mut r).unwrap();
        assert_eq!(dt.single_symbol, Some(42));
        for _ in 0..10 {
            assert_eq!(dt.decode(&mut r), Some(42));
        }
    }

    #[test]
    fn empty_table_case() {
        let freqs = [0u64; 256];
        let table = HuffmanTable::from_frequencies(&freqs);
        assert!(table.empty);
        let mut w = BitWriter::new();
        table.write_header(&mut w);
        let bytes = w.finish();
        assert_eq!(bytes.len(), 2); // 5+5 bits -> 2 bytes
        let mut r = BitReader::new(&bytes);
        let dt = HuffmanDecoderTable::parse(&mut r).unwrap();
        assert!(dt.empty);
    }

    #[test]
    fn many_symbols_presence_bitmap() {
        // Use 40 symbols to force the presence-bitmap path.
        let mut freqs = [0u64; 256];
        for i in 0..40u8 {
            freqs[i as usize] = (40 - i as u64) * 3;
        }
        let table = HuffmanTable::from_frequencies(&freqs);
        let mut w = BitWriter::new();
        table.write_header(&mut w);
        for i in 0..200u8 {
            table.write_symbol(&mut w, i % 40);
        }
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        let dt = HuffmanDecoderTable::parse(&mut r).unwrap();
        for i in 0..200u8 {
            assert_eq!(dt.decode(&mut r), Some(i % 40));
        }
    }

    #[test]
    fn length_bits_table() {
        assert_eq!(length_bits_for_range(0), 0);
        assert_eq!(length_bits_for_range(1), 1);
        assert_eq!(length_bits_for_range(2), 2);
        assert_eq!(length_bits_for_range(3), 2);
        assert_eq!(length_bits_for_range(4), 3);
        assert_eq!(length_bits_for_range(7), 3);
        assert_eq!(length_bits_for_range(8), 4);
        assert_eq!(length_bits_for_range(15), 4);
        assert_eq!(length_bits_for_range(16), 5);
        assert_eq!(length_bits_for_range(31), 5);
    }
}
