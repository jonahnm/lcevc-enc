//! MSB-first bit writer plus the LCEVC multibyte / byte-alignment helpers.
//!
//! All syntax elements in the LCEVC bitstream are written MSB first (most
//! significant bit of the first byte first), exactly as parsed by the
//! reference decoder (lcevcdec, `BitStream` in `bitstream.c`).

#[derive(Default)]
pub struct BitWriter {
    pub bytes: Vec<u8>,
    bit_count: u32,
}

impl BitWriter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Total number of bits written so far.
    pub fn bit_count(&self) -> u32 {
        self.bit_count
    }

    /// Number of bytes of output so far (may be less than `bytes.len()` until
    /// the final partial byte is flushed).
    pub fn byte_count(&self) -> usize {
        self.bytes.len()
    }

    /// Write `n` bits of `value` (MSB first). Only the low `n` bits are used.
    pub fn write_bits(&mut self, value: u64, n: u32) {
        debug_assert!(n <= 64);
        debug_assert!(n == 64 || value < (1u64 << n));
        for i in (0..n).rev() {
            let bit = ((value >> i) & 1) as u8;
            let byte_idx = (self.bit_count / 8) as usize;
            if self.bit_count % 8 == 0 {
                self.bytes.push(bit << 7);
            } else {
                self.bytes[byte_idx] |= bit << (7 - (self.bit_count % 8));
            }
            self.bit_count += 1;
        }
    }

    /// Write a single bit.
    pub fn write_bit(&mut self, bit: bool) {
        self.write_bits(bit as u64, 1);
    }

    /// Write one byte (8 bits, MSB first).
    pub fn write_byte(&mut self, byte: u8) {
        self.write_bits(byte as u64, 8);
    }

    /// Write a 16-bit big-endian value.
    pub fn write_u16(&mut self, value: u16) {
        self.write_byte((value >> 8) as u8);
        self.write_byte((value & 0xFF) as u8);
    }

    /// Byte-align: emit zero bits (LCEVC `byte_alignment()` syntax: the
    /// alignment bits are `alignment_bit_equal_to_zero` per
    /// ISO/IEC 23094-2 FDAM 1 7.3.12) so that the stream ends on a byte
    /// boundary. If already aligned, nothing is written.
    pub fn byte_alignment(&mut self) {
        while self.bit_count % 8 != 0 {
            self.write_bit(false);
        }
    }

    /// Byte-align by padding with zero bits only (used where the decoder just
    /// skips whole bytes and the padding pattern is not normative).
    pub fn byte_align_zero(&mut self) {
        while self.bit_count % 8 != 0 {
            self.write_bit(false);
        }
    }

    /// Whether the stream is currently byte-aligned.
    pub fn is_aligned(&self) -> bool {
        self.bit_count % 8 == 0
    }

    /// Finalize and return the bytes, padding the final partial byte with
    /// zero bits.
    pub fn finish(mut self) -> Vec<u8> {
        self.byte_align_zero();
        self.bytes
    }
}

/// Encode a value as an LCEVC "multibyte" (7-bit groups, MSB group first,
/// continuation flag in bit 7 of every group except the last).
pub fn write_multibyte(w: &mut BitWriter, mut value: u64) {
    // Determine number of 7-bit groups.
    let mut groups = [0u8; 10];
    let mut n = 0;
    loop {
        groups[n] = (value & 0x7F) as u8;
        n += 1;
        value >>= 7;
        if value == 0 {
            break;
        }
    }
    // Write most significant group first.
    for i in (0..n).rev() {
        let cont = if i > 0 { 0x80 } else { 0x00 };
        w.write_byte(groups[i] | cont);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bits_msb_first() {
        let mut w = BitWriter::new();
        w.write_bits(0b1010, 4);
        w.write_bits(0b110011, 6);
        let bytes = w.finish();
        assert_eq!(bytes, vec![0b10101100, 0b11000000]);
    }

    #[test]
    fn byte_alignment() {
        let mut w = BitWriter::new();
        w.write_bits(0b101, 3);
        w.byte_alignment();
        let bytes = w.finish();
        assert_eq!(bytes, vec![0b10100000]);
    }

    #[test]
    fn multibyte_roundtrip() {
        for v in [0u64, 1, 127, 128, 16383, 16384, 1 << 21, u32::MAX as u64] {
            let mut w = BitWriter::new();
            write_multibyte(&mut w, v);
            let bytes = w.finish();
            // Decode back (mirror of the reference decoder's bytestreamReadMultiByte).
            let mut acc = 0u64;
            let mut nbytes = 0;
            for b in &bytes {
                acc = (acc << 7) | (b & 0x7F) as u64;
                nbytes += 1;
                if b & 0x80 == 0 {
                    break;
                }
            }
            assert_eq!(acc, v, "value {v}");
            assert!(bytes[nbytes - 1] & 0x80 == 0);
        }
    }
}
