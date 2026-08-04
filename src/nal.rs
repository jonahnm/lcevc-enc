//! NAL unit and process-block framing, matching the reference decoder
//! (lcevcdec `unencapsulate()` + `parseBlock()`).
//!
//! Per-access-unit enhancement data layout:
//!
//! ```text
//! [0x00 0x00 0x01] [nal_header:2] [process blocks...] [0x80]
//! ```
//!
//! * Start code: `00 00 01` (3 bytes).
//! * NAL header: `forbidden_zero_bit(1)=0 forbidden_one_bit(1)=1
//!   nal_unit_type(5) reserved_flag(9)=0b111111111`. IDR pictures use
//!   nal_unit_type 29 (header byte 0x7B), others 28 (0x79); second byte is
//!   0xFF. The reference decoder verifies `(b0 & 0xC1) == 0x41 && b1 == 0xFF`.
//! * RBSP stop byte: the final byte of the NAL unit must be 0x80
//!   (rbsp_stop_one_bit followed by zeros); the decoder validates this.
//! * Emulation prevention: after two consecutive 0x00 bytes an 0x03 byte is
//!   inserted before any data byte <= 0x03. The decoder skips 0x03 whenever
//!   it follows two 0x00 bytes.
//!
//! Process block header (one byte): `payload_size_type(3) payload_type(5)`.
//! Size types 0..5 are the literal sizes 0..5; type 7 (custom) is followed by
//! a multibyte size.

use crate::bitstream::{write_multibyte, BitWriter};

pub const NAL_TYPE_NON_IDR: u8 = 28;
pub const NAL_TYPE_IDR: u8 = 29;

pub const BLOCK_SEQUENCE_CONFIG: u8 = 0;
pub const BLOCK_GLOBAL_CONFIG: u8 = 1;
pub const BLOCK_PICTURE_CONFIG: u8 = 2;
pub const BLOCK_ENCODED_DATA: u8 = 3;
pub const BLOCK_ENCODED_DATA_TILED: u8 = 4;
pub const BLOCK_ADDITIONAL_INFO: u8 = 5;
pub const BLOCK_FILLER: u8 = 6;

/// Build one enhancement NAL unit (with start code, header, emulation
/// prevention and RBSP stop byte) from the given process blocks, each
/// described by (payload_type, payload bytes).
pub fn build_nal_unit(idr: bool, blocks: &[(u8, Vec<u8>)]) -> Vec<u8> {
    let mut rbsp: Vec<u8> = Vec::new();

    // NAL unit header.
    let nal_type = if idr { NAL_TYPE_IDR } else { NAL_TYPE_NON_IDR };
    let b0: u8 = 0x41 | (nal_type << 1); // fzb=0, fob=1, type, reserved bit0=0
    rbsp.push(b0);
    rbsp.push(0xFF); // reserved_flag (8 remaining bits)

    // Process blocks: each payload is preceded by its header byte and,
    // for custom sizes, the multibyte size.
    for (payload_type, payload) in blocks {
        let size = payload.len();
        rbsp.push(block_header_byte(*payload_type, size));
        if size > 5 {
            let mut w = BitWriter::new();
            write_multibyte(&mut w, size as u64);
            rbsp.extend(w.finish());
        }
        rbsp.extend_from_slice(payload);
    }

    // Emulation prevention: insert 0x03 before a byte <= 0x03 when exactly
    // two consecutive 0x00 bytes precede it. The zero counter mirrors the
    // decoder's `unencapsulate` loop so the round trip is lossless.
    let mut out = Vec::with_capacity(rbsp.len() + 16);
    out.push(0x00);
    out.push(0x00);
    out.push(0x01);
    let mut zeros = 0u32;
    for &b in &rbsp {
        if zeros == 2 && b <= 0x03 {
            out.push(0x03);
            zeros = 0;
        }
        if b == 0x00 {
            zeros += 1;
        } else {
            zeros = 0;
        }
        out.push(b);
    }
    // RBSP stop byte (always 0x80; never needs emulation prevention).
    out.push(0x80);
    out
}

/// Header byte for a block whose payload size fits in a size type 0..5.
/// Sizes above 5 use the custom size type (7) with a multibyte size.
pub fn block_header_byte(payload_type: u8, payload_size: usize) -> u8 {
    if payload_size <= 5 {
        ((payload_size as u8) << 5) | (payload_type & 0x1F)
    } else {
        (7u8 << 5) | (payload_type & 0x1F)
    }
}

/// Header byte for a custom-sized block (size follows as a multibyte).
pub fn block_header_custom(payload_type: u8) -> u8 {
    (7u8 << 5) | (payload_type & 0x1F)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nal_unit_roundtrip_matches_decoder() {
        // Payload bytes chosen to force emulation prevention patterns.
        let payload = vec![0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x03, 0x04, 0xAA, 0x00, 0x00, 0x05];
        let nal = build_nal_unit(false, &[(BLOCK_ENCODED_DATA, payload)]);

        // Header.
        assert_eq!(&nal[0..3], &[0x00, 0x00, 0x01]);
        assert_eq!(nal[3], 0x79);
        assert_eq!(nal[4], 0xFF);
        // Stop byte.
        assert_eq!(*nal.last().unwrap(), 0x80);

        // Reverse the reference decoder's unencapsulate(): skip start code +
        // header, drop 0x03 after two 0x00, drop last byte.
        let mut rbsp = Vec::new();
        let mut i = 3usize;
        let mut zeros = 0u32;
        while i < nal.len() - 1 {
            let b = nal[i];
            i += 1;
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
        // Expect: header + payload with the inserted 0x03 removed.
        let mut expected = vec![0x79, 0xFF];
        // Block header: size 12 -> custom, then multibyte 12, then payload.
        expected.push(0xE3); // size 7 (custom) + type 3 (encoded data)
        expected.push(0x0C); // multibyte 12
        expected.extend_from_slice(&[0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x03, 0x04, 0xAA, 0x00, 0x00, 0x05]);
        assert_eq!(rbsp, expected);
    }

    #[test]
    fn small_block_headers() {
        // 2-byte payload: size type 2, type 3 (encoded data).
        assert_eq!(block_header_byte(BLOCK_ENCODED_DATA, 2), (2 << 5) | 3);
    }

}
