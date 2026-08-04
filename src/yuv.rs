//! Raw YUV and Y4M (yuv4mpeg) input reading.
//!
//! Supports 8-bit (yuv420p) and 10-bit (yuv420p10le) planar input. The Y4M
//! reader is streaming: frames are consumed incrementally from any `Read`
//! (e.g. stdin), so a source can be piped in without raw YUV on disk.

use crate::config::ChromaFormat;
use crate::frame::Picture;

fn plane_size(plane: &crate::frame::Plane) -> usize {
    plane.width * plane.height
}

/// Read one yuv420p frame of the given depth (8 or 10) from a byte slice.
/// For 10-bit input the samples are little-endian 16-bit values.
pub fn read_yuv420_frame(data: &[u8], width: usize, height: usize, depth: u8) -> Result<Picture, String> {
    let mut pic = Picture::new(width, height, ChromaFormat::C420);
    let mut off = 0;
    for plane in &mut pic.planes {
        let n = plane_size(plane);
        match depth {
            8 => {
                if off + n > data.len() {
                    return Err(format!("input too short: need {} bytes, have {}", off + n, data.len()));
                }
                for (dst, &b) in plane.data.iter_mut().zip(&data[off..off + n]) {
                    *dst = b as u16;
                }
                off += n;
            }
            10 => {
                let bytes = n * 2;
                if off + bytes > data.len() {
                    return Err(format!("input too short: need {} bytes, have {}", off + bytes, data.len()));
                }
                for (dst, chunk) in plane.data.iter_mut().zip(data[off..off + bytes].chunks_exact(2)) {
                    *dst = u16::from_le_bytes([chunk[0], chunk[1]]);
                }
                off += bytes;
            }
            other => return Err(format!("unsupported bit depth {other}")),
        }
    }
    Ok(pic)
}

/// Streaming Y4M (yuv4mpeg) input reader.
///
/// Consumes the `YUV4MPEG2` header (parsing W/H/F/C) and then yields one
/// picture per `FRAME` marker. `C420p10` and `C420jpeg`/`C420p8` headers are
/// accepted; 10-bit input at encode depth 8 is right-shifted to 8 bits.
pub struct Y4mReader {
    inner: Box<dyn std::io::Read>,
    pub width: usize,
    pub height: usize,
    pub fps_num: u32,
    pub fps_den: u32,
    pub input_depth: u8,
    depth: u8,
    eof: bool,
}

impl Y4mReader {
    /// Read the Y4M header from the stream.
    pub fn new<R: std::io::Read + 'static>(inner: R, depth: u8) -> Result<Self, String> {
        let mut inner: Box<dyn std::io::Read> = Box::new(inner);
        let mut head = Vec::new();
        let mut buf = [0u8; 4096];
        // Read until the header line terminator.
        loop {
            let n = inner.read(&mut buf).map_err(|e| format!("y4m read error: {e}"))?;
            if n == 0 {
                return Err("y4m: unexpected EOF in header".into());
            }
            head.extend_from_slice(&buf[..n]);
            if head.len() > 64 * 1024 {
                return Err("y4m: header too long".into());
            }
            if let Some(pos) = head.iter().position(|&b| b == b'\n') {
                let line = String::from_utf8_lossy(&head[..pos]).to_string();
                let rest = head.split_off(pos + 1);
                inner = Box::new(std::io::Read::chain(std::io::Cursor::new(rest), inner));
                let mut width = 0usize;
                let mut height = 0usize;
                let mut fps_num = 0u32;
                let mut fps_den = 0u32;
                let mut chroma = 0u8; // 0 = 420 (8-bit), 1 = 420p10
                for tok in line.split_whitespace() {
                    if tok == "YUV4MPEG2" {
                        continue;
                    }
                    if let Some(v) = tok.strip_prefix('W') {
                        width = v.parse().map_err(|_| format!("y4m: bad W {v}"))?;
                    } else if let Some(v) = tok.strip_prefix('H') {
                        height = v.parse().map_err(|_| format!("y4m: bad H {v}"))?;
                    } else if let Some(v) = tok.strip_prefix('F') {
                        let (a, b) = v.split_once(':').ok_or("y4m: bad F")?;
                        fps_num = a.parse().map_err(|_| format!("y4m: bad F {v}"))?;
                        fps_den = b.parse().map_err(|_| format!("y4m: bad F {v}"))?;
                    } else if let Some(v) = tok.strip_prefix('C') {
                        match v {
                            "420jpeg" | "420p8" | "420" => chroma = 0,
                            "420p10" => chroma = 1,
                            other => return Err(format!("y4m: unsupported colorspace {other}")),
                        }
                    }
                }
                if width == 0 || height == 0 || fps_num == 0 {
                    return Err(format!("y4m: incomplete header: {line}"));
                }
                let input_depth = if chroma == 1 { 10 } else { 8 };
                if depth != 8 && depth != 10 {
                    return Err(format!("unsupported encode depth {depth}"));
                }
                let mut r = Y4mReader {
                    inner,
                    width,
                    height,
                    fps_num,
                    fps_den: if fps_den == 0 { 1 } else { fps_den },
                    input_depth,
                    depth,
                    eof: false,
                };
                // Skip any leading FRAME marker lines.
                r.skip_frame_marker()?;
                return Ok(r);
            }
        }
    }

    fn skip_frame_marker(&mut self) -> Result<(), String> {
        // Consume a leading "FRAME\n" marker (after the header, or between
        // frames) — but only if present at the current position.
        let mut buf = [0u8; 6];
        let n = fill(&mut *self.inner, &mut buf).map_err(|e| e.to_string())?;
        if n < 6 {
            // Not enough bytes for a marker; leave what we read in the
            // buffer for the frame read.
            let old = std::mem::replace(&mut self.inner, Box::new(std::io::empty()));
            self.inner = Box::new(std::io::Read::chain(std::io::Cursor::new(buf[..n].to_vec()), old));
            return Ok(());
        }
        if &buf == b"FRAME\n" {
            return Ok(());
        }
        let old = std::mem::replace(&mut self.inner, Box::new(std::io::empty()));
        self.inner = Box::new(std::io::Read::chain(std::io::Cursor::new(buf.to_vec()), old));
        Ok(())
    }

    /// Read the next frame, or `None` at EOF.
    pub fn next_frame(&mut self) -> Result<Option<Picture>, String> {
        if self.eof {
            return Ok(None);
        }
        self.skip_frame_marker()?;
        let depth = self.depth;
        let input_depth = self.input_depth;
        let w = self.width;
        let h = self.height;
        let pic = read_yuv420_frame_from_reader(&mut *self.inner, w, h, input_depth)?;
        let pic = match pic {
            Some(p) => p,
            None => {
                self.eof = true;
                return Ok(None);
            }
        };
        if depth == input_depth {
            Ok(Some(pic))
        } else {
            // Downconvert 10 -> 8 (no upconversion path).
            let mut out = Picture::new(w, h, ChromaFormat::C420);
            for (op, ip) in out.planes.iter_mut().zip(pic.planes.iter()) {
                for (o, &v) in op.data.iter_mut().zip(ip.data.iter()) {
                    *o = v >> 2;
                }
            }
            Ok(Some(out))
        }
    }
}

fn fill(reader: &mut dyn std::io::Read, buf: &mut [u8]) -> std::io::Result<usize> {
    let mut filled = 0;
    while filled < buf.len() {
        match reader.read(&mut buf[filled..]) {
            Ok(0) => break,
            Ok(k) => filled += k,
            Err(e) => return Err(e),
        }
    }
    Ok(filled)
}

fn read_yuv420_frame_from_reader(
    reader: &mut dyn std::io::Read,
    width: usize,
    height: usize,
    depth: u8,
) -> Result<Option<Picture>, String> {
    let n = width * height + 2 * ((width / 2) * (height / 2));
    let bytes = n * if depth == 10 { 2 } else { 1 };
    let mut buf = vec![0u8; bytes];
    let filled = fill(reader, &mut buf).map_err(|e| format!("read error: {e}"))?;
    if filled == 0 {
        return Ok(None);
    }
    if filled < bytes {
        return Err(format!("truncated frame: got {filled} bytes, need {bytes}"));
    }
    read_yuv420_frame(&buf, width, height, depth).map(Some)
}

/// Read a raw yuv420p frame from a file handle (legacy path).
pub fn read_yuv420_frame_file(
    reader: &mut dyn std::io::Read,
    width: usize,
    height: usize,
    depth: u8,
) -> Result<Option<Picture>, String> {
    read_yuv420_frame_from_reader(reader, width, height, depth)
}
