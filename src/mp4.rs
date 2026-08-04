//! Minimal MP4 muxer for the LCEVC dual-track file (VVC base + LCEVC
//! enhancement track with an `sbas` track reference), matching the layout
//! ffmpeg-master's mov demuxer expects for its LCEVC stream-group support.
//!
//! The VVC samples are stored Annex-B (start codes; the parameter sets are
//! in-band in every access unit) and both tracks carry an empty `vvcC`/`lvcC`
//! box, which makes the demuxer expose no extradata so the `lcevc_merge`
//! bitstream filter raw-copies the enhancement into `AV_PKT_DATA_LCEVC`.

use std::io::Write;

fn box_be(fourcc: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(payload.len() + 8);
    out.extend_from_slice(&((payload.len() + 8) as u32).to_be_bytes());
    out.extend_from_slice(fourcc);
    out.extend_from_slice(payload);
    out
}

fn u16_be(v: u16) -> [u8; 2] {
    v.to_be_bytes()
}

fn u32_be(v: u32) -> [u8; 4] {
    v.to_be_bytes()
}

/// Split an Annex-B bitstream into NAL units (3- or 4-byte start codes).
/// Returns (data ranges) as (start, len) pairs *including* the start code.
pub fn split_nals(data: &[u8]) -> Vec<(usize, usize)> {    let mut out = Vec::new();
    let mut i = 0;
    while i + 4 <= data.len() {
        let sc4 = data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 0 && data[i + 3] == 1;
        let sc3 = data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 && (i + 3 >= data.len() || data[i + 3] != 0);
        if sc4 || sc3 {
            let start = i;
            let h = if sc4 { 4 } else { 3 };
            i += h;
            let mut end = i;
            while end < data.len() {
                let e4 = end + 3 < data.len()
                    && data[end] == 0 && data[end + 1] == 0 && data[end + 2] == 0 && data[end + 3] == 1;
                let e3 = end + 2 < data.len()
                    && data[end] == 0 && data[end + 1] == 0 && data[end + 2] == 1
                    && (end + 3 >= data.len() || data[end + 3] != 0);
                if e4 || e3 {
                    break;
                }
                end += 1;
            }
            out.push((start, end - start));
            i = end;
        } else {
            i += 1;
        }
    }
    out
}

/// Split the VVC base bitstream into access units: every AUD NAL (type 20)
/// starts a new unit. Returns the byte ranges of each AU (start codes kept).
pub fn split_aus(base: &[u8]) -> Vec<(usize, usize)> {
    let nals = split_nals(base);
    let mut aus = Vec::new();
    let mut start: Option<usize> = None;
    for (pos, _len) in &nals {
        let h = if base[*pos + 2] == 1 { 3 } else { 4 };
        let b1 = base[pos + h + 1];
        let ntype = (b1 >> 3) & 0x1F;
        if ntype == 20 {
            if let Some(s) = start {
                aus.push((s, pos - s));
            }
            start = Some(*pos);
        }
    }
    if let Some(s) = start {
        aus.push((s, base.len() - s));
    }
    if aus.is_empty() && !nals.is_empty() {
        aus.push((0, base.len()));
    }
    aus
}

/// Split the LCEVC bitstream into per-picture NAL units.
pub fn split_lcevc_nals(data: &[u8]) -> Vec<(usize, usize)> {
    split_nals(data)
}

/// Strip the Annex-B start code (3 or 4 bytes) from a NAL unit.
pub fn strip_start_code(nal: &[u8]) -> &[u8] {
    if nal.len() >= 4 && nal[0] == 0 && nal[1] == 0 && nal[2] == 0 && nal[3] == 1 {
        &nal[4..]
    } else if nal.len() >= 3 && nal[0] == 0 && nal[1] == 0 && nal[2] == 1 {
        &nal[3..]
    } else {
        nal
    }
}

/// The 78-byte visual sample entry header (before the width/height).
fn visual_sample_entry_head() -> Vec<u8> {
    let mut v = Vec::new();
    v.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 1]); // reserved(6) + data_ref_index(1)
    v.extend_from_slice(&[0, 0, 0, 0]); // pre_defined + reserved
    v.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]); // pre_defined(12)
    v
}

fn visual_tail() -> Vec<u8> {
    let mut v = Vec::new();
    v.extend_from_slice(&[0x00, 0x48, 0x00, 0x00]); // horiz resolution 72dpi
    v.extend_from_slice(&[0x00, 0x48, 0x00, 0x00]); // vert resolution
    v.extend_from_slice(&[0, 0, 0, 0]); // reserved
    v.extend_from_slice(&u16_be(1)); // frame_count
    v.extend_from_slice(&[0u8; 32]); // compressor name
    v.extend_from_slice(&[0x00, 0x18]); // depth 24
    v.extend_from_slice(&[0xff, 0xff]); // pre_defined
    v
}

fn btrt_box(buffer: u32, avg: u32, max: u32) -> Vec<u8> {
    let mut p = Vec::new();
    p.extend_from_slice(&u32_be(buffer));
    p.extend_from_slice(&u32_be(avg));
    p.extend_from_slice(&u32_be(max));
    box_be(b"btrt", &p)
}

fn mvhd_box(duration_ticks: u32, next_track_id: u32) -> Vec<u8> {
    let mut p = Vec::new();
    p.extend_from_slice(&[0, 0, 0, 0]); // version + flags
    p.extend_from_slice(&[0; 8]); // creation + modification
    p.extend_from_slice(&u32_be(1000)); // timescale
    p.extend_from_slice(&u32_be(duration_ticks));
    p.extend_from_slice(&u32_be(0x00010000)); // rate 1.0
    p.extend_from_slice(&u16_be(0x0100)); // volume
    p.extend_from_slice(&[0, 0]); // reserved
    p.extend_from_slice(&[0; 8]); // reserved
    p.extend_from_slice(&u32_be(0x00010000));
    p.extend_from_slice(&[0; 8]);
    p.extend_from_slice(&u32_be(0x00010000));
    p.extend_from_slice(&[0; 12]);
    p.extend_from_slice(&u32_be(0x40000000)); // matrix
    p.extend_from_slice(&[0; 4]); // matrix (9th entry)
    p.extend_from_slice(&[0; 24]); // pre_defined
    p.extend_from_slice(&u32_be(next_track_id));
    box_be(b"mvhd", &p)
}

fn tkhd_box(track_id: u32, width: u16, height: u16) -> Vec<u8> {
    let mut p = Vec::new();
    p.extend_from_slice(&[0, 0, 0, 3]); // version 0, flags enabled|in-movie
    p.extend_from_slice(&[0; 8]);
    p.extend_from_slice(&u32_be(track_id));
    p.extend_from_slice(&[0; 4]);
    p.extend_from_slice(&u32_be(1)); // duration (ticks, informational)
    p.extend_from_slice(&[0; 8]);
    p.extend_from_slice(&[0, 0, 0, 0]); // layer + alternate group
    p.extend_from_slice(&[0, 0, 0, 0]); // volume + reserved
    p.extend_from_slice(&u32_be(0x00010000));
    p.extend_from_slice(&[0; 8]);
    p.extend_from_slice(&u32_be(0x00010000));
    p.extend_from_slice(&[0; 12]);
    p.extend_from_slice(&u32_be(0x40000000)); // matrix
    p.extend_from_slice(&[0; 4]); // matrix (9th entry)
    p.extend_from_slice(&u32_be((width as u32) << 16)); // 16.16 fixed point
    p.extend_from_slice(&u32_be((height as u32) << 16));
    box_be(b"tkhd", &p)
}

fn elst_box(segment_duration: u32) -> Vec<u8> {
    let mut p = Vec::new();
    p.extend_from_slice(&[0, 0, 0, 0]); // version + flags
    p.extend_from_slice(&u32_be(1)); // entry count
    p.extend_from_slice(&u32_be(segment_duration)); // segment duration (movie timescale)
    p.extend_from_slice(&u32_be(0)); // media time
    p.extend_from_slice(&u32_be(0x00010000)); // media rate
    box_be(b"elst", &p)
}

fn mdhd_box(timescale: u32, duration_ticks: u32) -> Vec<u8> {
    let mut p = Vec::new();
    p.extend_from_slice(&[0, 0, 0, 0]);
    p.extend_from_slice(&[0; 8]);
    p.extend_from_slice(&u32_be(timescale));
    p.extend_from_slice(&u32_be(duration_ticks));
    p.extend_from_slice(&[0xc4, 0x00, 0, 0]); // language + pre_defined
    box_be(b"mdhd", &p)
}

fn hdlr_box(handler: &[u8; 4], name: &str) -> Vec<u8> {
    let mut p = Vec::new();
    p.extend_from_slice(&[0, 0, 0, 0]); // version + flags
    p.extend_from_slice(&[0; 4]); // pre_defined
    p.extend_from_slice(handler);
    p.extend_from_slice(&[0; 8]); // reserved
    let mut nameb = name.as_bytes().to_vec();
    nameb.push(0);
    while nameb.len() < 17 {
        nameb.push(0);
    }
    p.extend_from_slice(&nameb);
    box_be(b"hdlr", &p)
}

fn vmhd_box() -> Vec<u8> {
    let mut p = Vec::new();
    p.extend_from_slice(&[0, 0, 0, 1]); // version + flags
    p.extend_from_slice(&[0; 8]); // graphics mode + opcolor
    box_be(b"vmhd", &p)
}

fn dref_box() -> Vec<u8> {
    let mut dref = Vec::new();
    dref.extend_from_slice(&[0, 0, 0, 0]);
    dref.extend_from_slice(&u32_be(1));
    let url = [0u8, 0, 0, 1];
    dref.extend_from_slice(&u32_be(12));
    dref.extend_from_slice(b"url ");
    dref.extend_from_slice(&url);
    let dref = box_be(b"dref", &dref);
    box_be(b"dinf", &dref)
}

fn colr_box(colour: &crate::config::ColourInfo) -> Vec<u8> {
    let mut p = Vec::new();
    p.extend_from_slice(b"nclx");
    p.extend_from_slice(&u16_be(colour.primaries));
    p.extend_from_slice(&u16_be(colour.transfer));
    p.extend_from_slice(&u16_be(colour.matrix));
    p.push(if colour.full_range { 0x80 } else { 0 });
    box_be(b"colr", &p)
}

fn stsd_box_vvc(width: u16, height: u16, colour: Option<&crate::config::ColourInfo>) -> Vec<u8> {
    let mut entry = Vec::new();
    entry.extend_from_slice(b"vvc1");
    entry.extend_from_slice(&visual_sample_entry_head());
    entry.extend_from_slice(&u16_be(width));
    entry.extend_from_slice(&u16_be(height));
    entry.extend_from_slice(&visual_tail());
    // empty vvcC (version 0 + flags): 12-byte box
    let mut vvcc = Vec::new();
    vvcc.extend_from_slice(&u32_be(12));
    vvcc.extend_from_slice(b"vvcC");
    vvcc.extend_from_slice(&[0, 0, 0, 0]);
    entry.extend_from_slice(&vvcc);
    if let Some(c) = colour {
        entry.extend_from_slice(&colr_box(c));
    }
    entry.extend_from_slice(&btrt_box(15, 0x6992aa, 0x6992aa));
    let mut stsd = Vec::new();
    stsd.extend_from_slice(&[0, 0, 0, 0]);
    stsd.extend_from_slice(&u32_be(1));
    stsd.extend_from_slice(&u32_be((entry.len() + 4) as u32));
    stsd.extend_from_slice(&entry);
    box_be(b"stsd", &stsd)
}

fn stsd_box_lcevc(width: u16, height: u16) -> Vec<u8> {
    let mut entry = Vec::new();
    entry.extend_from_slice(b"lvc1");
    entry.extend_from_slice(&visual_sample_entry_head());
    entry.extend_from_slice(&u16_be(width));
    entry.extend_from_slice(&u16_be(height));
    entry.extend_from_slice(&visual_tail());
    // Proper lvcC (version 1 + lengthSizeMinusOne=3, i.e. 4-byte lengths):
    // [version][profile:3][lengthSizeMinusOne:4][reserved:9][nb_arrays:1] = 14 bytes.
    // The ffmpeg merge bsf derives the sample NAL length size from byte 4.
    let mut lvcc = Vec::new();
    lvcc.extend_from_slice(&u32_be(14));
    lvcc.extend_from_slice(b"lvcC");
    lvcc.extend_from_slice(&[1, 0, 0, 0, 0xC0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    entry.extend_from_slice(&lvcc);
    entry.extend_from_slice(&btrt_box(0, 0x1922c555, 0x1922c555));
    let mut stsd = Vec::new();
    stsd.extend_from_slice(&[0, 0, 0, 0]);
    stsd.extend_from_slice(&u32_be(1));
    stsd.extend_from_slice(&u32_be((entry.len() + 4) as u32));
    stsd.extend_from_slice(&entry);
    box_be(b"stsd", &stsd)
}

fn stts_box(n: u32, delta: u32) -> Vec<u8> {
    let mut p = Vec::new();
    p.extend_from_slice(&[0, 0, 0, 0]);
    p.extend_from_slice(&u32_be(1)); // entry count
    p.extend_from_slice(&u32_be(n)); // sample count
    p.extend_from_slice(&u32_be(delta)); // frame duration in ticks
    box_be(b"stts", &p)
}

fn stsc_box() -> Vec<u8> {
    let mut p = Vec::new();
    p.extend_from_slice(&[0, 0, 0, 0]);
    p.extend_from_slice(&u32_be(1)); // entry count
    p.extend_from_slice(&u32_be(1)); // first chunk
    p.extend_from_slice(&u32_be(1)); // samples per chunk
    p.extend_from_slice(&u32_be(1)); // sample description index
    box_be(b"stsc", &p)
}

fn stsz_box(sizes: &[u32]) -> Vec<u8> {
    let mut p = Vec::new();
    p.extend_from_slice(&[0, 0, 0, 0]);
    p.extend_from_slice(&u32_be(0)); // sample size (0 = per-entry)
    p.extend_from_slice(&u32_be(sizes.len() as u32));
    for s in sizes {
        p.extend_from_slice(&u32_be(*s));
    }
    box_be(b"stsz", &p)
}

fn stco_box(offsets: &[u32]) -> Vec<u8> {
    let mut p = Vec::new();
    p.extend_from_slice(&[0, 0, 0, 0]);
    p.extend_from_slice(&u32_be(offsets.len() as u32));
    for o in offsets {
        p.extend_from_slice(&u32_be(*o));
    }
    box_be(b"stco", &p)
}

fn stbl_box(stsd: Vec<u8>, n: u32, delta: u32, sizes: &[u32], offsets: &[u32]) -> Vec<u8> {
    let mut p = Vec::new();
    p.extend_from_slice(&stsd);
    p.extend_from_slice(&stts_box(n, delta));
    p.extend_from_slice(&stsc_box());
    p.extend_from_slice(&stsz_box(sizes));
    p.extend_from_slice(&stco_box(offsets));
    box_be(b"stbl", &p)
}

fn minf_box(stsd: Vec<u8>, n: u32, delta: u32, sizes: &[u32], offsets: &[u32]) -> Vec<u8> {
    let mut p = Vec::new();
    p.extend_from_slice(&vmhd_box());
    p.extend_from_slice(&dref_box());
    p.extend_from_slice(&stbl_box(stsd, n, delta, sizes, offsets));
    box_be(b"minf", &p)
}

fn mdia_box(handler: &[u8; 4], name: &str, timescale: u32, duration: u32, delta: u32,
            stsd: Vec<u8>, n: u32, sizes: &[u32], offsets: &[u32]) -> Vec<u8> {
    let mut p = Vec::new();
    p.extend_from_slice(&mdhd_box(timescale, duration));
    p.extend_from_slice(&hdlr_box(handler, name));
    p.extend_from_slice(&minf_box(stsd, n, delta, sizes, offsets));
    box_be(b"mdia", &p)
}

/// Build the full MP4 file.
///
/// * `base_aus` — per-frame VVC access units (Annex-B, start codes kept).
/// * `enh_nals` — per-frame LCEVC NAL units (start codes kept).
/// * `width`/`height` — the base (LOQ2) picture dimensions.
/// * `fps` — frame rate (used for the media timescale).
pub fn mux_mp4(
    path: &str,
    base_aus: &[Vec<u8>],
    enh_nals: &[Vec<u8>],
    width: u16,
    height: u16,
    fps: u32,
    colour: Option<&crate::config::ColourInfo>,
) -> Result<(), String> {
    let n = base_aus.len().min(enh_nals.len());
    if n == 0 {
        return Err("no frames to mux".into());
    }
    let timescale = 12800u32;
    let delta = (timescale / fps.max(1)).max(1);
    let duration_ticks = (n as u32) * delta;
    let movie_duration = ((n as u32) * 1000 / fps.max(1)).max(1);

    // ---- ftyp + free ----
    let mut out = Vec::new();
    {
        let mut ftyp = Vec::new();
        ftyp.extend_from_slice(b"isom");
        ftyp.extend_from_slice(&u32_be(0x200));
        ftyp.extend_from_slice(b"isomiso2mp41");
        out.extend_from_slice(&box_be(b"ftyp", &ftyp));
    }
    out.extend_from_slice(&u32_be(8));
    out.extend_from_slice(b"free");

    // ---- mdat: interleaved (base[i], enh[i]) ----
    let mdat_start = out.len() as u32;
    let mut base_off = Vec::with_capacity(n);
    let mut enh_off = Vec::with_capacity(n);
    let mut base_sizes = Vec::with_capacity(n);
    let mut enh_sizes = Vec::with_capacity(n);
    {
        out.extend_from_slice(&u32_be(0)); // mdat size placeholder
        out.extend_from_slice(b"mdat");
        for i in 0..n {
            base_off.push(out.len() as u32 - mdat_start);
            out.extend_from_slice(&base_aus[i]);
            base_sizes.push(base_aus[i].len() as u32);

            enh_off.push(out.len() as u32 - mdat_start);
            // LCEVC samples are stored length-prefixed (4-byte BE length,
            // no start code) so the merge bsf can rebuild Annex-B NALs.
            let body = strip_start_code(&enh_nals[i]);
            out.extend_from_slice(&u32_be(body.len() as u32));
            out.extend_from_slice(&body);
            enh_sizes.push(4 + body.len() as u32);
        }
        let mdat_size = (out.len() as u32 - mdat_start) as u32;
        out[mdat_start as usize..mdat_start as usize + 4].copy_from_slice(&u32_be(mdat_size));
    }
    let mdat_end = out.len() as u32;
    let _ = mdat_end;

    // ---- moov ----
    // stco chunk offsets must be absolute file offsets: mdat starts at
    // mdat_start, so add it to the per-sample relative offsets.
    let base_off_abs: Vec<u32> = base_off.iter().map(|o| mdat_start + o).collect();
    let enh_off_abs: Vec<u32> = enh_off.iter().map(|o| mdat_start + o).collect();

    let mut vvc_stsd = stsd_box_vvc(width, height, colour);
    let mut lcevc_stsd = stsd_box_lcevc(width, height);

    let mut trak_vvc = Vec::new();
    {
        trak_vvc.extend_from_slice(&tkhd_box(1, width, height));
        let mut edts = Vec::new();
        edts.extend_from_slice(&elst_box(movie_duration));
        trak_vvc.extend_from_slice(&box_be(b"edts", &edts));
        trak_vvc.extend_from_slice(&mdia_box(
            b"vide", "VideoHandler", timescale, duration_ticks, delta,
            std::mem::take(&mut vvc_stsd), n as u32, &base_sizes, &base_off_abs,
        ));
    }
    let mut trak_lcevc = Vec::new();
    {
        trak_lcevc.extend_from_slice(&tkhd_box(2, width, height));
        let mut edts = Vec::new();
        edts.extend_from_slice(&elst_box(movie_duration));
        trak_lcevc.extend_from_slice(&box_be(b"edts", &edts));
        let mut tref = Vec::new();
        tref.extend_from_slice(&u32_be(12));
        tref.extend_from_slice(b"sbas");
        tref.extend_from_slice(&u32_be(1));
        trak_lcevc.extend_from_slice(&box_be(b"tref", &tref));
        trak_lcevc.extend_from_slice(&mdia_box(
            b"vide", "VideoHandler", timescale, duration_ticks, delta,
            std::mem::take(&mut lcevc_stsd), n as u32, &enh_sizes, &enh_off_abs,
        ));
    }
    let mut moov = Vec::new();
    moov.extend_from_slice(&mvhd_box(movie_duration, 3));
    moov.extend_from_slice(&box_be(b"trak", &trak_vvc));
    moov.extend_from_slice(&box_be(b"trak", &trak_lcevc));
    let moov_bytes = box_be(b"moov", &moov);
    out.extend_from_slice(&moov_bytes);

    let mut f = std::fs::File::create(path).map_err(|e| e.to_string())?;
    f.write_all(&out).map_err(|e| e.to_string())?;
    Ok(())
}

