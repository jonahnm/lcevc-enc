//! lcevc_enc — LCEVC (ISO/IEC 23094-2) enhancement encoder with a VVC base.
//!
//! Usage:
//!   lcevc_enc -i input.yuv -s WxH -f FPS --frames N -o out.lcevc [options]
//!
//! Options include:
//!   --base-mode raw|vvc          base codec (raw = lossless base for tests)
//!   --base-out out.266           write the VVC base bitstream
//!   --step-width-l1 N            L1 quantizer step width (s8.7 units)
//!   --step-width-l2 N            L2 quantizer step width
//!   --transform 2x2|4x4          transform type (default 4x4)
//!   --upsampler nearest|bilinear|cubic|modified-cubic
//!   --scaling-l1 0|1|2           base -> L1 scaling (default 2)
//!   --scaling-l2 0|1|2           L1 -> L2 scaling (default 2)
//!   --predicted-average on|off   predicted average during upscaling
//!   --temporal on|off            temporal prediction (default off)
//!   --tiles none|512x256|1024x512|WxH
//!   --tile-size-compression 0|1|2
//!   --verify                     decode the stream with the built-in mirror
//!                                decoder and compare with the encoder output

use lcevc_enc::base::{self, BaseMode};
use lcevc_enc::config::{ChromaFormat, LcevcConfig, ScalingMode, TileDimensions, TransformType, UpsampleType};
use lcevc_enc::encoder::Encoder;
use lcevc_enc::nal::build_nal_unit;
use lcevc_enc::yuv;
use std::io::Write;

fn parse_size(s: &str) -> Result<(u32, u32), String> {
    let (w, h) = s.split_once('x').ok_or("expected WxH")?;
    Ok((w.parse().map_err(|_| "bad width")?, h.parse().map_err(|_| "bad height")?))
}

/// Process one encoded frame: assemble the NAL, stats, dumps, status.
#[allow(clippy::too_many_arguments)]
fn process_encoded_frame(
    encoded: &lcevc_enc::encoder::EncodedFrame,
    frame: &lcevc_enc::frame::Picture,
    frame_start: std::time::Instant,
    cfg: &lcevc_enc::config::LcevcConfig,
    bit_depth: u8,
    frames: u32,
    run_start: std::time::Instant,
    no_psnr: bool,
    frame_count: &mut u32,
    total_bytes: &mut usize,
    total_sse: &mut u64,
    total_samples: &mut u64,
    out_file: &mut std::fs::File,
    base_dump: &mut Option<std::fs::File>,
    recon_dump: &mut Option<std::fs::File>,
) -> Result<Vec<u8>, String> {
        let mut payloads = Vec::new();
        if encoded.idr {
            payloads.push((lcevc_enc::nal::BLOCK_SEQUENCE_CONFIG, cfg.write_sequence_config()));
            payloads.push((lcevc_enc::nal::BLOCK_GLOBAL_CONFIG, cfg.write_global_config()));
        }
        payloads.push((lcevc_enc::nal::BLOCK_PICTURE_CONFIG, encoded.picture_config.clone()));
        payloads.push((if cfg.is_tiled() { lcevc_enc::nal::BLOCK_ENCODED_DATA_TILED } else { lcevc_enc::nal::BLOCK_ENCODED_DATA }, encoded.encoded_data.clone()));

        let nal = build_nal_unit(encoded.idr, &payloads);
        let nal_bytes = nal.len();
        *total_bytes += nal_bytes;

        if let Some(f) = base_dump.as_mut() {
            lcevc_enc::base::write_yuv420(&encoded.base_picture, bit_depth, f)
                .map_err(|e| e.to_string())?;
        }
        if let Some(f) = recon_dump.as_mut() {
            lcevc_enc::base::write_yuv420(&encoded.output, bit_depth, f)
                .map_err(|e| e.to_string())?;
        }

        let psnr = if no_psnr {
            f64::NAN
        } else {
            let mut frame_sse = 0u64;
            let mut frame_samples = 0u64;
            for p in 0..frame.planes.len() {
                for (a, b) in frame.planes[p].data.iter().zip(encoded.output.planes[p].data.iter()) {
                    let d = *a as i64 - *b as i64;
                    frame_sse += (d * d) as u64;
                    frame_samples += 1;
                }
            }
            *total_sse += frame_sse;
            *total_samples += frame_samples;
            if frame_sse == 0 {
                f64::INFINITY
            } else {
                let max_val = ((1u64 << bit_depth) - 1) as f64;
                10.0 * (max_val * max_val * frame_samples as f64 / frame_sse as f64).log10()
            }
        };

        let now = std::time::Instant::now();
        let frame_secs = now.duration_since(frame_start).as_secs_f64();
        let fps = if frame_secs > 0.0 { 1.0 / frame_secs } else { 0.0 };
        let total_frames = *frame_count + 1;
        let elapsed = now.duration_since(run_start).as_secs_f64();
        let avg_fps = if elapsed > 0.0 { total_frames as f64 / elapsed } else { 0.0 };
        let eta = if frames != u32::MAX && total_frames < frames && avg_fps > 0.0 {
            format!(", ETA {:.0}s", (frames - total_frames) as f64 / avg_fps)
        } else {
            String::new()
        };
        let frames_total = if frames == u32::MAX { "?".to_string() } else { frames.to_string() };
        if psnr.is_nan() {
            eprintln!(
                "frame {total_frames}/{}: {frame_secs:6.2} s ({fps:5.2} fps, avg {avg_fps:5.2}){eta}, \
                 enhancement {nal_bytes:>6} bytes",
                frames_total,
            );
        } else {
            eprintln!(
                "frame {total_frames}/{}: {frame_secs:6.2} s ({fps:5.2} fps, avg {avg_fps:5.2}){eta}, \
                 enhancement {nal_bytes:>6} bytes, PSNR {psnr:.2} dB",
                frames_total,
            );
        }
        *frame_count += 1;
        Ok(nal)
}

/// Encode one GOP's enhancement against its decoded base.
fn process_gop(
    gop_frames: &[lcevc_enc::frame::Picture],
    decoded: &[lcevc_enc::frame::Picture],
    target_bytes: Option<usize>,
    fps: u32,
    encoder: &mut lcevc_enc::encoder::Encoder,
    cfg: &lcevc_enc::config::LcevcConfig,
    bit_depth: u8,
    frames: u32,
    run_start: std::time::Instant,
    no_psnr: bool,
    frame_count: &mut u32,
    total_bytes: &mut usize,
    total_sse: &mut u64,
    total_samples: &mut u64,
    out_file: &mut std::fs::File,
    base_dump: &mut Option<std::fs::File>,
    recon_dump: &mut Option<std::fs::File>,
) -> Result<(), String> {
    for (f, base) in gop_frames.iter().zip(decoded.iter()) {
        let frame_start = std::time::Instant::now();
        let encoded = match target_bytes {
            Some(tb) => {
                let (ef, _sw1, _sw2) = encoder.encode_frame_rc(f, base, tb)?;
                ef
            }
            None => encoder.encode_frame_with_base(f, base)?,
        };
        let _nal = process_encoded_frame(
            &encoded, f, frame_start, cfg, bit_depth, frames, run_start, no_psnr,
            frame_count, total_bytes, total_sse, total_samples, out_file, base_dump,
            recon_dump,
        )?;
    }
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if let Err(e) = run(&args) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn usage() -> String {
    "usage: lcevc_enc -i input.yuv -s WxH -f fps [-o out.lcevc] [options]\n\
     \n\
     options:\n\
       --base-mode raw|vvc          base codec (default raw)\n\
       --base-out FILE              write the VVC base bitstream\n\
       --frames N                   number of frames to encode (default all)\n\
       --step-width-l1 N            default 512\n\
       --step-width-l2 N            default 128\n\
       --transform 2x2|4x4          default 4x4\n\
       --upsampler nearest|bilinear|cubic|modified-cubic   default bilinear\n\
       --scaling-l1 0|1|2           default 2\n\
       --scaling-l2 0|1|2           default 2\n\
       --predicted-average on|off   default on\n\
       --temporal on|off            default off\n\
       --tiles none|512x256|1024x512|WxH   default none\n\
       --tile-size-compression 0|1|2       default 0\n\
       --bit-depth 8|10             sample depth of the enhanced output (default 8)\n\
       --input-format raw|y4m       input format; y4m reads the header from the\n\
                                    stream (default: y4m when the input starts\n\
                                    with the YUV4MPEG2 magic, else raw)\n\
       --vvc-preset fast|medium|slow  vvenc preset for the VVC base (default medium)\n\
       --vvc-qp N                   vvenc QP for the VVC base (default 30)\n\
       --base-gop N                 base frames per vvenc invocation (inter\n\
                                    prediction; default 30, 1 = per-frame IDR)\n\
       --target-kbps N             target enhancement bitrate (rate control;\n\
                                    the step widths are chosen per frame to hit it)\n\
       -i -                        read from stdin (raw or y4m)\n\
       --mux FILE.mp4               also mux the VVC base + enhancement into\n\
                                    an MP4 with an LCEVC stream group\n\
                                    (requires --base-mode vvc and --base-out)\n\
       --verify                     self-check with the built-in mirror decoder\n"
        .to_string()
}

fn run(args: &[String]) -> Result<(), String> {
    let mut input = None;
    let mut output = None;
    let mut base_out = None;
    let mut width = 0u32;
    let mut height = 0u32;
    let mut fps = 30u32;
    let mut frames = u32::MAX;
    let mut base_mode = "raw".to_string();
    let mut sw_l1 = 512u32;
    let mut sw_l2 = 256u32;
    let mut transform = TransformType::Dds;
    let mut upsampler = UpsampleType::Linear;
    let mut scaling_l1 = ScalingMode::Scale2D;
    let mut scaling_l2 = ScalingMode::Scale2D;
    let mut predicted_average = true;
    let mut temporal = false;
    let mut temporal_sw_modifier: u8 = 48;
    let mut tiles = TileDimensions::None;
    let mut custom_tile = None;
    let mut tile_size_compression = 0u8;
    let mut verify = false;
    let mut no_psnr = false;
    let mut no_rdoq = false;
    let mut colour: Option<lcevc_enc::config::ColourInfo> = None;
    let mut dump_base = false;
    let mut dump_recon = false;
    let mut bit_depth = 8u8;
    let mut input_format: Option<String> = None;
    let mut mux_out: Option<String> = None;
    let mut mux_only = false;
    let mut base_gop = 30usize;
    let mut base_gop_seconds: Option<f64> = None;
    let mut target_kbps: Option<u32> = None;
    let mut total_kbps: Option<u32> = None;
    let mut qm_beta: f64 = 0.3;
    let mut vvc_preset = "medium".to_string();
    let mut vvc_qp = "30".to_string();

    let mut i = 1;
    while i < args.len() {
        let a = &args[i];
        let next = |i: &mut usize| -> Result<String, String> {
            *i += 1;
            args.get(*i).cloned().ok_or_else(|| format!("missing value for {a}"))
        };
        match a.as_str() {
            "-h" | "--help" => {
                print!("{}", usage());
                return Ok(());
            }
            "-i" | "--input" => input = Some(next(&mut i)?),
            "-o" | "--out" => output = Some(next(&mut i)?),
            "--base-out" => base_out = Some(next(&mut i)?),
            "-s" | "--size" => {
                let (w, h) = parse_size(&next(&mut i)?)?;
                width = w;
                height = h;
            }
            "-f" | "--fps" => fps = next(&mut i)?.parse().map_err(|_| "bad fps")?,
            "--frames" => frames = next(&mut i)?.parse().map_err(|_| "bad frames")?,
            "--base-mode" => base_mode = next(&mut i)?,
            "--step-width-l1" => sw_l1 = next(&mut i)?.parse().map_err(|_| "bad step width")?,
            "--step-width-l2" => sw_l2 = next(&mut i)?.parse().map_err(|_| "bad step width")?,
            "--transform" => {
                transform = match next(&mut i)?.as_str() {
                    "2x2" => TransformType::Dd,
                    "4x4" => TransformType::Dds,
                    _ => return Err("bad transform".into()),
                }
            }
            "--upsampler" => {
                upsampler = match next(&mut i)?.as_str() {
                    "nearest" => UpsampleType::Nearest,
                    "bilinear" | "linear" => UpsampleType::Linear,
                    "cubic" => UpsampleType::Cubic,
                    "modified-cubic" => UpsampleType::ModifiedCubic,
                    "adaptive" => {
                        let taps = next(&mut i)?;
                        let v: Vec<i16> = taps
                            .split(',')
                            .map(|t| t.parse().map_err(|_| "bad kernel tap"))
                            .collect::<Result<_, _>>()?;
                        if v.len() != 4 {
                            return Err("adaptive kernel needs 4 taps".into());
                        }
                        UpsampleType::Adaptive { taps: [v[0], v[1], v[2], v[3]] }
                    }
                    _ => return Err("bad upsampler".into()),
                }
            }
            "--scaling-l1" => scaling_l1 = parse_scaling(&next(&mut i)?)?,
            "--scaling-l2" => scaling_l2 = parse_scaling(&next(&mut i)?)?,
            "--predicted-average" => predicted_average = parse_on_off(&next(&mut i)?)?,
            "--temporal" => temporal = parse_on_off(&next(&mut i)?)?,
            "--temporal-sw-modifier" => temporal_sw_modifier = next(&mut i)?.parse().map_err(|_| "bad modifier")?,
            "--tiles" => {
                let v = next(&mut i)?;
                (tiles, custom_tile) = match v.as_str() {
                    "none" => (TileDimensions::None, None),
                    "512x256" => (TileDimensions::T512x256, None),
                    "1024x512" => (TileDimensions::T1024x512, None),
                    _ => {
                        let (w, h) = parse_size(&v)?;
                        (TileDimensions::Custom, Some((w as u16, h as u16)))
                    }
                };
            }
            "--tile-size-compression" => {
                tile_size_compression = next(&mut i)?.parse().map_err(|_| "bad compression")?
            }
            "--bit-depth" => {
                bit_depth = next(&mut i)?.parse().map_err(|_| "bad bit depth")?;
                if bit_depth != 8 && bit_depth != 10 {
                    return Err("bit depth must be 8 or 10".into());
                }
            }
            "--input-format" => input_format = Some(next(&mut i)?),
            "--mux" => mux_out = Some(next(&mut i)?),
            "--mux-only" => mux_only = true,
            "--base-gop" => base_gop = next(&mut i)?.parse().map_err(|_| "bad base-gop")?,
            "--base-gop-seconds" => base_gop_seconds = Some(next(&mut i)?.parse().map_err(|_| "bad base-gop-seconds")?),
            "--target-kbps" => target_kbps = Some(next(&mut i)?.parse().map_err(|_| "bad target-kbps")?),
            "--total-kbps" => total_kbps = Some(next(&mut i)?.parse().map_err(|_| "bad total-kbps")?),
            "--qm-beta" => qm_beta = next(&mut i)?.parse().map_err(|_| "bad qm-beta")?,
            "--vvc-preset" => vvc_preset = next(&mut i)?,
            "--vvc-qp" => vvc_qp = next(&mut i)?,
            "--verify" => verify = true,
            "--no-psnr" => no_psnr = true,
            "--no-rdoq" => no_rdoq = true,
            "--color" => {
                let spec = next(&mut i)?;
                let parts: Vec<&str> = spec.split(':').collect();
                if parts.len() < 3 {
                    return Err("--color expects CP:TC:MC (e.g. bt2020:smpte2084:bt2020nc)".into());
                }
                let (primaries, pn) = colour_primaries(parts[0]);
                let (transfer, tn) = colour_transfer(parts[1]);
                let (matrix, mn) = colour_matrix(parts[2]);
                let full = parts.get(3).map(|r| *r == "pc").unwrap_or(false);
                colour = Some(lcevc_enc::config::ColourInfo {
                    primaries_name: pn.to_string(),
                    transfer_name: tn.to_string(),
                    matrix_name: mn.to_string(),
                    primaries,
                    transfer,
                    matrix,
                    full_range: full,
                });
            }
            "--dump-base" => dump_base = true,
            "--dump-recon" => dump_recon = true,
            _ => return Err(format!("unknown option {a}\n{}", usage())),
        }
        i += 1;
    }

    let input = input.ok_or_else(|| format!("missing input\n{}", usage()))?;

    if mux_only {
        let mux_path = mux_out.as_ref().ok_or("--mux-only requires --mux FILE")?;
        let base_path = base_out.as_ref().ok_or("--mux-only requires --base-out")?;
        if width == 0 || height == 0 {
            return Err(format!("missing size\n{}", usage()));
        }
        let cfg = LcevcConfig {
            width: width as u16,
            height: height as u16,
            base_depth: bit_depth,
            enhancement_depth: bit_depth,
            colour: colour.clone(),
            scaling_l1,
            scaling_l2,
            ..Default::default()
        };
        cfg.validate()?;
        let base_data = std::fs::read(base_path).map_err(|e| e.to_string())?;
        let lcevc_out = output.clone().unwrap_or_else(|| "out.lcevc".to_string());
        let lcevc_data = std::fs::read(&lcevc_out).map_err(|e| e.to_string())?;
        let aus = lcevc_enc::mp4::split_aus(&base_data);
        let nals = lcevc_enc::mp4::split_lcevc_nals(&lcevc_data);
        let mut base_aus: Vec<Vec<u8>> = Vec::new();
        for (s, l) in &aus {
            base_aus.push(base_data[*s..*s + *l].to_vec());
        }
        let mut enh_nals: Vec<Vec<u8>> = Vec::new();
        for (s, l) in &nals {
            enh_nals.push(lcevc_data[*s..*s + *l].to_vec());
        }
        let (bw, bh) = {
            let d = cfg.loq_dimensions();
            (d[2].0, d[2].1)
        };
        lcevc_enc::mp4::mux_mp4(mux_path, &base_aus, &enh_nals, bw, bh, fps, colour.as_ref(), None)?;
        eprintln!("muxed {mux_path} ({} frames, base {bw}x{bh})", base_aus.len());
        return Ok(());
    }

    // Frame source: raw YUV file/stdin or streaming Y4M. Opened before the
    // config so that Y4M can supply the size/frame-rate from its header.
    enum FrameSource {
        Raw(Box<dyn std::io::Read>),
        Y4m(lcevc_enc::yuv::Y4mReader),
    }
    let mut frame_source = if input == "-" {
        let stdin = std::io::stdin();
        if input_format.as_deref() == Some("y4m") {
            let r: Box<dyn std::io::Read> = Box::new(stdin.lock());
            FrameSource::Y4m(lcevc_enc::yuv::Y4mReader::new(r, bit_depth)?)
        } else {
            let mut probe = [0u8; 9];
            let mut locked = stdin.lock();
            let mut filled = 0usize;
            while filled < 9 {
                match std::io::Read::read(&mut locked, &mut probe[filled..]) {
                    Ok(0) => break,
                    Ok(k) => filled += k,
                    Err(e) => return Err(format!("read error: {e}")),
                }
            }
            if filled >= 9 && &probe[..9] == b"YUV4MPEG2" {
                let chain: Box<dyn std::io::Read> =
                    Box::new(std::io::Read::chain(std::io::Cursor::new(probe.to_vec()), locked));
                FrameSource::Y4m(lcevc_enc::yuv::Y4mReader::new(chain, bit_depth)?)
            } else {
                let chain: Box<dyn std::io::Read> =
                    Box::new(std::io::Read::chain(std::io::Cursor::new(probe[..filled].to_vec()), locked));
                FrameSource::Raw(chain)
            }
        }
    } else {
        let file = std::fs::File::open(&input).map_err(|e| e.to_string())?;
        match input_format.as_deref() {
            Some("y4m") => {
                let r: Box<dyn std::io::Read> = Box::new(file);
                FrameSource::Y4m(lcevc_enc::yuv::Y4mReader::new(r, bit_depth)?)
            }
            Some("raw") | None => {
                let r: Box<dyn std::io::Read> = Box::new(file);
                FrameSource::Raw(r)
            }
            Some(other) => return Err(format!("unknown input format {other}")),
        }
    };
    if let FrameSource::Y4m(r) = &frame_source {
        width = r.width as u32;
        height = r.height as u32;
        if r.fps_num != 0 {
            fps = (r.fps_num as f64 / r.fps_den as f64).round() as u32;
        }
    }
    if width == 0 || height == 0 {
        return Err(format!("missing size\n{}", usage()));
    }
    // --base-gop-seconds: GOP size in seconds (resolved against the stream
    // frame rate once the Y4M header is known).
    if let Some(secs) = base_gop_seconds {
        let rate = if fps > 0 {
            fps as f64
        } else if let FrameSource::Y4m(r) = &frame_source {
            if r.fps_num != 0 {
                r.fps_num as f64 / r.fps_den as f64
            } else {
                0.0
            }
        } else {
            0.0
        };
        if rate <= 0.0 {
            return Err("--base-gop-seconds needs the frame rate (Y4M header or --fps)".into());
        }
        base_gop = (secs * rate).ceil().max(1.0) as usize;
    }

    let mut cfg = LcevcConfig {
        width: width as u16,
        height: height as u16,
        chroma: ChromaFormat::C420,
        base_depth: bit_depth,
        enhancement_depth: bit_depth,
        transform,
        upsampler,
        scaling_l1,
        scaling_l2,
        predicted_average,
        temporal_enabled: temporal,
        temporal_step_width_modifier: temporal_sw_modifier,
        tile_dimensions: tiles,
        custom_tile_size: custom_tile,
        tile_size_compression,
        colour: colour.clone(),
        ..Default::default()
    };
    cfg.level = lcevc_enc::config::level_for_sample_rate(width, height, fps);
    cfg.validate()?;

    let base_out_path = base_out.clone().unwrap_or_default();
    base::set_base_codec(
        match base_mode.as_str() {
            "raw" => BaseMode::Raw,
            "vvc" => BaseMode::Vvc,
            other => return Err(format!("unknown base mode {other}")),
        },
        &format!("preset={vvc_preset},qp={vvc_qp}"),
        &base_out_path,
    );

    let out_path = output.unwrap_or_else(|| "out.lcevc".to_string());
    let mut out_file = std::fs::File::create(&out_path).map_err(|e| e.to_string())?;

    let mut base_dump = if dump_base {
        Some(std::fs::File::create("base_dump.yuv").map_err(|e| e.to_string())?)
    } else {
        None
    };
    let mut recon_dump = if dump_recon {
        Some(std::fs::File::create("recon_dump.yuv").map_err(|e| e.to_string())?)
    } else {
        None
    };

    let base_bitstream: Option<std::fs::File> = match &base_out {
        Some(p) => Some(std::fs::File::create(p).map_err(|e| e.to_string())?),
        None => None,
    };
    let _ = base_bitstream;

    let mut encoder = Encoder::new(cfg.clone(), sw_l1, sw_l2);
    encoder.qm_beta = qm_beta;
    encoder.rdoq = !no_rdoq;

    let mut frame_count = 0u32;
    let mut total_bytes = 0usize;
    let mut total_sse = 0u64;
    let mut total_samples = 0u64;
    let mut verify_frames: Vec<lcevc_enc::frame::Picture> = Vec::new();
    let run_start = std::time::Instant::now();

    let gop_size = base_gop.max(1);
    let depth = cfg.sample_depth();

    // Continuous base streaming: one long-running vvenc encode (raw YUV in
    // via stdin, VVC out via a pipe) feeding a decoder whose output the
    // enhancement consumes frame-by-frame. Memory stays bounded regardless
    // of the keyframe interval (the vvenc intra period is refresh_sec).
    // Base-size EMA: the base stream grows one access unit per received
    // frame, but the vvenc encoder's reorder delay means the file size lags
    // the frame count. Using the per-frame size delta (smoothed) avoids
    // underestimating the base bitrate, which would otherwise overshoot the
    // total budget.
    let tb_for = |base_size: u64, n: u64| -> Option<usize> {
        if let Some(total) = total_kbps {
            let base_bps = if n > 0 { base_size * 8 * fps as u64 / n } else { 0 };
            let budget = (total as u64 * 1000).saturating_sub(base_bps);
            return Some(((budget / 8) / fps.max(1) as u64).max(1) as usize);
        }
        target_kbps.map(|kbps| ((kbps as usize * 1000) / 8).max(1) / (fps.max(1) as usize).max(1))
    };
    // Difficulty-weighted allocation state: frames whose base-only
    // prediction error is large (motion/complexity) get a bigger share of
    // the enhancement budget, smoothed with one frame of lag.
    let mut rc_sse_prev: u64 = 0;
    let mut rc_sse_avg: f64 = 0.0;
    let mut rc_w_avg: f64 = 1.0;
    fn weighted_tb(t: usize, prev: u64, avg: &mut f64, w_avg: &mut f64) -> usize {
        if *avg > 0.0 {
            let w = (prev as f64 / *avg).clamp(0.5, 2.0);
            *avg = *avg * 0.95 + prev as f64 * 0.05;
            // Normalise so the average allocation stays at the target.
            *w_avg = *w_avg * 0.95 + w * 0.05;
            let wn = w / *w_avg;
            ((t as f64) * wn).round().max(1.0) as usize
        } else {
            t
        }
    }

        let mut read_count = 0u32;
        let mut read_next = |frame_source: &mut FrameSource,
                         width: u32,
                         height: u32,
                         bit_depth: u8,
                         verify_frames: &mut Vec<lcevc_enc::frame::Picture>,
                         verify: bool|
         -> Result<Option<lcevc_enc::frame::Picture>, String> {
            if read_count >= frames {
                return Ok(None);
            }
            let frame = match frame_source {
                FrameSource::Raw(f) => {
                    yuv::read_yuv420_frame_file(&mut **f, width as usize, height as usize, bit_depth)?
                }
                FrameSource::Y4m(r) => r.next_frame()?,
            };
            if let Some(f) = &frame {
                read_count += 1;
                if verify && verify_frames.len() < 2 {
                    verify_frames.push(f.clone());
                }
            }
            Ok(frame)
        };
    let mut au_pocs_out: Vec<u64> = Vec::new();
    if base_mode == "vvc" && gop_size > 1 {
        let base_out_stream = if base_out_path.is_empty() { None } else { Some(base_out_path.as_str()) };
        let refresh_sec = base_gop_seconds
            .or_else(|| Some(base_gop as f64 / fps.max(1) as f64))
            .unwrap_or(1.0)
            .ceil()
            .max(1.0) as u32;
        let stream_qp: i32 = vvc_qp.parse().unwrap_or(30);
        let mut streamer = match lcevc_enc::base::VvcStreamer::start_preset(&cfg, base_out_stream, refresh_sec, stream_qp, &vvc_preset) {
            Ok(s) => s,
            Err(e) => {
                // The vvenc library is unavailable: fall back to chunked
                // GOP encoding (a keyframe every `chunk` frames).
                eprintln!("warning: streaming vvenc unavailable ({e}); using chunked GOP encoding");
                let chunk = 10usize;
                let mut gop_buf: Vec<lcevc_enc::frame::Picture> = Vec::new();
                let mut chunk_count = 0u32;
                let mut prev_size: u64 = 0;
                loop {
                    while gop_buf.len() < chunk {
                        match read_next(&mut frame_source, width, height, bit_depth, &mut verify_frames, verify)? {
                            Some(f) => gop_buf.push(f),
                            None => break,
                        }
                    }
                    if gop_buf.is_empty() {
                        break;
                    }
                    let (bw, bh) = {
                        let d = cfg.loq_dimensions();
                        (d[2].0 as usize, d[2].1 as usize)
                    };
                    let mut base_gop: Vec<lcevc_enc::frame::Picture> = Vec::with_capacity(gop_buf.len());
                    for f in &gop_buf {
                        let mut bp = lcevc_enc::frame::Picture::new(bw, bh, cfg.chroma);
                        for p in 0..cfg.num_planes() {
                            let l1 = lcevc_enc::upscale::downscale_plane(&f.planes[p], cfg.scaling_l2, depth);
                            let base_t = lcevc_enc::upscale::downscale_plane(&l1, cfg.scaling_l1, depth);
                            bp.planes[p] = base_t;
                        }
                        base_gop.push(bp);
                    }
                    let base_out_opt = if base_out_path.is_empty() { None } else { Some(base_out_path.as_str()) };
                    let decoded = lcevc_enc::base::encode_decode_base_gop(&cfg, &base_gop, base_out_opt)?;
                    let size = std::fs::metadata(&base_out_path).map(|m| m.len()).unwrap_or(0);
                    let delta = size.saturating_sub(prev_size);
                    prev_size = size;
                    let tb = if let Some(total) = total_kbps {
                        if !gop_buf.is_empty() {
                            let base_bps = delta * 8 * fps as u64 / gop_buf.len() as u64;
                            let budget = (total as u64 * 1000).saturating_sub(base_bps);
                            Some(((budget / 8) / fps.max(1) as u64).max(1) as usize)
                        } else {
                            None
                        }
                    } else {
                        target_kbps.map(|kbps| ((kbps as usize * 1000) / 8).max(1) / (fps.max(1) as usize).max(1))
                    };
                    process_gop(
                        &gop_buf, &decoded, tb, fps, &mut encoder, &cfg, bit_depth,
                        frames, run_start, no_psnr, &mut frame_count, &mut total_bytes,
                        &mut total_sse, &mut total_samples, &mut out_file, &mut base_dump,
                        &mut recon_dump,
                    )?;
                    chunk_count += 1;
                    gop_buf.clear();
                }
                let _ = chunk_count;
                return Ok(());
            }
        };
        // Read-ahead queue: the VVC decoder's B-frame reordering delays its
        // output by a few frames, so the encoder must stay ahead of the
        // decoder (sending future frames) or the receive blocks forever.
        let mut queue: std::collections::VecDeque<lcevc_enc::frame::Picture> =
            std::collections::VecDeque::new();
        let mut sent = 0u64;
        let mut eof = false;
        // The base access units are emitted in the VVC decode order (B-frame
        // reordering), so the enhancement NALs must be written to the
        // output in the same order or the muxed file pairs each base
        // picture with the wrong enhancement. Buffer the NALs per source
        // frame and release them as the corresponding access unit (its POC
        // is the source frame index) is emitted.
        let mut au_pocs: std::collections::VecDeque<u64> = std::collections::VecDeque::new();
        let mut el_buf: Vec<Option<Vec<u8>>> = Vec::new();
        let mut el_done = 0usize;
        let mut nwritten = 0usize;
        // The base pipeline's latency (the vvenc encoder's reorder holdback
        // plus the decoder's) is several tens of frames, so the encoder
        // must stay that far ahead of the decoder or the receive blocks
        // forever. Prime the pipeline, then always send the next source
        // frame before receiving, keeping the in-flight distance bounded.
        for _ in 0..96 {
            match read_next(&mut frame_source, width, height, bit_depth, &mut verify_frames, verify)? {
                Some(f) => {
                    if let Some(p) = streamer.send_frame(&f, cfg.scaling_l1, cfg.scaling_l2)? {
                        au_pocs.push_back(p);
                    }
                    queue.push_back(f);
                    sent += 1;
                }
                None => {
                    eof = true;
                    break;
                }
            }
        }
        loop {
            if !eof {
                match read_next(&mut frame_source, width, height, bit_depth, &mut verify_frames, verify)? {
                    Some(f) => {
                        if let Some(p) = streamer.send_frame(&f, cfg.scaling_l1, cfg.scaling_l2)? {
                            au_pocs.push_back(p);
                        }
                        queue.push_back(f);
                        sent += 1;
                    }
                    None => eof = true,
                }
            } else {
                // Source exhausted: the pipeline still holds ~90 frames in
                // flight, but the decoder can only release them once the
                // encoder is flushed and its input closes. Do that now and
                // drain below.
                break;
            }
            match streamer.recv_frame()? {
                Some(base) => {
                    let prof = std::env::var("LCEVC_PROF").is_ok();
                    let t_recv = std::time::Instant::now();
                    if let Some(frame) = queue.pop_front() {
                let base_size = std::fs::metadata(&base_out_path).map(|m| m.len()).unwrap_or(0);
                let tb = tb_for(base_size, sent).map(|t| weighted_tb(t, rc_sse_prev, &mut rc_sse_avg, &mut rc_w_avg));
                let frame_start = std::time::Instant::now();
                let encoded = match tb {
                    Some(tb) => {
                        let (ef, _sw1, _sw2) = encoder.encode_frame_rc(&frame, &base, tb)?;
                        ef
                    }
                    None => encoder.encode_frame_with_base(&frame, &base)?,
                };
                let t_enc = std::time::Instant::now();
                if prof {
                    eprintln!("PROF recv_wait={:?} encode={:?}", t_recv.elapsed(), t_enc.duration_since(frame_start));
                }
                rc_sse_prev = encoded.base_only_sse;
                if rc_sse_avg == 0.0 {
                    rc_sse_avg = rc_sse_prev as f64;
                }
                let nal = process_encoded_frame(
                    &encoded, &frame, frame_start, &cfg, bit_depth, frames, run_start,
                    no_psnr, &mut frame_count, &mut total_bytes, &mut total_sse,
                    &mut total_samples, &mut out_file, &mut base_dump, &mut recon_dump,
                )?;
                if el_done == el_buf.len() {
                    el_buf.push(None);
                }
                el_buf[el_done] = Some(nal);
                el_done += 1;
                while let Some(&poc) = au_pocs.front() {
                    let idx = poc as usize;
                    if idx >= el_buf.len() || el_buf[idx].is_none() {
                        break;
                    }
                    if let Some(n) = el_buf[idx].take() {
                        out_file.write_all(&n).map_err(|e| e.to_string())?;
                        au_pocs.pop_front();
                    }
                }
                    }
                }
                None => break,
            }
        }
        // End of input: flush the encoder (delivering the remaining access
        // units) and close the decoder's input so its reordered backlog is
        // emitted, then drain everything.
        for p in streamer.finish_flush()? {
            au_pocs.push_back(p);
        }
        while let Some(base) = streamer.recv_frame()? {
            if let Some(frame) = queue.pop_front() {
                let prof = std::env::var("LCEVC_PROF").is_ok();
                let t_recv = std::time::Instant::now();
                let base_size = std::fs::metadata(&base_out_path).map(|m| m.len()).unwrap_or(0);
                let tb = tb_for(base_size, sent).map(|t| weighted_tb(t, rc_sse_prev, &mut rc_sse_avg, &mut rc_w_avg));
                let frame_start = std::time::Instant::now();
                let encoded = match tb {
                    Some(tb) => {
                        let (ef, _sw1, _sw2) = encoder.encode_frame_rc(&frame, &base, tb)?;
                        ef
                    }
                    None => encoder.encode_frame_with_base(&frame, &base)?,
                };
                if prof {
                    eprintln!("PROF flush recv_wait={:?} encode={:?}", t_recv.elapsed(), frame_start.elapsed());
                }
                rc_sse_prev = encoded.base_only_sse;
                if rc_sse_avg == 0.0 {
                    rc_sse_avg = rc_sse_prev as f64;
                }
                let nal = process_encoded_frame(
                    &encoded, &frame, frame_start, &cfg, bit_depth, frames, run_start,
                    no_psnr, &mut frame_count, &mut total_bytes, &mut total_sse,
                    &mut total_samples, &mut out_file, &mut base_dump, &mut recon_dump,
                )?;
                if el_done == el_buf.len() {
                    el_buf.push(None);
                }
                el_buf[el_done] = Some(nal);
                el_done += 1;
                while let Some(&poc) = au_pocs.front() {
                    let idx = poc as usize;
                    if idx >= el_buf.len() || el_buf[idx].is_none() {
                        break;
                    }
                    if let Some(n) = el_buf[idx].take() {
                        out_file.write_all(&n).map_err(|e| e.to_string())?;
                        au_pocs.pop_front();
                        nwritten += 1;
                    }
                }
            }
        }
        while let Some(&poc) = au_pocs.front() {
            let idx = poc as usize;
            if idx >= el_buf.len() || el_buf[idx].is_none() {
                break;
            }
            if let Some(n) = el_buf[idx].take() {
                out_file.write_all(&n).map_err(|e| e.to_string())?;
                au_pocs.pop_front();
            }
        }
        eprintln!("DEBUG au_pocs={} el_done={} nwritten={} front={:?} left={:?}", streamer.au_pocs.len(), el_done, nwritten, au_pocs.front(), el_buf.iter().enumerate().filter(|(_, v)| v.is_some()).map(|(i, _)| i).take(20).collect::<Vec<_>>());
        au_pocs_out = std::mem::take(&mut streamer.au_pocs);
        streamer.finish()?;
    } else {
        let mut read_count = 0u32;
        loop {
            if read_count >= frames {
                break;
            }
            let frame = match &mut frame_source {
                FrameSource::Raw(f) => {
                    yuv::read_yuv420_frame_file(&mut **f, width as usize, height as usize, bit_depth)?
                }
                FrameSource::Y4m(r) => r.next_frame()?,
            };
            let frame = match frame {
                Some(f) => f,
                None => break,
            };
            read_count += 1;
            if verify && verify_frames.len() < 2 {
                verify_frames.push(frame.clone());
            }
            let frame_start = std::time::Instant::now();
            let encoded = encoder.encode_frame(&frame)?;
            process_encoded_frame(
                &encoded, &frame, frame_start, &cfg, bit_depth, frames, run_start,
                no_psnr, &mut frame_count, &mut total_bytes, &mut total_sse,
                &mut total_samples, &mut out_file, &mut base_dump, &mut recon_dump,
            )?;
        }
    }

    let psnr = if no_psnr || total_sse == 0 {
        f64::NAN
    } else {
        let max_val = ((1u64 << bit_depth) - 1) as f64;
        10.0 * (max_val * max_val * total_samples as f64 / total_sse as f64).log10()
    };
    if psnr.is_nan() {
        eprintln!("encoded {frame_count} frames -> {out_path} ({total_bytes} bytes)");
    } else {
        eprintln!(
            "encoded {frame_count} frames -> {out_path} ({total_bytes} bytes), encoder reconstruction PSNR {psnr:.2} dB"
        );
    }
    eprintln!(
        "step widths: L1 = {sw_l1}, L2 = {sw_l2}; transform {:?}; upsampler {:?}; scaling L1 {:?} L2 {:?}",
        cfg.transform, cfg.upsampler, cfg.scaling_l1, cfg.scaling_l2,
    );

    if let Some(mux_path) = &mux_out {
        if base_mode != "vvc" {
            return Err("--mux requires --base-mode vvc".into());
        }
        let base_path = base_out.as_ref().ok_or("--mux requires --base-out")?;
        let base_data = std::fs::read(base_path).map_err(|e| e.to_string())?;
        let lcevc_data = std::fs::read(&out_path).map_err(|e| e.to_string())?;
        let aus = lcevc_enc::mp4::split_aus(&base_data);
        let nals = lcevc_enc::mp4::split_lcevc_nals(&lcevc_data);
        let mut base_aus: Vec<Vec<u8>> = Vec::new();
        for (s, l) in &aus {
            base_aus.push(base_data[*s..*s + *l].to_vec());
        }
        let mut enh_nals: Vec<Vec<u8>> = Vec::new();
        for (s, l) in &nals {
            enh_nals.push(lcevc_data[*s..*s + *l].to_vec());
        }
        let (bw, bh) = {
            let d = cfg.loq_dimensions();
            (d[2].0, d[2].1)
        };
        lcevc_enc::mp4::mux_mp4(mux_path, &base_aus, &enh_nals, bw, bh, fps, colour.as_ref(),
                                Some(&au_pocs_out))?;
        eprintln!("muxed {mux_path} ({} frames, base {bw}x{bh})", base_aus.len());
    }

    if verify {
        // Decode the stream with the built-in mirror decoder and compare
        // against the encoder's reconstruction, using the first two input
        // frames captured during the encode (works for stdin/Y4M too).
        if verify_frames.len() < 2 {
            return Err("verify needs at least two input frames".into());
        }
        let mut temporal_buffers: Vec<lcevc_enc::frame::PlaneS16> = (0..cfg.num_planes())
            .map(|p| {
                let (w, h) = cfg.plane_dimensions(0, p);
                lcevc_enc::frame::PlaneS16::new(w as usize, h as usize)
            })
            .collect();
        let frame = verify_frames[0].clone();
        let mut enc = Encoder::new(cfg.clone(), sw_l1, sw_l2);
        let f0 = enc.encode_frame(&frame).unwrap();
        // Decode the first frame (IDR) to initialise the temporal buffer.
        {
            let mut p0 = Vec::new();
            p0.push((lcevc_enc::nal::BLOCK_SEQUENCE_CONFIG, cfg.write_sequence_config()));
            p0.push((lcevc_enc::nal::BLOCK_GLOBAL_CONFIG, cfg.write_global_config()));
            p0.push((lcevc_enc::nal::BLOCK_PICTURE_CONFIG, f0.picture_config.clone()));
            p0.push((lcevc_enc::nal::BLOCK_ENCODED_DATA, f0.encoded_data.clone()));
            let nal = build_nal_unit(f0.idr, &p0);
            let (_, blocks) = lcevc_enc::decoder::parse_nal(&nal).unwrap();
            let _ = lcevc_enc::decoder::decode_frame(&cfg, &blocks, &f0.base_picture, &mut temporal_buffers).unwrap();
        }
        // Encode a second frame and decode it with the mirror.
        let f1 = enc.encode_frame(&verify_frames[1]).unwrap();
        let mut p1 = Vec::new();
        p1.push((lcevc_enc::nal::BLOCK_PICTURE_CONFIG, f1.picture_config.clone()));
        p1.push((lcevc_enc::nal::BLOCK_ENCODED_DATA, f1.encoded_data.clone()));
        let nal = build_nal_unit(f1.idr, &p1);
        let (_, blocks) = lcevc_enc::decoder::parse_nal(&nal).unwrap();
        let decoded = lcevc_enc::decoder::decode_frame(&cfg, &blocks, &f1.base_picture, &mut temporal_buffers).unwrap();
        let mut sse = 0i64;
        let mut maxd = 0i64;
        let mut n = 0i64;
        for p in 0..3 {
            for (a, b) in decoded.planes[p].data.iter().zip(f1.output.planes[p].data.iter()) {
                let d = (*a as i64 - *b as i64).abs();
                sse += d * d;
                maxd = maxd.max(d);
                n += 1;
            }
        }
        let psnr = if sse == 0 {
            f64::INFINITY
        } else {
            10.0 * (255.0f64 * 255.0 * n as f64 / sse as f64).log10()
        };
        eprintln!("self-check: {n} samples, PSNR {psnr:.3} dB, max diff {maxd}");
        if maxd > 0 {
            return Err("self-check failed: reconstruction mismatch".into());
        }
        eprintln!("self-check passed: the stream decodes bit-exactly with the mirror decoder");
    }

    Ok(())
}

fn colour_primaries(name: &str) -> (u16, &'static str) {
    match name {
        "bt709" => (1, "bt709"),
        "bt470bg" => (5, "bt470bg"),
        "smpte170m" => (6, "smpte170m"),
        "smpte240m" => (7, "smpte240m"),
        "film" => (8, "film"),
        "bt2020" => (9, "bt2020"),
        _ => (2, "bt709"),
    }
}

fn colour_transfer(name: &str) -> (u16, &'static str) {
    match name {
        "bt709" => (1, "bt709"),
        "bt470bg" | "gamma22" => (4, "bt470bg"),
        "smpte170m" => (6, "smpte170m"),
        "linear" => (8, "linear"),
        "bt2020_10" => (14, "bt2020_10"),
        "smpte2084" => (16, "smpte2084"),
        "arib-std-b67" => (18, "arib-std-b67"),
        _ => (2, "bt709"),
    }
}

fn colour_matrix(name: &str) -> (u16, &'static str) {
    match name {
        "gbr" | "rgb" => (0, "gbr"),
        "bt709" => (1, "bt709"),
        "bt470bg" => (5, "bt470bg"),
        "smpte170m" => (6, "smpte170m"),
        "bt2020nc" => (9, "bt2020nc"),
        "bt2020c" => (10, "bt2020c"),
        _ => (2, "bt709"),
    }
}

fn parse_scaling(s: &str) -> Result<ScalingMode, String> {
    match s {
        "0" => Ok(ScalingMode::Scale0D),
        "1" => Ok(ScalingMode::Scale1D),
        "2" => Ok(ScalingMode::Scale2D),
        _ => Err("bad scaling mode".into()),
    }
}

fn parse_on_off(s: &str) -> Result<bool, String> {
    match s {
        "on" | "1" | "true" => Ok(true),
        "off" | "0" | "false" => Ok(false),
        _ => Err("expected on/off".into()),
    }
}
