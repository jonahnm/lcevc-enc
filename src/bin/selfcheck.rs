//! Self-check: encode a picture, decode the stream with the built-in mirror
//! decoder and compare against the encoder's reconstruction.

use lcevc_enc::base::{self, BaseMode};
use lcevc_enc::config::{ChromaFormat, LcevcConfig};
use lcevc_enc::decoder::{decode_frame, parse_nal};
use lcevc_enc::encoder::Encoder;
use lcevc_enc::nal::build_nal_unit;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let input = args.get(1).expect("usage: selfcheck <input.yuv> [sw_l1] [sw_l2]");
    let sw_l1: u32 = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(16);
    let sw_l2: u32 = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(8);

    base::set_base_codec(BaseMode::Raw, "", "");
    let mut cfg = LcevcConfig {
        width: 1280,
        height: 720,
        chroma: ChromaFormat::C420,
        ..Default::default()
    };
    cfg.temporal_enabled = std::env::args().nth(6).map(|s| s == "1").unwrap_or(false);
    cfg.scaling_l1 = std::env::args().nth(4).and_then(|s| s.parse::<u8>().ok())
        .map(|v| match v { 1 => lcevc_enc::config::ScalingMode::Scale1D,
                            _ => lcevc_enc::config::ScalingMode::Scale2D }).unwrap_or(lcevc_enc::config::ScalingMode::Scale2D);
    cfg.scaling_l2 = std::env::args().nth(5).and_then(|s| s.parse::<u8>().ok())
        .map(|v| match v { 1 => lcevc_enc::config::ScalingMode::Scale1D,
                            _ => lcevc_enc::config::ScalingMode::Scale2D }).unwrap_or(lcevc_enc::config::ScalingMode::Scale2D);
    cfg.validate().unwrap();

    let data = std::fs::read(input).unwrap();
    let n = (1280 * 720 * 3) / 2;
    let frame = lcevc_enc::yuv::read_yuv420_frame(&data[..n], 1280, 720, 8).unwrap();

    let mut enc = Encoder::new(cfg.clone(), sw_l1, sw_l2);
    let frame0 = enc.encode_frame(&frame).unwrap();
    let encoded = enc.encode_frame(&frame).unwrap();

    let mut payloads = Vec::new();
    payloads.push((lcevc_enc::nal::BLOCK_PICTURE_CONFIG, encoded.picture_config.clone()));
    payloads.push((lcevc_enc::nal::BLOCK_ENCODED_DATA, encoded.encoded_data.clone()));
    let nal = build_nal_unit(encoded.idr, &payloads);
    std::fs::write("/tmp/opencode/selfcheck.lcevc", &nal).unwrap();

    let (idr, blocks) = parse_nal(&nal).unwrap();
    assert_eq!(idr, encoded.idr);

    let mut temporal_buffers: Vec<lcevc_enc::frame::PlaneS16> = (0..cfg.num_planes())
        .map(|p| {
            let (w, h) = cfg.plane_dimensions(0, p);
            lcevc_enc::frame::PlaneS16::new(w as usize, h as usize)
        })
        .collect();
    let frame0_out = decode_frame(&cfg, &parse_nal(&{
        let mut p0 = Vec::new();
        p0.push((lcevc_enc::nal::BLOCK_SEQUENCE_CONFIG, cfg.write_sequence_config()));
        p0.push((lcevc_enc::nal::BLOCK_GLOBAL_CONFIG, cfg.write_global_config()));
        p0.push((lcevc_enc::nal::BLOCK_PICTURE_CONFIG, frame0.picture_config.clone()));
        p0.push((lcevc_enc::nal::BLOCK_ENCODED_DATA, frame0.encoded_data.clone()));
        build_nal_unit(frame0.idr, &p0)
    }).unwrap().1, &frame0.base_picture, &mut temporal_buffers).unwrap();
    {
        let mut sse = 0i64; let mut n = 0i64;
        for p in 0..3 {
            for (a, b) in frame0_out.planes[p].data.iter().zip(frame0.output.planes[p].data.iter()) {
                let d = (*a as i64 - *b as i64).abs(); sse += d * d; n += 1;
            }
        }
        let psnr = if sse == 0 { f64::INFINITY } else { 10.0 * (255.0f64 * 255.0 * n as f64 / sse as f64).log10() };
        println!("frame0 selfcheck: PSNR {psnr:.3} dB");
    }
    let decoded = decode_frame(&cfg, &blocks, &encoded.base_picture, &mut temporal_buffers).unwrap();

    let mut sse = 0i64;
    let mut maxd = 0i64;
    let mut n = 0i64;
    let mut first = None;
    for p in 0..3 {
        for (i, (a, b)) in decoded.planes[p].data.iter().zip(encoded.output.planes[p].data.iter()).enumerate() {
            let d = (*a as i64 - *b as i64).abs();
            sse += d * d;
            if d > maxd { maxd = d; }
            if d > 0 && first.is_none() { first = Some((p, i, *a, *b)); }
            n += 1;
        }
    }
    if let Some((p, i, a, b)) = first {
        println!("first diff: plane {p} sample {i} (x={}, y={}): dec={a} enc={b}", i % 1280, i / 1280);
    }
    let psnr = if sse == 0 {
        f64::INFINITY
    } else {
        10.0 * (255.0f64 * 255.0 * n as f64 / sse as f64).log10()
    };
    println!("selfcheck: {n} samples, PSNR vs encoder: {psnr:.3} dB, max diff {maxd}");
    if maxd > 0 {
        std::process::exit(1);
    }
}
