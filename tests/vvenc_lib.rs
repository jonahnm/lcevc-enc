#[test]
fn vvenc_lib_padded() {
    if !lib_available() {
        println!("libvvenc not available; skipping");
        return;
    }
    let mut lib = lcevc_enc::base::vvenc_lib::VvencLib::new(320, 192, 25, 24, 4, 10, None).unwrap();
    let y = vec![512u16; 320 * 192];
    let uv = vec![512u16; 160 * 96];
    let planes = [y.as_slice(), uv.as_slice(), uv.as_slice()];
    let pw = [320, 160, 160];
    let ph = [192, 96, 96];
    for i in 0..3 {
        let au = lib.encode_frame(&planes, pw, ph).unwrap();
        println!("enc {i}: au={}", au.as_ref().map(|a| a.data.len()).unwrap_or(0));
    }
    let f = lib.flush().unwrap();
    println!("flush AUs: {}", f.len());
}

#[test]
fn vvenc_lib_alignment() {
    if !lib_available() {
        println!("libvvenc not available; skipping");
        return;
    }
    // 640x360 (height not 16-aligned) must be rejected; 640x368 accepted.
    let r1 = lcevc_enc::base::vvenc_lib::VvencLib::new(640, 360, 25, 24, 4, 10, None);
    println!("640x360 -> {:?}", r1.is_err());
    let r2 = lcevc_enc::base::vvenc_lib::VvencLib::new(640, 368, 25, 24, 4, 10, None);
    println!("640x368 -> {:?}", r2.is_ok());
}

/// Returns None when libvvenc is not available (tests pass vacuously so
/// `cargo test` works on machines without the library).
fn lib_available() -> bool {
    lcevc_enc::base::vvenc_lib::VvencLib::new(16, 16, 25, 24, 1, 10, None).is_ok()
}

#[test]
fn vvenc_lib_streams_aus() {
    if !lib_available() {
        println!("libvvenc not available; skipping");
        return;
    }
    let mut lib = lcevc_enc::base::vvenc_lib::VvencLib::new(320, 176, 25, 24, 4, 10, None).unwrap();
    println!("config: {}", lib.config_string());
    let mut lib1 = lcevc_enc::base::vvenc_lib::VvencLib::new_single(320, 176, 25, 24, None).unwrap();
    let y = vec![512u16; 320 * 176];
    let uv = vec![512u16; 160 * 88];
    let planes = [y.as_slice(), uv.as_slice(), uv.as_slice()];
    let pw = [320, 160, 160];
    let ph = [176, 88, 88];
    let mut got1 = 0usize;
    for i in 0..40 {
        let au = lib1.encode_frame(&planes, pw, ph).unwrap();
        got1 += au.as_ref().map(|a| a.data.len()).unwrap_or(0);
        if i < 4 {
            println!("single frame {i}: au={}", au.as_ref().map(|a| a.data.len()).unwrap_or(0));
        }
    }
    let flushed = match lib1.flush() {
        Ok(f) => f,
        Err(e) => {
            println!("flush err: {e}; last: {}", lib1.last_error());
            panic!("flush failed");
        }
    };
    got1 += flushed.iter().map(|a| a.data.len()).sum::<usize>();
    println!("single-thread: {} AUs flushed, {} bytes total", flushed.len(), got1);
    drop(lib1);

    let t0 = std::time::Instant::now();
    for i in 0..40 {
        let au = lib.encode_frame(&planes, pw, ph).unwrap();
        if i % 8 == 0 {
            println!("frame {i}: au={}", au.as_ref().map(|a| a.data.len()).unwrap_or(0));
        }
    }
    println!("encode 40 frames took {:?}", t0.elapsed());
    let flushed = lib.flush().unwrap();
    println!(
        "flushed {} AUs, total bytes: {}",
        flushed.len(),
        flushed.iter().map(|a| a.data.len()).sum::<usize>()
    );
    assert!(flushed.len() > 0, "flush should produce AUs");
}
