//! Minimal dynamic bindings to the vvenc (VVC) encoder library.
//!
//! The library is loaded at runtime (dlopen / LoadLibrary) so the binary
//! keeps building without a link-time dependency. The struct layouts are
//! taken from the public vvenc headers (1.14.0) and verified against the
//! shipped dev package; `vvenc_config` is treated as an opaque blob whose
//! parameters are set by name through `vvenc_set_param`, so only its total
//! size must match (47312 bytes on x86_64/aarch64, defaults aligned).

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};

const VVENC_CONFIG_SIZE: usize = 47312;
const VVENC_VERSION: &str = "1.14.0";
const VVENC_FASTER: i32 = 0;

// ---------------------------------------------------------------------------
// dynamic loading

type VoidPtr = *mut std::ffi::c_void;

#[cfg(unix)]
mod dynlib {
    use std::ffi::{c_char, c_int, c_void};
    extern "C" {
        fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
        fn dlclose(handle: *mut c_void) -> c_int;
    }
    const RTLD_NOW: c_int = 2;

    pub fn open(names: &[&str]) -> Result<*mut c_void, String> {
        let mut last = "library not found".to_string();
        for name in names {
            let cname = std::ffi::CString::new(*name).unwrap();
            unsafe {
                let h = dlopen(cname.as_ptr(), RTLD_NOW);
                if !h.is_null() {
                    return Ok(h);
                }
            }
            last = format!("{name} not found");
        }
        Err(last)
    }

    pub fn symbol(handle: *mut c_void, name: &str) -> Result<*mut c_void, String> {
        let cname = std::ffi::CString::new(name).unwrap();
        unsafe {
            let s = dlsym(handle, cname.as_ptr());
            if s.is_null() {
                Err(format!("symbol {name} not found"))
            } else {
                Ok(s)
            }
        }
    }

    pub fn close(handle: *mut c_void) {
        unsafe { dlclose(handle); }
    }
}

#[cfg(windows)]
mod dynlib {
    use std::ffi::c_void;
    extern "system" {
        fn LoadLibraryA(name: *const u8) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const u8) -> *mut c_void;
        fn FreeLibrary(module: *mut c_void) -> i32;
    }

    pub fn open(names: &[&str]) -> Result<*mut c_void, String> {
        let mut last = "library not found".to_string();
        for name in names {
            let mut buf = name.as_bytes().to_vec();
            buf.push(0);
            unsafe {
                let h = LoadLibraryA(buf.as_ptr());
                if !h.is_null() {
                    return Ok(h);
                }
            }
            last = format!("{name} not found");
        }
        Err(last)
    }

    pub fn symbol(handle: *mut c_void, name: &str) -> Result<*mut c_void, String> {
        let mut buf = name.as_bytes().to_vec();
        buf.push(0);
        unsafe {
            let s = GetProcAddress(handle, buf.as_ptr());
            if s.is_null() {
                Err(format!("symbol {name} not found"))
            } else {
                Ok(s)
            }
        }
    }

    pub fn close(handle: *mut c_void) {
        unsafe { FreeLibrary(handle); }
    }
}

// When the vendored vvenc was linked statically (build.rs emitted
// `cargo:rustc-cfg=have_vvenc`), the symbols resolve directly; otherwise
// the library is loaded at runtime.
#[cfg(have_vvenc)]
extern "C" {
    fn vvenc_init_default(
        c: *mut VvencConfig, w: i32, h: i32, fps: i32, br: i32, qp: i32, preset: i32,
    ) -> i32;
    fn vvenc_set_param(c: *mut VvencConfig, n: *const std::ffi::c_char, v: *const std::ffi::c_char) -> i32;
    fn vvenc_set_msg_callback(c: *mut VvencConfig, ctx: *mut std::ffi::c_void, f: Option<VvencMsgFn>) -> std::ffi::c_void;
    fn vvenc_encoder_create() -> *mut std::ffi::c_void;
    fn vvenc_encoder_open(e: *mut std::ffi::c_void, c: *mut VvencConfig) -> i32;
    fn vvenc_get_headers(e: *mut std::ffi::c_void, au: *mut VvencAccessUnit) -> i32;
    fn vvenc_encode(e: *mut std::ffi::c_void, y: *const VvencYuvBuffer, au: *mut VvencAccessUnit, done: *mut bool) -> i32;
    fn vvenc_encoder_close(e: *mut std::ffi::c_void) -> i32;
    fn vvenc_get_config(e: *mut std::ffi::c_void, c: *mut VvencConfig) -> i32;
    fn vvenc_get_config_as_string(c: *const VvencConfig, lvl: i32) -> *const std::ffi::c_char;
    fn vvenc_get_last_error(e: *mut std::ffi::c_void) -> *const std::ffi::c_char;
    fn vvenc_get_version() -> *const std::ffi::c_char;
}

#[cfg(have_vvenc)]
const STATIC_SYMS: &[(&str, VoidPtr)] = &[
    ("vvenc_init_default", vvenc_init_default as *const () as VoidPtr),
    ("vvenc_set_param", vvenc_set_param as *const () as VoidPtr),
    ("vvenc_set_msg_callback", vvenc_set_msg_callback as *const () as VoidPtr),
    ("vvenc_encoder_create", vvenc_encoder_create as *const () as VoidPtr),
    ("vvenc_encoder_open", vvenc_encoder_open as *const () as VoidPtr),
    ("vvenc_get_headers", vvenc_get_headers as *const () as VoidPtr),
    ("vvenc_encode", vvenc_encode as *const () as VoidPtr),
    ("vvenc_encoder_close", vvenc_encoder_close as *const () as VoidPtr),
    ("vvenc_get_config", vvenc_get_config as *const () as VoidPtr),
    ("vvenc_get_config_as_string", vvenc_get_config_as_string as *const () as VoidPtr),
    ("vvenc_get_last_error", vvenc_get_last_error as *const () as VoidPtr),
    ("vvenc_get_version", vvenc_get_version as *const () as VoidPtr),
];

struct Lib {
    handle: Option<VoidPtr>,
}

impl Lib {
    fn open() -> Result<Lib, String> {
        #[cfg(have_vvenc)]
        {
            let _ = STATIC_SYMS;
            return Ok(Lib { handle: None });
        }
        #[cfg(not(have_vvenc))]
        {
            #[cfg(unix)]
            let names = ["libvvenc.so.1.14", "libvvenc.so.1", "libvvenc.so"];
            #[cfg(windows)]
            let names = ["vvenc.dll"];
            let handle = dynlib::open(&names)?;
            Ok(Lib { handle: Some(handle) })
        }
    }

    fn sym(&self, name: &str) -> Result<VoidPtr, String> {
        #[cfg(have_vvenc)]
        {
            STATIC_SYMS
                .iter()
                .find(|(n, _)| *n == name)
                .map(|(_, p)| *p)
                .ok_or_else(|| format!("static symbol {name} missing"))
        }
        #[cfg(not(have_vvenc))]
        {
            dynlib::symbol(self.handle.expect("libvvenc handle"), name)
        }
    }
}

impl Drop for Lib {
    fn drop(&mut self) {
        #[cfg(not(have_vvenc))]
        if let Some(h) = self.handle {
            dynlib::close(h);
        }
    }
}

// ---------------------------------------------------------------------------
// struct layouts (from vvenc.h, stable API)

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct VvencYuvPlane {
    ptr: *mut i16,
    width: i32,
    height: i32,
    stride: i32,
}

#[repr(C)]
struct VvencYuvBuffer {
    planes: [VvencYuvPlane; 3],
    sequence_number: u64,
    cts: i64,
    cts_valid: bool,
}

impl Default for VvencYuvBuffer {
    fn default() -> Self {
        VvencYuvBuffer {
            planes: [VvencYuvPlane::default(); 3],
            sequence_number: 0,
            cts: 0,
            cts_valid: false,
        }
    }
}

#[repr(C)]
struct VvencAccessUnit {
    payload: *mut u8,
    payload_size: i32,
    payload_used_size: i32,
    cts: i64,
    dts: i64,
    cts_valid: bool,
    dts_valid: bool,
    rap: bool,
    slice_type: i32,
    ref_pic: bool,
    temporal_layer: i32,
    poc: u64,
    status: i32,
    essential_bytes: i32,
    info_string: [i8; 1024],
}

impl Default for VvencAccessUnit {
    fn default() -> Self {
        VvencAccessUnit {
            payload: std::ptr::null_mut(),
            payload_size: 0,
            payload_used_size: 0,
            cts: 0,
            dts: 0,
            cts_valid: false,
            dts_valid: false,
            rap: false,
            slice_type: 0,
            ref_pic: false,
            temporal_layer: 0,
            poc: 0,
            status: 0,
            essential_bytes: 0,
            info_string: [0i8; 1024],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VvencConfig {
    bytes: [u8; VVENC_CONFIG_SIZE],
}

impl Default for VvencConfig {
    fn default() -> Self {
        VvencConfig {
            bytes: [0u8; VVENC_CONFIG_SIZE],
        }
    }
}

// Formatting a C va_list needs the CRT's vsnprintf, which is not reliably
// resolvable on every Windows toolchain (MSVC's import libraries disagree
// on the exported name); the callback therefore formats only on Unix and
// prints a minimal diagnostic on Windows.
#[cfg(unix)]
extern "C" {
    fn vsnprintf(s: *mut std::ffi::c_char, n: usize, fmt: *const std::ffi::c_char, ap: *mut std::ffi::c_void) -> std::ffi::c_int;
}

// ---------------------------------------------------------------------------
// logging callback

type VvencMsgFn = unsafe extern "C" fn(
    *mut std::ffi::c_void,
    std::ffi::c_int,
    *const std::ffi::c_char,
    *mut std::ffi::c_void,
);

/// Callback invoked by vvenc for its messages; formats the va_list into a
/// line and prints it (warnings and errors only).
extern "C" fn vvenc_msg_cb(
    _ctx: *mut std::ffi::c_void,
    level: std::ffi::c_int,
    fmt: *const std::ffi::c_char,
    args: *mut std::ffi::c_void,
) {
    if level < 1 {
        return;
    }
    let tag = match level {
        1 => "error",
        2 => "warning",
        _ => "info",
    };
    #[cfg(unix)]
    {
        let mut buf = [0i8; 2048];
        let n = unsafe { vsnprintf(buf.as_mut_ptr() as *mut std::ffi::c_char, buf.len(), fmt, args) };
        if n > 0 {
            let msg = unsafe { std::ffi::CStr::from_ptr(buf.as_ptr() as *const std::ffi::c_char) }
                .to_string_lossy()
                .into_owned();
            eprintln!("vvenc[{tag}]: {msg}");
            return;
        }
    }
    let _ = (fmt, args);
    eprintln!("vvenc[{tag}]: (message)");
}

// ---------------------------------------------------------------------------
// the encoder wrapper

type EncoderHandle = VoidPtr;

/// A vvenc encoder instance loaded at runtime, streaming one access unit
/// per encoded frame (plus the reorder-delayed AUs when flushed at EOF).
pub struct VvencLib {
    _lib: Box<Lib>,
    enc: EncoderHandle,
    yuv: Box<VvencYuvBuffer>,
    au: Box<VvencAccessUnit>,
    au_payload: Vec<u8>,
    frames_in: u64,
    width: usize,
    height: usize,
}

pub struct EncodedAu {
    pub data: Vec<u8>,
    pub poc: u64,
    pub rap: bool,
}

impl VvencLib {
    /// Convenience: a single-threaded encoder (used for diagnostics).
    pub fn new_single(
        width: usize,
        height: usize,
        framerate: i32,
        qp: i32,
        colour: Option<&crate::config::ColourInfo>,
    ) -> Result<VvencLib, String> {
        Self::new_with(width, height, framerate, qp, 1, 10, colour, false)
    }

    /// Load libvvenc and create an encoder for the given source picture.
    pub fn new(
        width: usize,
        height: usize,
        framerate: i32,
        qp: i32,
        threads: i32,
        refresh_sec: i32,
        colour: Option<&crate::config::ColourInfo>,
    ) -> Result<VvencLib, String> {
        Self::new_with(width, height, framerate, qp, threads, refresh_sec, colour, true)
    }

    fn new_with(
        width: usize,
        height: usize,
        framerate: i32,
        qp: i32,
        threads: i32,
        refresh_sec: i32,
        colour: Option<&crate::config::ColourInfo>,
        multithread: bool,
    ) -> Result<VvencLib, String> {
        let lib = Box::new(Lib::open()?);
        let mut cfg: Box<VvencConfig> = Box::default();

                // Logging callback so encoder messages (warnings/errors) are not
        // silently dropped.
        {
            type FnSetMsg = unsafe extern "C" fn(*mut VvencConfig, *mut std::ffi::c_void, Option<VvencMsgFn>) -> std::ffi::c_void;
            if let Ok(sym) = lib.sym("vvenc_set_msg_callback") {
                let set_msg: FnSetMsg = unsafe { std::mem::transmute(sym) };
                unsafe { set_msg(&mut *cfg, std::ptr::null_mut(), Some(vvenc_msg_cb)); }
            }
        }

        type FnInitDefault = unsafe extern "C" fn(
            *mut VvencConfig, i32, i32, i32, i32, i32, i32,
        ) -> i32;
        let f = unsafe {
            std::mem::transmute::<VoidPtr, FnInitDefault>(lib.sym("vvenc_init_default")?)
        };
        let cfg_ptr = &mut *cfg;
        let rc = unsafe { f(cfg_ptr, width as i32, height as i32, framerate, 0, qp, VVENC_FASTER) };
        if rc != 0 {
            return Err(format!("vvenc_init_default failed: {rc}"));
        }

        type FnSetParam = unsafe extern "C" fn(*mut VvencConfig, *const std::ffi::c_char, *const std::ffi::c_char) -> i32;
        let set = unsafe {
            std::mem::transmute::<VoidPtr, FnSetParam>(lib.sym("vvenc_set_param")?)
        };
        let mut set_str = |name: &str, value: &str| -> Result<(), String> {
            let n = std::ffi::CString::new(name).unwrap();
            let v = std::ffi::CString::new(value).unwrap();
            let rc = unsafe { set(cfg_ptr, n.as_ptr(), v.as_ptr()) };
            if rc != 0 {
                return Err(format!("vvenc_set_param({name}={value}) failed: {rc}"));
            }
            Ok(())
        };

        set_str("preset", "faster")?;
        set_str("threads", &threads.to_string())?;
        if multithread {
            set_str("mtprofile", "3")?;
        } else {
            set_str("mtprofile", "0")?;
        }
        set_str("refreshsec", &refresh_sec.max(1).to_string())?;
        set_str("decodingrefreshtype", "idr")?;
        set_str("gopsize", "16")?;
        set_str("inputbitdepth", "10")?;
        set_str("internalbitdepth", "10")?;
        set_str("POC0IDR", "1")?;
        if let Some(c) = colour {
            let mode = match (c.transfer_name.as_str(), c.matrix_name.as_str()) {
                ("smpte2084", "bt2020nc") => "pq_2020",
                ("smpte2084", _) => "pq",
                ("arib-std-b67", "bt2020nc") => "hlg_2020",
                ("arib-std-b67", _) => "hlg",
                _ => "",
            };
            if !mode.is_empty() {
                set_str("hdr", mode)?;
            }
        }

        type FnCreate = unsafe extern "C" fn() -> EncoderHandle;
        let create = unsafe {
            std::mem::transmute::<VoidPtr, FnCreate>(lib.sym("vvenc_encoder_create")?)
        };
        let enc = unsafe { create() };
        if enc.is_null() {
            return Err("vvenc_encoder_create failed".into());
        }

        type FnOpen = unsafe extern "C" fn(EncoderHandle, *mut VvencConfig) -> i32;
        let open = unsafe {
            std::mem::transmute::<VoidPtr, FnOpen>(lib.sym("vvenc_encoder_open")?)
        };
        let rc = unsafe { open(enc, cfg_ptr) };
        if rc != 0 {
            return Err(format!("vvenc_encoder_open failed: {rc}"));
        }

        #[cfg(have_vvenc)]
        eprintln!("vvenc: using statically linked vvenc {VVENC_VERSION}");
        #[cfg(not(have_vvenc))]
        eprintln!("vvenc: using runtime-loaded library");

        // Payload buffer large enough for any coded AU at these dimensions.
        let payload_cap = width * height * 2;
        let au_payload = vec![0u8; payload_cap];
        let mut au = Box::new(VvencAccessUnit::default());
        au.payload = au_payload.as_ptr() as *mut u8;
        au.payload_size = payload_cap as i32;

        Ok(VvencLib {
            _lib: lib,
            enc,
            yuv: Box::default(),
            au,
            au_payload,
            frames_in: 0,
            width,
            height,
        })
    }

    /// Fetch the parameter sets (VPS/SPS/PPS) that must be written to the
    /// stream before the first frame's access unit.
    pub fn get_headers(&mut self) -> Result<Vec<u8>, String> {
        type FnHdr = unsafe extern "C" fn(EncoderHandle, *mut VvencAccessUnit) -> i32;
        let f: FnHdr = unsafe { std::mem::transmute(self._lib.sym("vvenc_get_headers")?) };
        self.au.payload_used_size = 0;
        let rc = unsafe { f(self.enc, &mut *self.au) };
        if rc != 0 {
            return Err(format!("vvenc_get_headers failed: {rc}"));
        }
        let used = self.au.payload_used_size;
        if used < 0 || used as usize > self.au_payload.len() {
            return Err("vvenc header payload overflow".into());
        }
        Ok(self.au_payload[..used as usize].to_vec())
    }

    /// Encode one frame; returns the access unit if the encoder emitted
    /// one (with reordering enabled, frames are held back until their
    /// temporal layer can be emitted).
    pub fn encode_frame(
        &mut self,
        planes: &[&[u16]; 3],
        plane_w: [i32; 3],
        plane_h: [i32; 3],
    ) -> Result<Option<EncodedAu>, String> {
        self.yuv.planes[0] = VvencYuvPlane {
            ptr: planes[0].as_ptr() as *mut i16,
            width: plane_w[0],
            height: plane_h[0],
            stride: plane_w[0],
        };
        self.yuv.planes[1] = VvencYuvPlane {
            ptr: planes[1].as_ptr() as *mut i16,
            width: plane_w[1],
            height: plane_h[1],
            stride: plane_w[1],
        };
        self.yuv.planes[2] = VvencYuvPlane {
            ptr: planes[2].as_ptr() as *mut i16,
            width: plane_w[2],
            height: plane_h[2],
            stride: plane_w[2],
        };
        self.yuv.sequence_number = self.frames_in;
        self.frames_in += 1;

        type FnEncode = unsafe extern "C" fn(
            EncoderHandle,
            *const VvencYuvBuffer,
            *mut VvencAccessUnit,
            *mut bool,
        ) -> i32;
        let enc_fn = unsafe {
            std::mem::transmute::<VoidPtr, FnEncode>(self._lib.sym("vvenc_encode")?)
        };
        let mut done = false;
        self.au.payload_used_size = 0;
        let rc = unsafe { enc_fn(self.enc, &*self.yuv, &mut *self.au, &mut done) };
        if rc != 0 {
            return Err(format!("vvenc_encode failed: {rc}"));
        }
        // The access unit is valid when the encoder filled the payload
        // (the `done` flag is only meaningful while flushing).
        let used = self.au.payload_used_size;
        if used <= 0 {
            return Ok(None);
        }
        if used as usize > self.au_payload.len() {
            return Err("vvenc AU payload overflow".into());
        }
        Ok(Some(EncodedAu {
            data: self.au_payload[..used as usize].to_vec(),
            poc: self.au.poc,
            rap: self.au.rap,
        }))
    }

    /// Flush the encoder at end of stream; returns all remaining AUs.
    pub fn flush(&mut self) -> Result<Vec<EncodedAu>, String> {
        type FnEncode = unsafe extern "C" fn(
            EncoderHandle,
            *const VvencYuvBuffer,
            *mut VvencAccessUnit,
            *mut bool,
        ) -> i32;
        let enc_fn = unsafe {
            std::mem::transmute::<VoidPtr, FnEncode>(self._lib.sym("vvenc_encode")?)
        };
        let mut out = Vec::new();
        for _ in 0..10000 {
            let mut done = false;
            self.au.payload_used_size = 0;
            let rc = unsafe { enc_fn(self.enc, std::ptr::null(), &mut *self.au, &mut done) };
            if rc != 0 {
                // The encoder reports RESTART_REQUIRED once the flush has
                // already completed; treat that as the end of the stream.
                if !out.is_empty() {
                    break;
                }
                return Err(format!("vvenc flush failed: {rc}"));
            }
            let used = self.au.payload_used_size;
            if used > 0 {
                out.push(EncodedAu {
                    data: self.au_payload[..used as usize].to_vec(),
                    poc: self.au.poc,
                    rap: self.au.rap,
                });
            }
            if done {
                break;
            }
        }
        Ok(out)
    }

    pub fn last_error(&self) -> String {
        type FnErr = unsafe extern "C" fn(EncoderHandle) -> *const std::ffi::c_char;
        if let Ok(sym) = self._lib.sym("vvenc_get_last_error") {
            let f: FnErr = unsafe { std::mem::transmute(sym) };
            let p = unsafe { f(self.enc) };
            if !p.is_null() {
                return unsafe { std::ffi::CStr::from_ptr(p) }.to_string_lossy().into_owned();
            }
        }
        "no error info".into()
    }

    pub fn frames_in(&self) -> u64 {
        self.frames_in
    }

    pub fn config_string(&self) -> String {
        let mut cfg: Box<VvencConfig> = Box::default();
        type FnGetCfg = unsafe extern "C" fn(EncoderHandle, *mut VvencConfig) -> i32;
        if let Ok(sym) = self._lib.sym("vvenc_get_config") {
            let f: FnGetCfg = unsafe { std::mem::transmute(sym) };
            if unsafe { f(self.enc, &mut *cfg) } == 0 {
                type FnStr = unsafe extern "C" fn(*const VvencConfig, i32) -> *const std::ffi::c_char;
                if let Ok(s2) = self._lib.sym("vvenc_get_config_as_string") {
                    let f2: FnStr = unsafe { std::mem::transmute(s2) };
                    let p = unsafe { f2(&*cfg, 6) };
                    if !p.is_null() {
                        return unsafe { std::ffi::CStr::from_ptr(p) }.to_string_lossy().into_owned();
                    }
                }
            }
        }
        "unavailable".into()
    }

    pub fn version(&self) -> String {
        type FnVersion = unsafe extern "C" fn() -> *const std::ffi::c_char;
        if let Ok(sym) = self._lib.sym("vvenc_get_version") {
            let f: FnVersion = unsafe { std::mem::transmute(sym) };
            let p = unsafe { f() };
            if !p.is_null() {
                return unsafe { std::ffi::CStr::from_ptr(p) }.to_string_lossy().into_owned();
            }
        }
        "unknown".into()
    }
}

impl Drop for VvencLib {
    fn drop(&mut self) {
        if !self.enc.is_null() {
            type FnClose = unsafe extern "C" fn(EncoderHandle) -> i32;
            if let Ok(sym) = self._lib.sym("vvenc_encoder_close") {
                let close: FnClose = unsafe { std::mem::transmute(sym) };
                unsafe { close(self.enc); }
            }
            self.enc = std::ptr::null_mut();
        }
    }
}

/// Frame queue shared between the decode pump thread and the caller.
pub struct DecodedQueue {
    pub frames: Mutex<VecDeque<crate::frame::Picture>>,
    pub cv: Condvar,
    pub done: Mutex<bool>,
}

impl DecodedQueue {
    pub fn new() -> Arc<DecodedQueue> {
        Arc::new(DecodedQueue {
            frames: Mutex::new(VecDeque::new()),
            cv: Condvar::new(),
            done: Mutex::new(false),
        })
    }

    /// Pop a decoded frame, waiting for one to arrive.
    pub fn pop(&self) -> Option<crate::frame::Picture> {
        let mut frames = self.frames.lock().unwrap();
        loop {
            if let Some(f) = frames.pop_front() {
                return Some(f);
            }
            if *self.done.lock().unwrap() {
                return None;
            }
            frames = self.cv.wait(frames).unwrap();
        }
    }

    /// Push a decoded frame and wake any waiter.
    pub fn push(&self, f: crate::frame::Picture) {
        self.frames.lock().unwrap().push_back(f);
        self.cv.notify_one();
    }

    /// Mark the stream ended and wake any waiter.
    pub fn end(&self) {
        *self.done.lock().unwrap() = true;
        self.cv.notify_all();
    }
}
