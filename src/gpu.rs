// muguet - gpu.rs
// Constantin Clerc (c22dev) - 0.1

use crate::mem_base_len;
use regex::Regex;
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};
use wasmtime::SharedMemory;
use wgpu::*;

const RING_DATA: u64 = 64;
const BIND_SRV: u32 = 0x8;
const BIND_RT: u32 = 0x20;
const BIND_DS: u32 = 0x40;
const BIND_UAV: u32 = 0x80;
const RING_CHUNK: u64 = 8 << 20;
const UBO_WINDOW: u64 = 65536;
const FRAME_SLOTS: usize = 3;
const ZERO_VB_SIZE: u64 = 4096;
const OCCLUSION_CAPACITY: u32 = 2048;
const OCCLUSION_VISIBLE_COUNT: u64 = 65536;
const BG_LIMIT: usize = if cfg!(target_os = "android") { 4096 } else { 24000 };

pub struct SurfaceHandoff {
    pub instance: Instance,
    pub surface: Surface<'static>,
    pub window: Arc<winit::window::Window>,
}

/// Where the renderer presents. The wasmtime path owns a winit window; the
/// wasm2c engine path presents into an Android SurfaceView, which has no winit
/// window and a size the Java side reports.
pub enum SurfaceSrc {
    Window(Arc<winit::window::Window>),
    Fixed(u32, u32),
}

/// The engine path's render target: the Java SurfaceView's ANativeWindow.
/// The wgpu surface is built from it by the renderer thread, so it always
/// comes from the same instance as the device. The pointer is kept as an
/// integer because a raw pointer is not Send and this lives in a static.
pub struct EngineSurface {
    pub ptr: usize,
    pub size: (u32, u32),
}

pub static SURFACE: Mutex<Option<SurfaceHandoff>> = Mutex::new(None);
/// The engine path's surface. The Java side attaches it before the engine
/// boots; a later change is picked up by the renderer thread through the flag.
pub static ENGINE_SURFACE: Mutex<Option<EngineSurface>> = Mutex::new(None);
pub static ENGINE_SURFACE_CHANGED: AtomicBool = AtomicBool::new(false);
pub static STARTED: AtomicBool = AtomicBool::new(false);

/// The engine's shared memory, as far as the ring protocol is concerned.
///
/// The wasmtime path uses wasmtime's own atomics. The wasm2c engine path uses
/// the engine core's own futex, which the C core registers here at startup:
/// the renderer never links against the engine library, so the dependency
/// stays one way (C -> Rust) and the wake-up semantics stay the engine's.
pub trait RingMem: Send + Sync {
    /// The engine's linear memory.
    fn base(&self) -> *mut u8;
    /// One 32-bit word of that memory, seen as an atomic.
    fn word(&self, addr: u64) -> &AtomicI32 {
        unsafe { &*(self.base().add(addr as usize) as *const AtomicI32) }
    }
    fn wait32(&self, addr: u64, expected: u32, timeout: Option<Duration>) -> u32;
    fn notify(&self, addr: u64) -> u32;
}

struct WasmMem {
    mem: SharedMemory,
}

impl RingMem for WasmMem {
    fn base(&self) -> *mut u8 {
        mem_base_len(&self.mem).0
    }
    fn wait32(&self, addr: u64, expected: u32, timeout: Option<Duration>) -> u32 {
        let _ = self.mem.atomic_wait32(addr, expected, timeout);
        0
    }
    fn notify(&self, addr: u64) -> u32 {
        let _ = self.mem.atomic_notify(addr, u32::MAX);
        0
    }
}

pub type WaitFn = extern "C" fn(u64, u32, i64) -> u32;
pub type NotifyFn = extern "C" fn(u64) -> u32;

static HOOKS: Mutex<(Option<WaitFn>, Option<NotifyFn>)> = Mutex::new((None, None));

/// Called by the engine core once its futex is up. Both pointers are optional:
/// a missing one degrades to a sleep instead of a crash.
#[no_mangle]
pub extern "C" fn muguet_gpu_set_hooks(wait: Option<WaitFn>, notify: Option<NotifyFn>) {
    *HOOKS.lock().unwrap() = (wait, notify);
}

struct EngineMem {
    base: usize,
}

impl RingMem for EngineMem {
    fn base(&self) -> *mut u8 {
        self.base as *mut u8
    }
    fn wait32(&self, addr: u64, expected: u32, timeout: Option<Duration>) -> u32 {
        // Copy the pointer out before the call: the hook blocks, and holding
        // the lock across it would stall the notify side.
        let wait = HOOKS.lock().unwrap().0;
        match wait {
            Some(f) => {
                let ns = timeout.map(|t| t.as_nanos() as i64).unwrap_or(-1);
                f(addr, expected, ns)
            }
            None => {
                std::thread::sleep(Duration::from_millis(1));
                0
            }
        }
    }
    fn notify(&self, addr: u64) -> u32 {
        let notify = HOOKS.lock().unwrap().1;
        match notify {
            Some(f) => f(addr),
            None => 0,
        }
    }
}

pub static OCCLUDED: AtomicBool = AtomicBool::new(false);

pub static SUSPENDED: AtomicBool = AtomicBool::new(false);

pub static PRESENTS: AtomicU64 = AtomicU64::new(0);

pub fn backends() -> Backends {
    if cfg!(target_vendor = "apple") { Backends::METAL } else { Backends::VULKAN }
}

#[cfg(target_os = "android")]
static SURFACE_RELEASE: Mutex<Option<std::sync::mpsc::Sender<()>>> = Mutex::new(None);

#[cfg(target_os = "android")]
pub fn release_android_surface() {
    // Serialize against start() taking the initial surface handoff.
    let mut handoff = SURFACE.lock().unwrap();
    handoff.take();
    if !STARTED.load(Ordering::SeqCst) { return; }
    let (tx, rx) = std::sync::mpsc::channel();
    *SURFACE_RELEASE.lock().unwrap() = Some(tx);
    drop(handoff);
    if rx.recv().is_err() { log("GPU surface release failed"); }
}

pub fn start(mem: &SharedMemory, ring: u64, size: u64, dbg: u64) {
    start_with(Arc::new(WasmMem { mem: mem.clone() }), ring, size, dbg);
}

/// Entry point for the wasm2c engine core.
///
/// The engine's graphics worker calls this instead of the Metal renderer. The
/// ring it passes is the same command stream the browser's wgpu worker
/// consumes, so everything below this point is engine agnostic.
#[no_mangle]
pub extern "C" fn muguet_gpu_start(memory: *mut u8, _memory_size: u64, ring: u64, ring_size: u64, debug: u64) {
    start_with(Arc::new(EngineMem { base: memory as usize }), ring, ring_size, debug);
}

/// Android: the Java SurfaceView hands over its ANativeWindow. The window is
/// only recorded here; the renderer thread builds the wgpu surface from it,
/// because the surface has to belong to the instance that owns the device.
#[no_mangle]
pub extern "C" fn muguet_gpu_attach_window(window: *mut std::ffi::c_void, width: u32, height: u32) {
    #[cfg(target_os = "android")]
    {
        if window.is_null() {
            *ENGINE_SURFACE.lock().unwrap() = None;
            ENGINE_SURFACE_CHANGED.store(true, Ordering::SeqCst);
            return;
        }
        *ENGINE_SURFACE.lock().unwrap() = Some(EngineSurface { ptr: window as usize, size: (width.max(1), height.max(1)) });
        ENGINE_SURFACE_CHANGED.store(true, Ordering::SeqCst);
    }
    #[cfg(not(target_os = "android"))]
    {
        let _ = (window, width, height);
    }
}

/// Builds the engine path's wgpu surface from the Java SurfaceView's window.
#[cfg(target_os = "android")]
fn engine_surface_for(instance: Instance, e: EngineSurface) -> Option<(Surface<'static>, SurfaceSrc)> {
    use winit::raw_window_handle::{AndroidNdkWindowHandle, HasDisplayHandle, HasWindowHandle, RawWindowHandle};
    // wgpu wants the handle to be Send + Sync, so the window pointer is kept
    // as an integer here and turned back into a pointer on the way out.
    struct AndroidWindow {
        ptr: usize,
    }
    impl HasWindowHandle for AndroidWindow {
        fn window_handle(&self) -> Result<winit::raw_window_handle::WindowHandle<'_>, winit::raw_window_handle::HandleError> {
            // The pointer comes from the Java SurfaceView and is never null.
            let ptr = unsafe { std::ptr::NonNull::new_unchecked(self.ptr as *mut std::ffi::c_void) };
            Ok(unsafe {
                winit::raw_window_handle::WindowHandle::borrow_raw(RawWindowHandle::AndroidNdk(AndroidNdkWindowHandle::new(ptr)))
            })
        }
    }
    impl HasDisplayHandle for AndroidWindow {
        fn display_handle(&self) -> Result<winit::raw_window_handle::DisplayHandle<'_>, winit::raw_window_handle::HandleError> {
            Ok(winit::raw_window_handle::DisplayHandle::android())
        }
    }
    if e.ptr == 0 {
        return None;
    }
    match instance.create_surface(SurfaceTarget::DisplayAndWindow(Box::new(AndroidWindow { ptr: e.ptr }))) {
        Ok(s) => Some((s, SurfaceSrc::Fixed(e.size.0, e.size.1))),
        Err(err) => {
            log(&format!("engine surface failed: {err}"));
            None
        }
    }
}

#[cfg(not(target_os = "android"))]
fn engine_surface_for(_instance: Instance, _e: EngineSurface) -> Option<(Surface<'static>, SurfaceSrc)> {
    None
}

/// Starts the renderer thread for either engine.
///
/// The wasmtime path hands over a wasmtime SharedMemory; the wasm2c engine
/// path hands over the engine's own linear memory plus the futex hooks the
/// engine core registered.
fn start_with(mem: Arc<dyn RingMem>, ring: u64, size: u64, dbg: u64) {
    let handoff = {
        let mut surface = SURFACE.lock().unwrap();
        if STARTED.swap(true, Ordering::SeqCst) {
            return;
        }
        surface.take()
    };
    let engine_surface = ENGINE_SURFACE.lock().unwrap().take();
    // The surface is taken here, so the change flag has to be cleared with it:
    // otherwise the run loop would read an empty slot and drop the surface.
    ENGINE_SURFACE_CHANGED.store(false, Ordering::SeqCst);
    let mem2 = mem.clone();
    std::thread::Builder::new()
        .name("gpu".into())
        .stack_size(32 << 20)
        .spawn(move || {
            crate::set_interactive_qos();
            let mut gpu = match Gpu::new(mem2.clone(), ring, size, dbg, handoff, engine_surface) {
                Ok(g) => g,
                Err(e) => {
                    log(&format!("init failed: {e:#}"));
                    #[cfg(target_os = "android")]
                    crate::android::status(&format!("Graphics initialization failed: {e:#}"));
                    drain_forever(mem2.as_ref(), ring);
                }
            };
            gpu.run();
        })
        .expect("spawn gpu thread");
}

fn log(s: &str) {
    eprintln!("[wgpu] {s}");
}

fn drain_forever(mem: &dyn RingMem, ring: u64) -> ! {
    let a = |w: u64| mem.word(ring + 4 * w);
    a(2).store(2, Ordering::SeqCst);
    loop {
        #[cfg(target_os = "android")]
        if let Some(reply) = SURFACE_RELEASE.lock().unwrap().take() { let _ = reply.send(()); }
        let h = a(0).load(Ordering::SeqCst);
        a(1).store(h, Ordering::SeqCst);
        let _ = mem.notify(ring + 4);
        std::thread::sleep(Duration::from_millis(5));
    }
}

struct Fmt {
    format: TextureFormat,
    views: Vec<TextureFormat>,
    a8: bool,
}
fn map_format(f: u32, bind: u32) -> Fmt {
    use TextureFormat as T;
    let ds = bind & BIND_DS != 0;
    let (format, views, a8) = match f {
        2 => (T::Rgba32Float, vec![], false),
        9 | 10 | 11 => (T::Rgba16Float, vec![], false),
        16 => (T::Rg32Float, vec![], false),
        23 | 24 => (T::Rgb10a2Unorm, vec![], false),
        26 => (T::Rg11b10Ufloat, vec![], false),
        27 | 28 => (T::Rgba8Unorm, vec![T::Rgba8UnormSrgb], false),
        29 => (T::Rgba8UnormSrgb, vec![T::Rgba8Unorm], false),
        90 | 87 | 88 => (T::Bgra8Unorm, vec![T::Bgra8UnormSrgb], false),
        91 => (T::Bgra8UnormSrgb, vec![T::Bgra8Unorm], false),
        33 | 34 | 35 => (T::Rg16Float, vec![], false),
        39 => (if ds { T::Depth32Float } else { T::R32Float }, vec![], false),
        40 => (T::Depth32Float, vec![], false),
        41 => (T::R32Float, vec![], false),
        19 | 20 => (T::Depth32FloatStencil8, vec![], false),
        44 | 45 => (T::Depth24PlusStencil8, vec![], false),
        48 | 49 => (T::Rg8Unorm, vec![], false),
        53 => (if ds { T::Depth16Unorm } else { T::R16Float }, vec![], false),
        55 => (T::Depth16Unorm, vec![], false),
        54 => (T::R16Float, vec![], false),
        60 | 61 => (T::R8Unorm, vec![], false),
        63 => (T::R8Snorm, vec![], false),
        51 => (T::Rg8Snorm, vec![], false),
        31 => (T::Rgba8Snorm, vec![], false),
        67 => (T::Rgb9e5Ufloat, vec![], false),
        65 => (T::Rgba8Unorm, vec![], true),
        56 => (T::Depth16Unorm, vec![], false),
        21 | 22 | 46 | 47 => (T::Depth24PlusStencil8, vec![], false),
        70 | 71 => (T::Bc1RgbaUnorm, vec![T::Bc1RgbaUnormSrgb], false),
        72 => (T::Bc1RgbaUnormSrgb, vec![T::Bc1RgbaUnorm], false),
        73 | 74 => (T::Bc2RgbaUnorm, vec![T::Bc2RgbaUnormSrgb], false),
        75 => (T::Bc2RgbaUnormSrgb, vec![T::Bc2RgbaUnorm], false),
        76 | 77 => (T::Bc3RgbaUnorm, vec![T::Bc3RgbaUnormSrgb], false),
        78 => (T::Bc3RgbaUnormSrgb, vec![T::Bc3RgbaUnorm], false),
        79 | 80 => (T::Bc4RUnorm, vec![], false),
        81 => (T::Bc4RSnorm, vec![], false),
        82 | 83 => (T::Bc5RgUnorm, vec![], false),
        84 => (T::Bc5RgSnorm, vec![], false),
        94 | 95 => (T::Bc6hRgbUfloat, vec![], false),
        96 => (T::Bc6hRgbFloat, vec![], false),
        97 | 98 => (T::Bc7RgbaUnorm, vec![T::Bc7RgbaUnormSrgb], false),
        99 => (T::Bc7RgbaUnormSrgb, vec![T::Bc7RgbaUnorm], false),
        other => {
            log_once(&format!("fmt{other}"), &format!("no WebGPU format for DXGI {other} (using rgba8unorm)"));
            (T::Rgba8Unorm, vec![], false)
        }
    };
    Fmt { format, views, a8 }
}

fn is_depth(f: TextureFormat) -> bool {
    f.is_depth_stencil_format() || matches!(f, TextureFormat::Stencil8)
}

const STENCIL_PROXY: TextureFormat = TextureFormat::Rg8Uint;

fn has_stencil(f: TextureFormat) -> bool {
    matches!(f, TextureFormat::Depth24PlusStencil8 | TextureFormat::Depth32FloatStencil8 | TextureFormat::Stencil8)
}
fn is_bc(f: TextureFormat) -> bool {
    f.is_compressed()
}
fn block_bytes(f: TextureFormat) -> u32 {
    if !is_bc(f) { return 0; }
    f.block_copy_size(None).unwrap_or(16)
}
fn texel_bytes(f: TextureFormat) -> u32 {
    if is_bc(f) || is_depth(f) { return 0; }
    f.block_copy_size(None).unwrap_or(0)
}

// D3D reports the logical mip dimensions, including 1x1 and 2x2 BC mips.
// WebGPU copies whole blocks, even when the mip is smaller than one block.
fn copy_extent(format: TextureFormat, w: u32, h: u32, d: u32) -> Extent3d {
    Extent3d { width: w, height: h, depth_or_array_layers: d }.physical_size(format)
}

static LOGGED: LazyLock<Mutex<std::collections::HashSet<String>>> = LazyLock::new(|| Mutex::new(Default::default()));
fn log_once(key: &str, msg: &str) {
    if LOGGED.lock().unwrap().insert(key.to_string()) {
        log(msg);
    }
}

fn cmp(i: u32) -> Option<CompareFunction> {
    use CompareFunction as C;
    Some(match i { 1 => C::Never, 2 => C::Less, 3 => C::Equal, 4 => C::LessEqual, 5 => C::Greater, 6 => C::NotEqual, 7 => C::GreaterEqual, 8 => C::Always, _ => return None })
}
fn stencil_op(i: u32) -> StencilOperation {
    use StencilOperation as S;
    match i { 2 => S::Zero, 3 => S::Replace, 4 => S::IncrementClamp, 5 => S::DecrementClamp, 6 => S::Invert, 7 => S::IncrementWrap, 8 => S::DecrementWrap, _ => S::Keep }
}
fn blend_factor(i: u32, default: BlendFactor) -> BlendFactor {
    use BlendFactor as B;
    match i {
        1 => B::Zero, 2 => B::One, 3 => B::Src, 4 => B::OneMinusSrc, 5 => B::SrcAlpha, 6 => B::OneMinusSrcAlpha,
        7 => B::DstAlpha, 8 => B::OneMinusDstAlpha, 9 => B::Dst, 10 => B::OneMinusDst, 11 => B::SrcAlphaSaturated,
        14 => B::Constant, 15 => B::OneMinusConstant, 16 => B::Src1, 17 => B::OneMinusSrc1, 18 => B::Src1Alpha, 19 => B::OneMinusSrc1Alpha,
        _ => default,
    }
}
fn blend_op(i: u32) -> BlendOperation {
    use BlendOperation as O;
    match i { 2 => O::Subtract, 3 => O::ReverseSubtract, 4 => O::Min, 5 => O::Max, _ => O::Add }
}
fn topology(i: u32) -> (PrimitiveTopology, bool) {
    use PrimitiveTopology as P;
    match i { 1 => (P::PointList, false), 2 => (P::LineList, false), 3 => (P::LineStrip, true), 5 => (P::TriangleStrip, true), _ => (P::TriangleList, false) }
}
fn address(i: u32) -> AddressMode {
    match i { 2 | 5 => AddressMode::MirrorRepeat, 3 | 4 => AddressMode::ClampToEdge, _ => AddressMode::Repeat }
}
fn vertex_format(dxgi: u32) -> Option<VertexFormat> {
    use VertexFormat as V;
    Some(match dxgi {
        2 => V::Float32x4, 6 => V::Float32x3, 16 => V::Float32x2, 41 => V::Float32,
        3 => V::Uint32x4, 7 => V::Uint32x3, 17 => V::Uint32x2, 42 => V::Uint32,
        4 => V::Sint32x4, 8 => V::Sint32x3, 18 => V::Sint32x2, 43 => V::Sint32,
        10 => V::Float16x4, 34 => V::Float16x2,
        11 => V::Unorm16x4, 35 => V::Unorm16x2,
        12 => V::Uint16x4, 36 => V::Uint16x2,
        13 => V::Snorm16x4, 37 => V::Snorm16x2,
        14 => V::Sint16x4, 38 => V::Sint16x2,
        28 => V::Unorm8x4, 30 => V::Uint8x4, 31 => V::Snorm8x4, 32 => V::Sint8x4,
        49 => V::Unorm8x2, 50 => V::Uint8x2, 51 => V::Snorm8x2, 52 => V::Sint8x2,
        87 => V::Unorm8x4Bgra, 24 => V::Unorm10_10_10_2,
        _ => return None,
    })
}
fn wf(w: u32) -> f32 { f32::from_bits(w) }

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind { Uniform, Storage, ReadOnlyStorage, Sampler, SamplerCmp, Texture, StorageTexture }
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Sample { Float, Uint, Sint, Depth, Unfilterable }

#[derive(Clone, Debug)]
struct Binding {
    group: u32,
    binding: u32,
    kind: Kind,
    dim: TextureViewDimension,
    sample: Sample,
    ms: bool,
    format: Option<TextureFormat>,
    format_name: String,
    access: StorageTextureAccess,
}
struct Info {
    bindings: Vec<Binding>,
    outputs: Vec<u32>,
}

static RE_DECL: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"@group\((\d+)u?\)\s*@binding\((\d+)u?\)\s*var(?:<([^>]*)>)?\s+(\w+)\s*:\s*([^;]+);").unwrap());
static RE_TEXTYPE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^texture_(\w+?)(?:<([^>]*)>)?$").unwrap());
static RE_FRAG_SIG: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"@fragment\s*fn\s+main\s*\((?:[^()]|\([^)]*\))*\)\s*->\s*([^{]+)\{").unwrap());
static RE_LOC: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"@location\((\d+)u?\)").unwrap());
// The clip distances the vertex stage writes, and the fields of a struct.
static RE_ENABLE_CLIP: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?m)^\s*enable\s+clip_distances\s*;\s*").unwrap());
static RE_CLIP_FIELD: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"@builtin\(clip_distances\)\s*(\w+)\s*:\s*array<f32,\s*(\d+)u?>\s*,").unwrap());
static RE_STRUCT_FIELD: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?:@\w+\([^)]*\)\s*)*(\w+)\s*:\s*(?:array<[^>]*>|[^,]+),").unwrap());
// The fragment entry point, up to the opening brace.
static RE_FRAG_ENTRY: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(@fragment\s*fn\s+main\s*\()([\s\S]*?)(\)\s*(?:->\s*(?:@\w+\([^)]*\)\s*)*\w+(?:<[^>]*>)?\s*)?\{)").unwrap());
static RE_UNIFORM_DECL: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"@group\((\d+)u?\)\s*@binding\((\d+)u?\)\s*var<uniform>\s+(\w+)\s*:\s*(\w+)\s*;").unwrap());
static RE_ARRAY_N: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"array<[^>]*>,\s*(\d+)u?>").unwrap());
static RE_FIRST_FN: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\n(?:@\w+|fn)\s").unwrap());
static RE_FLAT: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"@location\((\d+)u?\)\s*@interpolate\(flat[^)]*\)").unwrap());
static RE_VS_LOC_MEMBER: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"@location\((\d+)u\)(\s*\n\s*\w+\s*:)").unwrap());
static RE_VS_MAIN: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"@vertex\s*fn\s+main\s*\([\s\S]*?\)\s*->\s*(\w+)\s*\{").unwrap());
static RE_MAIN_INNER: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(\n\s*main_inner\([^;]*\);)").unwrap());

fn view_dim(name: &str) -> TextureViewDimension {
    match name {
        "2d_array" | "2d-array" => TextureViewDimension::D2Array,
        "3d" => TextureViewDimension::D3,
        "cube" => TextureViewDimension::Cube,
        "cube_array" | "cube-array" => TextureViewDimension::CubeArray,
        "1d" => TextureViewDimension::D1,
        _ => TextureViewDimension::D2,
    }
}

fn wgsl_storage_format(name: &str) -> Option<TextureFormat> {
    use TextureFormat as T;
    Some(match name {
        "rgba8unorm" => T::Rgba8Unorm, "rgba8snorm" => T::Rgba8Snorm, "rgba8uint" => T::Rgba8Uint, "rgba8sint" => T::Rgba8Sint,
        "bgra8unorm" => T::Bgra8Unorm, "rgba16uint" => T::Rgba16Uint, "rgba16sint" => T::Rgba16Sint, "rgba16float" => T::Rgba16Float,
        "r32uint" => T::R32Uint, "r32sint" => T::R32Sint, "r32float" => T::R32Float, "rg32uint" => T::Rg32Uint, "rg32sint" => T::Rg32Sint,
        "rg32float" => T::Rg32Float, "rgba32uint" => T::Rgba32Uint, "rgba32sint" => T::Rgba32Sint, "rgba32float" => T::Rgba32Float,
        "r8unorm" => T::R8Unorm, "rg8unorm" => T::Rg8Unorm, "r16float" => T::R16Float, "rg16float" => T::Rg16Float,
        "rgb10a2unorm" => T::Rgb10a2Unorm, "rg11b10ufloat" => T::Rg11b10Ufloat, "r16uint" => T::R16Uint, "rg16uint" => T::Rg16Uint,
        _ => return None,
    })
}
fn wgsl_format_name(f: TextureFormat) -> Option<&'static str> {
    use TextureFormat as T;
    Some(match f {
        T::Rgba8Unorm => "rgba8unorm", T::Rgba8Snorm => "rgba8snorm", T::Rgba8Uint => "rgba8uint", T::Rgba8Sint => "rgba8sint",
        T::Bgra8Unorm => "bgra8unorm", T::Rgba16Uint => "rgba16uint", T::Rgba16Sint => "rgba16sint", T::Rgba16Float => "rgba16float",
        T::R32Uint => "r32uint", T::R32Sint => "r32sint", T::R32Float => "r32float", T::Rg32Uint => "rg32uint", T::Rg32Sint => "rg32sint",
        T::Rg32Float => "rg32float", T::Rgba32Uint => "rgba32uint", T::Rgba32Sint => "rgba32sint", T::Rgba32Float => "rgba32float",
        T::R8Unorm => "r8unorm", T::Rg8Unorm => "rg8unorm", T::R16Float => "r16float", T::Rg16Float => "rg16float",
        T::Rgb10a2Unorm => "rgb10a2unorm", T::Rg11b10Ufloat => "rg11b10ufloat",
        _ => return None,
    })
}

fn parse_wgsl(text: &str) -> Info {
    let mut bindings = Vec::new();
    for m in RE_DECL.captures_iter(text) {
        let group: u32 = m[1].parse().unwrap_or(0);
        let binding: u32 = m[2].parse().unwrap_or(0);
        let space: String = m.get(3).map(|x| x.as_str()).unwrap_or("").chars().filter(|c| !c.is_whitespace()).collect();
        let ty = m[5].trim().to_string();
        let mut b = Binding {
            group, binding, kind: Kind::Texture, dim: TextureViewDimension::D2, sample: Sample::Float, ms: false,
            format: None, format_name: String::new(), access: StorageTextureAccess::WriteOnly,
        };
        if space == "uniform" {
            b.kind = Kind::Uniform;
        } else if space.starts_with("storage") {
            b.kind = if space.contains("read_write") { Kind::Storage } else { Kind::ReadOnlyStorage };
        } else if ty == "sampler" {
            b.kind = Kind::Sampler;
        } else if ty == "sampler_comparison" {
            b.kind = Kind::SamplerCmp;
        } else {
            let (name, arg) = match RE_TEXTYPE.captures(&ty) {
                Some(t) => (t[1].to_string(), t.get(2).map(|x| x.as_str().to_string()).unwrap_or_else(|| "f32".into())),
                None => ("2d".to_string(), "f32".to_string()),
            };
            if let Some(d) = name.strip_prefix("storage_") {
                b.kind = Kind::StorageTexture;
                b.dim = view_dim(d);
                b.format_name = arg.split(',').next().unwrap_or("").trim().to_string();
                b.format = wgsl_storage_format(&b.format_name);
                b.access = if arg.contains("read_write") { StorageTextureAccess::ReadWrite } else if arg.contains("read") { StorageTextureAccess::ReadOnly } else { StorageTextureAccess::WriteOnly };
            } else if let Some(d) = name.strip_prefix("depth_") {
                b.dim = view_dim(d.strip_prefix("multisampled_").unwrap_or(d));
                b.sample = Sample::Depth;
                b.ms = d.starts_with("multisampled");
            } else if name.starts_with("multisampled_") {
                b.dim = TextureViewDimension::D2;
                b.ms = true;
                b.sample = Sample::Unfilterable;
            } else {
                b.dim = view_dim(&name);
                b.sample = match arg.as_str() { "u32" => Sample::Uint, "i32" => Sample::Sint, _ => Sample::Float };
            }
        }
        bindings.push(b);
    }
    let mut outputs = Vec::new();
    if let Some(sig) = RE_FRAG_SIG.captures(text) {
        let r = sig[1].trim();
        if let Some(l) = RE_LOC.captures(r) {
            outputs.push(l[1].parse().unwrap_or(0));
        } else if let Ok(st) = Regex::new(&format!(r"struct\s+{}\s*\{{([^}}]*)\}}", regex::escape(r))) {
            if let Some(s) = st.captures(text) {
                for m in RE_LOC.captures_iter(&s[1]) {
                    outputs.push(m[1].parse().unwrap_or(0));
                }
            }
        }
    }
    Info { bindings, outputs }
}

struct PackMember { binding: u32, size: u64, offset: u64 }
struct Pack { members: Vec<PackMember>, size: u64 }

fn pack_uniforms(text: &str) -> (String, Option<Pack>) {
    struct M { binding: u32, name: String, ty: String, size: u64 }
    let mut members: Vec<M> = Vec::new();
    let mut group = 0;
    let out = RE_UNIFORM_DECL.replace_all(text, |c: &regex::Captures| {
        let b: u32 = c[2].parse().unwrap_or(0);
        if b == 15 { return c[0].to_string(); }
        group = c[1].parse().unwrap_or(0);
        let ty = c[4].to_string();
        let size = Regex::new(&format!(r"struct\s+{}\s*\{{([^}}]*)\}}", regex::escape(&ty)))
            .ok()
            .and_then(|st| st.captures(text).map(|s| s[1].to_string()))
            .and_then(|body| RE_ARRAY_N.captures(&body).map(|a| a[1].parse::<u64>().unwrap_or(1) * 16))
            .unwrap_or(16);
        members.push(M { binding: b, name: c[3].to_string(), ty, size });
        String::new()
    });
    if members.is_empty() {
        return (text.to_string(), None);
    }
    let mut out = out.into_owned();
    members.sort_by_key(|m| m.binding);
    let mut off = 0;
    let mut pm = Vec::new();
    for m in &members {
        pm.push(PackMember { binding: m.binding, size: m.size, offset: off });
        off += m.size;
    }
    for m in &members {
        let re = Regex::new(&format!(r"\b{}\b", regex::escape(&m.name))).unwrap();
        out = re.replace_all(&out, format!("cbpack_v.{}", m.name).as_str()).into_owned();
    }
    let st = format!(
        "struct cbpack_t {{\n{}\n}}\n@group({}u) @binding(0u) var<uniform> cbpack_v : cbpack_t;\n",
        members.iter().map(|m| format!("  {} : {},", m.name, m.ty)).collect::<Vec<_>>().join("\n"),
        group
    );
    let out = match RE_FIRST_FN.find(&out) {
        Some(at) => format!("{}\n{}{}", &out[..at.start()], st, &out[at.start()..]),
        None => out + &st,
    };
    (out, Some(Pack { members: pm, size: off }))
}

fn vs_output_struct(text: &str) -> Option<Regex> {
    let m = RE_VS_MAIN.captures(text)?;
    Regex::new(&format!(r"(struct\s+{}\s*\{{)([^}}]*)(\}})", regex::escape(&m[1]))).ok()
}
fn locations_in(text: &str, vs: bool) -> Vec<u32> {
    let body = if vs {
        vs_output_struct(text).and_then(|re| re.captures(text).map(|m| m[2].to_string())).unwrap_or_default()
    } else {
        RE_FRAG_ENTRY.captures(text).map(|m| m[2].to_string()).unwrap_or_default()
    };
    RE_LOC.captures_iter(&body).filter_map(|x| x[1].parse().ok()).collect()
}
fn remap_locations(text: &str, vs: bool, map: &HashMap<u32, u32>) -> String {
    let swap = |s: &str| -> String {
        RE_LOC_U.replace_all(s, |c: &regex::Captures| {
            let n: u32 = c[1].parse().unwrap_or(0);
            match map.get(&n) { Some(t) => format!("@location({}{})", t, &c[2]), None => c[0].to_string() }
        }).into_owned()
    };
    let re = if vs { vs_output_struct(text) } else { Some(RE_FRAG_ENTRY.clone()) };
    match re {
        Some(re) => re.replace(text, |c: &regex::Captures| format!("{}{}{}", &c[1], swap(&c[2]), &c[3])).into_owned(),
        None => text.to_string(),
    }
}
static RE_LOC_U: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"@location\((\d+)(u?)\)").unwrap());

#[derive(Clone, Copy)]
struct TexDesc { w: u32, h: u32, d: u32, mips: u32, samples: u32, dim: TextureDimension }
struct Proxy {
    tex: Texture,
    format: TextureFormat,
    dirty: bool,
    passes: Vec<(TextureView, BindGroup)>,
}
struct Tex {
    tex: Texture,
    desc: TexDesc,
    format: TextureFormat,
    compressed_source: Option<TextureFormat>,
    views: HashMap<(TextureFormat, u32, u32), TextureView>,
    a8: bool,
    enc: u64,
    in_bg: bool,
    proxy_depth: Option<Proxy>,
    proxy_stencil: Option<Proxy>,
    proxy_fdepth: Option<Proxy>,
    blit_bg: Option<BindGroup>,
    mip_passes: Option<Vec<(TextureView, BindGroup)>>,
}
struct Buf {
    size: u64,
    stride: u32,
    gpu: Option<Buffer>,
    bytes: u64,
    shadow: Option<Vec<u8>>,
    version: u64,
    enc: u64,
    in_bg: bool,
    counter: Option<Buffer>,
}

impl Buf {
    fn update_shadow(&mut self, offset: u64, data: &[u8]) -> anyhow::Result<()> {
        let Some(shadow) = self.shadow.as_mut() else { return Ok(()) };
        let count = (data.len() as u64).min(self.size.saturating_sub(offset));
        if count == 0 { return Ok(()); }
        let from = usize::try_from(offset)?;
        let end = usize::try_from(offset + count)?;
        if shadow.len() < end {
            shadow.try_reserve_exact(end - shadow.len())?;
            shadow.resize(end, 0);
        }
        shadow[from..end].copy_from_slice(&data[..count as usize]);
        self.version += 1;
        Ok(())
    }

    fn copy_shadow(&self, offset: u64, output: &mut [u8]) {
        // Preserve the zero-filled tail of the former full-size shadow buffer.
        output.fill(0);
        let Some(shadow) = self.shadow.as_ref() else { return };
        let Ok(from) = usize::try_from(offset) else { return };
        let count = (output.len() as u64).min(self.size.saturating_sub(offset))
            .min(shadow.len().saturating_sub(from) as u64) as usize;
        if count > 0 { output[..count].copy_from_slice(&shadow[from..from + count]); }
    }
}

// Bound both cached descriptors and their FIFO bookkeeping. Replacing a key
// does not add another queue entry; dropping an entry releases our ownership,
// while handles already recorded in a GPU submission remain alive in wgpu.
struct FifoCache<V> {
    entries: HashMap<Vec<u64>, V>,
    order: VecDeque<Vec<u64>>,
    capacity: usize,
    evictions: u64,
}

impl<V> FifoCache<V> {
    fn new(capacity: usize) -> Self {
        Self { entries: HashMap::new(), order: VecDeque::new(), capacity, evictions: 0 }
    }
    fn get(&self, key: &[u64]) -> Option<&V> { self.entries.get(key) }
    fn len(&self) -> usize { self.entries.len() }
    fn insert(&mut self, key: Vec<u64>, value: V) {
        if self.capacity == 0 { return; }
        if let Some(existing) = self.entries.get_mut(&key) {
            *existing = value;
            return;
        }
        while self.entries.len() >= self.capacity {
            let oldest = self.order.pop_front().expect("cached entries have FIFO keys");
            self.entries.remove(&oldest);
            self.evictions += 1;
        }
        self.order.push_back(key.clone());
        self.entries.insert(key, value);
    }
}
#[derive(Clone, Copy)]
struct SigEl { sem: u32, idx: u32, reg: u32 }
struct ShaderObj {
    stage: u32,
    hash: String,
    sig: Vec<SigEl>,
    program: Option<Option<Rc<Program>>>,
}
struct Program {
    module: ShaderModule,
    info: Info,
    consts: Option<Buffer>,
    pack: Option<Pack>,
    text: String,
    flat: Vec<u32>,
    group_bindings: BTreeMap<u32, Vec<Binding>>,
    variants: RefCell<HashMap<String, Option<ShaderModule>>>,
    label: String,
    uid: u64,
    vid: bool,
    iid: bool,
    out_locs: Vec<u32>,
    in_locs: Vec<u32>,
    // Set when the vertex stage wrote clip distances that the device cannot do
    // in hardware: they became ordinary varyings and the pixel stage has to
    // discard the fragments outside the clip volume.
    clip: Option<ClipInfo>,
}
#[derive(Clone)]
struct ClipInfo {
    count: usize,
    locations: Vec<u32>,
}
#[derive(Clone, Copy)]
struct El { sem: u32, idx: u32, fmt: u32, slot: u32, offset: u32, cls: u32 }
struct LayoutObj { els: Vec<El>, slots: Vec<(u32, u32)> }
struct StateObj { words: Vec<u32>, sampler: Option<Sampler> }
struct ViewObj {
    res: u32,
    is_buf: bool,
    fmt: u32,
    vdim: u32,
    a: u32,
    b: u32,
    c: u32,
    d: u32,
    views: HashMap<String, TextureView>,
}
struct QueryObj { qtype: u32, addr: u64, slots: Vec<u32> }

enum Obj {
    Tex(Tex),
    Buf(Buf),
    Shader(ShaderObj),
    Layout(Rc<LayoutObj>),
    State(Rc<StateObj>),
    Srv(ViewObj),
    Uav(ViewObj),
    Query(QueryObj),
}

#[derive(Clone, Default)]
struct Vb { buf: u32, stride: u32, offset: u32 }
#[derive(Clone, Default)]
struct Cb { buf: u32, first: u32 }
#[derive(Clone, Default, PartialEq)]
struct Rt { tex: u32, mip: u32, slice: u32, fmt: u32 }
#[derive(Clone, Default, PartialEq)]
struct Ds { tex: u32, mip: u32, slice: u32, fmt: u32, flags: u32 }

struct State {
    shader: [u32; 3],
    layout: u32,
    topo: u32,
    vb: Vec<Vb>,
    ib: (u32, u32, u32),
    cb: [Vec<Cb>; 3],
    srv: [Vec<u32>; 3],
    uav: [Vec<u32>; 3],
    smp: [Vec<u32>; 3],
    rt: Vec<Rt>,
    ds: Ds,
    blend: (u32, [f64; 4]),
    dss: (u32, u32),
    rs: u32,
    viewports: Vec<[f32; 6]>,
    scissors: Vec<[u32; 4]>,
}
impl State {
    fn new() -> Self {
        State {
            shader: [0; 3], layout: 0, topo: 4,
            vb: vec![Vb::default(); 16], ib: (0, 0, 0),
            cb: std::array::from_fn(|_| vec![Cb::default(); 16]),
            srv: std::array::from_fn(|_| vec![0; 128]),
            uav: std::array::from_fn(|_| vec![0; 8]),
            smp: std::array::from_fn(|_| vec![0; 16]),
            rt: vec![Rt::default(); 8], ds: Ds::default(),
            blend: (0, [1.0; 4]), dss: (0, 0), rs: 0,
            viewports: vec![], scissors: vec![],
        }
    }
}

#[derive(Clone)]
struct PassInfo {
    rt_formats: Vec<Option<TextureFormat>>,
    rt_ids: Vec<u32>,
    samples: u32,
    ds_format: Option<TextureFormat>,
    ro_depth: bool,
    ro_stencil: bool,
    width: u32,
    height: u32,
}
#[derive(Default)]
struct PassSt {
    pl: Option<u64>,
    bg: [Option<(u64, u32)>; 2],
    vb: Vec<Option<(u64, u64)>>,
    ib: Option<(u64, IndexFormat, u64)>,
    vp: Option<[f32; 6]>,
    sc: Option<[u32; 4]>,
    bf: Option<[f64; 4]>,
    sref: Option<u32>,
}

struct Pl {
    pipeline: RenderPipeline,
    uid: u64,
    vb_req: Vec<VbReq>,
    label: String,
}

struct PipeJob {
    label: String,
    layout: PipelineLayout,
    vmod: ShaderModule,
    fmod: Option<ShaderModule>,
    vbs: Vec<Option<(u64, VertexStepMode, Vec<VertexAttribute>)>>,
    primitive: PrimitiveState,
    depth_stencil: Option<DepthStencilState>,
    multisample: MultisampleState,
    targets: Vec<Option<ColorTargetState>>,
}
enum PipeEntry { Pending, Done(Option<Rc<Pl>>) }

// A validation-only scope misses driver/compiler errors and allocation failures.
// Device loss is reported separately by wgpu, so also check the callback's flag
// before accepting a newly created resource.
fn gpu_scoped<T>(device: &Device, lost: &AtomicBool, what: &str, f: impl FnOnce(&Device) -> T) -> Option<T> {
    if lost.load(Ordering::SeqCst) { return None; }
    let oom = device.push_error_scope(ErrorFilter::OutOfMemory);
    let internal = device.push_error_scope(ErrorFilter::Internal);
    let validation = device.push_error_scope(ErrorFilter::Validation);
    let v = f(device);
    let errors = [pollster::block_on(validation.pop()), pollster::block_on(internal.pop()), pollster::block_on(oom.pop())];
    let mut failed = lost.load(Ordering::SeqCst);
    if failed && errors.iter().all(Option::is_none) {
        log_once(&format!("lost:{what}"), &format!("{what} failed: device lost during resource creation (see device-loss message)"));
    }
    for e in errors.into_iter().flatten() {
        failed = true;
        let msg = error_chain(&e);
        let key: String = msg.chars().take(160).collect();
        let detail: String = msg.chars().take(1500).collect();
        log_once(&format!("{what}:{key}"), &format!("{what} failed: {detail}"));
    }
    if failed { None } else { Some(v) }
}

fn compile_pipeline(device: &Device, lost: &AtomicBool, job: &PipeJob) -> Option<RenderPipeline> {
    let vbs: Vec<Option<VertexBufferLayout>> = job.vbs.iter().map(|v| v.as_ref().map(|(stride, step, attrs)| VertexBufferLayout { array_stride: *stride, step_mode: *step, attributes: attrs })).collect();
    let desc = RenderPipelineDescriptor {
        label: Some(&job.label),
        layout: Some(&job.layout),
        vertex: VertexState { module: &job.vmod, entry_point: Some("main"), compilation_options: Default::default(), buffers: &vbs },
        primitive: job.primitive,
        depth_stencil: job.depth_stencil.clone(),
        multisample: job.multisample,
        fragment: job.fmod.as_ref().map(|m| FragmentState { module: m, entry_point: Some("main"), compilation_options: Default::default(), targets: &job.targets }),
        multiview_mask: None,
        cache: None,
    };
    let what = format!("pipeline {} (targets {:?}, depth {:?}, samples {})", job.label, job.targets, job.depth_stencil.as_ref().map(|d| d.format), job.multisample.count);
    gpu_scoped(device, lost, &what, |d| d.create_render_pipeline(&desc))
}

#[derive(Clone, Copy)]
struct VbReq { slot: u32, instance: bool, stride: u64, last: u64 }

#[derive(Clone, Copy, Default)]
struct PendingClear { color: Option<Color>, depth: Option<f32>, stencil: Option<u32> }

struct Chunk { buf: Buffer, id: u64, base: u64, stage: Vec<u8>, lo: u64, hi: u64 }
struct BgResult { bg: BindGroup, uid: u64, dyn_off: Option<u32> }

struct PendingMap {
    buf: Buffer,
    // 0 = pending, 1 = mapped, -1 = failed. A failed map has no range to read
    // and must not be unmapped, particularly after the device destroys it.
    status: Arc<AtomicI32>,
    kind: MapKind,
}
enum MapKind {
    Readback { flag: u64, rows: Vec<(u64, u64, u64)> },
    Occlusion { list: Vec<(u64, u32, u64, bool, usize)> },
}

pub struct Gpu {
    mem: Arc<dyn RingMem>,
    base: *mut u8,
    ring: u64,
    device: Device,
    queue: Queue,
    adapter: Adapter,
    surface: Option<(Surface<'static>, SurfaceSrc)>,
    surface_format: TextureFormat,
    surface_size: (u32, u32),
    objs: Vec<Option<Obj>>,
    enc: Option<CommandEncoder>,
    enc_epoch: u64,
    pass: Option<RenderPass<'static>>,
    pass_key: Option<Vec<u32>>,
    pass_info: Option<PassInfo>,
    pass_st: PassSt,
    s: State,
    pdirty: bool,
    last_pl: Option<Rc<Pl>>,
    shader_index: HashMap<String, serde_json::Value>,
    shader_dir: PathBuf,
    next_uid: u64,
    bgl_cache: HashMap<(String, u32, u32), (BindGroupLayout, u64)>,
    empty_bgl: Option<(BindGroupLayout, u64)>,
    f32_filterable: bool,
    pipelines: HashMap<Vec<i64>, PipeEntry>,
    pipe_jobs: Option<std::sync::mpsc::Sender<(u64, PipeJob)>>,
    pipe_results: std::sync::mpsc::Receiver<(u64, Option<RenderPipeline>)>,
    pipe_pending: HashMap<u64, (Vec<i64>, Vec<VbReq>, String)>,
    pending_skips: u64,
    pending_skips0: u64,
    compute: HashMap<String, Option<(ComputePipeline, BindGroupLayout, u64, Rc<Program>)>>,
    bind_groups: FifoCache<(BindGroup, u64)>,
    ubo_slots: Vec<Vec<Chunk>>,
    ubo_shared: Vec<(Buffer, u64)>,
    ubo_slot: usize,
    ubo_chunk: usize,
    ubo_off: u64,
    dummy_tex: HashMap<String, TextureView>,
    dummy_smp: Option<Sampler>,
    dummy_cmp: Option<Sampler>,
    dummy_bufs: HashMap<u32, Buffer>,
    zero_vb: Option<Buffer>,
    zero_src: Option<Buffer>,
    scratch: Option<Buffer>,
    proxy_pipelines: HashMap<&'static str, RenderPipeline>,
    f2d_pipeline: Option<RenderPipeline>,
    blit: Option<RenderPipeline>,
    mip_pipelines: HashMap<TextureFormat, RenderPipeline>,
    mip_sampler: Option<Sampler>,
    occlusion_set: Option<QuerySet>,
    occlusion_next: u32,
    occlusion_active: Option<u32>,
    occlusion_pass_slot: Option<u32>,
    pending_resolve: Vec<(u32, u32, Vec<u32>)>,
    pending_events: Vec<(u32, u32, u64)>,
    pending_maps: Vec<PendingMap>,
    counter_src: Option<Buffer>,
    counter_slot: u32,
    frame_slot_done: Vec<Option<SubmissionIndex>>,
    last_submit: Option<SubmissionIndex>,
    vb_avail: [u64; 16],
    dbg_idx: u64,

    draws: u64,
    skipped: u64,
    presents: u64,
    pipe_creates: u64,
    fps_t0: Instant,
    fps_n0: u64,
    fps_d0: u64,
    errors: Arc<AtomicU64>,
    lost: Arc<AtomicBool>,

    inflight: Arc<AtomicU64>,
    instance: Instance,

    pending_clears: HashMap<u32, Vec<(u32, u32, PendingClear)>>,
    clears_folded: u64,
    submits: u64,
    submits0: u64,
    passes: u64,
    passes0: u64,
}

unsafe impl Send for Gpu {}

pub fn max_msaa() -> u32 {
    static CACHED: LazyLock<u32> = LazyLock::new(max_msaa_uncached);
    *CACHED
}

fn max_msaa_uncached() -> u32 {
    let mut d = InstanceDescriptor::new_without_display_handle();
    d.backends = backends();
    let Ok(adapter) = pollster::block_on(Instance::new(d).request_adapter(&RequestAdapterOptions::default())) else { return msaa_cap(4) };
    let formats = [TextureFormat::Rgba8Unorm, TextureFormat::Rgba16Float, TextureFormat::Rg11b10Ufloat, TextureFormat::Depth32FloatStencil8];
    let supported = [16, 8, 4, 2]
        .into_iter()
        .find(|&n| formats.iter().all(|f| adapter.get_texture_format_features(*f).flags.sample_count_supported(n)))
        .unwrap_or(1);
    msaa_cap(supported)
}

// MSAA is the largest single fragment-cost lever on phone GPUs: 4x costs
// substantially more than the desktop default returns. Desktop keeps the
// adapter maximum; Android is capped unless MUGUET_MSAA_MAX overrides it.
fn msaa_cap(supported: u32) -> u32 {
    let override_ = std::env::var("MUGUET_MSAA_MAX").ok().and_then(|v| v.parse::<u32>().ok());
    let cap = override_.unwrap_or(if cfg!(target_os = "android") { 2 } else { u32::MAX });
    supported.min(cap).max(1)
}

impl Gpu {
    fn new(mem: Arc<dyn RingMem>, ring: u64, _size: u64, dbg: u64, handoff: Option<SurfaceHandoff>, engine_surface: Option<EngineSurface>) -> anyhow::Result<Gpu> {
        // The memory the engine and the renderer share. Both engines hand over
        // their own, and the ring lives inside it.
        let base = mem.base();
        let (instance, surface) = match handoff {
            Some(h) => (h.instance, Some((h.surface, SurfaceSrc::Window(h.window)))),
            None => {
                let mut d = InstanceDescriptor::new_without_display_handle();
                d.backends = backends();
                let instance = Instance::new(d);
                // The engine path's surface must come from this instance: a
                // surface and its device cannot live in different ones.
                match engine_surface.and_then(|e| engine_surface_for(instance.clone(), e)) {
                    Some(s) => (instance, Some(s)),
                    None => (instance, None),
                }
            }
        };
        let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            compatible_surface: surface.as_ref().map(|s| &s.0),
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))?;
        let want = Features::TEXTURE_COMPRESSION_BC | Features::TEXTURE_COMPRESSION_BC_SLICED_3D | Features::DEPTH32FLOAT_STENCIL8
            | Features::FLOAT32_FILTERABLE | Features::INDIRECT_FIRST_INSTANCE | Features::RG11B10UFLOAT_RENDERABLE
            | Features::BGRA8UNORM_STORAGE | Features::DEPTH_CLIP_CONTROL | Features::CLIP_DISTANCES | Features::DUAL_SOURCE_BLENDING

            | Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
        let info = adapter.get_info();
        let mut features = want & adapter.features();
        // The Adreno Vulkan driver loses the device when a shader uses clip
        // distances ("Unexpected error variant (driver implementation is at
        // fault)"), which leaves the world black. The software emulation draws
        // the same picture, so it is used there automatically.
        let adreno = driver_needs_software_clip(&info.name, &info.driver, &info.driver_info);
        let software_clip = match std::env::var("MUGUET_GPU_SOFTWARE_CLIP").as_deref() {
            Ok("1") => true,
            Ok("0") => false,
            _ => adreno,
        };
        let cpu_bc = std::env::var("MUGUET_GPU_CPU_BC").map(|v| v == "1").unwrap_or(false);
        if software_clip { features &= !Features::CLIP_DISTANCES; }
        if cpu_bc { features &= !(Features::TEXTURE_COMPRESSION_BC | Features::TEXTURE_COMPRESSION_BC_SLICED_3D); }
        let (device, queue) = pollster::block_on(adapter.request_device(&DeviceDescriptor {
            label: Some("muguet"),
            required_features: features,
            required_limits: adapter.limits(),
            ..Default::default()
        }))?;
        let errors = Arc::new(AtomicU64::new(0));
        let lost = Arc::new(AtomicBool::new(false));
        {
            let lost = lost.clone();
            device.set_device_lost_callback(move |reason, message| {
                lost.store(true, Ordering::SeqCst);
                log(&format!("device lost: {reason:?}: {message}"));
                #[cfg(target_os = "android")]
                crate::android::status(&format!("Graphics device lost: {message}. Export the logs from the launcher."));
            });
        }
        {
            let errors = errors.clone();
            let mem2 = mem.clone();
            let kinds: Mutex<HashMap<String, u64>> = Mutex::new(HashMap::new());
            device.on_uncaptured_error(Arc::new(move |e: Error| {
                errors.fetch_add(1, Ordering::Relaxed);
                mem2.word(ring + 16).fetch_add(1, Ordering::Relaxed);
                let msg = error_chain(&e);
                let key: String = msg.chars().filter(|c| !c.is_ascii_digit()).take(200).collect();
                let mut k = kinds.lock().unwrap();
                let n = k.entry(key).or_insert(0);
                *n += 1;
                if *n == 1 && k.len() <= 60 {
                    log(&format!("validation error: {}", &msg[..msg.len().min(900)]));
                }
            }));
        }
        log(&format!("device ready: {} ({:?}), driver {} {}, features {:?}", info.name, info.backend, info.driver, info.driver_info, features));
        if software_clip {
            log(&format!("clip distances: software emulation ({})", if adreno { "Adreno driver" } else { "requested" }));
        } else {
            log("clip distances: hardware");
        }
        if cpu_bc { log("diagnostic: CPU BC decoding requested"); }
        if !features.contains(Features::TEXTURE_COMPRESSION_BC) {
            log("BC textures will be decoded on CPU (higher memory use and loading cost)");
        }
        let surface_format = surface
            .as_ref()
            .map(|(s, _)| {
                let caps = s.get_capabilities(&adapter);
                caps.formats.iter().copied().find(|f| !f.is_srgb()).unwrap_or(caps.formats[0])
            })
            .unwrap_or(TextureFormat::Bgra8Unorm);
        let shader_dir = PathBuf::from(std::env::var("MUGUET_SHADERS").unwrap_or_default());
        let index_text = std::fs::read_to_string(shader_dir.join("index.json")).unwrap_or_else(|e| {
            log(&format!("no shader index at {}: {e}", shader_dir.display()));
            "{}".into()
        });
        let shader_index: HashMap<String, serde_json::Value> = serde_json::from_str(&index_text).unwrap_or_default();
        log(&format!("shader index: {} entries from {}", shader_index.len(), shader_dir.display()));

        let (res_tx, pipe_results) = std::sync::mpsc::channel::<(u64, Option<RenderPipeline>)>();
        let pipe_jobs = {
            let (job_tx, job_rx) = std::sync::mpsc::channel::<(u64, PipeJob)>();
            let job_rx = Arc::new(Mutex::new(job_rx));
            for i in 0..4 {
                let (dev, lost, rx, tx) = (device.clone(), lost.clone(), job_rx.clone(), res_tx.clone());
                std::thread::Builder::new().name(format!("pipeline-compile-{i}")).spawn(move || loop {
                    let job = rx.lock().unwrap().recv();
                    let Ok((id, job)) = job else { return };
                    let p = compile_pipeline(&dev, &lost, &job);
                    if tx.send((id, p)).is_err() { return; }
                }).expect("spawn compile thread");
            }
            Some(job_tx)
        };
        Ok(Gpu {
            instance: instance.clone(), pending_clears: HashMap::new(), clears_folded: 0, pipe_jobs, pipe_results, pipe_pending: HashMap::new(), pending_skips: 0, pending_skips0: 0,
            base, mem, ring, device, queue, adapter, surface, surface_format, surface_size: (0, 0),
            objs: Vec::new(), enc: None, enc_epoch: 1, pass: None, pass_key: None, pass_info: None, pass_st: PassSt::default(),
            s: State::new(), pdirty: true, last_pl: None, shader_index, shader_dir, next_uid: 1,
            bgl_cache: HashMap::new(), empty_bgl: None, f32_filterable: features.contains(Features::FLOAT32_FILTERABLE), pipelines: HashMap::new(), compute: HashMap::new(), bind_groups: FifoCache::new(BG_LIMIT),
            ubo_slots: (0..FRAME_SLOTS).map(|_| Vec::new()).collect(), ubo_shared: Vec::new(), ubo_slot: 0, ubo_chunk: 0, ubo_off: 0,
            dummy_tex: HashMap::new(), dummy_smp: None, dummy_cmp: None, dummy_bufs: HashMap::new(), zero_vb: None, zero_src: None, scratch: None,
            proxy_pipelines: HashMap::new(), f2d_pipeline: None, blit: None, mip_pipelines: HashMap::new(), mip_sampler: None,
            occlusion_set: None, occlusion_next: 0, occlusion_active: None, occlusion_pass_slot: None,
            pending_resolve: Vec::new(), pending_events: Vec::new(), pending_maps: Vec::new(), counter_src: None, counter_slot: 0,
            frame_slot_done: (0..FRAME_SLOTS).map(|_| None).collect(), last_submit: None, vb_avail: [0; 16],
            dbg_idx: if dbg != 0 { dbg + 424 } else { 0 },
            draws: 0, skipped: 0, presents: 0, pipe_creates: 0, fps_t0: Instant::now(), fps_n0: 0, fps_d0: 0, errors, lost, inflight: Arc::new(AtomicU64::new(0)), submits: 0, submits0: 0, passes: 0, passes0: 0,
        })
    }

    #[inline]
    fn u(&self, w: u64) -> u32 {
        unsafe { std::ptr::read_volatile(self.base.add((w * 4) as usize) as *const u32) }
    }
    #[inline]
    fn f(&self, w: u64) -> f32 {
        f32::from_bits(self.u(w))
    }
    fn bytes(&self, addr: u64, n: u64) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.base.add(addr as usize), n as usize) }
    }
    fn hdr(&self, w: u64) -> &AtomicI32 {
        self.mem.word(self.ring + 4 * w)
    }
    fn uid(&mut self) -> u64 {
        self.next_uid += 1;
        self.next_uid
    }

    fn obj(&self, id: u32) -> Option<&Obj> {
        self.objs.get(id as usize).and_then(|o| o.as_ref())
    }
    fn obj_mut(&mut self, id: u32) -> Option<&mut Obj> {
        self.objs.get_mut(id as usize).and_then(|o| o.as_mut())
    }
    fn set_obj(&mut self, id: u32, o: Obj) {
        let i = id as usize;
        if i > 64 << 20 {
            log_once("hugeid", &format!("object id {id} out of range: ignored"));
            return;
        }
        if self.objs.len() <= i {
            self.objs.resize_with(i + 1, || None);
        }
        self.objs[i] = Some(o);
    }
    fn tex(&self, id: u32) -> Option<&Tex> {
        match self.obj(id) { Some(Obj::Tex(t)) => Some(t), _ => None }
    }
    fn tex_mut(&mut self, id: u32) -> Option<&mut Tex> {
        match self.obj_mut(id) { Some(Obj::Tex(t)) => Some(t), _ => None }
    }
    fn buf(&self, id: u32) -> Option<&Buf> {
        match self.obj(id) { Some(Obj::Buf(b)) => Some(b), _ => None }
    }
    fn buf_mut(&mut self, id: u32) -> Option<&mut Buf> {
        match self.obj_mut(id) { Some(Obj::Buf(b)) => Some(b), _ => None }
    }
    fn view_obj(&self, id: u32) -> Option<&ViewObj> {
        match self.obj(id) { Some(Obj::Srv(v)) | Some(Obj::Uav(v)) => Some(v), _ => None }
    }
    fn view_obj_mut(&mut self, id: u32) -> Option<&mut ViewObj> {
        match self.obj_mut(id) { Some(Obj::Srv(v)) | Some(Obj::Uav(v)) => Some(v), _ => None }
    }
    fn state(&self, id: u32) -> Option<Rc<StateObj>> {
        match self.obj(id) { Some(Obj::State(s)) => Some(s.clone()), _ => None }
    }
    fn use_tex(&mut self, id: u32) {
        let e = self.enc_epoch;
        if let Some(t) = self.tex_mut(id) { t.enc = e; }
    }
    fn use_buf(&mut self, id: u32) {
        let e = self.enc_epoch;
        if let Some(b) = self.buf_mut(id) { b.enc = e; }
    }
    fn used(&self, id: u32) -> bool {
        match self.obj(id) {
            Some(Obj::Tex(t)) => t.enc == self.enc_epoch,
            Some(Obj::Buf(b)) => b.enc == self.enc_epoch,
            _ => false,
        }
    }

    fn scoped<T>(&self, what: &str, f: impl FnOnce(&Device) -> T) -> Option<T> {
        gpu_scoped(&self.device, &self.lost, what, f)
    }

    fn encoder(&mut self) -> &mut CommandEncoder {
        if self.enc.is_none() {
            self.enc = Some(self.device.create_command_encoder(&CommandEncoderDescriptor::default()));
        }
        self.enc.as_mut().unwrap()
    }
    fn end_pass(&mut self) {
        if let Some(mut p) = self.pass.take() {
            if self.occlusion_pass_slot.take().is_some() {
                p.end_occlusion_query();
            }
            drop(p);
            self.pass_key = None;
            self.pass_info = None;
            self.pass_st = PassSt::default();
        }
    }
    fn flush(&mut self) {
        self.end_pass();
        if self.lost.load(Ordering::SeqCst) {
            // Release engine waits without submitting to an invalid device.
            self.enc.take();
            for (id, seq, _) in std::mem::take(&mut self.pending_resolve) {
                if let Some(Obj::Query(q)) = self.obj(id) { self.query_store(q.addr, OCCLUSION_VISIBLE_COUNT, seq); }
            }
            for (id, seq, value) in std::mem::take(&mut self.pending_events) {
                if let Some(Obj::Query(q)) = self.obj(id) { self.query_store(q.addr, value, seq); }
            }
            self.process_maps(false);
            return;
        }
        self.flush_uniforms();
        let resolves = if self.pending_resolve.is_empty() { None } else { self.record_resolves() };
        if let Some(enc) = self.enc.take() {
            let idx = self.queue.submit([enc.finish()]);
            self.last_submit = Some(idx);
            self.submits += 1;
            self.on_submitted();
            self.counter_slot = 0;
        }
        if let Some(r) = resolves {
            self.map_pending(r.buf, r.kind);
        }
        self.enc_epoch += 1;
    }

    fn run(&mut self) -> ! {
        let mut tail = RING_DATA as i32;
        self.hdr(1).store(tail, Ordering::SeqCst);
        self.hdr(2).store(1, Ordering::SeqCst);
        log("running");
        loop {
            #[cfg(target_os = "android")]
            if let Some(reply) = SURFACE_RELEASE.lock().unwrap().take() {
                self.flush();
                let _ = self.device.poll(PollType::Wait { submission_index: None, timeout: None });
                self.surface = None;
                self.surface_size = (0, 0);
                let _ = reply.send(());
            }
            if SUSPENDED.load(Ordering::Relaxed) {
                #[cfg(target_os = "android")]
                {
                    std::thread::sleep(Duration::from_millis(5));
                    continue;
                }
                #[cfg(not(target_os = "android"))]
                {
                let _ = self.device.poll(PollType::Wait { submission_index: None, timeout: None });
                while SUSPENDED.load(Ordering::Relaxed) { std::thread::sleep(Duration::from_millis(50)); }
                }
            }
            #[cfg(target_os = "android")]
            if let Some(h) = SURFACE.lock().unwrap().take() {
                let caps = h.surface.get_capabilities(&self.adapter);
                if caps.formats.contains(&self.surface_format) {
                    self.surface = Some((h.surface, SurfaceSrc::Window(h.window)));
                    self.surface_size = (0, 0);
                } else {
                    crate::android::status("The resumed surface is incompatible with the GPU; restart the game.");
                }
            }
            // The engine path: the Java SurfaceView is recreated on rotation
            // and after backgrounding, so its window arrives as an event.
            #[cfg(target_os = "android")]
            if ENGINE_SURFACE_CHANGED.swap(false, Ordering::SeqCst) {
                let next = ENGINE_SURFACE.lock().unwrap().take();
                self.flush();
                let _ = self.device.poll(PollType::Wait { submission_index: None, timeout: None });
                self.surface = None;
                self.surface_size = (0, 0);
                if let Some(e) = next {
                    if let Some((s, src)) = engine_surface_for(self.instance.clone(), e) {
                        let caps = s.get_capabilities(&self.adapter);
                        self.surface_format = caps.formats.iter().copied().find(|f| !f.is_srgb()).unwrap_or(caps.formats[0]);
                        self.surface = Some((s, src));
                    }
                }
            }
            self.process_maps(false);
            if self.inflight.load(Ordering::SeqCst) > 0 {
                let _ = self.device.poll(PollType::Poll);
            }
            let head = self.hdr(0).load(Ordering::SeqCst);
            if head == tail {
                if !self.pending_events.is_empty() || !self.pending_resolve.is_empty() {
                    self.flush();
                }
                self.hdr(5).store(1, Ordering::SeqCst);
                if self.hdr(0).load(Ordering::SeqCst) != tail {
                    self.hdr(5).store(0, Ordering::SeqCst);
                    continue;
                }
                let timeout = if self.pending_maps.is_empty() && self.inflight.load(Ordering::SeqCst) == 0 { 100 } else { 1 };
                let _ = self.mem.wait32(self.ring, tail as u32, Some(Duration::from_millis(timeout)));
                self.hdr(5).store(0, Ordering::SeqCst);
                continue;
            }
            tail = self.run_commands(tail, head);
            self.hdr(1).store(tail, Ordering::SeqCst);
            let _ = self.mem.notify(self.ring + 4);
        }
    }

    fn run_commands(&mut self, mut tail: i32, head: i32) -> i32 {
        let ring_w = self.ring / 4;
        let mut since = 0u32;
        let mut executed = 0i32;
        while tail != head {
            let base = ring_w + (tail as u64 >> 2);
            let op = self.u(base);
            let n = self.u(base + 1) as u64;
            if op == 1 {
                tail = RING_DATA as i32;
                continue;
            }
            let p = base + 2;
            self.exec(op, p, n);
            executed += 1;
            tail += (n * 4 + 8) as i32;
            since += 1;
            if since >= 512 {
                since = 0;
                self.hdr(3).fetch_add(executed, Ordering::Relaxed);
                executed = 0;
                self.hdr(1).store(tail, Ordering::SeqCst);
                let _ = self.mem.notify(self.ring + 4);
                self.process_maps(false);
                if self.inflight.load(Ordering::SeqCst) > 0 { let _ = self.device.poll(PollType::Poll); }
            }
        }
        self.hdr(3).fetch_add(executed, Ordering::Relaxed);
        tail
    }

    fn exec(&mut self, op: u32, p: u64, n: u64) {
        // After device loss, only commands that complete engine waits or write
        // diagnostics remain useful. Keep draining the ring so the HUD can
        // display the failure and the user can return to the launcher.
        if self.lost.load(Ordering::SeqCst) && !matches!(op, 7 | 8 | 10 | 42 | 44 | 46) { return; }
        let mb = self.base;
        let u = move |w: u64| unsafe { std::ptr::read_volatile(mb.add(((p + w) * 4) as usize) as *const u32) };
        match op {
            0 | 2 => {}
            3 => self.create_texture(u(0), u(1), u(2), u(3), u(4), u(5), u(6), u(7), u(8), u(9)),
            4 => self.destroy(u(0)),
            5 => self.clear_rt(u(0), u(1), u(2), [self.f(p + 3), self.f(p + 4), self.f(p + 5), self.f(p + 6)]),
            6 => self.clear_ds(u(0), u(1), u(2), u(3), self.f(p + 4), u(5)),
            7 => self.present(u(0)),
            8 => {
                self.flush();
                if !self.lost.load(Ordering::SeqCst) { let _ = self.device.poll(PollType::Wait { submission_index: None, timeout: None }); }
                let addr = u(0) as u64 + u(1) as u64 * 4294967296;
                self.mem.word(addr).store(u(2) as i32, Ordering::SeqCst);
                let _ = self.mem.notify(addr);
            }
            9 => {}
            10 => {
                let len = u(0) as u64;
                let text = String::from_utf8_lossy(self.bytes((p + 1) * 4, len)).into_owned();
                if text == "stats" { self.log_stats(); } else { log(&text); }
            }
            11 => self.create_buffer(u(0), u(1), u(2), u(3), u(4)),
            12 | 47 => self.upload_buffer(u(0), u(1) as u64, u(2) as u64, (p + 3) * 4, op == 47),
            13 => self.upload_texture(u(0), u(1), u(2), u(3), u(4), u(5), u(6), u(7), u(8), u(9), u(10), u(11) as u64, (p + 12) * 4),
            14 => self.create_shader(p),
            15 => self.create_layout(p),
            16 => self.create_state(p, n),
            17 => self.create_view(p, false),
            37 => self.create_view(p, true),
            18 => {
                let st = u(0) as usize;
                if st < 3 && self.s.shader[st] != u(1) { self.s.shader[st] = u(1); self.pdirty = true; }
            }
            19 => if self.s.layout != u(0) { self.s.layout = u(0); self.pdirty = true; },
            20 => if self.s.topo != u(0) { self.s.topo = u(0); self.pdirty = true; },
            21 => {
                let (s, c) = (u(0) as usize, u(1) as usize);
                for i in 0..c {
                    if s + i >= 16 { break; }
                    let (nb, ns, no) = (u(2 + i as u64 * 3), u(3 + i as u64 * 3), u(4 + i as u64 * 3));
                    let v = &mut self.s.vb[s + i];
                    if v.stride != ns || (v.buf == 0) != (nb == 0) { self.pdirty = true; }
                    *v = Vb { buf: nb, stride: ns, offset: no };
                }
            }
            22 => {
                if self.s.ib.1 != u(1) || (self.s.ib.0 == 0) != (u(0) == 0) { self.pdirty = true; }
                self.s.ib = (u(0), u(1), u(2));
            }
            23 => {
                let (st, s, c) = (u(0) as usize, u(1) as usize, u(2) as usize);
                for i in 0..c {
                    if s + i >= 16 || st >= 3 { break; }
                    self.s.cb[st][s + i] = Cb { buf: u(3 + i as u64 * 3), first: u(4 + i as u64 * 3) };
                }
            }
            24 => {
                let (st, s, c) = (u(0) as usize, u(1) as usize, u(2) as usize);
                for i in 0..c { if s + i < 128 && st < 3 { self.s.srv[st][s + i] = u(3 + i as u64); } }
            }
            25 => {
                let (st, s, c) = (u(0) as usize, u(1) as usize, u(2) as usize);
                for i in 0..c { if s + i < 16 && st < 3 { self.s.smp[st][s + i] = u(3 + i as u64); } }
            }
            26 => {
                for i in 0..8 {
                    let q = i as u64 * 4;
                    self.s.rt[i] = Rt { tex: u(q), mip: u(q + 1), slice: u(q + 2), fmt: u(q + 3) };
                }
                self.s.ds = Ds { tex: u(32), mip: u(33), slice: u(34), fmt: u(35), flags: u(36) };
                self.pdirty = true;
            }
            27 => {
                if self.s.blend.0 != u(0) { self.pdirty = true; }
                self.s.blend = (u(0), [self.f(p + 1) as f64, self.f(p + 2) as f64, self.f(p + 3) as f64, self.f(p + 4) as f64]);
            }
            28 => {
                if self.s.dss.0 != u(0) { self.pdirty = true; }
                self.s.dss = (u(0), u(1));
            }
            29 => if self.s.rs != u(0) { self.s.rs = u(0); self.pdirty = true; },
            30 => {
                let c = u(0) as usize;
                self.s.viewports = (0..c).map(|i| std::array::from_fn(|k| self.f(p + 1 + i as u64 * 6 + k as u64))).collect();
            }
            31 => {
                let c = u(0) as usize;
                self.s.scissors = (0..c).map(|i| std::array::from_fn(|k| self.u(p + 1 + i as u64 * 4 + k as u64))).collect();
            }
            32 => self.do_draw(p),
            33 => self.reset_state(),
            34 => self.copy_region(u(0), u(1), u(2), u(3), u(4), u(5), u(6), u(7) != 0, u(8), u(9), u(10), u(11), u(12), u(13)),
            35 => self.copy_resource(u(0), u(1)),
            36 => self.resolve(u(0), u(1), u(2), u(3)),
            38 => {
                let (st, s0, c) = (u(0) as usize, u(1) as usize, u(2) as usize);
                for i in 0..c {
                    if s0 + i >= 8 || st >= 3 { break; }
                    let id = u(3 + i as u64);
                    self.s.uav[st][s0 + i] = id;
                    let init = u(3 + c as u64 + i as u64);
                    if init != 0xffff_ffff { self.set_uav_counter(id, init); }
                }
            }
            39 => self.do_dispatch(u(0), u(1), u(2)),
            40 => self.clear_uav(u(0), [u(2), u(3), u(4), u(5)]),
            41 => self.generate_mips(u(0)),
            42 => self.readback(p),
            43 => self.copy_counter(u(0), u(1) as u64, u(2)),
            44 => self.create_query(u(0), u(1), u(2) as u64 + u(3) as u64 * 4294967296),
            45 => self.begin_query(u(0)),
            46 => self.end_query(u(0), u(1)),
            other => log_once(&format!("op{other}"), &format!("unknown opcode {other}")),
        }
    }

    fn reset_state(&mut self) {
        self.pdirty = true;
        self.s = State::new();
        self.end_pass();
    }

    fn destroy(&mut self, id: u32) {
        self.pending_clears.remove(&id);
        if self.is_bound(id) { self.pdirty = true; }
        if self.obj(id).is_none() { return; }
        if self.used(id) { self.flush(); }
        if let Some(o) = self.objs.get_mut(id as usize).and_then(|o| o.take()) {
            match o {
                Obj::Tex(t) => t.tex.destroy(),
                Obj::Buf(b) => { if let Some(g) = b.gpu { g.destroy(); } }
                _ => {}
            }
        }
    }
    fn is_bound(&self, id: u32) -> bool {
        let s = &self.s;
        s.layout == id || s.blend.0 == id || s.dss.0 == id || s.rs == id || s.ib.0 == id || s.shader.contains(&id) || s.vb.iter().any(|v| v.buf == id)
    }

    #[allow(clippy::too_many_arguments)]
    fn create_texture(&mut self, id: u32, dim: u32, w: u32, h: u32, d: u32, mips: u32, array: u32, dxgi: u32, bind: u32, samples: u32) {
        let mut m = map_format(dxgi, bind);
        let (bw, bh) = m.format.block_dimensions();
        let unsupported_bc = is_bc(m.format) && (!self.device.features().contains(Features::TEXTURE_COMPRESSION_BC)
            || dim == 3 && !self.device.features().contains(Features::TEXTURE_COMPRESSION_BC_SLICED_3D)
            || dim == 1 || w.max(1) % bw != 0 || h.max(1) % bh != 0);
        let compressed_source = unsupported_bc.then_some(m.format);
        if unsupported_bc {
            m.format = crate::texture::fallback_format(m.format);
            m.views = match m.format {
                TextureFormat::Rgba8Unorm => vec![TextureFormat::Rgba8UnormSrgb],
                TextureFormat::Rgba8UnormSrgb => vec![TextureFormat::Rgba8Unorm],
                _ => vec![],
            };
        }
        if m.format == TextureFormat::Depth32FloatStencil8 && !self.device.features().contains(Features::DEPTH32FLOAT_STENCIL8) {
            m.format = TextureFormat::Depth24PlusStencil8;
        }
        let mut usage = TextureUsages::COPY_SRC | TextureUsages::COPY_DST;
        if bind & BIND_SRV != 0 { usage |= TextureUsages::TEXTURE_BINDING; }
        if bind & (BIND_RT | BIND_DS) != 0 { usage |= TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING; }
        if bind & BIND_UAV != 0 { usage |= TextureUsages::STORAGE_BINDING; }
        if bind & (BIND_SRV | BIND_RT | BIND_DS | BIND_UAV) == 0 { usage |= TextureUsages::TEXTURE_BINDING; }
        let samples = samples.max(1);
        if samples > 1 { usage &= !(TextureUsages::COPY_SRC | TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING); }

        let allowed = self.adapter.get_texture_format_features(m.format).allowed_usages;
        usage &= allowed | TextureUsages::COPY_SRC | TextureUsages::COPY_DST;
        let dimension = match dim { 3 => TextureDimension::D3, 1 => TextureDimension::D1, _ => TextureDimension::D2 };
        let size = Extent3d { width: w.max(1), height: if dim == 1 { 1 } else { h.max(1) }, depth_or_array_layers: if dim == 3 { d.max(1) } else { array.max(1) } };
        let views = m.views.clone();
        let desc = TextureDescriptor {
            label: None, size, mip_level_count: mips.max(1), sample_count: samples, dimension, format: m.format, usage, view_formats: &views,
        };
        let tex = match self.scoped(&format!("createTexture id={id} dim={dim} {}x{}x{} mips={mips} dxgi={dxgi} bind=0x{bind:x} samples={samples} -> {:?} usage {:?}", size.width, size.height, size.depth_or_array_layers, m.format, usage), |d| d.create_texture(&desc)) { Some(t) => t, None => return };
        self.set_obj(id, Obj::Tex(Tex {
            tex, desc: TexDesc { w: size.width, h: size.height, d: size.depth_or_array_layers, mips: mips.max(1), samples, dim: dimension },
            format: m.format, compressed_source, views: HashMap::new(), a8: m.a8, enc: 0, in_bg: false, proxy_depth: None, proxy_stencil: None,
            proxy_fdepth: None, blit_bg: None, mip_passes: None,
        }));
    }

    fn create_buffer(&mut self, id: u32, size: u32, bind: u32, misc: u32, stride: u32) {
        let size = size as u64;
        // The shadow of a constant buffer holds only what the engine uploaded.
        // Allocating the full buffer size upfront costs host memory for every
        // constant buffer the engine creates, and it never shrinks.
        let mut b = Buf { size, stride, gpu: None, bytes: 0, shadow: None, version: 0, enc: 0, in_bg: false, counter: None };
        if bind & 4 != 0 { b.shadow = Some(Vec::new()); }
        let mut usage = BufferUsages::COPY_DST | BufferUsages::COPY_SRC;
        if bind & 1 != 0 { usage |= BufferUsages::VERTEX; }
        if bind & 2 != 0 { usage |= BufferUsages::INDEX; }
        if bind & (8 | 0x80) != 0 { usage |= BufferUsages::STORAGE; }
        if misc & 0x10 != 0 { usage |= BufferUsages::INDIRECT; }
        if bind & (1 | 2 | 8 | 0x80) != 0 || misc & 0x10 != 0 {
            b.bytes = ((size + 3) & !3).max(4);
            b.gpu = Some(self.device.create_buffer(&BufferDescriptor { label: None, size: b.bytes, usage, mapped_at_creation: false }));
        }
        self.set_obj(id, Obj::Buf(b));
    }

    fn upload_buffer(&mut self, id: u32, offset: u64, size: u64, data_addr: u64, no_overwrite: bool) {
        let data = self.bytes(data_addr, size).to_vec();
        let used = self.used(id);
        let Some(b) = self.buf_mut(id) else { return };
        if let Err(e) = b.update_shadow(offset, &data) {
            log_once("shadow-allocation", &format!("constant buffer shadow upload id={id} offset={offset} size={size} failed: {e:#}"));
            return;
        }
        let Some(g) = b.gpu.clone() else { return };
        if !no_overwrite && used { self.flush(); }
        let n = size & !3;
        if n > 0 { self.queue.write_buffer(&g, offset, &data[..n as usize]); }
        if size & 3 != 0 {
            let mut tail = [0u8; 4];
            tail[..(size - n) as usize].copy_from_slice(&data[n as usize..]);
            self.queue.write_buffer(&g, offset + n, &tail);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn upload_texture(&mut self, id: u32, mip: u32, slice: u32, x: u32, y: u32, z: u32, w: u32, h: u32, d: u32, bpr: u32, rows: u32, size: u64, data_addr: u64) {
        if self.tex(id).is_none() { return; }
        if self.pending_clears.contains_key(&id) { self.materialize_clears(id); self.flush(); }
        if self.used(id) { self.flush(); }
        let t = self.tex_mut(id).unwrap();
        if let Some(p) = t.proxy_fdepth.as_mut() { p.dirty = true; }
        let three = t.desc.dim == TextureDimension::D3;
        let (tex, format, a8, compressed_source) = (t.tex.clone(), t.format, t.a8, t.compressed_source);
        let (w, h) = (w.min((t.desc.w >> mip).max(1).saturating_sub(x)), h.min((t.desc.h >> mip).max(1).saturating_sub(y)));
        if w == 0 || h == 0 { return; }
        let mut data = self.bytes(data_addr, size).to_vec();
        let mut bpr = bpr;
        let mut rows = rows;
        if let Some(format) = compressed_source {
            match crate::texture::decode(format, &data, w, h, d.max(1), bpr, rows) {
                Ok((decoded, pitch)) => { data = decoded; bpr = pitch; rows = h; }
                Err(e) => { log_once("bc-decode", &format!("BC upload failed: {e:#}")); return; }
            }
        }
        if a8 {
            let mut out = vec![0u8; (w * h * 4) as usize];
            for r in 0..h as usize {
                for c in 0..w as usize {
                    out[(r * w as usize + c) * 4 + 3] = data.get(r * bpr as usize + c).copied().unwrap_or(0);
                }
            }
            data = out;
            bpr = w * 4;
        }
        self.queue.write_texture(
            TexelCopyTextureInfo { texture: &tex, mip_level: mip, origin: Origin3d { x, y, z: if three { z } else { slice } }, aspect: TextureAspect::All },
            &data,
            TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bpr), rows_per_image: Some(rows.max(1)) },
            copy_extent(format, w, h, d.max(1)),
        );
    }

    fn create_shader(&mut self, p: u64) {
        let id = self.u(p);
        let stage = self.u(p + 1);
        let hash = format!("{:08x}{:08x}", self.u(p + 3), self.u(p + 2));
        let nin = self.u(p + 4) as u64;
        let sig = (0..nin).map(|i| SigEl { sem: self.u(p + 5 + i * 4), idx: self.u(p + 6 + i * 4), reg: self.u(p + 7 + i * 4) }).collect();
        self.set_obj(id, Obj::Shader(ShaderObj { stage, hash, sig, program: None }));
    }

    fn create_layout(&mut self, p: u64) {
        let id = self.u(p);
        let n = self.u(p + 1) as u64;
        let els: Vec<El> = (0..n)
            .map(|i| {
                let q = p + 2 + i * 7;
                El { sem: self.u(q), idx: self.u(q + 1), fmt: self.u(q + 2), slot: self.u(q + 3), offset: self.u(q + 4), cls: self.u(q + 5) }
            })
            .collect();
        let mut slots = Vec::new();
        let mut seen = 0u32;
        for e in &els {
            if e.slot < 32 && (seen >> e.slot) & 1 == 0 {
                seen |= 1 << e.slot;
                slots.push((e.slot, e.cls));
            }
        }
        self.set_obj(id, Obj::Layout(Rc::new(LayoutObj { els, slots })));
    }

    fn create_state(&mut self, p: u64, n: u64) {
        let id = self.u(p);
        let kind = self.u(p + 1);
        let words: Vec<u32> = (2..n).map(|i| self.u(p + i)).collect();
        let sampler = if kind == 3 { Some(self.make_sampler(&words)) } else { None };
        self.set_obj(id, Obj::State(Rc::new(StateObj { words, sampler })));
    }

    fn make_sampler(&self, w: &[u32]) -> Sampler {
        let g = |i: usize| w.get(i).copied().unwrap_or(0);
        let filter = g(0);
        let aniso = (filter & 0x7f) == 0x55;
        let lin = |b: bool| if b { FilterMode::Linear } else { FilterMode::Nearest };
        let lin_m = |b: bool| if b { MipmapFilterMode::Linear } else { MipmapFilterMode::Nearest };
        let lod_min = wf(g(11)).max(0.0);
        let mut lod_max = wf(g(12)).clamp(0.0, 32.0);
        if lod_max < lod_min { lod_max = lod_min; }
        let mut d = SamplerDescriptor {
            label: None,
            address_mode_u: address(g(1)), address_mode_v: address(g(2)), address_mode_w: address(g(3)),
            mag_filter: lin((filter >> 2) & 1 != 0), min_filter: lin((filter >> 4) & 1 != 0), mipmap_filter: lin_m(filter & 1 != 0),
            lod_min_clamp: lod_min, lod_max_clamp: lod_max,
            compare: if filter & 0x80 != 0 { Some(cmp(g(6)).unwrap_or(CompareFunction::Always)) } else { None },
            anisotropy_clamp: 1, border_color: None,
        };
        if aniso {
            d.mag_filter = FilterMode::Linear;
            d.min_filter = FilterMode::Linear;
            d.mipmap_filter = MipmapFilterMode::Linear;
            d.anisotropy_clamp = g(5).clamp(1, 16) as u16;
        }
        self.device.create_sampler(&d)
    }

    fn create_view(&mut self, p: u64, uav: bool) {
        let v = ViewObj {
            res: self.u(p + 2), is_buf: self.u(p + 1) == 1, fmt: self.u(p + 3), vdim: self.u(p + 4),
            a: self.u(p + 5), b: self.u(p + 6), c: self.u(p + 7), d: self.u(p + 8), views: HashMap::new(),
        };
        let id = self.u(p);
        self.set_obj(id, if uav { Obj::Uav(v) } else { Obj::Srv(v) });
    }

    fn shader_program(&mut self, id: u32) -> Option<Rc<Program>> {
        let (hash, stage) = match self.obj(id) {
            Some(Obj::Shader(s)) => {
                if let Some(p) = &s.program { return p.clone(); }
                (s.hash.clone(), s.stage)
            }
            _ => return None,
        };
        let prog = self.load_program(&hash, stage);
        if let Some(Obj::Shader(s)) = self.obj_mut(id) { s.program = Some(prog.clone()); }
        prog
    }

    fn load_program(&mut self, hash: &str, stage: u32) -> Option<Rc<Program>> {
        let rec = self.shader_index.get(hash).cloned();
        let ok = rec.as_ref().and_then(|r| r.get("ok")).and_then(|v| v.as_bool()).unwrap_or(false);
        let label = rec.as_ref().map(|r| format!("{}::{}", r["effect"].as_str().unwrap_or("?"), r["program"].as_str().unwrap_or("?"))).unwrap_or_else(|| hash.to_string());
        if !ok {
            log_once(&format!("noshader{hash}"), &format!("no WGSL for {} shader {hash} ({label})", ["vs", "ps", "cs"].get(stage as usize).unwrap_or(&"?")));
            return None;
        }
        let source = match std::fs::read_to_string(self.shader_dir.join(format!("{hash}.wgsl"))) {
            Ok(s) => s,
            Err(e) => { log_once(&format!("nosrc{hash}"), &format!("shader source {hash}: {e}")); return None; }
        };
        let (lowered, clip) = if self.device.features().contains(Features::CLIP_DISTANCES) {
            (source.clone(), None)
        } else {
            Self::lower_clip_distances(&source)
        };
        let (text, pack) = pack_uniforms(&lowered);
        let module = self.scoped(&format!("shader {label}"), |d| d.create_shader_module(ShaderModuleDescriptor { label: Some(&label), source: ShaderSource::Wgsl(text.as_str().into()) }))?;
        let info = parse_wgsl(&text);
        let mut consts = None;
        if rec.as_ref().and_then(|r| r.get("consts")).map(|c| !c.is_null() && c != &serde_json::Value::Bool(false)).unwrap_or(false) {
            if let Ok(t) = std::fs::read_to_string(self.shader_dir.join(format!("{hash}.consts.json"))) {
                let arr: Vec<serde_json::Value> = serde_json::from_str(&t).unwrap_or_default();
                let n = arr.len().max(1);
                let mut data = vec![0u32; n * 4];
                for (i, v) in arr.iter().enumerate() {
                    data[i * 4] = v.as_i64().map(|x| x as u32).or_else(|| v.as_f64().map(|x| x as i64 as u32)).unwrap_or(0);
                }
                let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
                let b = self.device.create_buffer(&BufferDescriptor { label: None, size: (bytes.len() as u64).max(16), usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST, mapped_at_creation: false });
                self.queue.write_buffer(&b, 0, &bytes);
                consts = Some(b);
            }
        }
        let flat = RE_FLAT.captures_iter(&text).filter_map(|m| m[1].parse().ok()).collect();
        let mut group_bindings: BTreeMap<u32, Vec<Binding>> = BTreeMap::new();
        for b in &info.bindings { group_bindings.entry(b.group).or_default().push(b.clone()); }
        for v in group_bindings.values_mut() { v.sort_by_key(|b| b.binding); }
        let uid = self.uid();
        let vid = text.contains("vertex_index");
        let iid = text.contains("instance_index");
        let out_locs = if stage == 0 { locations_in(&text, true) } else { vec![] };
        let in_locs = if stage == 1 { locations_in(&text, false) } else { vec![] };
        Some(Rc::new(Program { module, info, consts, pack, text, flat, group_bindings, variants: RefCell::new(HashMap::new()), label, uid, vid, iid, out_locs, in_locs, clip }))
    }

    fn variant(&self, prog: &Program, key: &str, make: impl FnOnce(&str) -> String) -> Option<ShaderModule> {
        if let Some(m) = prog.variants.borrow().get(key) { return m.clone(); }
        let text = make(&prog.text);
        let label = format!("{} [{key}]", prog.label);
        let m = self.scoped(&format!("shader variant {label}"), |d| d.create_shader_module(ShaderModuleDescriptor { label: Some(&label), source: ShaderSource::Wgsl(text.as_str().into()) }));
        prog.variants.borrow_mut().insert(key.to_string(), m.clone());
        m
    }

    fn inter_stage_remap(&self, vs: &Program, ps: Option<&Program>) -> Option<(HashMap<u32, u32>, String, bool)> {
        let max = self.device.limits().max_inter_stage_shader_variables.max(16);
        // Clip distances in hardware count against the inter-stage limit; when
        // they were lowered into ordinary varyings they do not.
        let clip_slots = RE_CLIP_FIELD.captures(&vs.text).and_then(|m| m[2].parse::<u32>().ok()).unwrap_or(0).div_ceil(4);
        let lim = max.saturating_sub(clip_slots);
        if !vs.out_locs.iter().any(|&l| l >= lim) { return None; }
        let ins = ps.map(|p| p.in_locs.clone()).unwrap_or_default();
        let mut used: std::collections::HashSet<u32> = vs.out_locs.iter().chain(ins.iter()).copied().filter(|&l| l < lim).collect();
        let mut map = HashMap::new();
        let mut free = 0;
        for &l in &vs.out_locs {
            if l < lim { continue; }
            while used.contains(&free) { free += 1; }
            if free >= lim { return None; }
            map.insert(l, free);
            used.insert(free);
        }
        let mut pairs: Vec<_> = map.iter().map(|(a, b)| format!("{a}>{b}")).collect();
        pairs.sort();
        let ps_hit = ins.iter().any(|l| map.contains_key(l));
        Some((map, pairs.join(","), ps_hit))
    }

    fn vertex_module(&self, vs: &Program, flat: &[u32], remap: Option<&(HashMap<u32, u32>, String, bool)>) -> Option<ShaderModule> {
        if flat.is_empty() && remap.is_none() { return Some(vs.module.clone()); }
        let key = format!("flat {}{}", flat.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","), remap.map(|r| format!("|{}", r.1)).unwrap_or_default());
        self.variant(vs, &key, |text| {
            let t = RE_VS_LOC_MEMBER.replace_all(text, |c: &regex::Captures| {
                let loc: u32 = c[1].parse().unwrap_or(0);
                if flat.contains(&loc) { format!("@location({}u) @interpolate(flat){}", &c[1], &c[2]) } else { c[0].to_string() }
            }).into_owned();
            match remap { Some(r) => remap_locations(&t, true, &r.0), None => t }
        })
    }

    fn pixel_module(&self, ps: &Program, a2c: bool, remap: Option<&(HashMap<u32, u32>, String, bool)>, clip: Option<&ClipInfo>) -> Option<ShaderModule> {
        let ps_remap = remap.filter(|r| r.2);
        if ps_remap.is_none() && !a2c && clip.is_none() { return Some(ps.module.clone()); }
        let key = format!("{}{}{}", remap.map(|r| format!("remap|{}", r.1)).unwrap_or_default(), if a2c { "|a2c" } else { "" }, clip.map(|c| format!("|clip{}:{:?}", c.count, c.locations)).unwrap_or_default());
        let m = self.variant(ps, &key, |text| {
            let mut t = text.to_string();
            if a2c { t = RE_MAIN_INNER.replace(&t, "$1\n  if (o0.w < 0.5f) { discard; }").into_owned(); }
            if let Some(r) = ps_remap { t = remap_locations(&t, false, &r.0); }
            Self::clip_fragment(&t, clip, remap)
        });
        if m.is_none() && ps_remap.is_none() && clip.is_none() { Some(ps.module.clone()) } else { m }
    }

    fn layout_entry(b: &Binding, stage: u32, f32_filterable: bool) -> BindGroupLayoutEntry {
        let visibility = match stage { 0 => ShaderStages::VERTEX, 1 => ShaderStages::FRAGMENT, _ => ShaderStages::COMPUTE };
        let ty = match b.kind {
            Kind::Uniform => BindingType::Buffer { ty: BufferBindingType::Uniform, has_dynamic_offset: b.binding != 15, min_binding_size: None },
            Kind::Storage => BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
            Kind::ReadOnlyStorage => BindingType::Buffer { ty: BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
            Kind::Sampler => BindingType::Sampler(SamplerBindingType::Filtering),
            Kind::SamplerCmp => BindingType::Sampler(SamplerBindingType::Comparison),
            Kind::Texture => BindingType::Texture {
                sample_type: match b.sample {
                    Sample::Float => TextureSampleType::Float { filterable: f32_filterable },
                    Sample::Unfilterable => TextureSampleType::Float { filterable: false },
                    Sample::Uint => TextureSampleType::Uint,
                    Sample::Sint => TextureSampleType::Sint,
                    Sample::Depth => TextureSampleType::Depth,
                },
                view_dimension: b.dim,
                multisampled: b.ms,
            },
            Kind::StorageTexture => BindingType::StorageTexture { access: b.access, format: b.format.unwrap_or(TextureFormat::Rgba8Unorm), view_dimension: b.dim },
        };
        BindGroupLayoutEntry { binding: b.binding, visibility, ty, count: None }
    }

    fn bgl_for(&mut self, prog: &Program, stage: u32, group: u32) -> (BindGroupLayout, u64) {
        let key = (prog.label.clone() + "#" + &prog.uid.to_string(), stage, group);
        if let Some(b) = self.bgl_cache.get(&key) { return b.clone(); }
        let entries: Vec<_> = prog.info.bindings.iter().filter(|b| b.group == group).map(|b| Self::layout_entry(b, stage, self.f32_filterable)).collect();
        let bgl = self.device.create_bind_group_layout(&BindGroupLayoutDescriptor { label: None, entries: &entries });
        let uid = self.uid();
        self.bgl_cache.insert(key, (bgl.clone(), uid));
        (bgl, uid)
    }
    fn empty_bgl(&mut self) -> BindGroupLayout {
        if self.empty_bgl.is_none() {
            let b = self.device.create_bind_group_layout(&BindGroupLayoutDescriptor { label: None, entries: &[] });
            let uid = self.uid();
            self.empty_bgl = Some((b, uid));
        }
        self.empty_bgl.as_ref().unwrap().0.clone()
    }

    fn target_view(&mut self, id: u32, dxgi: u32, mip: u32, slice: u32) -> Option<(TextureView, TextureFormat)> {
        let t = self.tex(id)?;
        let fmt = if dxgi != 0 { map_format(dxgi, 0).format } else { t.format };
        let fmt = if is_depth(t.format) { t.format } else if t.compressed_source.is_some() { crate::texture::fallback_format(fmt) } else { fmt };
        if let Some(v) = t.views.get(&(fmt, mip, slice)) { return Some((v.clone(), fmt)); }
        let three = t.desc.dim == TextureDimension::D3;
        let tex = t.tex.clone();
        let v = self.scoped("target view", |_| tex.create_view(&TextureViewDescriptor {
            format: Some(fmt), dimension: Some(if three { TextureViewDimension::D3 } else { TextureViewDimension::D2 }),
            base_mip_level: mip, mip_level_count: Some(1), base_array_layer: if three { 0 } else { slice }, array_layer_count: Some(1),
            ..Default::default()
        }))?;
        self.tex_mut(id).unwrap().views.insert((fmt, mip, slice), v.clone());
        Some((v, fmt))
    }

    // Clip distances on a device that cannot do them in hardware, or when the
    // diagnostic override requests software clipping.
    fn lower_clip_distances(source: &str) -> (String, Option<ClipInfo>) {
        let clean = RE_ENABLE_CLIP.replace_all(source, "").into_owned();
        let Some(main) = RE_VS_MAIN.captures(&clean) else { return (clean, None) };
        let name = main[1].to_string();
        let Ok(re) = Regex::new(&format!(r"(struct\s+{}\s*\{{)([^}}]*)(\}})", regex::escape(&name))) else { return (source.to_string(), None) };
        let Some(st) = re.captures(&clean) else { return (source.to_string(), None) };
        let body = st[2].to_string();
        let Some(field) = RE_CLIP_FIELD.captures(&body) else { return (clean, None) };
        let count: usize = field[2].parse().unwrap_or(0);
        if count < 1 || count > 8 { return (source.to_string(), None); }
        let fields: Vec<String> = RE_STRUCT_FIELD.captures_iter(&body).map(|m| m[1].to_string()).collect();
        let Some(index) = fields.iter().position(|f| f.as_str() == &field[1]) else { return (source.to_string(), None) };
        let Ok(ctor) = Regex::new(&format!(r"return\s+{}\s*\(([^;]*)\)\s*;", regex::escape(&name))) else { return (source.to_string(), None) };
        let Some(ret) = ctor.captures(&clean) else { return (source.to_string(), None) };
        // Commas in vector/array constructors are not output-struct separators.
        let args = wgsl_arguments(&ret[1]);
        if args.len() != fields.len() { return (source.to_string(), None); }
        let value = args[index].clone();
        // The clip distances travel to the pixel stage as ordinary varyings.
        // Pick locations the shader does not already use.
        let used: std::collections::HashSet<u32> = RE_LOC.captures_iter(&body).filter_map(|m| m[1].parse().ok()).collect();
        let slots = count.div_ceil(4) as u32;
        let mut base = 32;
        while base < 256 && !(base..base + slots).all(|l| !used.contains(&l)) { base += 1; }
        let locations: Vec<u32> = (0..slots).map(|i| base + i).collect();
        let replacement: Vec<String> = locations.iter().enumerate().map(|(i, l)| format!("@location({l}u) gtaClip{i} : vec4<f32>,")).collect();
        let new_body = body.replacen(&field[0], &replacement.join("\n"), 1);
        let new_struct = format!("{}{}{}", &st[1], new_body, &st[3]);
        let mut text = re.replace(&clean, new_struct.as_str()).into_owned();
        let vectors: Vec<String> = locations.iter().enumerate().map(|(i, _)| {
            let comps: Vec<String> = (0..4).map(|j| if i * 4 + j < count { format!("({value})[{}u]", i * 4 + j) } else { "1.0f".to_string() }).collect();
            format!("vec4<f32>({})", comps.join(", "))
        }).collect();
        let mut new_args = args;
        new_args.splice(index..index + 1, vectors);
        let new_ret = format!("return {}({});", name, new_args.join(", "));
        text = ctor.replace(&text, new_ret.as_str()).into_owned();
        (text, Some(ClipInfo { count, locations }))
    }

    // The pixel stage half of the clip emulation. Keep the original body and
    // everything after the entry point when injecting the discard test.
    fn clip_fragment(source: &str, clip: Option<&ClipInfo>, remap: Option<&(HashMap<u32, u32>, String, bool)>) -> String {
        let Some(clip) = clip else { return source.to_string() };
        let params: Vec<String> = clip.locations.iter().enumerate().map(|(i, loc)| {
            let l = remap.and_then(|r| r.0.get(loc)).copied().unwrap_or(*loc);
            format!("@location({l}u) gtaClip{i}: vec4<f32>")
        }).collect();
        let test: Vec<String> = (0..clip.locations.len()).map(|i| format!("any(gtaClip{i} < vec4<f32>(0.0f))")).collect();
        let Some(m) = RE_FRAG_ENTRY.captures(source) else { return source.to_string() };
        let ps_params = m[2].trim().trim_end_matches(',').trim_end();
        let new_params = if ps_params.is_empty() { params.join(", ") } else { format!("{}, {}", ps_params, params.join(", ")) };
        let entry = m.get(0).unwrap();
        format!("{}{}{}{}\n if ({}) {{ discard; }}\n{}", &source[..entry.start()], &m[1], new_params, &m[3], test.join(" || "), &source[entry.end()..])
    }

    fn mark_color_dirty(&mut self, id: u32) {
        if let Some(t) = self.tex_mut(id) { if let Some(p) = t.proxy_fdepth.as_mut() { p.dirty = true; } }
    }
    fn mark_depth_dirty(&mut self, id: u32) {
        if let Some(t) = self.tex_mut(id) {
            if let Some(p) = t.proxy_depth.as_mut() { p.dirty = true; }
            if let Some(p) = t.proxy_stencil.as_mut() { p.dirty = true; }
        }
    }

    fn ensure_pass(&mut self) -> bool {
        let mut key = Vec::with_capacity(37);
        for r in &self.s.rt { key.extend([r.tex, r.mip, r.slice, r.fmt]); }
        let d = &self.s.ds;
        key.extend([d.tex, d.mip, d.slice, d.fmt, d.flags]);
        if self.pass.is_some() && self.pass_key.as_ref() == Some(&key) { return true; }
        self.end_pass();
        let mut colors: Vec<Option<(TextureView, TextureFormat)>> = Vec::new();
        let mut samples = 1;
        let mut any = false;
        let (mut pw, mut ph) = (1, 1);
        for i in 0..8 {
            let r = self.s.rt[i].clone();
            if r.tex == 0 || self.tex(r.tex).is_none() { colors.push(None); continue; }
            match self.target_view(r.tex, r.fmt, r.mip, r.slice) {
                Some(v) => {
                    self.mark_color_dirty(r.tex);
                    let t = self.tex(r.tex).unwrap();
                    samples = t.desc.samples;
                    pw = (t.desc.w >> r.mip).max(1);
                    ph = (t.desc.h >> r.mip).max(1);
                    colors.push(Some(v));
                    any = true;
                    self.use_tex(r.tex);
                }
                None => colors.push(None),
            }
        }
        while matches!(colors.last(), Some(None)) { colors.pop(); }
        let ds = self.s.ds.clone();
        let mut ds_att = None;
        let mut ds_format = None;
        let (ro_d, ro_s) = (ds.flags & 1 != 0, ds.flags & 2 != 0);
        if ds.tex != 0 && self.tex(ds.tex).is_some() {
            if let Some((v, f)) = self.target_view(ds.tex, 0, ds.mip, ds.slice) {
                if !ro_d && !ro_s { self.mark_depth_dirty(ds.tex); }
                let t = self.tex(ds.tex).unwrap();
                ds_format = Some(f);
                samples = t.desc.samples;
                if !self.s.rt.iter().any(|r| r.tex != 0) { pw = (t.desc.w >> ds.mip).max(1); ph = (t.desc.h >> ds.mip).max(1); }
                ds_att = Some((v, f));
                any = true;
                self.use_tex(ds.tex);
            }
        }
        if !any { return false; }

        if (ro_d || ro_s) && ds_att.is_some() && self.pending_clears.contains_key(&ds.tex) { self.materialize_clears(ds.tex); }

        let take = |me: &mut Self, tex: u32, mip: u32, slice: u32| -> Option<PendingClear> {
            let list = me.pending_clears.get_mut(&tex)?;
            let i = list.iter().position(|e| e.0 == mip && e.1 == slice)?;
            let c = list.swap_remove(i).2;
            if list.is_empty() { me.pending_clears.remove(&tex); }
            me.clears_folded += 1;
            Some(c)
        };
        let mut color_loads = Vec::with_capacity(colors.len());
        for (i, c) in colors.iter().enumerate() {
            let r = self.s.rt[i].clone();
            let load = match c {
                Some(_) => match take(self, r.tex, r.mip, r.slice) {
                    Some(pc) => {
                        pc.color.map(LoadOp::Clear).unwrap_or(LoadOp::Load)
                    }
                    None => LoadOp::Load,
                },
                None => LoadOp::Load,
            };
            color_loads.push(load);
        }
        let ds_clear = if ds_att.is_some() { take(self, ds.tex, ds.mip, ds.slice) } else { None };
        let color_atts: Vec<Option<RenderPassColorAttachment>> = colors.iter().zip(&color_loads).map(|(c, load)| c.as_ref().map(|(v, _)| RenderPassColorAttachment {
            view: v, depth_slice: None, resolve_target: None, ops: Operations { load: *load, store: StoreOp::Store },
        })).collect();
        let dsa = ds_att.as_ref().map(|(v, f)| RenderPassDepthStencilAttachment {
            view: v,
            depth_ops: if ro_d { None } else { Some(Operations { load: ds_clear.and_then(|c| c.depth).map(LoadOp::Clear).unwrap_or(LoadOp::Load), store: StoreOp::Store }) },
            stencil_ops: if has_stencil(*f) && !ro_s { Some(Operations { load: ds_clear.and_then(|c| c.stencil).map(LoadOp::Clear).unwrap_or(LoadOp::Load), store: StoreOp::Store }) } else { None },
        });
        let occ = self.occlusion_set.clone();
        let pass = self.encoder().begin_render_pass(&RenderPassDescriptor {
            label: None, color_attachments: &color_atts, depth_stencil_attachment: dsa, timestamp_writes: None,
            occlusion_query_set: occ.as_ref(), multiview_mask: None,
        }).forget_lifetime();
        self.pass = Some(pass);
        self.passes += 1;
        self.pass_key = Some(key);
        self.pdirty = true;
        self.pass_info = Some(PassInfo {
            rt_formats: colors.iter().map(|c| c.as_ref().map(|x| x.1)).collect(),
            rt_ids: self.s.rt.iter().map(|r| r.tex).collect(),
            samples, ds_format, ro_depth: ds_att.is_some() && ro_d, ro_stencil: ds_att.is_some() && ro_s, width: pw, height: ph,
        });
        self.pass_st = PassSt::default();
        self.begin_occlusion_in_pass();
        true
    }

    fn proxy_pipeline(&mut self, kind: &'static str) -> RenderPipeline {
        if let Some(p) = self.proxy_pipelines.get(kind) { return p.clone(); }
        let code = if kind == "depth" {
            "@group(0) @binding(0) var src: texture_depth_2d;
             @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f { var p = array<vec2f, 3>(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3)); return vec4f(p[i], 0, 1); }
             @fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f { return vec4f(textureLoad(src, vec2i(pos.xy), 0), 0, 0, 1); }"
        } else {
            "@group(0) @binding(0) var src: texture_2d<u32>;
             @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f { var p = array<vec2f, 3>(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3)); return vec4f(p[i], 0, 1); }
             @fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4<u32> { let s = textureLoad(src, vec2i(pos.xy), 0).x; return vec4<u32>(s, s, 0u, 1u); }"
        };
        let module = self.device.create_shader_module(ShaderModuleDescriptor { label: None, source: ShaderSource::Wgsl(code.into()) });
        let fmt = if kind == "depth" { TextureFormat::R32Float } else { STENCIL_PROXY };
        let pl = self.simple_pipeline(&module, Some(fmt), None);
        self.proxy_pipelines.insert(kind, pl.clone());
        pl
    }

    fn simple_pipeline(&self, module: &ShaderModule, color: Option<TextureFormat>, depth: Option<DepthStencilState>) -> RenderPipeline {
        let targets = [color.map(|f| ColorTargetState { format: f, blend: None, write_mask: ColorWrites::ALL })];
        self.device.create_render_pipeline(&RenderPipelineDescriptor {
            label: None, layout: None,
            vertex: VertexState { module, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
            primitive: PrimitiveState::default(),
            depth_stencil: depth,
            multisample: MultisampleState::default(),
            fragment: Some(FragmentState { module, entry_point: Some("fs"), compilation_options: Default::default(), targets: if color.is_some() { &targets } else { &[] } }),
            multiview_mask: None,
            cache: None,
        })
    }

    fn depth_proxy(&mut self, id: u32, stencil: bool) -> Option<(Texture, TextureFormat)> {
        self.materialize_clears(id);
        let pl = self.proxy_pipeline(if stencil { "stencil" } else { "depth" });
        let (desc, tex) = { let t = self.tex(id)?; (t.desc, t.tex.clone()) };
        let dev = self.device.clone();
        let t = self.tex_mut(id).unwrap();
        let slot = if stencil { &mut t.proxy_stencil } else { &mut t.proxy_depth };
        if slot.is_none() {
            let fmt = if stencil { STENCIL_PROXY } else { TextureFormat::R32Float };
            let ptex = dev.create_texture(&TextureDescriptor {
                label: None, size: Extent3d { width: desc.w, height: desc.h, depth_or_array_layers: desc.d }, mip_level_count: desc.mips, sample_count: 1,
                dimension: TextureDimension::D2, format: fmt, usage: TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT, view_formats: &[],
            });
            let mut passes = Vec::new();
            for mip in 0..desc.mips {
                for layer in 0..desc.d {
                    let src = tex.create_view(&TextureViewDescriptor {
                        dimension: Some(TextureViewDimension::D2), base_mip_level: mip, mip_level_count: Some(1), base_array_layer: layer, array_layer_count: Some(1),
                        aspect: if stencil { TextureAspect::StencilOnly } else { TextureAspect::DepthOnly }, ..Default::default()
                    });
                    let dst = ptex.create_view(&TextureViewDescriptor { base_mip_level: mip, mip_level_count: Some(1), base_array_layer: layer, array_layer_count: Some(1), dimension: Some(TextureViewDimension::D2), ..Default::default() });
                    let bg = dev.create_bind_group(&BindGroupDescriptor { label: None, layout: &pl.get_bind_group_layout(0), entries: &[BindGroupEntry { binding: 0, resource: BindingResource::TextureView(&src) }] });
                    passes.push((dst, bg));
                }
            }
            *slot = Some(Proxy { tex: ptex, format: fmt, dirty: true, passes });
        }
        let px = slot.as_mut().unwrap();
        let out = (px.tex.clone(), px.format);
        if px.dirty {
            px.dirty = false;
            let passes: Vec<_> = px.passes.iter().map(|(v, b)| (v.clone(), b.clone())).collect();
            self.end_pass();
            let enc = self.encoder();
            for (dst, bg) in &passes {
                let mut p = enc.begin_render_pass(&RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(RenderPassColorAttachment { view: dst, depth_slice: None, resolve_target: None, ops: Operations { load: LoadOp::Clear(Color::BLACK), store: StoreOp::Store } })],
                    depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None, multiview_mask: None,
                });
                p.set_pipeline(&pl);
                p.set_bind_group(0, bg, &[]);
                p.draw(0..3, 0..1);
            }
            self.use_tex(id);
        }
        Some(out)
    }

    fn float_to_depth_proxy(&mut self, id: u32) -> Option<(Texture, TextureFormat)> {
        self.materialize_clears(id);
        if self.f2d_pipeline.is_none() {
            let module = self.device.create_shader_module(ShaderModuleDescriptor { label: None, source: ShaderSource::Wgsl(
                "@group(0) @binding(0) var src: texture_2d<f32>;
                 @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f { var p = array<vec2f, 3>(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3)); return vec4f(p[i], 0, 1); }
                 @fragment fn fs(@builtin(position) pos: vec4f) -> @builtin(frag_depth) f32 { return textureLoad(src, vec2i(pos.xy), 0).x; }".into()) });
            let ds = DepthStencilState { format: TextureFormat::Depth32Float, depth_write_enabled: Some(true), depth_compare: Some(CompareFunction::Always), stencil: Default::default(), bias: Default::default() };
            self.f2d_pipeline = Some(self.simple_pipeline(&module, None, Some(ds)));
        }
        let pl = self.f2d_pipeline.clone().unwrap();
        let (desc, tex) = { let t = self.tex(id)?; (t.desc, t.tex.clone()) };
        let dev = self.device.clone();
        let t = self.tex_mut(id).unwrap();
        if t.proxy_fdepth.is_none() {
            let ptex = dev.create_texture(&TextureDescriptor {
                label: None, size: Extent3d { width: desc.w, height: desc.h, depth_or_array_layers: desc.d }, mip_level_count: desc.mips, sample_count: 1,
                dimension: TextureDimension::D2, format: TextureFormat::Depth32Float, usage: TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT, view_formats: &[],
            });
            let mut passes = Vec::new();
            for mip in 0..desc.mips {
                for layer in 0..desc.d {
                    let src = tex.create_view(&TextureViewDescriptor { dimension: Some(TextureViewDimension::D2), base_mip_level: mip, mip_level_count: Some(1), base_array_layer: layer, array_layer_count: Some(1), ..Default::default() });
                    let dst = ptex.create_view(&TextureViewDescriptor { base_mip_level: mip, mip_level_count: Some(1), base_array_layer: layer, array_layer_count: Some(1), dimension: Some(TextureViewDimension::D2), ..Default::default() });
                    let bg = dev.create_bind_group(&BindGroupDescriptor { label: None, layout: &pl.get_bind_group_layout(0), entries: &[BindGroupEntry { binding: 0, resource: BindingResource::TextureView(&src) }] });
                    passes.push((dst, bg));
                }
            }
            t.proxy_fdepth = Some(Proxy { tex: ptex, format: TextureFormat::Depth32Float, dirty: true, passes });
        }
        let px = t.proxy_fdepth.as_mut().unwrap();
        let out = (px.tex.clone(), px.format);
        if px.dirty {
            px.dirty = false;
            let passes: Vec<_> = px.passes.iter().map(|(v, b)| (v.clone(), b.clone())).collect();
            self.end_pass();
            let enc = self.encoder();
            for (dst, bg) in &passes {
                let mut p = enc.begin_render_pass(&RenderPassDescriptor {
                    label: None, color_attachments: &[],
                    depth_stencil_attachment: Some(RenderPassDepthStencilAttachment { view: dst, depth_ops: Some(Operations { load: LoadOp::Clear(1.0), store: StoreOp::Store }), stencil_ops: None }),
                    timestamp_writes: None, occlusion_query_set: None, multiview_mask: None,
                });
                p.set_pipeline(&pl);
                p.set_bind_group(0, bg, &[]);
                p.draw(0..3, 0..1);
            }
            self.use_tex(id);
        }
        Some(out)
    }

    fn prepare_depth_proxies(&mut self, prog: &Program, stage: usize) {
        for b in prog.info.bindings.iter().filter(|b| b.kind == Kind::Texture) {
            let id = *self.s.srv[stage].get((b.binding as usize).wrapping_sub(32)).unwrap_or(&0);
            let Some(v) = self.view_obj(id) else { continue };
            if v.is_buf { continue; }
            let res = v.res;
            let Some(t) = self.tex(res) else { continue };
            if is_depth(t.format) && b.sample != Sample::Depth {
                self.depth_proxy(res, b.sample == Sample::Uint);
            } else if !is_depth(t.format) && b.sample == Sample::Depth {
                self.float_to_depth_proxy(res);
            }
        }
    }

    fn srv_view(&mut self, srv_id: u32, b: &Binding) -> Option<TextureView> {
        let (res, fmt, a, bb, c, d) = { let v = self.view_obj(srv_id)?; (v.res, v.fmt, v.a, v.b, v.c, v.d) };
        let t = self.tex(res)?;
        let (tformat, desc, decoded_bc) = (t.format, t.desc, t.compressed_source.is_some());
        let mut proxied = false;
        let (tex, format) = if is_depth(tformat) && b.sample != Sample::Depth {
            proxied = true;
            self.depth_proxy(res, b.sample == Sample::Uint)?
        } else if !is_depth(tformat) && b.sample == Sample::Depth {
            proxied = true;
            self.float_to_depth_proxy(res)?
        } else {
            (self.tex(res)?.tex.clone(), tformat)
        };
        let key = format!("{:?}{}", b.dim, if proxied { ":p" } else { "" });
        if let Some(v) = self.view_obj(srv_id).and_then(|v| v.views.get(&key)) { return Some(v.clone()); }
        let total = desc.mips;
        let mip_count = if bb == 0xffff_ffff || bb == 0 { total.saturating_sub(a) } else { bb };
        let layers = desc.d;
        if matches!(b.dim, TextureViewDimension::Cube | TextureViewDimension::CubeArray) && layers.saturating_sub(c) < 6 {
            log_once(&format!("cubeview{res}"), &format!("cube SRV over a texture with {layers} layer(s): bound as an empty cube"));
            return None;
        }
        let mut vd = TextureViewDescriptor {
            dimension: Some(b.dim),
            base_mip_level: a.min(total - 1),
            mip_level_count: Some(mip_count.min(total.saturating_sub(a)).max(1)),
            ..Default::default()
        };
        if matches!(b.dim, TextureViewDimension::D2Array | TextureViewDimension::Cube | TextureViewDimension::CubeArray | TextureViewDimension::D2) {
            let cnt = match b.dim { TextureViewDimension::Cube => 6, TextureViewDimension::D2 => 1, _ => if d != 0 { d } else { layers - c } };
            vd.base_array_layer = c.min(layers - 1);
            vd.array_layer_count = Some(cnt.min(layers.saturating_sub(c)).max(1));
        }
        if fmt != 0 && !is_depth(format) && !proxied {
            let mf = map_format(fmt, 0).format;
            let mf = if decoded_bc { crate::texture::fallback_format(mf) } else { mf };
            if mf != format && mf.is_srgb() != format.is_srgb() && mf.remove_srgb_suffix() == format.remove_srgb_suffix() { vd.format = Some(mf); }
        }
        if is_depth(format) && !proxied {
            vd.aspect = if fmt == 47 || fmt == 22 { TextureAspect::StencilOnly } else { TextureAspect::DepthOnly };
        }
        let v = self.scoped("SRV view", |_| tex.create_view(&vd))?;
        if !proxied {
            if let Some(vo) = self.view_obj_mut(srv_id) { vo.views.insert(key, v.clone()); }
        }
        Some(v)
    }

    fn uniform_alloc(&mut self, size: u64) -> (usize, u64) {
        let size = (size + 255) & !255;
        let slot = self.ubo_slot;
        let need_new = match self.ubo_slots[slot].get(self.ubo_chunk) { None => true, Some(_) => self.ubo_off + size + UBO_WINDOW > RING_CHUNK };
        if need_new {
            if self.ubo_slots[slot].get(self.ubo_chunk).is_some() { self.ubo_chunk += 1; self.ubo_off = 0; }
            if self.ubo_slots[slot].get(self.ubo_chunk).is_none() {
                while self.ubo_shared.len() <= self.ubo_chunk {
                    let buf = self.device.create_buffer(&BufferDescriptor { label: Some("uniform ring"), size: RING_CHUNK * FRAME_SLOTS as u64, usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST, mapped_at_creation: false });
                    let id = self.uid();
                    self.ubo_shared.push((buf, id));
                }
                let (buf, id) = self.ubo_shared[self.ubo_chunk].clone();
                while self.ubo_slots[slot].len() <= self.ubo_chunk {
                    self.ubo_slots[slot].push(Chunk { buf: buf.clone(), id, base: slot as u64 * RING_CHUNK, stage: Vec::new(), lo: u64::MAX, hi: 0 });
                }
            }
        }
        let off = self.ubo_off;
        self.ubo_off += size;
        (self.ubo_chunk, off)
    }
    fn flush_uniforms(&mut self) {
        for slot in &mut self.ubo_slots {
            for c in slot.iter_mut() {
                if c.lo >= c.hi { continue; }
                let hi = ((c.hi + 3) & !3).min(c.stage.len() as u64);
                self.queue.write_buffer(&c.buf, c.base + c.lo, &c.stage[c.lo as usize..hi as usize]);
                c.lo = u64::MAX;
                c.hi = 0;
            }
        }
    }

    fn pack_copy(&mut self, prog: &Program, stage: usize) -> Option<(usize, u64)> {
        let pack = prog.pack.as_ref()?;
        if pack.size > UBO_WINDOW {
            log_once(&format!("packbig{}", prog.label), &format!("{}: constant buffers total {} bytes (> 64 KB)", prog.label, pack.size));
            return None;
        }
        let (ci, off) = self.uniform_alloc(pack.size);
        let mut block = vec![0u8; pack.size as usize];
        for m in &pack.members {
            let cb = self.s.cb[stage].get(m.binding as usize).cloned().unwrap_or_default();
            if let Some(b) = self.buf(cb.buf) {
                b.copy_shadow(cb.first as u64 * 16, &mut block[m.offset as usize..(m.offset + m.size) as usize]);
            }
        }
        let chunk = &mut self.ubo_slots[self.ubo_slot][ci];
        let end = (off + pack.size) as usize;
        if chunk.stage.len() < end { chunk.stage.resize((end.max(262144)).next_power_of_two().min(RING_CHUNK as usize), 0); }
        chunk.stage[off as usize..end].copy_from_slice(&block);
        chunk.lo = chunk.lo.min(off);
        chunk.hi = chunk.hi.max(end as u64);
        Some((ci, off))
    }

    fn dummy_texture(&mut self, b: &Binding) -> TextureView {
        let key = format!("{:?}|{:?}|{}", b.dim, b.sample, b.ms);
        if let Some(v) = self.dummy_tex.get(&key) { return v.clone(); }
        let format = match b.sample { Sample::Uint => TextureFormat::Rgba8Uint, Sample::Sint => TextureFormat::Rgba8Sint, Sample::Depth => TextureFormat::Depth32Float, _ => TextureFormat::Rgba8Unorm };
        let cube = matches!(b.dim, TextureViewDimension::Cube | TextureViewDimension::CubeArray);
        let dimension = match b.dim { TextureViewDimension::D3 => TextureDimension::D3, TextureViewDimension::D1 => TextureDimension::D1, _ => TextureDimension::D2 };
        let mut usage = TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;
        if b.ms { usage |= TextureUsages::RENDER_ATTACHMENT; usage.remove(TextureUsages::COPY_DST); }
        let tex = self.device.create_texture(&TextureDescriptor {
            label: Some("dummy"), size: Extent3d { width: 1, height: 1, depth_or_array_layers: if cube { 6 } else { 1 } }, mip_level_count: 1,
            sample_count: if b.ms { 4 } else { 1 }, dimension, format, usage, view_formats: &[],
        });
        let v = tex.create_view(&TextureViewDescriptor { dimension: Some(b.dim), ..Default::default() });
        self.dummy_tex.insert(key, v.clone());
        v
    }
    fn dummy_sampler(&mut self, cmp_: bool) -> Sampler {
        if cmp_ {
            self.dummy_cmp.get_or_insert_with(|| self.device.create_sampler(&SamplerDescriptor { compare: Some(CompareFunction::Always), ..Default::default() })).clone()
        } else {
            self.dummy_smp.get_or_insert_with(|| self.device.create_sampler(&SamplerDescriptor::default())).clone()
        }
    }
    fn dummy_buffer(&mut self, binding: u32) -> Buffer {
        let dev = self.device.clone();
        self.dummy_bufs.entry(binding).or_insert_with(|| dev.create_buffer(&BufferDescriptor { label: Some("dummy"), size: 65536, usage: BufferUsages::STORAGE | BufferUsages::COPY_DST, mapped_at_creation: false })).clone()
    }
    fn counter_buffer(&mut self, buf_id: u32) -> Option<Buffer> {
        let dev = self.device.clone();
        let b = self.buf_mut(buf_id)?;
        Some(b.counter.get_or_insert_with(|| dev.create_buffer(&BufferDescriptor { label: Some("uav counter"), size: 16, usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST, mapped_at_creation: false })).clone())
    }

    fn stage_bind_group(&mut self, stage: usize, prog: &Program, group: u32, bgl: &(BindGroupLayout, u64)) -> Option<BgResult> {
        enum Res { Buf(Buffer, u64, Option<u64>), View(TextureView), Smp(Sampler) }
        let bindings = prog.group_bindings.get(&group).cloned().unwrap_or_default();
        let mut entries: Vec<(u32, Res)> = Vec::new();
        let mut key: Vec<u64> = vec![bgl.1, prog.uid, stage as u64];
        let mut dyn_off = None;
        for b in &bindings {
            match b.kind {
                Kind::Uniform => {
                    if b.binding == 15 {
                        match &prog.consts {
                            Some(c) => entries.push((15, Res::Buf(c.clone(), 0, None))),
                            None => { log_once(&format!("c15{}", prog.label), &format!("shader {} declares binding 15 but has no constant table", prog.label)); return None; }
                        }
                        continue;
                    }
                    let (ci, off) = self.pack_copy(prog, stage)?;
                    let c = &self.ubo_slots[self.ubo_slot][ci];
                    entries.push((0, Res::Buf(c.buf.clone(), 0, Some(prog.pack.as_ref().unwrap().size))));
                    dyn_off = Some((c.base + off) as u32);
                    key.push(c.id);
                }
                Kind::Sampler | Kind::SamplerCmp => {
                    let sid = *self.s.smp[stage].get((b.binding as usize).wrapping_sub(16)).unwrap_or(&0);
                    let want_cmp = b.kind == Kind::SamplerCmp;
                    let st = self.state(sid);
                    let s = match st.as_ref().and_then(|s| s.sampler.clone().filter(|_| ((s.words.first().copied().unwrap_or(0) & 0x80) != 0) == want_cmp)) {
                        Some(s) => { key.push(sid as u64); s }
                        None => { key.push(0); self.dummy_sampler(want_cmp) }
                    };
                    entries.push((b.binding, Res::Smp(s)));
                }
                Kind::Texture => {
                    let id = *self.s.srv[stage].get((b.binding as usize).wrapping_sub(32)).unwrap_or(&0);
                    let is_tex = self.view_obj(id).map(|v| !v.is_buf).unwrap_or(false);
                    if is_tex { if let Some(r) = self.view_obj(id).map(|v| v.res) { if self.pending_clears.contains_key(&r) { self.materialize_clears(r); } } }
                    let v = if is_tex { self.srv_view(id, b) } else { None };
                    match v {
                        Some(v) => {
                            let res = self.view_obj(id).unwrap().res;
                            self.use_tex(res);
                            if let Some(t) = self.tex_mut(res) { t.in_bg = true; }
                            key.push(id as u64);
                            key.push(0);
                            entries.push((b.binding, Res::View(v)));
                        }
                        None => { key.push(0); let d = self.dummy_texture(b); entries.push((b.binding, Res::View(d))); }
                    }
                }
                Kind::ReadOnlyStorage | Kind::Storage => {
                    let is_counter = b.binding >= 176;
                    let is_uav = b.binding >= 160;
                    let id = if is_uav { *self.s.uav[stage].get((b.binding - if is_counter { 176 } else { 160 }) as usize).unwrap_or(&0) } else { *self.s.srv[stage].get((b.binding as usize).wrapping_sub(32)).unwrap_or(&0) };
                    let buf_id = self.view_obj(id).filter(|v| v.is_buf).map(|v| (v.res, v.a, v.b));
                    let mut res_off = 0;
                    let mut res_size = None;
                    let gb = if is_counter {
                        match buf_id { Some((r, _, _)) => self.counter_buffer(r), None => None }.unwrap_or_else(|| self.dummy_buffer(b.binding))
                    } else {
                        match buf_id.and_then(|(r, a, n)| self.buf(r).and_then(|bo| bo.gpu.clone().map(|g| (g, bo.stride, bo.bytes, a, n)))) {
                            Some((g, stride, bytes, first, num)) => {
                                let stride = if stride == 0 { 4 } else { stride } as u64;
                                let off = first as u64 * stride;
                                if off != 0 && off % 256 == 0 {
                                    res_off = off;
                                    if num != 0 && num != 0xffff_ffff { res_size = Some((num as u64 * stride).min(bytes - off)); }
                                } else if off != 0 {
                                    log_once(&format!("bufoff{}", prog.label), &format!("{}: buffer view offset {off} is not 256-aligned; binding the whole buffer", prog.label));
                                }
                                g
                            }
                            None => self.dummy_buffer(b.binding),
                        }
                    };
                    if let Some((r, _, _)) = buf_id {
                        self.use_buf(r);
                        if let Some(bo) = self.buf_mut(r) { bo.in_bg = true; }
                        key.push(id as u64);
                    } else {
                        key.push(0);
                    }
                    key.push(res_off);
                    entries.push((b.binding, Res::Buf(gb, res_off, res_size)));
                }
                Kind::StorageTexture => {
                    let uid = *self.s.uav[stage].get((b.binding as usize).wrapping_sub(160)).unwrap_or(&0);
                    let (res, a, bb, c, vdim) = match self.view_obj(uid).filter(|v| !v.is_buf) { Some(v) => (v.res, v.a, v.b, v.c, v.vdim), None => {
                        log_once(&format!("nouavtex{}", prog.label), &format!("{}: storage texture u{} has no UAV bound", prog.label, b.binding - 160));
                        return None;
                    } };
                    if self.pending_clears.contains_key(&res) { self.materialize_clears(res); }
                    let (tex, tformat, layers) = { let t = self.tex(res)?; (t.tex.clone(), t.format, t.desc.d) };
                    let k2 = format!("{:?}:{:?}", b.dim, tformat);
                    let v = match self.view_obj(uid).and_then(|v| v.views.get(&k2).cloned()) {
                        Some(v) => v,
                        None => {
                            let mut d = TextureViewDescriptor { dimension: Some(b.dim), base_mip_level: a, mip_level_count: Some(1), ..Default::default() };
                            if b.dim == TextureViewDimension::D2Array {
                                d.base_array_layer = if vdim == 5 { bb } else { 0 };
                                d.array_layer_count = Some(if vdim == 5 { c } else { layers });
                            } else if b.dim == TextureViewDimension::D2 {
                                d.base_array_layer = if vdim == 5 { bb } else { 0 };
                                d.array_layer_count = Some(1);
                            }
                            let v = self.scoped("UAV view", |_| tex.create_view(&d))?;
                            self.view_obj_mut(uid).unwrap().views.insert(k2, v.clone());
                            v
                        }
                    };
                    self.use_tex(res);
                    self.mark_color_dirty(res);
                    if let Some(t) = self.tex_mut(res) { t.in_bg = true; }
                    key.push(uid as u64);
                    entries.push((b.binding, Res::View(v)));
                }
            }
        }
        if let Some((bg, uid)) = self.bind_groups.get(&key) {
            return Some(BgResult { bg: bg.clone(), uid: *uid, dyn_off });
        }
        let wentries: Vec<BindGroupEntry> = entries.iter().map(|(binding, r)| BindGroupEntry {
            binding: *binding,
            resource: match r {
                Res::Buf(b, off, size) => BindingResource::Buffer(BufferBinding { buffer: b, offset: *off, size: size.and_then(std::num::NonZeroU64::new) }),
                Res::View(v) => BindingResource::TextureView(v),
                Res::Smp(s) => BindingResource::Sampler(s),
            },
        }).collect();
        let bg = self.scoped(&format!("bind group {}", prog.label), |d| d.create_bind_group(&BindGroupDescriptor { label: None, layout: &bgl.0, entries: &wentries }))?;
        let uid = self.uid();
        self.bind_groups.insert(key, (bg.clone(), uid));
        Some(BgResult { bg, uid, dyn_off })
    }

    fn get_pipeline(&mut self, vs_id: u32, ps_id: u32, vs: &Rc<Program>, ps: Option<&Rc<Program>>, layout: &Rc<LayoutObj>) -> Option<Rc<Pl>> {
        let pi = self.pass_info.clone()?;
        let (topo, strip) = topology(self.s.topo);
        let strip_fmt = if strip && self.s.ib.0 != 0 { Some(if self.s.ib.1 == 57 { IndexFormat::Uint16 } else { IndexFormat::Uint32 }) } else { None };
        let mut key: Vec<i64> = vec![vs.uid as i64, ps.map(|p| p.uid as i64).unwrap_or(0), self.s.layout as i64];
        let strides: Vec<u32> = (0..16).map(|i| { let v = &self.s.vb[i]; if v.buf != 0 && self.buf(v.buf).map(|b| b.gpu.is_some()).unwrap_or(false) { v.stride } else { 0 } }).collect();
        for &(slot, _) in &layout.slots { key.push(strides[slot as usize] as i64); }
        key.extend([self.s.topo as i64, match strip_fmt { Some(IndexFormat::Uint16) => 1, Some(_) => 2, None => 0 }, self.s.blend.0 as i64, self.s.dss.0 as i64, self.s.rs as i64]);
        for f in &pi.rt_formats { key.push(f.map(|f| format_id(f)).unwrap_or(-1)); }
        key.extend([-7, pi.ds_format.map(format_id).unwrap_or(-1), pi.samples as i64, pi.ro_depth as i64, pi.ro_stencil as i64]);
        let _ = (vs_id, ps_id);
        match self.pipelines.get(&key) {
            Some(PipeEntry::Done(p)) => return p.clone(),
            Some(PipeEntry::Pending) => { self.pending_skips += 1; return None; }
            None => {}
        }
        self.pipe_creates += 1;
        let Some((job, vb_req)) = self.create_pipeline(vs, ps, layout, &strides, topo, strip_fmt, &pi) else {
            self.pipelines.insert(key, PipeEntry::Done(None));
            return None;
        };
        if let Some(tx) = self.pipe_jobs.clone() {
            let id = self.uid();
            self.pipe_pending.insert(id, (key.clone(), vb_req, job.label.clone()));
            self.pipelines.insert(key, PipeEntry::Pending);
            let _ = tx.send((id, job));
            self.pending_skips += 1;
            return None;
        }
        let pl = compile_pipeline(&self.device, &self.lost, &job).map(|pipeline| { let uid = self.uid(); Rc::new(Pl { pipeline, uid, vb_req, label: job.label }) });
        self.pipelines.insert(key, PipeEntry::Done(pl.clone()));
        pl
    }

    #[allow(clippy::too_many_arguments)]
    fn create_pipeline(&mut self, vs: &Rc<Program>, ps: Option<&Rc<Program>>, layout: &Rc<LayoutObj>, strides: &[u32], topo: PrimitiveTopology, strip_fmt: Option<IndexFormat>, pi: &PassInfo) -> Option<(PipeJob, Vec<VbReq>)> {
        let vs_obj_sig = self.vs_sig()?;
        let label = format!("{} + {}", vs.label, ps.map(|p| p.label.as_str()).unwrap_or("-"));
        let bgl0 = if vs.info.bindings.iter().any(|b| b.group == 0) { self.bgl_for(vs, 0, 0).0 } else { self.empty_bgl() };
        let bgl1 = match ps { Some(p) if p.info.bindings.iter().any(|b| b.group == 1) => self.bgl_for(p, 1, 1).0, _ => self.empty_bgl() };
        let pl_layout = self.device.create_pipeline_layout(&PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&bgl0), Some(&bgl1)], immediate_size: 0 });
        let remap = self.inter_stage_remap(vs, ps.map(|p| p.as_ref()));
        let vmod = self.vertex_module(vs, ps.map(|p| p.flat.as_slice()).unwrap_or(&[]), remap.as_ref())?;

        let mut by_slot: BTreeMap<u32, (Vec<VertexAttribute>, VertexStepMode)> = BTreeMap::new();
        for e in &layout.els {
            let Some(s) = vs_obj_sig.iter().find(|s| s.sem == e.sem && s.idx == e.idx) else { continue };
            let Some(f) = vertex_format(e.fmt) else { log_once(&format!("vf{}", e.fmt), &format!("unsupported vertex format DXGI {}", e.fmt)); continue };
            let ent = by_slot.entry(e.slot).or_insert_with(|| (Vec::new(), if e.cls == 1 { VertexStepMode::Instance } else { VertexStepMode::Vertex }));
            ent.0.push(VertexAttribute { format: f, offset: e.offset as u64, shader_location: s.reg });
        }
        let max_slot = by_slot.keys().next_back().map(|s| *s as usize + 1).unwrap_or(0);
        let mut vb_layouts: Vec<Option<(u64, VertexStepMode, Vec<VertexAttribute>)>> = vec![None; max_slot];
        let mut vb_req = Vec::new();
        for (slot, (attrs, step)) in &by_slot {
            let stride = strides[*slot as usize] as u64;
            vb_layouts[*slot as usize] = Some((stride, *step, attrs.clone()));
            let last = attrs.iter().map(|a| a.offset + a.format.size()).max().unwrap_or(0);
            vb_req.push(VbReq { slot: *slot, instance: *step == VertexStepMode::Instance, stride, last });
        }

        let rs = if self.s.rs != 0 { self.state(self.s.rs) } else { None };
        let blend = if self.s.blend.0 != 0 { self.state(self.s.blend.0) } else { None };
        let dss = if self.s.dss.0 != 0 { self.state(self.s.dss.0) } else { None };
        let mut primitive = PrimitiveState { topology: topo, strip_index_format: strip_fmt, ..Default::default() };
        match &rs {
            Some(r) => {
                let w = &r.words;
                primitive.cull_mode = match w.get(1) { Some(2) => Some(Face::Front), Some(3) => Some(Face::Back), _ => None };
                primitive.front_face = if w.get(2).copied().unwrap_or(0) != 0 { FrontFace::Ccw } else { FrontFace::Cw };
                if w.get(6).copied() == Some(0) && self.device.features().contains(Features::DEPTH_CLIP_CONTROL) { primitive.unclipped_depth = true; }
            }
            None => { primitive.cull_mode = Some(Face::Back); primitive.front_face = FrontFace::Cw; }
        }
        let depth_stencil = pi.ds_format.map(|fmt| {
            let mut d = DepthStencilState { format: fmt, depth_write_enabled: Some(false), depth_compare: Some(CompareFunction::Always), stencil: StencilState::default(), bias: DepthBiasState::default() };
            if let Some(s) = &dss {
                let w = &s.words;
                let g = |i: usize| w.get(i).copied().unwrap_or(0);
                if g(0) != 0 { d.depth_write_enabled = Some(g(1) == 1); d.depth_compare = Some(cmp(g(2)).unwrap_or(CompareFunction::Less)); }
                if has_stencil(fmt) && g(3) != 0 {
                    d.stencil.read_mask = g(4) & 0xff;
                    d.stencil.write_mask = (g(4) >> 8) & 0xff;
                    d.stencil.front = StencilFaceState { fail_op: stencil_op(g(5)), depth_fail_op: stencil_op(g(6)), pass_op: stencil_op(g(7)), compare: cmp(g(8)).unwrap_or(CompareFunction::Always) };
                    d.stencil.back = StencilFaceState { fail_op: stencil_op(g(9)), depth_fail_op: stencil_op(g(10)), pass_op: stencil_op(g(11)), compare: cmp(g(12)).unwrap_or(CompareFunction::Always) };
                }
            }
            if pi.ro_depth { d.depth_write_enabled = Some(false); }
            if pi.ro_stencil { d.stencil.write_mask = 0; }
            if let Some(r) = &rs { d.bias = DepthBiasState { constant: r.words.get(3).copied().unwrap_or(0) as i32, slope_scale: wf(r.words.get(5).copied().unwrap_or(0)), clamp: wf(r.words.get(4).copied().unwrap_or(0)) }; }
            d
        });
        let a2c = blend.as_ref().map(|b| b.words.first().copied().unwrap_or(0) != 0).unwrap_or(false) && pi.samples <= 1 && ps.map(|p| p.info.outputs.contains(&0)).unwrap_or(false);
        let mut targets: Vec<Option<ColorTargetState>> = Vec::new();
        let fmod = match ps {
            Some(p) => {
                for (i, f) in pi.rt_formats.iter().enumerate() {
                    let Some(format) = f else { targets.push(None); continue };
                    if !p.info.outputs.contains(&(i as u32)) { targets.push(Some(ColorTargetState { format: *format, blend: None, write_mask: ColorWrites::empty() })); continue; }
                    let (bl, write) = blend_for(blend.as_deref(), i);

                    let blendable = self.adapter.get_texture_format_features(*format).flags.contains(TextureFormatFeatureFlags::BLENDABLE);
                    targets.push(Some(ColorTargetState { format: *format, blend: if blendable { bl } else { None }, write_mask: write }));
                }
                let missing: Vec<_> = p.info.outputs.iter().filter(|&&l| pi.rt_formats.get(l as usize).copied().flatten().is_none()).collect();
                if !missing.is_empty() {
                    log_once(&format!("psout{}", p.label), &format!("{} writes colour output(s) {:?} with no render target bound; the pass had {:?} (textures {:?})", p.label, missing, pi.rt_formats, pi.rt_ids));
                }
                Some(self.pixel_module(p, a2c, remap.as_ref(), vs.clip.as_ref())?)
            }
            None => None,
        };
        let multisample = MultisampleState { count: pi.samples.max(1), mask: !0, alpha_to_coverage_enabled: pi.samples > 1 && blend.as_ref().map(|b| b.words.first().copied().unwrap_or(0) != 0).unwrap_or(false) };
        Some((PipeJob { label, layout: pl_layout, vmod, fmod, vbs: vb_layouts, primitive, depth_stencil, multisample, targets }, vb_req))
    }

    fn collect_pipelines(&mut self) {
        while let Ok((id, pipeline)) = self.pipe_results.try_recv() {
            let Some((key, vb_req, label)) = self.pipe_pending.remove(&id) else { continue };
            let entry = pipeline.map(|pipeline| { let uid = self.uid(); Rc::new(Pl { pipeline, uid, vb_req, label }) });
            self.pipelines.insert(key, PipeEntry::Done(entry));
            self.pdirty = true;
        }
    }

    fn vs_sig(&self) -> Option<Vec<SigEl>> {
        match self.obj(self.s.shader[0]) { Some(Obj::Shader(s)) => Some(s.sig.clone()), _ => None }
    }

    fn skip(&mut self) {
        self.skipped += 1;
    }

    fn do_draw(&mut self, p: u64) {
        if self.dbg_idx != 0 {
            let lim = self.mem.word(self.dbg_idx).load(Ordering::Relaxed);
            if lim >= 0 { }
        }
        let mb = self.base;
        let u = move |w: u64| unsafe { std::ptr::read_volatile(mb.add(((p + w) * 4) as usize) as *const u32) };
        let (kind, count, inst, start, base_vertex, start_instance) = (u(0), u(1), u(2), u(3), u(4) as i32, u(5));
        self.collect_pipelines();
        let vs_id = self.s.shader[0];
        let ps_id = self.s.shader[1];
        let Some(vs) = self.shader_program(vs_id) else { self.skip(); return };
        let ps = if ps_id != 0 { match self.shader_program(ps_id) { Some(p) => Some(p), None => { self.skip(); return } } } else { None };
        let layout = match self.obj(self.s.layout) { Some(Obj::Layout(l)) => l.clone(), _ => { self.skip(); return } };
        self.prepare_depth_proxies(&vs, 0);
        if let Some(p) = &ps { self.prepare_depth_proxies(p, 1); }
        if !self.ensure_pass() { self.skip(); return; }
        let pl = if !self.pdirty && self.last_pl.is_some() {
            self.last_pl.clone()
        } else {
            let pl = self.get_pipeline(vs_id, ps_id, &vs, ps.as_ref(), &layout);
            self.last_pl = pl.clone();
            self.pdirty = false;
            pl
        };
        let Some(pl) = pl else { self.skip(); return };

        let g0 = if vs.group_bindings.contains_key(&0) {
            let bgl = self.bgl_for(&vs, 0, 0);
            match self.stage_bind_group(0, &vs, 0, &bgl) { Some(g) => Some(g), None => { self.skip(); return } }
        } else { None };
        let g1 = match &ps {
            Some(p) if p.group_bindings.contains_key(&1) => {
                let bgl = self.bgl_for(p, 1, 1);
                match self.stage_bind_group(1, p, 1, &bgl) { Some(g) => Some(g), None => { self.skip(); return } }
            }
            _ => None,
        };

        if self.pass.is_none() && !self.ensure_pass() { self.skip(); return; }

        let indexed = kind == 4 || (kind < 4 && (kind & 1) != 0);
        let (fold_v, fold_i) = (vs.vid, vs.iid);
        let v_first = if indexed || fold_v { 0 } else { start };
        let i_first = if fold_i { 0 } else { start_instance };

        let mut vb_binds: Vec<(u32, Buffer, u64, u64)> = Vec::new();
        for &(slot, cls) in &layout.slots {
            let vb = self.s.vb[slot as usize].clone();
            match self.buf(vb.buf).and_then(|b| b.gpu.clone().map(|g| (g, b.bytes))) {
                None => {
                    let dev = self.device.clone();
                    let z = self.zero_vb.get_or_insert_with(|| dev.create_buffer(&BufferDescriptor { label: Some("zero vb"), size: ZERO_VB_SIZE, usage: BufferUsages::VERTEX, mapped_at_creation: false })).clone();
                    self.vb_avail[slot as usize] = ZERO_VB_SIZE;
                    vb_binds.push((slot, z, 0, u64::MAX));
                }
                Some((g, bytes)) => {
                    let shift: i64 = if cls == 1 { if fold_i { start_instance as i64 * vb.stride as i64 } else { 0 } } else if fold_v { (if indexed { base_vertex as i64 } else { start as i64 }) * vb.stride as i64 } else { 0 };
                    let voff = (vb.offset as i64 + shift).max(0) as u64;
                    self.vb_avail[slot as usize] = bytes.saturating_sub(voff);
                    vb_binds.push((slot, g, voff, vb.buf as u64));
                    self.use_buf(vb.buf);
                }
            }
        }
        let mut ib_max = u64::MAX;
        let mut ib_bind = None;
        if indexed && kind < 4 {
            let (ibid, ifmt, ioff) = self.s.ib;
            let Some((g, bytes)) = self.buf(ibid).and_then(|b| b.gpu.clone().map(|g| (g, b.bytes))) else { self.skip(); return };
            let fmt = if ifmt == 57 { IndexFormat::Uint16 } else { IndexFormat::Uint32 };
            let isz = if fmt == IndexFormat::Uint16 { 2 } else { 4 };
            ib_max = (bytes.saturating_sub(ioff as u64) / isz).saturating_sub(start as u64);
            ib_bind = Some((g, fmt, ioff as u64, ibid));
            self.use_buf(ibid);
        } else if kind == 4 {
            let (ibid, ifmt, ioff) = self.s.ib;
            if let Some(g) = self.buf(ibid).and_then(|b| b.gpu.clone()) {
                ib_bind = Some((g, if ifmt == 57 { IndexFormat::Uint16 } else { IndexFormat::Uint32 }, ioff as u64, ibid));
                self.use_buf(ibid);
            }
        }
        let pi = self.pass_info.clone().unwrap();
        let rs_scissor = self.s.rs != 0 && self.state(self.s.rs).map(|r| r.words.get(7).copied().unwrap_or(0) != 0).unwrap_or(false);
        let scissor = if rs_scissor && !self.s.scissors.is_empty() {
            let c = self.s.scissors[0];
            let x0 = pi.width.min(c[0]);
            let y0 = pi.height.min(c[1]);
            let x1 = pi.width.min(x0.max(c[2]));
            let y1 = pi.height.min(y0.max(c[3]));
            [x0, y0, x1 - x0, y1 - y0]
        } else {
            [0, 0, pi.width, pi.height]
        };
        let viewport = self.s.viewports.first().copied();
        let bf = self.s.blend.1;
        let sref = self.s.dss.1;

        let (mut d_count, mut d_inst) = (count as i64, inst as i64);
        if kind < 4 {
            for r in &pl.vb_req {
                let avail = self.vb_avail[r.slot as usize];
                let n: i64 = if avail < r.last { 0 } else if r.stride != 0 { ((avail - r.last) / r.stride + 1) as i64 } else { i64::MAX };
                if r.instance { d_inst = d_inst.min(n - i_first as i64); } else if !indexed { d_count = d_count.min(n - v_first as i64); }
            }
            if indexed { d_count = d_count.min(ib_max as i64); }
            if d_count != count as i64 || d_inst != inst as i64 {
                log_once(&format!("clamp{}", pl.label), &format!("draw clamped to its buffers ({}): {count} x {inst} -> {d_count} x {d_inst}", pl.label));
                if d_count <= 0 || d_inst <= 0 { self.skip(); return; }
            }
        }
        let args = if kind >= 4 {
            match self.buf(u(1)).and_then(|b| b.gpu.clone()) { Some(g) => { self.use_buf(u(1)); Some((g, u(2) as u64)) } None => { self.skip(); return } }
        } else { None };

        let st = &mut self.pass_st;
        let pass = self.pass.as_mut().unwrap();
        if st.pl != Some(pl.uid) { pass.set_pipeline(&pl.pipeline); st.pl = Some(pl.uid); }
        for (i, g) in [g0, g1].into_iter().enumerate() {
            if let Some(g) = g {
                let d = g.dyn_off.unwrap_or(u32::MAX);
                if st.bg[i] != Some((g.uid, d)) {
                    match g.dyn_off { Some(o) => pass.set_bind_group(i as u32, &g.bg, &[o]), None => pass.set_bind_group(i as u32, &g.bg, &[]) }
                    st.bg[i] = Some((g.uid, d));
                }
            }
        }
        if st.vb.len() < 16 { st.vb.resize(16, None); }
        for (slot, g, off, ident) in &vb_binds {
            let key = (*ident, *off);
            if st.vb[*slot as usize] != Some(key) {
                pass.set_vertex_buffer(*slot, g.slice(*off..));
                st.vb[*slot as usize] = Some(key);
            }
        }
        if let Some((g, fmt, off, ident)) = &ib_bind {
            if st.ib != Some((*ident as u64, *fmt, *off)) {
                pass.set_index_buffer(g.slice(*off..), *fmt);
                st.ib = Some((*ident as u64, *fmt, *off));
            }
        }
        if let Some(v) = viewport {
            if st.vp != Some(v) {
                pass.set_viewport(v[0], v[1], v[2].max(1.0), v[3].max(1.0), v[4].clamp(0.0, 1.0), v[5].clamp(v[4].clamp(0.0, 1.0), 1.0));
                st.vp = Some(v);
            }
        }
        if st.sc != Some(scissor) { pass.set_scissor_rect(scissor[0], scissor[1], scissor[2], scissor[3]); st.sc = Some(scissor); }
        if st.bf != Some(bf) { pass.set_blend_constant(Color { r: bf[0], g: bf[1], b: bf[2], a: bf[3] }); st.bf = Some(bf); }
        if st.sref != Some(sref) { pass.set_stencil_reference(sref); st.sref = Some(sref); }
        match kind {
            4 => { let (g, off) = args.unwrap(); pass.draw_indexed_indirect(&g, off); }
            5.. => { let (g, off) = args.unwrap(); pass.draw_indirect(&g, off); }
            _ => {
                if indexed {
                    pass.draw_indexed(start..start + d_count as u32, if fold_v { 0 } else { base_vertex }, i_first..i_first + d_inst as u32);
                } else {
                    pass.draw(v_first..v_first + d_count as u32, i_first..i_first + d_inst as u32);
                }
            }
        }
        self.draws += 1;
    }

    fn do_dispatch(&mut self, x: u32, y: u32, z: u32) {
        let cs_id = self.s.shader[2];
        let Some(prog) = self.shader_program(cs_id) else { self.skipped += 1; return };
        self.prepare_depth_proxies(&prog, 2);
        self.end_pass();

        let mut subs = Vec::new();
        for b in prog.group_bindings.get(&0).cloned().unwrap_or_default() {
            if b.kind != Kind::StorageTexture { continue; }
            let uid = *self.s.uav[2].get((b.binding as usize).wrapping_sub(160)).unwrap_or(&0);
            let tf = self.view_obj(uid).filter(|v| !v.is_buf).and_then(|v| self.tex(v.res)).map(|t| t.format);
            if let (Some(tf), Some(bf)) = (tf, b.format) {
                if tf != bf { if let Some(name) = wgsl_format_name(tf) { subs.push((b.binding, b.format_name.clone(), name.to_string())); } }
            }
        }
        let key = format!("{}|{}", prog.uid, subs.iter().map(|s| format!("{}:{}", s.0, s.2)).collect::<Vec<_>>().join(";"));
        if !self.compute.contains_key(&key) {
            let cprog = if subs.is_empty() { prog.clone() } else {
                let mut text = prog.text.clone();
                for (binding, from, to) in &subs {
                    let re = Regex::new(&format!(r"(@binding\({binding}u?\)\s*var\s+\w+\s*:\s*texture_storage_\w+<){}", regex::escape(from))).unwrap();
                    text = re.replace(&text, format!("${{1}}{to}").as_str()).into_owned();
                }
                let label = format!("{} [uav formats]", prog.label);
                match self.scoped("compute variant", |d| d.create_shader_module(ShaderModuleDescriptor { label: Some(&label), source: ShaderSource::Wgsl(text.as_str().into()) })) {
                    Some(module) => {
                        let info = parse_wgsl(&text);
                        let mut gb: BTreeMap<u32, Vec<Binding>> = BTreeMap::new();
                        for b in &info.bindings { gb.entry(b.group).or_default().push(b.clone()); }
                        for v in gb.values_mut() { v.sort_by_key(|b| b.binding); }
                        let uid = self.uid();
                        Rc::new(Program { module, info, consts: prog.consts.clone(), pack: prog.pack.as_ref().map(|p| Pack { members: p.members.iter().map(|m| PackMember { binding: m.binding, size: m.size, offset: m.offset }).collect(), size: p.size }), text, flat: vec![], group_bindings: gb, variants: RefCell::new(HashMap::new()), label: prog.label.clone(), uid, vid: false, iid: false, out_locs: vec![], in_locs: vec![], clip: None })
                    }
                    None => { self.compute.insert(key.clone(), None); self.skipped += 1; return; }
                }
            };
            let entries: Vec<_> = cprog.info.bindings.iter().filter(|b| b.group == 0).map(|b| Self::layout_entry(b, 2, self.f32_filterable)).collect();
            let bgl = self.device.create_bind_group_layout(&BindGroupLayoutDescriptor { label: None, entries: &entries });
            let layout = self.device.create_pipeline_layout(&PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&bgl)], immediate_size: 0 });
            let label = cprog.label.clone();
            let pl = self.scoped(&format!("compute pipeline {label}"), |d| d.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some(&label), layout: Some(&layout), module: &cprog.module, entry_point: Some("main"), compilation_options: Default::default(), cache: None,
            }));
            let uid = self.uid();
            self.compute.insert(key.clone(), pl.map(|p| (p, bgl, uid, cprog)));
        }
        let Some((pl, bgl, bgl_uid, cprog)) = self.compute.get(&key).cloned().flatten() else { self.skipped += 1; return };
        let Some(g0) = self.stage_bind_group(2, &cprog, 0, &(bgl, bgl_uid)) else { self.skipped += 1; return };
        let enc = self.encoder();
        let mut cp = enc.begin_compute_pass(&ComputePassDescriptor::default());
        cp.set_pipeline(&pl);
        match g0.dyn_off { Some(o) => cp.set_bind_group(0, &g0.bg, &[o]), None => cp.set_bind_group(0, &g0.bg, &[]) }
        cp.dispatch_workgroups(x, y, z);
    }

    fn clear_view(&mut self, id: u32, mip: u32, slice: u32) -> Option<TextureView> {
        let t = self.tex(id)?;
        if t.desc.dim == TextureDimension::D3 {
            return Some(t.tex.create_view(&TextureViewDescriptor { base_mip_level: mip, mip_level_count: Some(1), base_array_layer: slice, array_layer_count: Some(1), dimension: Some(TextureViewDimension::D2), ..Default::default() }));
        }
        self.target_view(id, 0, mip, slice).map(|v| v.0)
    }

    fn in_open_pass(&self, id: u32) -> bool {
        self.pass.is_some() && self.pass_key.as_ref().map(|k| (0..9).any(|i| k[i * 4] == id)).unwrap_or(false)
    }
    fn clear_rt(&mut self, id: u32, mip: u32, slice: u32, c: [f32; 4]) {
        let Some(three) = self.tex(id).map(|t| t.desc.dim == TextureDimension::D3) else { return };
        self.mark_color_dirty(id);
        if three { self.materialize_clears(id); self.clear_rt_now(id, mip, slice, Color { r: c[0] as f64, g: c[1] as f64, b: c[2] as f64, a: c[3] as f64 }); return; }
        if self.in_open_pass(id) { self.end_pass(); }
        let list = self.pending_clears.entry(id).or_default();
        let color = Color { r: c[0] as f64, g: c[1] as f64, b: c[2] as f64, a: c[3] as f64 };
        match list.iter_mut().find(|e| e.0 == mip && e.1 == slice) {
            Some(e) => e.2.color = Some(color),
            None => list.push((mip, slice, PendingClear { color: Some(color), ..Default::default() })),
        }
    }
    fn clear_ds(&mut self, id: u32, mip: u32, slice: u32, flags: u32, depth: f32, stencil: u32) {
        let Some((fmt, three)) = self.tex(id).map(|t| (t.format, t.desc.dim == TextureDimension::D3)) else { return };
        self.mark_depth_dirty(id);
        let d = (flags & 1 != 0).then(|| depth.clamp(0.0, 1.0));
        let st = (flags & 2 != 0 && has_stencil(fmt)).then_some(stencil);
        if d.is_none() && st.is_none() { return; }
        if three { self.materialize_clears(id); self.clear_ds_now(id, mip, slice, d, st); return; }
        if self.in_open_pass(id) { self.end_pass(); }
        let list = self.pending_clears.entry(id).or_default();
        match list.iter_mut().find(|e| e.0 == mip && e.1 == slice) {
            Some(e) => { if d.is_some() { e.2.depth = d; } if st.is_some() { e.2.stencil = st; } }
            None => list.push((mip, slice, PendingClear { depth: d, stencil: st, ..Default::default() })),
        }
    }

    fn materialize_clears(&mut self, id: u32) {
        let Some(list) = self.pending_clears.remove(&id) else { return };
        self.end_pass();
        for (mip, slice, c) in list {
            if let Some(color) = c.color { self.clear_rt_now(id, mip, slice, color); }
            if c.depth.is_some() || c.stencil.is_some() { self.clear_ds_now(id, mip, slice, c.depth, c.stencil); }
        }
    }
    fn clear_rt_now(&mut self, id: u32, mip: u32, slice: u32, color: Color) {
        if self.tex(id).is_none() { return; }
        self.end_pass();
        let Some(view) = self.clear_view(id, mip, slice) else { return };
        self.use_tex(id);
        self.encoder().begin_render_pass(&RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(RenderPassColorAttachment { view: &view, depth_slice: None, resolve_target: None, ops: Operations { load: LoadOp::Clear(color), store: StoreOp::Store } })],
            depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None, multiview_mask: None,
        });
    }
    fn clear_ds_now(&mut self, id: u32, mip: u32, slice: u32, depth: Option<f32>, stencil: Option<u32>) {
        let Some(fmt) = self.tex(id).map(|t| t.format) else { return };
        self.end_pass();
        let Some(view) = self.clear_view(id, mip, slice) else { return };
        self.use_tex(id);
        self.encoder().begin_render_pass(&RenderPassDescriptor {
            label: None, color_attachments: &[],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: &view,
                depth_ops: Some(Operations { load: depth.map(LoadOp::Clear).unwrap_or(LoadOp::Load), store: StoreOp::Store }),
                stencil_ops: if has_stencil(fmt) { Some(Operations { load: stencil.map(LoadOp::Clear).unwrap_or(LoadOp::Load), store: StoreOp::Store }) } else { None },
            }),
            timestamp_writes: None, occlusion_query_set: None, multiview_mask: None,
        });
    }
    fn clear_uav(&mut self, id: u32, values: [u32; 4]) {
        let Some((res, is_buf, a)) = self.view_obj(id).map(|v| (v.res, v.is_buf, v.a)) else { return };
        let zero = values.iter().all(|&v| v == 0);
        if is_buf {
            let Some((g, bytes)) = self.buf(res).and_then(|b| b.gpu.clone().map(|g| (g, b.bytes))) else { return };
            self.end_pass();
            if zero {
                self.encoder().clear_buffer(&g, 0, None);
            } else {
                self.flush();
                let data: Vec<u8> = (0..bytes / 4).flat_map(|i| values[(i & 3) as usize].to_le_bytes()).collect();
                self.queue.write_buffer(&g, 0, &data);
            }
            self.use_buf(res);
            return;
        }
        self.materialize_clears(res);
        let Some((tex, fmt, desc)) = self.tex(res).map(|t| (t.tex.clone(), t.format, t.desc)) else { return };
        let bpt = texel_bytes(fmt);
        if bpt == 0 { log_once(&format!("clearuav{fmt:?}"), &format!("ClearUnorderedAccessView: texture format {fmt:?} not supported")); return; }
        let (w, h) = ((desc.w >> a).max(1), (desc.h >> a).max(1));
        let layers = if desc.dim == TextureDimension::D3 { (desc.d >> a).max(1) } else { desc.d };
        if zero {
            let bpr = (w * bpt).div_ceil(256) * 256;
            let need = (bpr * h) as u64;
            if self.zero_src.as_ref().map(|z| z.size() < need).unwrap_or(true) {
                self.zero_src = Some(self.device.create_buffer(&BufferDescriptor { label: Some("zero"), size: need.max(1 << 20), usage: BufferUsages::COPY_SRC, mapped_at_creation: false }));
            }
            let z = self.zero_src.clone().unwrap();
            self.end_pass();
            let enc = self.encoder();
            for l in 0..layers {
                enc.copy_buffer_to_texture(
                    TexelCopyBufferInfo { buffer: &z, layout: TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bpr), rows_per_image: Some(h) } },
                    TexelCopyTextureInfo { texture: &tex, mip_level: a, origin: Origin3d { x: 0, y: 0, z: l }, aspect: TextureAspect::All },
                    Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                );
            }
            self.use_tex(res);
            self.mark_color_dirty(res);
            return;
        }
        self.flush();
        let mut data = vec![0u8; (w * h * bpt) as usize];
        for i in 0..(w * h) as usize {
            for c in 0..(bpt / 4) as usize {
                data[i * bpt as usize + c * 4..i * bpt as usize + c * 4 + 4].copy_from_slice(&values[c & 3].to_le_bytes());
            }
        }
        for l in 0..layers {
            self.queue.write_texture(
                TexelCopyTextureInfo { texture: &tex, mip_level: a, origin: Origin3d { x: 0, y: 0, z: l }, aspect: TextureAspect::All },
                &data, TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * bpt), rows_per_image: Some(h) },
                Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            );
        }
    }

    fn sub_of(desc: &TexDesc, sub: u32) -> (u32, u32) {
        (sub % desc.mips, sub / desc.mips)
    }
    #[allow(clippy::too_many_arguments)]
    fn copy_region(&mut self, dst: u32, dst_sub: u32, dx: u32, dy: u32, dz: u32, src: u32, src_sub: u32, has_box: bool, l: u32, t: u32, f: u32, r: u32, b: u32, bk: u32) {
        self.materialize_clears(src);
        self.materialize_clears(dst);
        if self.obj(src).is_none() || self.obj(dst).is_none() { return; }
        self.end_pass();
        if let (Some(sb), Some(db)) = (self.buf(src), self.buf(dst)) {
            let (Some(sg), Some(dg)) = (sb.gpu.clone(), db.gpu.clone()) else { return };
            let size = if has_box { (r - l) as u64 } else { sb.size.min(db.size) };
            let size = size & !3;
            if size > 0 { self.encoder().copy_buffer_to_buffer(&sg, if has_box { l as u64 } else { 0 }, &dg, dx as u64, size); }
            self.use_buf(src);
            self.use_buf(dst);
            return;
        }
        let (Some(s), Some(d)) = (self.tex(src), self.tex(dst)) else { return };
        let (ss, ds) = (Self::sub_of(&s.desc, src_sub), Self::sub_of(&d.desc, dst_sub));
        let (mw, mh, md) = ((s.desc.w >> ss.0).max(1), (s.desc.h >> ss.0).max(1), if s.desc.dim == TextureDimension::D3 { (s.desc.d >> ss.0).max(1) } else { 1 });
        let (w, h, dep) = if has_box { (r - l, b - t, bk - f) } else { (mw, mh, md) };
        let three = s.desc.dim == TextureDimension::D3;
        let d3 = d.desc.dim == TextureDimension::D3;
        let (stex, dtex, sf, df) = (s.tex.clone(), d.tex.clone(), s.format, d.format);
        if sf != df && !three && !d3 && (block_bytes(sf) != 0) != (block_bytes(df) != 0) && block_bytes(sf).max(block_bytes(df)) == texel_bytes(sf).max(texel_bytes(df)) {
            let bytes = block_bytes(sf).max(block_bytes(df));
            let s_bc = block_bytes(sf) != 0;
            let (cw, ch) = if s_bc { (w.div_ceil(4), h.div_ceil(4)) } else { (w, h) };
            let bpr = (cw * bytes).div_ceil(256) * 256;
            let need = (bpr * ch) as u64;
            if self.scratch.as_ref().map(|x| x.size() < need).unwrap_or(true) {
                self.scratch = Some(self.device.create_buffer(&BufferDescriptor { label: Some("scratch"), size: need.max(1 << 16), usage: BufferUsages::COPY_SRC | BufferUsages::COPY_DST, mapped_at_creation: false }));
            }
            let sc = self.scratch.clone().unwrap();
            let enc = self.encoder();
            enc.copy_texture_to_buffer(
                TexelCopyTextureInfo { texture: &stex, mip_level: ss.0, origin: Origin3d { x: if has_box { l } else { 0 }, y: if has_box { t } else { 0 }, z: ss.1 }, aspect: TextureAspect::All },
                TexelCopyBufferInfo { buffer: &sc, layout: TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bpr), rows_per_image: Some(ch) } },
                Extent3d { width: if s_bc { cw * 4 } else { cw }, height: if s_bc { ch * 4 } else { ch }, depth_or_array_layers: 1 },
            );
            enc.copy_buffer_to_texture(
                TexelCopyBufferInfo { buffer: &sc, layout: TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bpr), rows_per_image: Some(ch) } },
                TexelCopyTextureInfo { texture: &dtex, mip_level: ds.0, origin: Origin3d { x: dx, y: dy, z: ds.1 }, aspect: TextureAspect::All },
                Extent3d { width: if !s_bc { cw * 4 } else { cw }, height: if !s_bc { ch * 4 } else { ch }, depth_or_array_layers: 1 },
            );
        } else {
            let enc = self.encoder();
            enc.copy_texture_to_texture(
                TexelCopyTextureInfo { texture: &stex, mip_level: ss.0, origin: Origin3d { x: if has_box { l } else { 0 }, y: if has_box { t } else { 0 }, z: if three { if has_box { f } else { 0 } } else { ss.1 } }, aspect: TextureAspect::All },
                TexelCopyTextureInfo { texture: &dtex, mip_level: ds.0, origin: Origin3d { x: dx, y: dy, z: if d3 { dz } else { ds.1 } }, aspect: TextureAspect::All },
                copy_extent(sf, w, h, if three { dep } else { 1 }),
            );
        }
        self.mark_depth_dirty(dst);
        self.mark_color_dirty(dst);
        self.use_tex(src);
        self.use_tex(dst);
    }
    fn copy_resource(&mut self, dst: u32, src: u32) {
        self.materialize_clears(src);
        self.materialize_clears(dst);
        if self.buf(src).is_some() { self.copy_region(dst, 0, 0, 0, 0, src, 0, false, 0, 0, 0, 0, 0, 0); return; }
        let (Some(s), Some(d)) = (self.tex(src), self.tex(dst)) else { return };
        let (stex, dtex, sf, sd, dd) = (s.tex.clone(), d.tex.clone(), s.format, s.desc, d.desc);
        self.end_pass();
        let enc = self.encoder();
        for mip in 0..sd.mips.min(dd.mips) {
            let (w, h) = ((sd.w >> mip).max(1), (sd.h >> mip).max(1));
            let layers = if sd.dim == TextureDimension::D3 { (sd.d >> mip).max(1) } else { sd.d };
            enc.copy_texture_to_texture(
                TexelCopyTextureInfo { texture: &stex, mip_level: mip, origin: Origin3d::ZERO, aspect: TextureAspect::All },
                TexelCopyTextureInfo { texture: &dtex, mip_level: mip, origin: Origin3d::ZERO, aspect: TextureAspect::All },
                copy_extent(sf, w, h, layers),
            );
        }
        self.mark_depth_dirty(dst);
        self.mark_color_dirty(dst);
        self.use_tex(src);
        self.use_tex(dst);
    }
    fn resolve(&mut self, dst: u32, dst_sub: u32, src: u32, src_sub: u32) {
        self.materialize_clears(src);
        self.materialize_clears(dst);
        let (Some(s), Some(d)) = (self.tex(src), self.tex(dst)) else { return };
        let (ss, ds) = (Self::sub_of(&s.desc, src_sub), Self::sub_of(&d.desc, dst_sub));
        let sv = s.tex.create_view(&TextureViewDescriptor { base_mip_level: ss.0, mip_level_count: Some(1), base_array_layer: ss.1, array_layer_count: Some(1), dimension: Some(TextureViewDimension::D2), ..Default::default() });
        let dv = d.tex.create_view(&TextureViewDescriptor { base_mip_level: ds.0, mip_level_count: Some(1), base_array_layer: ds.1, array_layer_count: Some(1), dimension: Some(TextureViewDimension::D2), ..Default::default() });
        self.end_pass();
        self.encoder().begin_render_pass(&RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(RenderPassColorAttachment { view: &sv, depth_slice: None, resolve_target: Some(&dv), ops: Operations { load: LoadOp::Load, store: StoreOp::Discard } })],
            depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None, multiview_mask: None,
        });
        self.mark_color_dirty(dst);
        self.use_tex(src);
        self.use_tex(dst);
    }

    fn generate_mips(&mut self, srv_id: u32) {
        let Some(res) = self.view_obj(srv_id).map(|v| v.res) else { return };
        self.materialize_clears(res);
        let Some((fmt, desc, tex)) = self.tex(res).map(|t| (t.format, t.desc, t.tex.clone())) else { return };
        if desc.mips < 2 { return; }
        if is_depth(fmt) || is_bc(fmt) || desc.dim == TextureDimension::D3 { log_once(&format!("mips{fmt:?}"), &format!("GenerateMips: unsupported format/dimension {fmt:?}")); return; }
        self.end_pass();
        if !self.mip_pipelines.contains_key(&fmt) {
            let module = self.device.create_shader_module(ShaderModuleDescriptor { label: None, source: ShaderSource::Wgsl(
                "@group(0) @binding(0) var src: texture_2d<f32>;
                 @group(0) @binding(1) var smp: sampler;
                 @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f { var p = array<vec2f, 3>(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3)); return vec4f(p[i], 0, 1); }
                 @fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
                   let dst = max(vec2f(1.0), floor(vec2f(textureDimensions(src, 0)) * 0.5));
                   return textureSampleLevel(src, smp, pos.xy / dst, 0.0);
                 }".into()) });
            let pl = self.simple_pipeline(&module, Some(fmt), None);
            self.mip_pipelines.insert(fmt, pl);
        }
        let pl = self.mip_pipelines[&fmt].clone();
        let dev = self.device.clone();
        let smp = self.mip_sampler.get_or_insert_with(|| dev.create_sampler(&SamplerDescriptor { mag_filter: FilterMode::Linear, min_filter: FilterMode::Linear, ..Default::default() })).clone();
        let t = self.tex_mut(res).unwrap();
        if t.mip_passes.is_none() {
            let mut passes = Vec::new();
            for layer in 0..desc.d {
                for mip in 1..desc.mips {
                    let src = tex.create_view(&TextureViewDescriptor { dimension: Some(TextureViewDimension::D2), base_mip_level: mip - 1, mip_level_count: Some(1), base_array_layer: layer, array_layer_count: Some(1), ..Default::default() });
                    let dst = tex.create_view(&TextureViewDescriptor { dimension: Some(TextureViewDimension::D2), base_mip_level: mip, mip_level_count: Some(1), base_array_layer: layer, array_layer_count: Some(1), ..Default::default() });
                    let bg = dev.create_bind_group(&BindGroupDescriptor { label: None, layout: &pl.get_bind_group_layout(0), entries: &[
                        BindGroupEntry { binding: 0, resource: BindingResource::TextureView(&src) }, BindGroupEntry { binding: 1, resource: BindingResource::Sampler(&smp) }] });
                    passes.push((dst, bg));
                }
            }
            t.mip_passes = Some(passes);
        }
        let passes = t.mip_passes.clone().unwrap();
        let enc = self.encoder();
        for (dst, bg) in &passes {
            let mut p = enc.begin_render_pass(&RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(RenderPassColorAttachment { view: dst, depth_slice: None, resolve_target: None, ops: Operations { load: LoadOp::Clear(Color::BLACK), store: StoreOp::Store } })],
                depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None, multiview_mask: None,
            });
            p.set_pipeline(&pl);
            p.set_bind_group(0, bg, &[]);
            p.draw(0..3, 0..1);
        }
        self.use_tex(res);
        self.mark_color_dirty(res);
    }

    fn readback(&mut self, p: u64) {
        let mb = self.base;
        let u = move |w: u64| unsafe { std::ptr::read_volatile(mb.add(((p + w) * 4) as usize) as *const u32) };
        let (src, sub, has_box) = (u(0), u(1), u(2) != 0);
        let (l, t, f, r, b, bk) = (u(3), u(4), u(5), u(6), u(7), u(8));
        let addr = u(9) as u64 + u(10) as u64 * 4294967296;
        let (row_pitch, slice_pitch, is_buf, size) = (u(11) as u64, u(12) as u64, u(13) != 0, u(14) as u64);
        let flag = u(15) as u64 + u(16) as u64 * 4294967296;
        let done = |me: &Self| { if flag != 0 { me.mem.word(flag).store(0, Ordering::SeqCst); let _ = me.mem.notify(flag); } };
        if self.lost.load(Ordering::SeqCst) { done(self); return; }
        self.materialize_clears(src);
        if self.obj(src).is_none() { done(self); return; }
        if is_buf {
            let Some((g, bsize)) = self.buf(src).and_then(|bo| bo.gpu.clone().map(|g| (g, bo.size))) else { done(self); return };
            let n = if has_box { (r - l) as u64 } else { size.min(bsize) };
            let padded = ((n + 3) & !3).max(4);
            let rb = self.device.create_buffer(&BufferDescriptor { label: Some("readback"), size: padded, usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ, mapped_at_creation: false });
            self.end_pass();
            let from = if has_box { l as u64 } else { 0 };
            let copy = padded.min(g.size().saturating_sub(from)) & !3;
            if copy > 0 { self.encoder().copy_buffer_to_buffer(&g, from, &rb, 0, copy); }
            self.flush();
            self.map_pending(rb, MapKind::Readback { flag, rows: vec![(0, addr, n)] });
            return;
        }
        let Some((tex, fmt, desc)) = self.tex(src).map(|x| (x.tex.clone(), x.format, x.desc)) else { done(self); return };
        if desc.samples > 1 || is_depth(fmt) { log_once(&format!("rbfmt{fmt:?}"), &format!("readback of {fmt:?} not supported")); done(self); return; }
        let (mip, layer) = (sub % desc.mips, sub / desc.mips);
        let (mw, mh) = ((desc.w >> mip).max(1), (desc.h >> mip).max(1));
        let three = desc.dim == TextureDimension::D3;
        let md = if three { (desc.d >> mip).max(1) } else { 1 };
        let (bw, bh) = fmt.block_dimensions();
        let bytes = fmt.block_copy_size(None).unwrap_or(4);
        let (x0, y0, z0) = if has_box { (l, t, f) } else { (0, 0, 0) };
        let (w, h, d) = if has_box { (r - l, b - t, bk - f) } else { (mw, mh, md) };
        let (bx, by) = (w.div_ceil(bw), h.div_ceil(bh));
        let row_bytes = (bx * bytes) as u64;
        let bpr = row_bytes.div_ceil(256) * 256;
        let rb_size = bpr * by as u64 * d as u64;
        let rb = self.device.create_buffer(&BufferDescriptor { label: Some("readback"), size: rb_size.max(4), usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ, mapped_at_creation: false });
        self.end_pass();
        self.encoder().copy_texture_to_buffer(
            TexelCopyTextureInfo { texture: &tex, mip_level: mip, origin: Origin3d { x: x0, y: y0, z: if three { z0 } else { layer } }, aspect: TextureAspect::All },
            TexelCopyBufferInfo { buffer: &rb, layout: TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bpr as u32), rows_per_image: Some(by) } },
            Extent3d { width: bx * bw, height: by * bh, depth_or_array_layers: if three { d } else { 1 } },
        );
        self.flush();
        let mut rows = Vec::new();
        for z in 0..d as u64 {
            for row in 0..by as u64 {
                rows.push(((z * by as u64 + row) * bpr, addr + z * slice_pitch + row * row_pitch, row_bytes));
            }
        }
        self.map_pending(rb, MapKind::Readback { flag, rows });
    }

    fn map_pending(&mut self, buf: Buffer, kind: MapKind) {
        let status = Arc::new(AtomicI32::new(0));
        let s2 = status.clone();
        buf.slice(..).map_async(MapMode::Read, move |r| {
            let value = match r {
                Ok(()) => 1,
                Err(e) => { log_once("map-async", &format!("map failed: {e}")); -1 }
            };
            s2.store(value, Ordering::SeqCst);
        });
        self.pending_maps.push(PendingMap { buf, status, kind });
    }

    fn process_maps(&mut self, wait: bool) {
        if self.pending_maps.is_empty() { return; }
        let _ = self.device.poll(if wait { PollType::Wait { submission_index: None, timeout: None } } else { PollType::Poll });
        let mut i = 0;
        while i < self.pending_maps.len() {
            let status = if self.lost.load(Ordering::SeqCst) { -1 } else { self.pending_maps[i].status.load(Ordering::SeqCst) };
            if status == 0 { i += 1; continue; }
            let pm = self.pending_maps.swap_remove(i);
            let PendingMap { buf, kind, .. } = pm;
            // A failed mapping (an invalid buffer, a lost device) must not take
            // the renderer down, and the engine must not wait forever for a
            // readback that will not arrive: report it and answer anyway.
            let mut mapped = false;
            if status == 1 {
                match buf.slice(..).get_mapped_range() {
                    Ok(range) => {
                        mapped = true;
                        match &kind {
                            MapKind::Readback { rows, .. } => {
                                for &(from, to, n) in rows {
                                    if from >= range.len() as u64 { continue; }
                                    let n = n.min(range.len() as u64 - from);
                                    unsafe { std::ptr::copy_nonoverlapping(range.as_ptr().add(from as usize), self.base.add(to as usize), n as usize); }
                                }
                            }
                            MapKind::Occlusion { list } => {
                                for &(byte, cnt, addr, consecutive, seq) in list {
                                    let mut sum = 0u64;
                                    for k in 0..cnt as u64 {
                                        let at = (byte + k * if consecutive { 8 } else { 256 }) as usize;
                                        if at + 8 <= range.len() { sum += u64::from_le_bytes(range[at..at + 8].try_into().unwrap()); }
                                    }
                                    self.query_store(addr, if sum != 0 { OCCLUSION_VISIBLE_COUNT } else { 0 }, seq as u32);
                                }
                            }
                        }
                        drop(range);
                        if !self.lost.load(Ordering::SeqCst) { buf.unmap(); }
                    }
                    Err(e) => {
                        log_once("maprange", &format!("readback mapping failed: {e}"));
                    }
                }
            }
            // A readback failure is not proof that the queried geometry is
            // occluded. Complete the query conservatively so it stays visible.
            if !mapped {
                if let MapKind::Occlusion { list } = &kind {
                    for &(_, _, addr, _, seq) in list { self.query_store(addr, OCCLUSION_VISIBLE_COUNT, seq as u32); }
                }
            }
            if let MapKind::Readback { flag, .. } = &kind {
                if *flag != 0 { self.mem.word(*flag).store(0, Ordering::SeqCst); let _ = self.mem.notify(*flag); }
            }
        }
    }

    fn query_store(&self, addr: u64, result: u64, seq: u32) {
        self.mem.word(addr + 8).store(result as u32 as i32, Ordering::SeqCst);
        self.mem.word(addr + 12).store((result >> 32) as u32 as i32, Ordering::SeqCst);
        self.mem.word(addr).store(seq as i32, Ordering::SeqCst);
        let _ = self.mem.notify(addr);
    }
    fn create_query(&mut self, id: u32, qtype: u32, addr: u64) {
        self.set_obj(id, Obj::Query(QueryObj { qtype, addr, slots: vec![] }));
        if self.lost.load(Ordering::SeqCst) { return; }
        if self.occlusion_set.is_none() {
            self.occlusion_set = Some(self.device.create_query_set(&QuerySetDescriptor { label: None, ty: QueryType::Occlusion, count: OCCLUSION_CAPACITY }));
        }
    }
    fn is_occlusion(t: u32) -> bool { t == 1 || t == 5 }
    fn begin_occlusion_in_pass(&mut self) {
        let Some(q) = self.occlusion_active else { return };
        if self.pass.is_none() || self.occlusion_pass_slot.is_some() { return; }
        if self.occlusion_next >= OCCLUSION_CAPACITY { log_once("qfull", "occlusion query set exhausted"); return; }
        let slot = self.occlusion_next;
        self.occlusion_next += 1;
        if let Some(Obj::Query(qo)) = self.obj_mut(q) { qo.slots.push(slot); }
        self.pass.as_mut().unwrap().begin_occlusion_query(slot);
        self.occlusion_pass_slot = Some(slot);
    }
    fn begin_query(&mut self, id: u32) {
        let Some(Obj::Query(q)) = self.obj_mut(id) else { return };
        if !Self::is_occlusion(q.qtype) { return; }
        q.slots.clear();
        self.occlusion_active = Some(id);
        self.begin_occlusion_in_pass();
    }
    fn end_query(&mut self, id: u32, seq: u32) {
        let Some((qtype, addr)) = (match self.obj(id) { Some(Obj::Query(q)) => Some((q.qtype, q.addr)), _ => None }) else { return };
        if self.lost.load(Ordering::SeqCst) {
            let value = if Self::is_occlusion(qtype) { OCCLUSION_VISIBLE_COUNT } else if qtype == 2 { START.elapsed().as_nanos() as u64 } else { 1 };
            self.query_store(addr, value, seq);
            return;
        }
        if Self::is_occlusion(qtype) {
            if self.occlusion_active == Some(id) {
                if self.occlusion_pass_slot.take().is_some() { if let Some(p) = self.pass.as_mut() { p.end_occlusion_query(); } }
                self.occlusion_active = None;
            }
            let slots = match self.obj_mut(id) { Some(Obj::Query(q)) => std::mem::take(&mut q.slots), _ => vec![] };
            self.pending_resolve.push((id, seq, slots));
        } else {
            let t = Instant::now().duration_since(*START).as_nanos() as u64;
            self.pending_events.push((id, seq, if qtype == 2 { t } else { 1 }));
            let _ = addr;
        }
    }
    fn record_resolves(&mut self) -> Option<PendingMap> {
        let list = std::mem::take(&mut self.pending_resolve);
        let mut layout = Vec::new();
        let mut bytes = 0u64;
        for (id, seq, slots) in list {
            let Some(addr) = (match self.obj(id) { Some(Obj::Query(q)) => Some(q.addr), _ => None }) else { continue };
            if slots.is_empty() { self.query_store(addr, 0, seq); continue; }
            let consecutive = slots.windows(2).all(|w| w[1] == w[0] + 1);
            layout.push((slots, bytes, addr, consecutive, seq));
            bytes += if consecutive { (layout.last().unwrap().0.len() as u64 * 8).div_ceil(256) * 256 } else { layout.last().unwrap().0.len() as u64 * 256 };
        }
        if self.occlusion_next > OCCLUSION_CAPACITY - 256 && self.occlusion_active.is_none() { self.occlusion_next = 0; }
        if bytes == 0 { return None; }
        let size = bytes.next_power_of_two().max(256);
        let rb = self.device.create_buffer(&BufferDescriptor { label: Some("occlusion resolve"), size, usage: BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC, mapped_at_creation: false });
        let map = self.device.create_buffer(&BufferDescriptor { label: Some("occlusion map"), size, usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ, mapped_at_creation: false });
        let set = self.occlusion_set.clone()?;
        let enc = self.encoder();
        let mut out = Vec::new();
        for (slots, at, addr, consecutive, seq) in &layout {
            if *consecutive {
                enc.resolve_query_set(&set, slots[0]..slots[0] + slots.len() as u32, &rb, *at);
            } else {
                for (i, s) in slots.iter().enumerate() { enc.resolve_query_set(&set, *s..*s + 1, &rb, at + i as u64 * 256); }
            }
            out.push((*at, slots.len() as u32, *addr, *consecutive, *seq as usize));
        }
        enc.copy_buffer_to_buffer(&rb, 0, &map, 0, bytes);
        let status = Arc::new(AtomicI32::new(0));
        Some(PendingMap { buf: map, status, kind: MapKind::Occlusion { list: out } })
    }
    fn on_submitted(&mut self) {
        if !self.pending_events.is_empty() {
            let evs = std::mem::take(&mut self.pending_events);
            let mem = self.mem.clone();
            let inflight = self.inflight.clone();
            inflight.fetch_add(1, Ordering::SeqCst);
            let addrs: Vec<(u64, u64, u32)> = evs.iter().filter_map(|(id, seq, v)| match self.obj(*id) { Some(Obj::Query(q)) => Some((q.addr, *v, *seq)), _ => None }).collect();
            self.queue.on_submitted_work_done(move || {
                for (addr, v, seq) in addrs {
                    mem.word(addr + 8).store(v as u32 as i32, Ordering::SeqCst);
                    mem.word(addr + 12).store((v >> 32) as u32 as i32, Ordering::SeqCst);
                    mem.word(addr).store(seq as i32, Ordering::SeqCst);
                    let _ = mem.notify(addr);
                }
                inflight.fetch_sub(1, Ordering::SeqCst);
            });
        }
    }

    fn set_uav_counter(&mut self, uav: u32, value: u32) {
        let Some(res) = self.view_obj(uav).filter(|v| v.is_buf).map(|v| v.res) else { return };
        if self.buf(res).is_none() { return; }
        if self.counter_src.is_none() {
            self.counter_src = Some(self.device.create_buffer(&BufferDescriptor { label: Some("counter src"), size: 1024 * 4, usage: BufferUsages::COPY_SRC | BufferUsages::COPY_DST, mapped_at_creation: false }));
        }
        if self.counter_slot >= 1024 { self.flush(); }
        let slot = self.counter_slot as u64;
        self.counter_slot += 1;
        let src = self.counter_src.clone().unwrap();
        self.queue.write_buffer(&src, slot * 4, &value.to_le_bytes());
        self.end_pass();
        let Some(cb) = self.counter_buffer(res) else { return };
        self.encoder().copy_buffer_to_buffer(&src, slot * 4, &cb, 0, 4);
        self.use_buf(res);
    }
    fn copy_counter(&mut self, dst: u32, offset: u64, uav: u32) {
        let Some(res) = self.view_obj(uav).filter(|v| v.is_buf).map(|v| v.res) else { return };
        let Some(dg) = self.buf(dst).and_then(|b| b.gpu.clone()) else { return };
        let Some(cb) = self.counter_buffer(res) else { return };
        self.end_pass();
        self.encoder().copy_buffer_to_buffer(&cb, 0, &dg, offset, 4);
        self.use_buf(dst);
        self.use_buf(res);
    }

    fn present(&mut self, id: u32) {
        if self.lost.load(Ordering::SeqCst) {
            self.flush();
            self.collect_pipelines();
            return;
        }
        self.materialize_clears(id);
        let visible = !OCCLUDED.load(Ordering::Relaxed);
        if let (Some(t), Some(_), true) = (self.tex(id), self.surface.as_ref(), visible) {
            let (w, h) = (t.desc.w, t.desc.h);
            let tex = t.tex.clone();
            self.end_pass();
            #[cfg(target_os = "android")]
            let (surface_w, surface_h) = match &self.surface.as_ref().unwrap().1 {
                SurfaceSrc::Window(w) => {
                    let s = w.inner_size();
                    (s.width.max(1), s.height.max(1))
                }
                SurfaceSrc::Fixed(w, h) => (*w, *h),
            };
            #[cfg(not(target_os = "android"))]
            let (surface_w, surface_h) = (w, h);
            if self.surface_size != (surface_w, surface_h) {
                let (surface, src) = self.surface.as_ref().unwrap();
                let caps = surface.get_capabilities(&self.adapter);
                surface.configure(&self.device, &SurfaceConfiguration {
                    usage: TextureUsages::RENDER_ATTACHMENT,
                    format: self.surface_format,
                    width: surface_w,
                    height: surface_h,
                    present_mode: if !cfg!(target_os = "android") && caps.present_modes.contains(&PresentMode::Mailbox) {
                        // Android Fifo matches the display and avoids rendering frames
                        // the compositor never shows when the game outruns vsync.
                        PresentMode::Mailbox
                    } else {
                        PresentMode::Fifo
                    },
                    desired_maximum_frame_latency: 2,
                    alpha_mode: caps.alpha_modes[0],
                    view_formats: vec![],
                    color_space: SurfaceColorSpace::Auto,
                });
                self.surface_size = (surface_w, surface_h);
                log(&format!("surface configured {surface_w}x{surface_h} {:?}", self.surface_format));
                if let SurfaceSrc::Window(w) = src {
                    w.request_redraw();
                }
            }
            if self.blit.is_none() {
                let module = self.device.create_shader_module(ShaderModuleDescriptor { label: Some("blit"), source: ShaderSource::Wgsl(
                    "@group(0) @binding(0) var src: texture_2d<f32>;
                     struct VOut { @builtin(position) position: vec4f, @location(0) uv: vec2f }
                     @vertex fn vs(@builtin(vertex_index) i: u32) -> VOut {
                         var p = array<vec2f, 3>(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
                         var o: VOut; o.position = vec4f(p[i], 0, 1); o.uv = p[i] * vec2f(0.5, -0.5) + vec2f(0.5); return o;
                     }
                     @fragment fn fs(@location(0) uv: vec2f) -> @location(0) vec4f {
                         let size = vec2i(textureDimensions(src));
                         return textureLoad(src, clamp(vec2i(uv * vec2f(size)), vec2i(0), size - vec2i(1)), 0);
                     }".into()) });
                self.blit = Some(self.simple_pipeline(&module, Some(self.surface_format), None));
            }
            let blit = self.blit.clone().unwrap();
            let dev = self.device.clone();
            let bg = {
                let t = self.tex_mut(id).unwrap();
                t.blit_bg.get_or_insert_with(|| {
                    let v = tex.create_view(&TextureViewDescriptor { dimension: Some(TextureViewDimension::D2), base_mip_level: 0, mip_level_count: Some(1), base_array_layer: 0, array_layer_count: Some(1), ..Default::default() });
                    dev.create_bind_group(&BindGroupDescriptor { label: None, layout: &blit.get_bind_group_layout(0), entries: &[BindGroupEntry { binding: 0, resource: BindingResource::TextureView(&v) }] })
                }).clone()
            };
            let frame = match self.surface.as_ref().unwrap().0.get_current_texture() {
                CurrentSurfaceTexture::Success(f) | CurrentSurfaceTexture::Suboptimal(f) => Some(f),
                other => { log_once("surface", &format!("surface texture unavailable: {other:?}")); self.surface_size = (0, 0); None }
            };
            if let Some(frame) = frame {
                let view = frame.texture.create_view(&TextureViewDescriptor::default());
                self.use_tex(id);
                {
                    let enc = self.encoder();
                    let mut p = enc.begin_render_pass(&RenderPassDescriptor {
                        label: None,
                        color_attachments: &[Some(RenderPassColorAttachment { view: &view, depth_slice: None, resolve_target: None, ops: Operations { load: LoadOp::Clear(Color::BLACK), store: StoreOp::Store } })],
                        depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None, multiview_mask: None,
                    });
                    p.set_pipeline(&blit);
                    p.set_bind_group(0, &bg, &[]);
                    p.draw(0..3, 0..1);
                }
                self.flush();
                self.queue.present(frame);
            } else {
                self.flush();
            }
        } else {
            self.flush();
        }
        self.collect_pipelines();
        self.presents += 1;
        PRESENTS.store(self.presents, Ordering::Relaxed);
        // Live frame rate for the engine shell's counter (the engine path has
        // no other way to ask the renderer how fast it is going).
        let n = FPS_FRAMES.fetch_add(1, Ordering::Relaxed) + 1;
        if n % 30 == 0 {
            let dt = FPS_START.elapsed().as_secs_f64();
            if dt > 0.0 {
                FPS_X100.store((n as f64 / dt * 100.0) as u64, Ordering::Relaxed);
            }
        }
        self.log_fps();

        self.frame_slot_done[self.ubo_slot] = self.last_submit.clone();
        self.ubo_slot = (self.ubo_slot + 1) % FRAME_SLOTS;
        self.ubo_chunk = 0;
        self.ubo_off = 0;
        if let Some(idx) = self.frame_slot_done[self.ubo_slot].take() {
            let _ = self.device.poll(PollType::Wait { submission_index: Some(idx), timeout: None });
        }
        self.process_maps(false);
    }

    fn log_fps(&mut self) {
        let now = Instant::now();
        let dt = now.duration_since(self.fps_t0).as_secs_f64();
        if dt >= 10.0 {
            let frames = self.presents - self.fps_n0;
            log(&format!(
                "fps: {:.1} ({} MB memory, game heap {} MB, {} frames, {} clears folded, {} draws/frame, {} draws waiting for a compiling pipeline, {:.1} submits/frame, {:.1} passes/frame, {} skipped total, pipelines {}, bind groups cached {}, validation errors {})",
                frames as f64 / dt, crate::footprint_mb(), crate::RT.get().map(|r| r.mem.data_size() >> 20).unwrap_or(0), frames, self.clears_folded, (self.draws - self.fps_d0) / frames.max(1), self.pending_skips - self.pending_skips0, (self.submits - self.submits0) as f64 / frames.max(1) as f64,
                (self.passes - self.passes0) as f64 / frames.max(1) as f64, self.skipped, self.pipe_creates, self.bind_groups.len(), self.errors.load(Ordering::Relaxed)
            ));
            if let Some(r) = self.instance.generate_report() {
                let h = &r.hub;
                log(&format!("live wgpu objects: buffers {} textures {} views {} bind groups {} samplers {} pipelines {} | engine objects {}",
                    h.buffers.num_kept_from_user, h.textures.num_kept_from_user, h.texture_views.num_kept_from_user,
                    h.bind_groups.num_kept_from_user, h.samplers.num_kept_from_user, h.render_pipelines.num_kept_from_user,
                    self.objs.iter().filter(|o| o.is_some()).count()));
            }
            let (shadow_len, shadow_capacity, shadow_declared) = self.objs.iter().fold((0u64, 0u64, 0u64), |(len, cap, declared), obj| {
                match obj {
                    Some(Obj::Buf(Buf { size, shadow: Some(shadow), .. })) => (len + shadow.len() as u64, cap + shadow.capacity() as u64, declared + size),
                    _ => (len, cap, declared),
                }
            });
            let uniform_staging: usize = self.ubo_slots.iter().flatten().map(|c| c.stage.capacity()).sum();
            log(&format!("host buffer memory: shadow data {} KiB, allocated {} KiB, declared {} KiB; uniform staging {} KiB; bind group cache {}/{} ({} evictions)",
                shadow_len >> 10, shadow_capacity >> 10, shadow_declared >> 10, uniform_staging >> 10,
                self.bind_groups.len(), self.bind_groups.capacity, self.bind_groups.evictions));
            self.submits0 = self.submits;
            self.pending_skips0 = self.pending_skips;
            self.passes0 = self.passes;
            self.fps_t0 = now;
            self.fps_n0 = self.presents;
            self.fps_d0 = self.draws;
        }
    }
    fn log_stats(&self) {
        log(&format!("stats: draws {} (skipped {}), presents {}, objects {}, pipelines {}", self.draws, self.skipped, self.presents, self.objs.iter().filter(|o| o.is_some()).count(), self.pipelines.len()));
    }
}

static START: LazyLock<Instant> = LazyLock::new(Instant::now);
static FPS_FRAMES: AtomicU64 = AtomicU64::new(0);
static FPS_X100: AtomicU64 = AtomicU64::new(0);
static FPS_START: LazyLock<Instant> = LazyLock::new(Instant::now);

/// The renderer's current frame rate, for the engine shell's counter.
#[no_mangle]
pub extern "C" fn muguet_gpu_fps() -> f64 {
    FPS_X100.load(Ordering::Relaxed) as f64 / 100.0
}

fn format_id(f: TextureFormat) -> i64 {
    let s = format!("{f:?}");
    let mut h: i64 = 1469598103934665603u64 as i64;
    for b in s.bytes() { h = (h ^ b as i64).wrapping_mul(1099511628211); }
    h
}

fn blend_for(state: Option<&StateObj>, rt: usize) -> (Option<BlendState>, ColorWrites) {
    let Some(s) = state else { return (None, ColorWrites::ALL) };
    let w = &s.words;
    let g = |i: usize| w.get(i).copied().unwrap_or(0);
    let r = if g(1) != 0 { rt } else { 0 };
    let q = 2 + r * 8;
    let write = ColorWrites::from_bits_truncate(g(q + 7) & 0xf);
    if g(q) == 0 { return (None, write); }
    (Some(BlendState {
        color: BlendComponent { src_factor: blend_factor(g(q + 1), BlendFactor::One), dst_factor: blend_factor(g(q + 2), BlendFactor::Zero), operation: blend_op(g(q + 3)) },
        alpha: BlendComponent { src_factor: blend_factor(g(q + 4), BlendFactor::One), dst_factor: blend_factor(g(q + 5), BlendFactor::Zero), operation: blend_op(g(q + 6)) },
    }), write)
}

fn error_chain(e: &dyn std::error::Error) -> String {
    let mut s = e.to_string();
    let mut cur = e.source();
    while let Some(c) = cur {
        s.push_str(": ");
        s.push_str(&c.to_string());
        cur = c.source();
    }
    s
}

fn wgsl_arguments(text: &str) -> Vec<String> {
    let mut args = Vec::new();
    let (mut depth, mut start) = (0u32, 0);
    for (at, c) in text.char_indices() {
        match c {
            '(' | '[' | '<' => depth += 1,
            ')' | ']' | '>' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => { args.push(text[start..at].trim().to_string()); start = at + 1; }
            _ => {}
        }
    }
    let last = text[start..].trim();
    if !last.is_empty() { args.push(last.to_string()); }
    args
}

/// The Adreno Vulkan drivers lose the device on hardware clip distances, so
/// the clipping is emulated in the fragment stage there.
fn driver_needs_software_clip(name: &str, driver: &str, driver_info: &str) -> bool {
    format!("{name} {driver} {driver_info}").to_lowercase().contains("adreno")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shadow_buffer(size: u64) -> Buf {
        Buf { size, stride: 0, gpu: None, bytes: 0, shadow: Some(Vec::new()), version: 0, enc: 0, in_bg: false, counter: None }
    }

    #[test]
    fn partial_constant_buffer_uploads_preserve_unwritten_uniforms() {
        let mut buffer = shadow_buffer(65536);
        buffer.update_shadow(16, &[10, 11, 12, 13]).unwrap();
        buffer.update_shadow(18, &[99, 88]).unwrap();
        buffer.update_shadow(0, &[1, 2]).unwrap();
        let mut packed = [0xcc; 32];
        buffer.copy_shadow(0, &mut packed);
        assert_eq!(&packed[..2], &[1, 2]);
        assert_eq!(&packed[2..16], &[0; 14]);
        assert_eq!(&packed[16..20], &[10, 11, 99, 88]);
        assert_eq!(&packed[20..], &[0; 12]);
        assert_eq!(buffer.shadow.as_ref().unwrap().len(), 20);
        assert!(buffer.shadow.as_ref().unwrap().capacity() < 65536);
        let mut ranged = [0xcc; 16];
        buffer.copy_shadow(16, &mut ranged);
        assert_eq!(&ranged[..4], &[10, 11, 99, 88]);
        assert_eq!(&ranged[4..], &[0; 12]);
    }

    #[test]
    fn shadow_uploads_stay_within_the_declared_buffer_size() {
        let mut buffer = shadow_buffer(32);
        buffer.update_shadow(30, &[7, 8, 9, 10]).unwrap();
        assert_eq!(buffer.shadow.as_ref().unwrap().len(), 32);
        assert_eq!(&buffer.shadow.as_ref().unwrap()[30..], &[7, 8]);
        let version = buffer.version;
        buffer.update_shadow(u64::MAX, &[1]).unwrap();
        buffer.update_shadow(32, &[]).unwrap();
        assert_eq!(buffer.version, version);
        assert_eq!(buffer.shadow.as_ref().unwrap().len(), 32);
        let mut packed = [0xcc; 4];
        buffer.copy_shadow(30, &mut packed);
        assert_eq!(packed, [7, 8, 0, 0]);
        buffer.copy_shadow(u64::MAX, &mut packed);
        assert_eq!(packed, [0; 4]);
        buffer.shadow = None;
        buffer.update_shadow(0, &[1, 2, 3, 4]).unwrap();
        assert!(buffer.shadow.is_none(), "vertex/storage buffers must not gain host shadows");
    }

    #[test]
    fn fifo_cache_replacements_and_churn_keep_bookkeeping_bounded() {
        let mut cache = FifoCache::new(2);
        cache.insert(vec![1], 10);
        cache.insert(vec![2], 20);
        cache.insert(vec![1], 11);
        assert_eq!(cache.order.len(), 2, "replacement must not add another FIFO key");
        cache.insert(vec![3], 30);
        assert!(cache.get(&[1]).is_none(), "replacement keeps the original FIFO age");
        assert_eq!(cache.get(&[2]), Some(&20));
        assert_eq!(cache.get(&[3]), Some(&30));
        for key in 4..20000 {
            cache.insert(vec![key], key);
            cache.insert(vec![key], key + 1);
            assert!(cache.len() <= 2);
            assert_eq!(cache.order.len(), cache.len());
        }
        let mut disabled = FifoCache::new(0);
        disabled.insert(vec![1], 1);
        assert_eq!(disabled.len(), 0);
        assert!(disabled.order.is_empty());
    }

    #[test]
    fn cache_eviction_releases_ownership_without_invalidating_recorded_work() {
        let mut cache = FifoCache::new(1);
        let resource = Arc::new(());
        let weak = Arc::downgrade(&resource);
        let recorded = resource.clone();
        cache.insert(vec![1], resource);
        cache.insert(vec![2], Arc::new(()));
        assert!(cache.get(&[1]).is_none());
        assert_eq!(weak.strong_count(), 1, "only recorded work should still own the evicted resource");
        drop(recorded);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn recorded_bind_groups_validate_after_cache_eviction() {
        let (device, queue) = Device::noop(&DeviceDescriptor::default());
        let lost = AtomicBool::new(false);
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[BindGroupLayoutEntry {
                binding: 0, visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            }],
        });
        let buffer = device.create_buffer(&BufferDescriptor { label: None, size: 16, usage: BufferUsages::STORAGE, mapped_at_creation: false });
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl("@group(0) @binding(0) var<storage, read_write> data: array<u32>; @compute @workgroup_size(1) fn main() { data[0] = 1u; }".into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: None, layout: Some(&pipeline_layout), module: &module, entry_point: Some("main"), compilation_options: Default::default(), cache: None,
        });
        let group = || device.create_bind_group(&BindGroupDescriptor {
            label: None, layout: &layout, entries: &[BindGroupEntry { binding: 0, resource: buffer.as_entire_binding() }],
        });
        let mut cache = FifoCache::new(1);
        cache.insert(vec![1], group());
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, cache.get(&[1]).unwrap(), &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        cache.insert(vec![2], group());
        assert!(cache.get(&[1]).is_none());
        assert!(gpu_scoped(&device, &lost, "submit after cache eviction", |_| queue.submit([encoder.finish()])).is_some());
    }

    fn validate_wgsl(source: &str) {
        let module = wgpu::naga::front::wgsl::parse_str(source).unwrap_or_else(|e| panic!("{}", e.emit_to_string(source)));
        wgpu::naga::valid::Validator::new(wgpu::naga::valid::ValidationFlags::all(), wgpu::naga::valid::Capabilities::empty())
            .validate(&module).unwrap_or_else(|e| panic!("{}", e.emit_to_string(source)));
    }

    #[test]
    fn native_bc_mip_uploads_and_copies_validate() {
        let (device, queue) = Device::noop(&DeviceDescriptor { required_features: Features::TEXTURE_COMPRESSION_BC, ..Default::default() });
        let lost = AtomicBool::new(false);
        for format in [TextureFormat::Bc1RgbaUnorm, TextureFormat::Bc3RgbaUnorm] {
            let make = || gpu_scoped(&device, &lost, "BC mip test texture", |d| d.create_texture(&TextureDescriptor {
                label: None, size: Extent3d { width: 16, height: 8, depth_or_array_layers: 2 }, mip_level_count: 5, sample_count: 1,
                dimension: TextureDimension::D2, format, usage: TextureUsages::COPY_SRC | TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING, view_formats: &[],
            })).unwrap();
            let (src, dst) = (make(), make());
            // This is the exact invalid logical-mip copy reported in all three
            // issue logs; the noop backend must still exercise core validation.
            let block = vec![0u8; format.block_copy_size(None).unwrap() as usize];
            assert!(gpu_scoped(&device, &lost, "unaligned BC mip control", |_| queue.write_texture(
                TexelCopyTextureInfo { texture: &src, mip_level: 4, origin: Origin3d::ZERO, aspect: TextureAspect::All },
                &block, TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(block.len() as u32), rows_per_image: Some(1) },
                Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            )).is_none());
            for mip in 0..5 {
                let (w, h) = ((16u32 >> mip).max(1), (8u32 >> mip).max(1));
                let (bw, bh) = format.block_dimensions();
                let pitch = w.div_ceil(bw) * format.block_copy_size(None).unwrap();
                let rows = h.div_ceil(bh);
                let data = vec![0u8; (pitch * rows * 2) as usize];
                let extent = copy_extent(format, w, h, 2);
                assert!(gpu_scoped(&device, &lost, "BC mip test upload", |_| queue.write_texture(
                    TexelCopyTextureInfo { texture: &src, mip_level: mip, origin: Origin3d::ZERO, aspect: TextureAspect::All },
                    &data, TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(pitch), rows_per_image: Some(rows) }, extent,
                )).is_some());
                assert!(gpu_scoped(&device, &lost, "BC mip test copy", |d| {
                    let mut encoder = d.create_command_encoder(&CommandEncoderDescriptor::default());
                    encoder.copy_texture_to_texture(
                        TexelCopyTextureInfo { texture: &src, mip_level: mip, origin: Origin3d::ZERO, aspect: TextureAspect::All },
                        TexelCopyTextureInfo { texture: &dst, mip_level: mip, origin: Origin3d::ZERO, aspect: TextureAspect::All }, extent,
                    );
                    queue.submit([encoder.finish()]);
                }).is_some());
            }
        }
        assert_eq!(copy_extent(TextureFormat::Rgba8Unorm, 3, 2, 1), Extent3d { width: 3, height: 2, depth_or_array_layers: 1 });
    }

    #[test]
    fn software_clip_preserves_and_validates_both_shader_stages() {
        let vertex = r#"
enable clip_distances;
struct VSOut {
    @builtin(position) position: vec4<f32>,
    @location(16u) uv: vec2<f32>,
    @builtin(clip_distances) distances: array<f32, 5u>,
}
@vertex fn main(@location(0u) position: vec4<f32>) -> VSOut {
    return VSOut(position, vec2<f32>(0.25f, 0.75f), array<f32, 5u>(1.0f, 2.0f, 3.0f, 4.0f, 5.0f));
}
"#;
        let pixel = r#"
struct PSOut { @location(0u) color: vec4<f32>, }
fn shade(uv: vec2<f32>) -> vec4<f32> { return vec4<f32>(uv, 0.5f, 1.0f); }
@fragment fn main(@location(16u) uv: vec2<f32>,) -> PSOut {
    let color = shade(uv);
    return PSOut(color);
}
fn after_entry() -> f32 { return 1.0f; }
"#;
        let (lowered, clip) = Gpu::lower_clip_distances(vertex);
        let clip = clip.expect("vertex clip distances must be lowered");
        assert_eq!(clip.count, 5);
        assert!(!lowered.contains("@builtin(clip_distances)"));
        assert!(!lowered.contains("enable clip_distances"));
        let remap = (HashMap::from([(16, 0), (32, 1), (33, 2)]), "16>0,32>1,33>2".into(), true);
        let vs = remap_locations(&lowered, true, &remap.0);
        let ps = Gpu::clip_fragment(&remap_locations(pixel, false, &remap.0), Some(&clip), Some(&remap));
        validate_wgsl(&vs);
        validate_wgsl(&ps);
        assert!(ps.contains("return PSOut(color);"));
        assert!(ps.contains("fn after_entry()"));
        assert!(ps.contains("discard;"));
        assert_eq!(locations_in(&vs, true), locations_in(&ps, false));
        let (device, _) = Device::noop(&DeviceDescriptor::default());
        let lost = AtomicBool::new(false);
        let module = |text: &str| gpu_scoped(&device, &lost, "clip test shader", |d| d.create_shader_module(ShaderModuleDescriptor {
            label: None, source: ShaderSource::Wgsl(text.into()),
        })).unwrap();
        let job = PipeJob {
            label: "software clip test".into(),
            layout: device.create_pipeline_layout(&PipelineLayoutDescriptor { label: None, bind_group_layouts: &[], immediate_size: 0 }),
            vmod: module(&vs), fmod: Some(module(&ps)),
            vbs: vec![Some((16, VertexStepMode::Vertex, vec![VertexAttribute { format: VertexFormat::Float32x4, offset: 0, shader_location: 0 }]))],
            primitive: PrimitiveState::default(), depth_stencil: None, multisample: MultisampleState::default(),
            targets: vec![Some(ColorTargetState { format: TextureFormat::Rgba8Unorm, blend: None, write_mask: ColorWrites::ALL })],
        };
        assert!(compile_pipeline(&device, &lost, &job).is_some());
    }

    #[test]
    fn software_clip_handles_depth_only_fragment_entry() {
        let source = "@fragment fn main() { let alpha = 1.0f; if (alpha < 0.5f) { discard; } }";
        let clip = ClipInfo { count: 1, locations: vec![0] };
        let text = Gpu::clip_fragment(source, Some(&clip), None);
        validate_wgsl(&text);
        assert!(text.contains("let alpha = 1.0f;"));
    }

    #[test]
    fn unsupported_clip_constructor_keeps_the_original_shader() {
        let source = r#"
enable clip_distances;
struct VSOut {
    @builtin(position) position: vec4<f32>,
    @builtin(clip_distances) distances: array<f32, 1u>,
}
@vertex fn main(@location(0u) position: vec4<f32>) -> VSOut {
    var result: VSOut;
    result.position = position;
    result.distances[0] = 1.0f;
    return result;
}
"#;
        let (text, clip) = Gpu::lower_clip_distances(source);
        assert_eq!(text, source);
        assert!(clip.is_none());
    }

    #[test]
    fn clip_varyings_avoid_the_locations_the_shader_uses() {
        let source = r#"
enable clip_distances;
struct VSOut {
    @builtin(position) position: vec4<f32>,
    @location(32u) a: vec4<f32>,
    @location(33u) b: vec4<f32>,
    @builtin(clip_distances) distances: array<f32, 5u>,
}
@vertex fn main(@location(0u) position: vec4<f32>) -> VSOut {
    return VSOut(position, vec4<f32>(1.0f), vec4<f32>(2.0f), array<f32, 5u>(1.0f, 2.0f, 3.0f, 4.0f, 5.0f));
}
"#;
        let (lowered, clip) = Gpu::lower_clip_distances(source);
        let clip = clip.expect("the clip distances must be lowered");
        assert_eq!(clip.locations, vec![34, 35]);
        assert!(lowered.contains("@location(34u) gtaClip0"));
        assert!(lowered.contains("@location(35u) gtaClip1"));
        assert!(lowered.contains("@location(32u) a"));
    }

    #[test]
    fn adreno_drivers_use_software_clip_distances() {
        assert!(driver_needs_software_clip("Adreno (TM) 840 (Vulkan)", "Qualcomm Technologies Inc. Adreno Vulkan Driver Driver Build: 718a07726a", ""));
        assert!(driver_needs_software_clip("Adreno (TM) 829 (Vulkan)", "", ""));
        assert!(!driver_needs_software_clip("Mali-G68 MC4 (Vulkan)", "ARM", ""));
        assert!(!driver_needs_software_clip("Apple M3 (Metal)", "Apple", ""));
    }
}
