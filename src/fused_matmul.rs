//! Column-tiled quantized matmul + fused gated-gelu.
//!
//! Provides `MatMul`, a drop-in replacement for candle's `QMatMul` that uses
//! column-tiled loops for better L1 cache behavior on x86.
//! Pre-dequantized weights use fbgemm-rs: INT8 packed GEMM on x86_64,
//! F32 packed GEMM elsewhere.

use candle_core::backend::BackendStorage;
use candle_core::quantized::k_quants::*;
use candle_core::quantized::{GgmlDType, GgmlType, QTensor};
use candle_core::{CpuStorage, CustomOp1, DType, Layout, Module, Result, Shape, Tensor};
use rayon::prelude::*;
use std::sync::Arc;

#[cfg(not(target_arch = "x86_64"))]
use fbgemm_rs::PackedMatrix;
#[cfg(target_arch = "x86_64")]
use fbgemm_rs::PackedBMatrixI8;
#[cfg(target_arch = "x86_64")]
use fbgemm_rs::I8GemmScratch;

#[cfg(target_arch = "x86_64")]
use std::cell::RefCell;

#[cfg(target_arch = "x86_64")]
thread_local! {
    static I8_SCRATCH: RefCell<I8GemmScratch> = RefCell::new(I8GemmScratch::new());
}

fn as_block_slice<T>(data: &[u8]) -> &[T] {
    let size = std::mem::size_of::<T>();
    let ptr = data.as_ptr();
    debug_assert_eq!(data.len() % size, 0);
    debug_assert_eq!((ptr as usize) % std::mem::align_of::<T>(), 0);
    unsafe { std::slice::from_raw_parts(ptr as *const T, data.len() / size) }
}

// ---- Column-tiled matmul ----

fn tiled_matmul_inner<T: GgmlType>(
    (m, k, n): (usize, usize, usize),
    lhs: &[f32],
    rhs_t: &[T],
    dst: &mut [f32],
) {
    let k_in_blocks = k.div_ceil(T::BLCK_SIZE);

    let mut lhs_b = vec![T::VecDotType::zeros(); m * k_in_blocks];
    for row_idx in 0..m {
        let lhs_b_row = &mut lhs_b[row_idx * k_in_blocks..(row_idx + 1) * k_in_blocks];
        let lhs_row = &lhs[row_idx * k..(row_idx + 1) * k];
        T::VecDotType::from_float(lhs_row, lhs_b_row);
    }

    let tile_n = 128.min(n);
    let tile_starts: Vec<usize> = (0..n).step_by(tile_n).collect();
    let dst_ptr = dst.as_mut_ptr() as usize;
    tile_starts.into_par_iter().for_each(|tile_start| {
        let tile_end = (tile_start + tile_n).min(n);
        // SAFETY: Non-overlapping column tiles — no two threads write same dst element.
        let dst = dst_ptr as *mut f32;
        for row_idx in 0..m {
            let lhs_row = &lhs_b[row_idx * k_in_blocks..(row_idx + 1) * k_in_blocks];
            for col_idx in tile_start..tile_end {
                let rhs_col = &rhs_t[col_idx * k_in_blocks..(col_idx + 1) * k_in_blocks];
                unsafe {
                    *dst.add(row_idx * n + col_idx) = T::vec_dot(k, rhs_col, lhs_row);
                }
            }
        }
    });
}

struct QTiledOp(Arc<QTensor>);

impl CustomOp1 for QTiledOp {
    fn name(&self) -> &'static str {
        "qtiled-matmul"
    }

    fn cpu_fwd(
        &self,
        storage: &CpuStorage,
        layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        if !layout.is_contiguous() {
            candle_core::bail!("input tensor is not contiguous {layout:?}")
        }
        let src_shape = layout.shape();
        let (n, k) = self.0.shape().dims2()?;
        if src_shape.rank() < 2 {
            candle_core::bail!("input tensor has only one dimension {layout:?}")
        }
        let mut dst_shape = src_shape.dims().to_vec();
        let last_k = dst_shape.pop().unwrap();
        if last_k != k {
            candle_core::bail!(
                "input tensor {layout:?} incompatible with {:?}",
                self.0.shape()
            )
        }
        dst_shape.push(n);
        let dst_shape = Shape::from(dst_shape);
        let m = dst_shape.elem_count() / n;

        if storage.dtype() != DType::F32 {
            candle_core::bail!("QTiledOp only supports f32 input")
        }
        let slice = storage.as_slice::<f32>()?;
        let slice = &slice[layout.start_offset()..layout.start_offset() + src_shape.elem_count()];
        let mut dst_storage = vec![0f32; dst_shape.elem_count()];

        let data = self.0.data()?;
        macro_rules! dispatch {
            ($ty:ty) => {
                tiled_matmul_inner::<$ty>(
                    (m, k, n),
                    slice,
                    as_block_slice::<$ty>(&data),
                    &mut dst_storage,
                )
            };
        }
        match self.0.dtype() {
            GgmlDType::Q4K => dispatch!(BlockQ4K),
            GgmlDType::Q5K => dispatch!(BlockQ5K),
            GgmlDType::Q6K => dispatch!(BlockQ6K),
            GgmlDType::Q8K => dispatch!(BlockQ8K),
            GgmlDType::Q2K => dispatch!(BlockQ2K),
            GgmlDType::Q3K => dispatch!(BlockQ3K),
            GgmlDType::Q4_0 => dispatch!(BlockQ4_0),
            GgmlDType::Q5_0 => dispatch!(BlockQ5_0),
            GgmlDType::Q8_0 => dispatch!(BlockQ8_0),
            dt => candle_core::bail!("QTiledOp: unsupported dtype {dt:?}"),
        }

        Ok((CpuStorage::F32(dst_storage), dst_shape))
    }
}

// ---- Fast tanh via Padé [3,3] rational approximation ----
// Pure polynomial ratio (no transcendentals) — auto-vectorizes with AVX2.

#[inline(always)]
fn fast_tanh(x: f32) -> f32 {
    let x = x.clamp(-4.97, 4.97);
    let x2 = x * x;
    x * (135135.0 + x2 * (17325.0 + x2 * (378.0 + x2)))
        / (135135.0 + x2 * (62370.0 + x2 * (3150.0 + 28.0 * x2)))
}

// ---- Fused gated-gelu matmul ----

fn fused_gated_gelu_inner<T: GgmlType>(
    (m, k, n): (usize, usize, usize),
    lhs: &[f32],
    rhs_gate: &[T],
    rhs_up: &[T],
    dst: &mut [f32],
) {
    let k_in_blocks = k.div_ceil(T::BLCK_SIZE);

    let mut lhs_b = vec![T::VecDotType::zeros(); m * k_in_blocks];
    for row_idx in 0..m {
        let lhs_b_row = &mut lhs_b[row_idx * k_in_blocks..(row_idx + 1) * k_in_blocks];
        let lhs_row = &lhs[row_idx * k..(row_idx + 1) * k];
        T::VecDotType::from_float(lhs_row, lhs_b_row);
    }

    // Half-sized tiles since we read from 2 weight matrices per column.
    let tile_n = 64.min(n);
    let tile_starts: Vec<usize> = (0..n).step_by(tile_n).collect();
    let dst_ptr = dst.as_mut_ptr() as usize;
    tile_starts.into_par_iter().for_each(|tile_start| {
        let tile_end = (tile_start + tile_n).min(n);
        let dst = dst_ptr as *mut f32;
        for row_idx in 0..m {
            let lhs_row = &lhs_b[row_idx * k_in_blocks..(row_idx + 1) * k_in_blocks];
            for col_idx in tile_start..tile_end {
                let gate_col =
                    &rhs_gate[col_idx * k_in_blocks..(col_idx + 1) * k_in_blocks];
                let up_col = &rhs_up[col_idx * k_in_blocks..(col_idx + 1) * k_in_blocks];
                let gate = T::vec_dot(k, gate_col, lhs_row);
                let up = T::vec_dot(k, up_col, lhs_row);
                let gate = 0.5 * gate
                    * (1.0
                        + fast_tanh(0.7978845608_f32 * gate * (1.0 + 0.044715 * gate * gate)));
                unsafe {
                    *dst.add(row_idx * n + col_idx) = gate * up;
                }
            }
        }
    });
}

struct QGatedMatMul(Arc<QTensor>, Arc<QTensor>);

impl CustomOp1 for QGatedMatMul {
    fn name(&self) -> &'static str {
        "qgated-matmul"
    }

    fn cpu_fwd(
        &self,
        storage: &CpuStorage,
        layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        if !layout.is_contiguous() {
            candle_core::bail!("input tensor is not contiguous {layout:?}")
        }
        let src_shape = layout.shape();
        let (n, k) = self.0.shape().dims2()?;
        let (n1, k1) = self.1.shape().dims2()?;
        if n != n1 || k != k1 {
            candle_core::bail!("gated matmul weight shape mismatch: ({n},{k}) vs ({n1},{k1})")
        }
        if src_shape.rank() < 2 {
            candle_core::bail!("input tensor has only one dimension {layout:?}")
        }
        let mut dst_shape = src_shape.dims().to_vec();
        let last_k = dst_shape.pop().unwrap();
        if last_k != k {
            candle_core::bail!(
                "input tensor {layout:?} incompatible with {:?}",
                self.0.shape()
            )
        }
        dst_shape.push(n);
        let dst_shape = Shape::from(dst_shape);
        let m = dst_shape.elem_count() / n;

        if storage.dtype() != DType::F32 {
            candle_core::bail!("QGatedMatMul only supports f32 input")
        }
        let slice = storage.as_slice::<f32>()?;
        let slice = &slice[layout.start_offset()..layout.start_offset() + src_shape.elem_count()];
        let mut dst_storage = vec![0f32; dst_shape.elem_count()];

        let gate_data = self.0.data()?;
        let up_data = self.1.data()?;

        macro_rules! dispatch {
            ($ty:ty) => {
                fused_gated_gelu_inner::<$ty>(
                    (m, k, n),
                    slice,
                    as_block_slice::<$ty>(&gate_data),
                    as_block_slice::<$ty>(&up_data),
                    &mut dst_storage,
                )
            };
        }
        match self.0.dtype() {
            GgmlDType::Q4K => dispatch!(BlockQ4K),
            GgmlDType::Q5K => dispatch!(BlockQ5K),
            GgmlDType::Q6K => dispatch!(BlockQ6K),
            GgmlDType::Q8K => dispatch!(BlockQ8K),
            GgmlDType::Q2K => dispatch!(BlockQ2K),
            GgmlDType::Q3K => dispatch!(BlockQ3K),
            GgmlDType::Q4_0 => dispatch!(BlockQ4_0),
            GgmlDType::Q5_0 => dispatch!(BlockQ5_0),
            GgmlDType::Q8_0 => dispatch!(BlockQ8_0),
            dt => candle_core::bail!("QGatedMatMul: unsupported dtype {dt:?}"),
        }

        Ok((CpuStorage::F32(dst_storage), dst_shape))
    }
}

// ---- fbgemm-rs INT8 GEMM (x86_64) ----

/// Quantize [N, K] row-major F32 weights to [K, N] row-major INT8 (transposed + quantized).
#[cfg(target_arch = "x86_64")]
fn quantize_weight_i8(w_f32: &[f32], n: usize, k: usize) -> (Vec<i8>, f32) {
    let max_abs = w_f32.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let b_scale = max_abs / 127.0;
    let inv_scale = if b_scale > 0.0 { 1.0 / b_scale } else { 0.0 };

    // Transpose [N, K] → [K, N] while quantizing
    let mut b_i8 = vec![0i8; k * n];
    for j in 0..n {
        for ki in 0..k {
            let val = w_f32[j * k + ki];
            b_i8[ki * n + j] = (val * inv_scale).round().clamp(-127.0, 127.0) as i8;
        }
    }
    (b_i8, b_scale)
}

#[cfg(target_arch = "x86_64")]
struct I8FbgemmOp {
    packed: Arc<PackedBMatrixI8>,
    b_scale: f32,
}

#[cfg(target_arch = "x86_64")]
impl CustomOp1 for I8FbgemmOp {
    fn name(&self) -> &'static str {
        "i8fbgemm-matmul"
    }

    fn cpu_fwd(
        &self,
        storage: &CpuStorage,
        layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        if !layout.is_contiguous() {
            candle_core::bail!("input tensor is not contiguous {layout:?}")
        }
        let src_shape = layout.shape();
        let k = self.packed.k();
        let n = self.packed.n();
        if src_shape.rank() < 2 {
            candle_core::bail!("input tensor has only one dimension {layout:?}")
        }
        let mut dst_shape = src_shape.dims().to_vec();
        let last_k = dst_shape.pop().unwrap();
        if last_k != k {
            candle_core::bail!(
                "input tensor {layout:?} incompatible with packed i8 matrix ({k}x{n})"
            )
        }
        dst_shape.push(n);
        let dst_shape = Shape::from(dst_shape);
        let m = dst_shape.elem_count() / n;

        if storage.dtype() != DType::F32 {
            candle_core::bail!("I8FbgemmOp only supports f32 input")
        }
        let slice = storage.as_slice::<f32>()?;
        let slice = &slice[layout.start_offset()..layout.start_offset() + src_shape.elem_count()];
        let mut dst_storage = vec![0f32; dst_shape.elem_count()];

        I8_SCRATCH.with_borrow_mut(|scratch| {
            fbgemm_rs::i8gemm_f32_par_with_scratch(m, slice, &self.packed, self.b_scale, &mut dst_storage, scratch);
        });

        Ok((CpuStorage::F32(dst_storage), dst_shape))
    }
}

// ---- fbgemm-rs F32 GEMM (non-x86_64 fallback) ----

#[cfg(not(target_arch = "x86_64"))]
struct FbgemmOp(Arc<PackedMatrix>);

#[cfg(not(target_arch = "x86_64"))]
impl CustomOp1 for FbgemmOp {
    fn name(&self) -> &'static str {
        "fbgemm-matmul"
    }

    fn cpu_fwd(
        &self,
        storage: &CpuStorage,
        layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        if !layout.is_contiguous() {
            candle_core::bail!("input tensor is not contiguous {layout:?}")
        }
        let src_shape = layout.shape();
        let k = self.0.k();
        let n = self.0.n();
        if src_shape.rank() < 2 {
            candle_core::bail!("input tensor has only one dimension {layout:?}")
        }
        let mut dst_shape = src_shape.dims().to_vec();
        let last_k = dst_shape.pop().unwrap();
        if last_k != k {
            candle_core::bail!(
                "input tensor {layout:?} incompatible with packed matrix ({k}x{n})"
            )
        }
        dst_shape.push(n);
        let dst_shape = Shape::from(dst_shape);
        let m = dst_shape.elem_count() / n;

        if storage.dtype() != DType::F32 {
            candle_core::bail!("FbgemmOp only supports f32 input")
        }
        let slice = storage.as_slice::<f32>()?;
        let slice = &slice[layout.start_offset()..layout.start_offset() + src_shape.elem_count()];
        let mut dst_storage = vec![0f32; dst_shape.elem_count()];

        fbgemm_rs::sgemm_simple_par(m, slice, &self.0, &mut dst_storage);

        Ok((CpuStorage::F32(dst_storage), dst_shape))
    }
}

// ---- MatMul: drop-in replacement for QMatMul ----

/// Drop-in replacement for `candle_core::quantized::QMatMul` that uses
/// column-tiled matmul for quantized weights. Pre-dequantized weights use
/// INT8 packed GEMM on x86_64, F32 packed GEMM elsewhere.
pub enum MatMul {
    QTensor(Arc<QTensor>),
    #[cfg(target_arch = "x86_64")]
    PackedI8 { packed: Arc<PackedBMatrixI8>, b_scale: f32 },
    #[cfg(not(target_arch = "x86_64"))]
    Packed(Arc<PackedMatrix>),
}

impl Clone for MatMul {
    fn clone(&self) -> Self {
        match self {
            Self::QTensor(qt) => Self::QTensor(qt.clone()),
            #[cfg(target_arch = "x86_64")]
            Self::PackedI8 { packed, b_scale } => Self::PackedI8 {
                packed: packed.clone(),
                b_scale: *b_scale,
            },
            #[cfg(not(target_arch = "x86_64"))]
            Self::Packed(p) => Self::Packed(p.clone()),
        }
    }
}

impl std::fmt::Debug for MatMul {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::QTensor(qt) => f.debug_tuple("QTensor").field(qt).finish(),
            #[cfg(target_arch = "x86_64")]
            Self::PackedI8 { packed, b_scale } => {
                write!(f, "PackedI8({}x{}, scale={})", packed.k(), packed.n(), b_scale)
            }
            #[cfg(not(target_arch = "x86_64"))]
            Self::Packed(p) => write!(f, "Packed({}x{})", p.k(), p.n()),
        }
    }
}

impl MatMul {
    pub fn from_qtensor(qt: Arc<QTensor>) -> Self {
        Self::QTensor(qt)
    }

    /// Pack a dequantized [N, K] weight tensor.
    /// x86_64: quantizes to INT8 and packs for i8gemm.
    /// Other: packs as F32 for sgemm.
    pub fn from_tensor(t: Tensor) -> Self {
        let (n, k) = t.dims2().expect("weight must be 2D for MatMul::from_tensor");
        let w_f32 = t
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .expect("weight to f32");

        #[cfg(target_arch = "x86_64")]
        {
            let (b_i8, b_scale) = quantize_weight_i8(&w_f32, n, k);
            let packed = PackedBMatrixI8::new(k, n, &b_i8);
            Self::PackedI8 { packed: Arc::new(packed), b_scale }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let packed = PackedMatrix::from_transposed(k, n, &w_f32);
            Self::Packed(Arc::new(packed))
        }
    }
}

impl Module for MatMul {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::QTensor(t) => xs.apply_op1_no_bwd(&QTiledOp(t.clone())),
            #[cfg(target_arch = "x86_64")]
            Self::PackedI8 { packed, b_scale } => {
                xs.apply_op1_no_bwd(&I8FbgemmOp {
                    packed: packed.clone(),
                    b_scale: *b_scale,
                })
            }
            #[cfg(not(target_arch = "x86_64"))]
            Self::Packed(p) => xs.apply_op1_no_bwd(&FbgemmOp(p.clone())),
        }
    }
}

/// Fused gated-gelu: `gelu(xs @ w0.T) * (xs @ w1.T)`.
pub fn forward_gated_gelu(w0: &MatMul, w1: &MatMul, xs: &Tensor) -> Result<Tensor> {
    match (w0, w1) {
        (MatMul::QTensor(w0), MatMul::QTensor(w1)) => {
            let op = QGatedMatMul(w0.clone(), w1.clone());
            xs.apply_op1_no_bwd(&op)
        }
        _ => {
            let gate = w0.forward(xs)?.gelu()?;
            let up = w1.forward(xs)?;
            gate.broadcast_mul(&up)
        }
    }
}
