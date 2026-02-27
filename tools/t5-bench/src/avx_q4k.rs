//! AVX2+FMA specialized Q4K dual-column matmul kernels.
//!
//! Processes two Q4K columns simultaneously sharing Q8K (LHS) data loads
//! and scale shuffle table lookups, cutting ~40% of memory operations in
//! the inner loop compared to two independent `vec_dot_q4k_q8k` calls.

// On non-x86_64 or without AVX2+FMA target features, this module is empty.
// Callers check cfg and fall back to the generic path.

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
mod imp {
    use core::arch::x86_64::*;

    use byteorder::{ByteOrder, LittleEndian};
    use candle_core::quantized::k_quants::{BlockQ4K, BlockQ8K, GgmlType, QK_K};
    use half::f16;
    use rayon::prelude::*;

    const BLOCK_Q4K_SIZE: usize = std::mem::size_of::<BlockQ4K>();
    const BLOCK_Q8K_SIZE: usize = std::mem::size_of::<BlockQ8K>();

    // Mirror structs with pub fields for accessing Q4K/Q8K block internals.
    // These match the candle #[repr(C)] layouts exactly.
    #[repr(C)]
    struct Q4KRaw {
        d: u16,           // f16 as raw bits
        dmin: u16,        // f16 as raw bits
        scales: [u8; 12],
        qs: [u8; 128],
    }
    const _: () = assert!(std::mem::size_of::<Q4KRaw>() == BLOCK_Q4K_SIZE);

    #[repr(C)]
    struct Q8KRaw {
        d: f32,
        qs: [i8; 256],
        bsums: [i16; 16],
    }
    const _: () = assert!(std::mem::size_of::<Q8KRaw>() == BLOCK_Q8K_SIZE);

    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    #[inline(always)]
    unsafe fn hsum_float_8(x: __m256) -> f32 {
        unsafe {
            let res = _mm256_extractf128_ps(x, 1);
            let res = _mm_add_ps(res, _mm256_castps256_ps128(x));
            let res = _mm_add_ps(res, _mm_movehl_ps(res, res));
            let res = _mm_add_ss(res, _mm_movehdup_ps(res));
            _mm_cvtss_f32(res)
        }
    }

    #[inline(always)]
    unsafe fn mm256_set_m128i(a: __m128i, b: __m128i) -> __m256i {
        unsafe { _mm256_insertf128_si256(_mm256_castsi128_si256(b), a, 1) }
    }

    #[inline(always)]
    unsafe fn get_scale_shuffle_k4(i: usize) -> __m256i {
        const K_SHUFFLE: [u8; 256] = [
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
            2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4,
            5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
            6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 8, 9, 8, 9, 8, 9, 8,
            9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 10,
            11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11,
            10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12,
            13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13,
            14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14,
            15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15,
        ];
        unsafe { _mm256_loadu_si256((K_SHUFFLE.as_ptr() as *const __m256i).add(i)) }
    }

    /// Decode Q4K packed 12-byte scales into [u32; 4] matching candle's format.
    #[inline(always)]
    fn decode_scales(scales: &[u8; 12], utmp: &mut [u32; 4]) {
        LittleEndian::read_u32_into(scales, &mut utmp[0..3]);
        utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
        let uaux = utmp[1] & KMASK1;
        utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
        utmp[2] = uaux;
        utmp[0] &= KMASK1;
    }

    /// Two Q4K·Q8K dot products simultaneously, sharing Q8K data loads.
    ///
    /// Equivalent to calling candle's `vec_dot_q4k_q8k` twice with the same
    /// Q8K (LHS) data but different Q4K (RHS) columns, but saves ~40% of
    /// memory loads in the inner loop by sharing Q8 and scale shuffle loads.
    #[inline(always)]
    unsafe fn vec_dot_q4k_q8k_dual(
        n: usize,
        col_a: &[Q4KRaw],
        col_b: &[Q4KRaw],
        lhs: &[Q8KRaw],
    ) -> (f32, f32) {
        unsafe {
            debug_assert!(n % QK_K == 0);
            let nb = n / QK_K;
            let m4 = _mm256_set1_epi8(0xF);

            let mut acc_a = _mm256_setzero_ps();
            let mut acc_b = _mm256_setzero_ps();
            let mut accm_a = _mm_setzero_ps();
            let mut accm_b = _mm_setzero_ps();

            for i in 0..nb {
                let y = &lhs[i];
                let xa = &col_a[i];
                let xb = &col_b[i];

                let yd = y.d;

                // Decode scales for both columns
                let mut utmp_a = [0u32; 4];
                let mut utmp_b = [0u32; 4];
                decode_scales(&xa.scales, &mut utmp_a);
                decode_scales(&xb.scales, &mut utmp_b);

                let da = yd * f16::from_bits(xa.d).to_f32();
                let dmina = -yd * f16::from_bits(xa.dmin).to_f32();
                let db = yd * f16::from_bits(xb.d).to_f32();
                let dminb = -yd * f16::from_bits(xb.dmin).to_f32();

                // Shared: load and hadd Q8K bsums
                let q8sums = _mm256_loadu_si256(y.bsums.as_ptr() as *const __m256i);
                let q8s = _mm_hadd_epi16(
                    _mm256_extracti128_si256(q8sums, 0),
                    _mm256_extracti128_si256(q8sums, 1),
                );

                // Min adjustment for column A
                let mins_and_scales_a = _mm256_cvtepu8_epi16(_mm_set_epi32(
                    utmp_a[3] as i32,
                    utmp_a[2] as i32,
                    utmp_a[1] as i32,
                    utmp_a[0] as i32,
                ));
                let prod_a =
                    _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales_a, 1), q8s);
                accm_a =
                    _mm_fmadd_ps(_mm_set1_ps(dmina), _mm_cvtepi32_ps(prod_a), accm_a);

                // Min adjustment for column B
                let mins_and_scales_b = _mm256_cvtepu8_epi16(_mm_set_epi32(
                    utmp_b[3] as i32,
                    utmp_b[2] as i32,
                    utmp_b[1] as i32,
                    utmp_b[0] as i32,
                ));
                let prod_b =
                    _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales_b, 1), q8s);
                accm_b =
                    _mm_fmadd_ps(_mm_set1_ps(dminb), _mm_cvtepi32_ps(prod_b), accm_b);

                // Broadcast scales for inner loop
                let sc128_a = _mm256_extracti128_si256(mins_and_scales_a, 0);
                let scales_a = mm256_set_m128i(sc128_a, sc128_a);
                let sc128_b = _mm256_extracti128_si256(mins_and_scales_b, 0);
                let scales_b = mm256_set_m128i(sc128_b, sc128_b);

                let mut sumi_a = _mm256_setzero_si256();
                let mut sumi_b = _mm256_setzero_si256();

                let mut q4a = xa.qs.as_ptr();
                let mut q4b = xb.qs.as_ptr();
                let mut q8 = y.qs.as_ptr();

                for j in 0..QK_K / 64 {
                    // Shared: load scale shuffle masks once
                    let scale_shuf_l = get_scale_shuffle_k4(2 * j);
                    let scale_shuf_h = get_scale_shuffle_k4(2 * j + 1);

                    let scale_l_a = _mm256_shuffle_epi8(scales_a, scale_shuf_l);
                    let scale_h_a = _mm256_shuffle_epi8(scales_a, scale_shuf_h);
                    let scale_l_b = _mm256_shuffle_epi8(scales_b, scale_shuf_l);
                    let scale_h_b = _mm256_shuffle_epi8(scales_b, scale_shuf_h);

                    // Shared: load Q8K data (32B low, 32B high)
                    let q8l = _mm256_loadu_si256(q8 as *const __m256i);
                    let q8h = _mm256_loadu_si256(q8.add(32) as *const __m256i);
                    q8 = q8.add(64);

                    // Column A
                    let q4bits_a = _mm256_loadu_si256(q4a as *const __m256i);
                    q4a = q4a.add(32);
                    let q4l_a = _mm256_and_si256(q4bits_a, m4);
                    let q4h_a =
                        _mm256_and_si256(_mm256_srli_epi16(q4bits_a, 4), m4);
                    let p16l_a = _mm256_madd_epi16(
                        scale_l_a,
                        _mm256_maddubs_epi16(q4l_a, q8l),
                    );
                    sumi_a = _mm256_add_epi32(sumi_a, p16l_a);
                    let p16h_a = _mm256_madd_epi16(
                        scale_h_a,
                        _mm256_maddubs_epi16(q4h_a, q8h),
                    );
                    sumi_a = _mm256_add_epi32(sumi_a, p16h_a);

                    // Column B (reuses q8l, q8h)
                    let q4bits_b = _mm256_loadu_si256(q4b as *const __m256i);
                    q4b = q4b.add(32);
                    let q4l_b = _mm256_and_si256(q4bits_b, m4);
                    let q4h_b =
                        _mm256_and_si256(_mm256_srli_epi16(q4bits_b, 4), m4);
                    let p16l_b = _mm256_madd_epi16(
                        scale_l_b,
                        _mm256_maddubs_epi16(q4l_b, q8l),
                    );
                    sumi_b = _mm256_add_epi32(sumi_b, p16l_b);
                    let p16h_b = _mm256_madd_epi16(
                        scale_h_b,
                        _mm256_maddubs_epi16(q4h_b, q8h),
                    );
                    sumi_b = _mm256_add_epi32(sumi_b, p16h_b);
                }

                acc_a = _mm256_fmadd_ps(
                    _mm256_set1_ps(da),
                    _mm256_cvtepi32_ps(sumi_a),
                    acc_a,
                );
                acc_b = _mm256_fmadd_ps(
                    _mm256_set1_ps(db),
                    _mm256_cvtepi32_ps(sumi_b),
                    acc_b,
                );
            }

            // Horizontal reduce acc_m (XMM, 4 floats -> scalar)
            let accm_a = _mm_add_ps(accm_a, _mm_movehl_ps(accm_a, accm_a));
            let accm_a = _mm_add_ss(accm_a, _mm_movehdup_ps(accm_a));
            let accm_b = _mm_add_ps(accm_b, _mm_movehl_ps(accm_b, accm_b));
            let accm_b = _mm_add_ss(accm_b, _mm_movehdup_ps(accm_b));

            (
                hsum_float_8(acc_a) + _mm_cvtss_f32(accm_a),
                hsum_float_8(acc_b) + _mm_cvtss_f32(accm_b),
            )
        }
    }

    /// Single Q4K·Q8K dot product (for odd-column remainder in tiled matmul).
    #[inline(always)]
    unsafe fn vec_dot_q4k_q8k_single(
        n: usize,
        col: &[Q4KRaw],
        lhs: &[Q8KRaw],
    ) -> f32 {
        unsafe {
            debug_assert!(n % QK_K == 0);
            let nb = n / QK_K;
            let m4 = _mm256_set1_epi8(0xF);

            let mut acc = _mm256_setzero_ps();
            let mut accm = _mm_setzero_ps();

            for i in 0..nb {
                let y = &lhs[i];
                let x = &col[i];

                let d = y.d * f16::from_bits(x.d).to_f32();
                let dmin = -y.d * f16::from_bits(x.dmin).to_f32();

                let mut utmp = [0u32; 4];
                decode_scales(&x.scales, &mut utmp);

                let q8sums = _mm256_loadu_si256(y.bsums.as_ptr() as *const __m256i);
                let q8s = _mm_hadd_epi16(
                    _mm256_extracti128_si256(q8sums, 0),
                    _mm256_extracti128_si256(q8sums, 1),
                );

                let mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(
                    utmp[3] as i32,
                    utmp[2] as i32,
                    utmp[1] as i32,
                    utmp[0] as i32,
                ));
                let prod =
                    _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
                accm = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), accm);

                let sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
                let scales = mm256_set_m128i(sc128, sc128);

                let mut sumi = _mm256_setzero_si256();
                let mut q4 = x.qs.as_ptr();
                let mut q8 = y.qs.as_ptr();

                for j in 0..QK_K / 64 {
                    let scale_l =
                        _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j));
                    let scale_h =
                        _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));

                    let q4bits = _mm256_loadu_si256(q4 as *const __m256i);
                    q4 = q4.add(32);
                    let q4l = _mm256_and_si256(q4bits, m4);
                    let q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

                    let q8l = _mm256_loadu_si256(q8 as *const __m256i);
                    q8 = q8.add(32);
                    let p16l = _mm256_maddubs_epi16(q4l, q8l);
                    let p16l = _mm256_madd_epi16(scale_l, p16l);
                    sumi = _mm256_add_epi32(sumi, p16l);

                    let q8h = _mm256_loadu_si256(q8 as *const __m256i);
                    q8 = q8.add(32);
                    let p16h = _mm256_maddubs_epi16(q4h, q8h);
                    let p16h = _mm256_madd_epi16(scale_h, p16h);
                    sumi = _mm256_add_epi32(sumi, p16h);
                }

                acc = _mm256_fmadd_ps(
                    _mm256_set1_ps(d),
                    _mm256_cvtepi32_ps(sumi),
                    acc,
                );
            }

            let accm = _mm_add_ps(accm, _mm_movehl_ps(accm, accm));
            let accm = _mm_add_ss(accm, _mm_movehdup_ps(accm));
            hsum_float_8(acc) + _mm_cvtss_f32(accm)
        }
    }

    fn as_raw_q4k(data: &[u8]) -> &[Q4KRaw] {
        debug_assert_eq!(data.len() % BLOCK_Q4K_SIZE, 0);
        unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const Q4KRaw,
                data.len() / BLOCK_Q4K_SIZE,
            )
        }
    }

    fn as_raw_q8k(blocks: &[BlockQ8K]) -> &[Q8KRaw] {
        unsafe { std::slice::from_raw_parts(blocks.as_ptr() as *const Q8KRaw, blocks.len()) }
    }

    /// Q4K-specialized fused gated-gelu using dual vec_dot.
    pub fn fused_gated_gelu_q4k(
        (m, k, n): (usize, usize, usize),
        lhs: &[f32],
        rhs_gate_data: &[u8],
        rhs_up_data: &[u8],
        dst: &mut [f32],
    ) {
        let k_in_blocks = k.div_ceil(QK_K);

        let mut lhs_b = vec![BlockQ8K::zeros(); m * k_in_blocks];
        for row_idx in 0..m {
            BlockQ8K::from_float(
                &lhs[row_idx * k..(row_idx + 1) * k],
                &mut lhs_b[row_idx * k_in_blocks..(row_idx + 1) * k_in_blocks],
            );
        }

        let gate_q4k = as_raw_q4k(rhs_gate_data);
        let up_q4k = as_raw_q4k(rhs_up_data);
        let lhs_q8k = as_raw_q8k(&lhs_b);

        let tile_n = 64.min(n);
        let tile_starts: Vec<usize> = (0..n).step_by(tile_n).collect();
        let dst_ptr = dst.as_mut_ptr() as usize;

        tile_starts.into_par_iter().for_each(|tile_start| {
            let tile_end = (tile_start + tile_n).min(n);
            let dst = dst_ptr as *mut f32;
            for row_idx in 0..m {
                let lhs_row =
                    &lhs_q8k[row_idx * k_in_blocks..(row_idx + 1) * k_in_blocks];
                for col_idx in tile_start..tile_end {
                    let gate_col =
                        &gate_q4k[col_idx * k_in_blocks..(col_idx + 1) * k_in_blocks];
                    let up_col =
                        &up_q4k[col_idx * k_in_blocks..(col_idx + 1) * k_in_blocks];
                    let (gate, up) =
                        unsafe { vec_dot_q4k_q8k_dual(k, gate_col, up_col, lhs_row) };
                    let gate = 0.5 * gate
                        * (1.0
                            + f32::tanh(
                                0.7978845608_f32 * gate * (1.0 + 0.044715 * gate * gate),
                            ));
                    unsafe {
                        *dst.add(row_idx * n + col_idx) = gate * up;
                    }
                }
            }
        });
    }

    /// Q4K-specialized tiled matmul using dual vec_dot for pairs of columns.
    pub fn tiled_matmul_q4k(
        (m, k, n): (usize, usize, usize),
        lhs: &[f32],
        rhs_data: &[u8],
        dst: &mut [f32],
    ) {
        let k_in_blocks = k.div_ceil(QK_K);

        let mut lhs_b = vec![BlockQ8K::zeros(); m * k_in_blocks];
        for row_idx in 0..m {
            BlockQ8K::from_float(
                &lhs[row_idx * k..(row_idx + 1) * k],
                &mut lhs_b[row_idx * k_in_blocks..(row_idx + 1) * k_in_blocks],
            );
        }

        let rhs_q4k = as_raw_q4k(rhs_data);
        let lhs_q8k = as_raw_q8k(&lhs_b);

        let tile_n = 128.min(n);
        let tile_starts: Vec<usize> = (0..n).step_by(tile_n).collect();
        let dst_ptr = dst.as_mut_ptr() as usize;

        tile_starts.into_par_iter().for_each(|tile_start| {
            let tile_end = (tile_start + tile_n).min(n);
            let dst = dst_ptr as *mut f32;
            for row_idx in 0..m {
                let lhs_row =
                    &lhs_q8k[row_idx * k_in_blocks..(row_idx + 1) * k_in_blocks];
                let mut col_idx = tile_start;
                // Process pairs of columns with dual vec_dot
                while col_idx + 1 < tile_end {
                    let col_a =
                        &rhs_q4k[col_idx * k_in_blocks..(col_idx + 1) * k_in_blocks];
                    let col_b = &rhs_q4k
                        [(col_idx + 1) * k_in_blocks..(col_idx + 2) * k_in_blocks];
                    let (va, vb) =
                        unsafe { vec_dot_q4k_q8k_dual(k, col_a, col_b, lhs_row) };
                    unsafe {
                        *dst.add(row_idx * n + col_idx) = va;
                        *dst.add(row_idx * n + col_idx + 1) = vb;
                    }
                    col_idx += 2;
                }
                // Odd remainder column
                if col_idx < tile_end {
                    let col =
                        &rhs_q4k[col_idx * k_in_blocks..(col_idx + 1) * k_in_blocks];
                    unsafe {
                        *dst.add(row_idx * n + col_idx) =
                            vec_dot_q4k_q8k_single(k, col, lhs_row);
                    }
                }
            }
        });
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
pub(crate) use imp::{fused_gated_gelu_q4k, tiled_matmul_q4k};
