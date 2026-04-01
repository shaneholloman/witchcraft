use fbgemm_rs::{sgemm_bf16, sgemm_bf16_simple, PackedMatrixBf16};

fn naive_matmul(m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            for p in 0..k {
                c[i * n + j] += a[i * k + p] * b[p * n + j];
            }
        }
    }
    c
}

/// Maximum relative error we tolerate from bf16 truncation.
/// bf16 truncates f32 mantissa from 23 to 7 bits, so individual values
/// lose up to ~0.8% precision. Accumulated over a dot product of length K,
/// the error can compound, so we use a generous tolerance.
const BF16_REL_TOL: f32 = 0.02;
const BF16_ABS_TOL: f32 = 0.1;

fn assert_close(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for i in 0..actual.len() {
        let err = (actual[i] - expected[i]).abs();
        let rel = if expected[i].abs() > 1.0 {
            err / expected[i].abs()
        } else {
            err
        };
        assert!(
            rel < BF16_REL_TOL || err < BF16_ABS_TOL,
            "{label}[{i}]: got {}, expected {}, rel_err={rel}, abs_err={err}",
            actual[i],
            expected[i],
        );
    }
}

// ---------- Basic correctness ----------

#[test]
fn bf16_identity_times_matrix() {
    let n = 4;
    let mut eye = vec![0.0f32; 16];
    for i in 0..n {
        eye[i * n + i] = 1.0;
    }
    let b: Vec<f32> = (1..=16).map(|x| x as f32).collect();
    let packed_b = PackedMatrixBf16::new(n, n, &b);

    let mut c = vec![0.0f32; 16];
    sgemm_bf16_simple(n, &eye, &packed_b, &mut c);

    // bf16 roundtrip of integer values 1..16 is exact (they fit in 7-bit mantissa)
    for i in 0..16 {
        assert!(
            (c[i] - b[i]).abs() < 1e-2,
            "index {i}: got {}, expected {}",
            c[i],
            b[i],
        );
    }
}

#[test]
fn bf16_small_matmul_matches_naive() {
    let (m, k, n) = (3, 5, 4);
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.5 - 3.0).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.3 + 1.0).collect();

    let expected = naive_matmul(m, k, n, &a, &b);

    let packed_b = PackedMatrixBf16::new(k, n, &b);
    let mut c = vec![0.0f32; m * n];
    sgemm_bf16_simple(m, &a, &packed_b, &mut c);

    assert_close(&c, &expected, "small_matmul");
}

#[test]
fn bf16_beta_accumulate() {
    let (m, k, n) = (2, 3, 2);
    let a = vec![1.0f32; m * k];
    let b = vec![1.0f32; k * n];
    let packed_b = PackedMatrixBf16::new(k, n, &b);

    // First pass: C = A*B (each element = k = 3.0)
    let mut c = vec![0.0f32; m * n];
    sgemm_bf16_simple(m, &a, &packed_b, &mut c);
    for &v in &c {
        assert!((v - 3.0).abs() < 0.5, "first pass: {v}");
    }

    // Second pass: C = 1.0*C + A*B (each element = 3 + 3 = 6.0)
    sgemm_bf16(m, &a, &packed_b, 1.0, &mut c);
    for &v in &c {
        assert!((v - 6.0).abs() < 0.5, "second pass: {v}");
    }
}

#[test]
fn bf16_beta_scale() {
    let (m, k, n) = (4, 8, 4);
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.1).collect();
    let packed_b = PackedMatrixBf16::new(k, n, &b);

    // C = A*B
    let mut c = vec![0.0f32; m * n];
    sgemm_bf16_simple(m, &a, &packed_b, &mut c);
    let c1 = c.clone();

    // C = 2.0*C + A*B  → should be 2*c1 + c1 = 3*c1
    sgemm_bf16(m, &a, &packed_b, 2.0, &mut c);
    let expected: Vec<f32> = c1.iter().map(|&v| 3.0 * v).collect();
    assert_close(&c, &expected, "beta_scale");
}

#[test]
fn bf16_negative_values() {
    let (m, k, n) = (4, 8, 4);
    let a: Vec<f32> = (0..m * k).map(|i| -1.0 + (i as f32) * 0.1).collect();
    let b: Vec<f32> = (0..k * n).map(|i| 2.0 - (i as f32) * 0.05).collect();

    let expected = naive_matmul(m, k, n, &a, &b);
    let packed_b = PackedMatrixBf16::new(k, n, &b);
    let mut c = vec![0.0f32; m * n];
    sgemm_bf16_simple(m, &a, &packed_b, &mut c);

    assert_close(&c, &expected, "negative_values");
}

#[test]
fn bf16_transposed_matches_row_major() {
    let (k, n) = (8, 12);
    let b_row_major: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.01 + 0.5).collect();

    // Column-major = transpose of row-major
    let mut b_col_major = vec![0.0f32; k * n];
    for i in 0..k {
        for j in 0..n {
            b_col_major[i + k * j] = b_row_major[i * n + j];
        }
    }

    let packed_row = PackedMatrixBf16::new(k, n, &b_row_major);
    let packed_col = PackedMatrixBf16::from_transposed(k, n, &b_col_major);

    let m = 6;
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
    let mut c_row = vec![0.0f32; m * n];
    let mut c_col = vec![0.0f32; m * n];

    sgemm_bf16_simple(m, &a, &packed_row, &mut c_row);
    sgemm_bf16_simple(m, &a, &packed_col, &mut c_col);

    for i in 0..m * n {
        assert!(
            (c_row[i] - c_col[i]).abs() < 1e-6,
            "index {i}: row={}, col={}",
            c_row[i],
            c_col[i],
        );
    }
}

// ---------- Fringe columns (N not divisible by BLOCK_COL_SIZE=16) ----------

#[test]
fn bf16_fringe_columns_n_is_1() {
    let (m, k, n) = (4, 8, 1);
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.2 + 1.0).collect();

    let expected = naive_matmul(m, k, n, &a, &b);
    let packed_b = PackedMatrixBf16::new(k, n, &b);
    let mut c = vec![0.0f32; m * n];
    sgemm_bf16_simple(m, &a, &packed_b, &mut c);

    assert_close(&c, &expected, "fringe_n=1");
}

#[test]
fn bf16_fringe_columns_n_is_17() {
    // 17 = 1 full block of 16 + fringe of 1
    let (m, k, n) = (4, 8, 17);
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.01).collect();

    let expected = naive_matmul(m, k, n, &a, &b);
    let packed_b = PackedMatrixBf16::new(k, n, &b);
    let mut c = vec![0.0f32; m * n];
    sgemm_bf16_simple(m, &a, &packed_b, &mut c);

    assert_close(&c, &expected, "fringe_n=17");
}

#[test]
fn bf16_fringe_columns_n_is_31() {
    // 31 = 1 full block of 16 + fringe of 15
    let (m, k, n) = (6, 16, 31);
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.01).collect();

    let expected = naive_matmul(m, k, n, &a, &b);
    let packed_b = PackedMatrixBf16::new(k, n, &b);
    let mut c = vec![0.0f32; m * n];
    sgemm_bf16_simple(m, &a, &packed_b, &mut c);

    assert_close(&c, &expected, "fringe_n=31");
}

// ---------- Single row (M=1, special x86 path) ----------

#[test]
fn bf16_single_row() {
    let (m, k, n) = (1, 32, 16);
    let a: Vec<f32> = (0..k).map(|i| (i as f32) * 0.1).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.01).collect();

    let expected = naive_matmul(m, k, n, &a, &b);
    let packed_b = PackedMatrixBf16::new(k, n, &b);
    let mut c = vec![0.0f32; n];
    sgemm_bf16_simple(1, &a, &packed_b, &mut c);

    assert_close(&c, &expected, "single_row");
}

#[test]
fn bf16_single_row_fringe() {
    let (m, k, n) = (1, 32, 19);
    let a: Vec<f32> = (0..k).map(|i| (i as f32) * 0.1).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.01).collect();

    let expected = naive_matmul(m, k, n, &a, &b);
    let packed_b = PackedMatrixBf16::new(k, n, &b);
    let mut c = vec![0.0f32; n];
    sgemm_bf16_simple(1, &a, &packed_b, &mut c);

    assert_close(&c, &expected, "single_row_fringe");
}

// ---------- Large K requiring multiple brow blocks (K > 512) ----------

#[test]
fn bf16_large_k_multiple_blocks() {
    let (m, k, n) = (4, 600, 8);
    let a: Vec<f32> = (0..m * k).map(|i| ((i % 100) as f32) * 0.001).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i % 100) as f32) * 0.001).collect();

    let expected = naive_matmul(m, k, n, &a, &b);
    let packed_b = PackedMatrixBf16::new(k, n, &b);
    let mut c = vec![0.0f32; m * n];
    sgemm_bf16_simple(m, &a, &packed_b, &mut c);

    assert_close(&c, &expected, "large_k_600");
}

#[test]
fn bf16_large_k_exact_block_boundary() {
    // K=512 exactly, one full brow block
    let (m, k, n) = (4, 512, 16);
    let a: Vec<f32> = (0..m * k).map(|i| ((i % 50) as f32) * 0.001).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i % 50) as f32) * 0.001).collect();

    let expected = naive_matmul(m, k, n, &a, &b);
    let packed_b = PackedMatrixBf16::new(k, n, &b);
    let mut c = vec![0.0f32; m * n];
    sgemm_bf16_simple(m, &a, &packed_b, &mut c);

    assert_close(&c, &expected, "large_k_512");
}

#[test]
fn bf16_large_k_two_full_blocks() {
    // K=1024 = 2 * 512
    let (m, k, n) = (4, 1024, 16);
    let a: Vec<f32> = (0..m * k).map(|i| ((i % 50) as f32) * 0.001).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i % 50) as f32) * 0.001).collect();

    let expected = naive_matmul(m, k, n, &a, &b);
    let packed_b = PackedMatrixBf16::new(k, n, &b);
    let mut c = vec![0.0f32; m * n];
    sgemm_bf16_simple(m, &a, &packed_b, &mut c);

    assert_close(&c, &expected, "large_k_1024");
}

// ---------- Sweep M sizes (covers MB_MAX=120 boundary and partition table) ----------

#[test]
fn bf16_sweep_m_1_to_130() {
    let k = 32;
    let n = 16;
    let b: Vec<f32> = (0..k * n).map(|i| ((i % 20) as f32) * 0.05).collect();
    let packed_b = PackedMatrixBf16::new(k, n, &b);

    for m in 1..=130 {
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 30) as f32) * 0.03).collect();
        let expected = naive_matmul(m, k, n, &a, &b);
        let mut c = vec![0.0f32; m * n];
        sgemm_bf16_simple(m, &a, &packed_b, &mut c);
        assert_close(&c, &expected, &format!("sweep_m={m}"));
    }
}

// ---------- Sweep N sizes (fringe coverage) ----------

#[test]
fn bf16_sweep_n_1_to_50() {
    let m = 6;
    let k = 16;

    for n in 1..=50 {
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 20) as f32) * 0.05).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 20) as f32) * 0.05).collect();
        let expected = naive_matmul(m, k, n, &a, &b);

        let packed_b = PackedMatrixBf16::new(k, n, &b);
        let mut c = vec![0.0f32; m * n];
        sgemm_bf16_simple(m, &a, &packed_b, &mut c);
        assert_close(&c, &expected, &format!("sweep_n={n}"));
    }
}

// ---------- Combination sweep: stress many M×K×N combos ----------

#[test]
fn bf16_stress_dimensions() {
    let ms = [1, 2, 3, 5, 6, 7, 12, 13, 60, 120, 121];
    let ks = [1, 2, 15, 16, 32, 100, 512, 513];
    let ns = [1, 2, 15, 16, 17, 31, 32, 33, 48];

    for &m in &ms {
        for &k in &ks {
            for &n in &ns {
                let a: Vec<f32> = (0..m * k)
                    .map(|i| ((i * 7 + 3) % 100) as f32 * 0.01 - 0.5)
                    .collect();
                let b: Vec<f32> = (0..k * n)
                    .map(|i| ((i * 11 + 5) % 100) as f32 * 0.01 - 0.5)
                    .collect();
                let expected = naive_matmul(m, k, n, &a, &b);

                let packed_b = PackedMatrixBf16::new(k, n, &b);
                let mut c = vec![0.0f32; m * n];
                sgemm_bf16_simple(m, &a, &packed_b, &mut c);
                assert_close(&c, &expected, &format!("stress m={m} k={k} n={n}"));
            }
        }
    }
}

// ---------- Zero matrices ----------

#[test]
fn bf16_zero_a() {
    let (m, k, n) = (4, 8, 16);
    let a = vec![0.0f32; m * k];
    let b: Vec<f32> = (0..k * n).map(|i| i as f32).collect();
    let packed_b = PackedMatrixBf16::new(k, n, &b);
    let mut c = vec![0.0f32; m * n];
    sgemm_bf16_simple(m, &a, &packed_b, &mut c);

    for &v in &c {
        assert!(v.abs() < 1e-6, "expected zero, got {v}");
    }
}

#[test]
fn bf16_zero_b() {
    let (m, k, n) = (4, 8, 16);
    let a: Vec<f32> = (0..m * k).map(|i| i as f32).collect();
    let b = vec![0.0f32; k * n];
    let packed_b = PackedMatrixBf16::new(k, n, &b);
    let mut c = vec![0.0f32; m * n];
    sgemm_bf16_simple(m, &a, &packed_b, &mut c);

    for &v in &c {
        assert!(v.abs() < 1e-6, "expected zero, got {v}");
    }
}

// ---------- Wide and tall matrices ----------

#[test]
fn bf16_wide_matrix() {
    let (m, k, n) = (4, 16, 128);
    let a: Vec<f32> = (0..m * k).map(|i| ((i % 20) as f32) * 0.05).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i % 30) as f32) * 0.03).collect();
    let expected = naive_matmul(m, k, n, &a, &b);

    let packed_b = PackedMatrixBf16::new(k, n, &b);
    let mut c = vec![0.0f32; m * n];
    sgemm_bf16_simple(m, &a, &packed_b, &mut c);

    assert_close(&c, &expected, "wide_matrix");
}

#[test]
fn bf16_tall_matrix() {
    let (m, k, n) = (256, 16, 4);
    let a: Vec<f32> = (0..m * k).map(|i| ((i % 20) as f32) * 0.05).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i % 30) as f32) * 0.03).collect();
    let expected = naive_matmul(m, k, n, &a, &b);

    let packed_b = PackedMatrixBf16::new(k, n, &b);
    let mut c = vec![0.0f32; m * n];
    sgemm_bf16_simple(m, &a, &packed_b, &mut c);

    assert_close(&c, &expected, "tall_matrix");
}

// ---------- Packing correctness ----------

#[test]
fn bf16_memory_savings() {
    let k = 768;
    let n = 3072;
    let src: Vec<f32> = vec![0.0; k * n];

    let packed_f32 = fbgemm_rs::PackedMatrix::new(k, n, &src);
    let packed_bf16 = PackedMatrixBf16::new(k, n, &src);

    let f32_bytes = packed_f32.k() * packed_f32.n() * 4;
    let bf16_bytes = packed_bf16.size_bytes();
    let ratio = f32_bytes as f64 / bf16_bytes as f64;
    assert!(
        ratio > 0.5 && ratio < 2.5,
        "expected bf16 to use roughly half the memory, ratio={ratio:.2}",
    );
}

#[test]
fn bf16_k_and_n_accessors() {
    let packed = PackedMatrixBf16::new(100, 200, &vec![0.0f32; 100 * 200]);
    assert_eq!(packed.k(), 100);
    assert_eq!(packed.n(), 200);
}

// ---------- Edge cases that should not crash ----------

#[test]
fn bf16_m_zero() {
    let (k, n) = (8, 16);
    let b = vec![1.0f32; k * n];
    let packed_b = PackedMatrixBf16::new(k, n, &b);
    let a: Vec<f32> = vec![];
    let mut c: Vec<f32> = vec![];
    sgemm_bf16_simple(0, &a, &packed_b, &mut c);
    // Should complete without crash
}

#[test]
#[should_panic(expected = "a must have length m * k")]
fn bf16_mismatched_a_panics() {
    let packed_b = PackedMatrixBf16::new(4, 4, &vec![0.0f32; 16]);
    let a = vec![0.0f32; 5]; // wrong length
    let mut c = vec![0.0f32; 8];
    sgemm_bf16_simple(2, &a, &packed_b, &mut c);
}

#[test]
#[should_panic(expected = "c must have length m * n")]
fn bf16_mismatched_c_panics() {
    let packed_b = PackedMatrixBf16::new(4, 4, &vec![0.0f32; 16]);
    let a = vec![0.0f32; 8];
    let mut c = vec![0.0f32; 5]; // wrong length
    sgemm_bf16_simple(2, &a, &packed_b, &mut c);
}

#[test]
#[should_panic(expected = "src length must be k * n")]
fn bf16_pack_mismatched_src_panics() {
    let _ = PackedMatrixBf16::new(4, 4, &vec![0.0f32; 10]);
}
