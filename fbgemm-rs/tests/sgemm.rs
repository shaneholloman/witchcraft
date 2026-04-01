use fbgemm_rs::{sgemm, sgemm_simple, PackedMatrix};

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

#[test]
fn identity_times_matrix() {
    let n = 4;
    // I(4x4) * B(4x4) = B
    let mut eye = vec![0.0f32; 16];
    for i in 0..n {
        eye[i * n + i] = 1.0;
    }
    let b: Vec<f32> = (1..=16).map(|x| x as f32).collect();
    let packed_b = PackedMatrix::new(n, n, &b);

    let mut c = vec![0.0f32; 16];
    sgemm_simple(n, &eye, &packed_b, &mut c);

    for i in 0..16 {
        assert!(
            (c[i] - b[i]).abs() < 1e-4,
            "index {}: got {}, expected {}",
            i,
            c[i],
            b[i]
        );
    }
}

#[test]
fn small_matmul_matches_naive() {
    let (m, k, n) = (3, 5, 4);
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.5 - 3.0).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.3 + 1.0).collect();

    let expected = naive_matmul(m, k, n, &a, &b);

    let packed_b = PackedMatrix::new(k, n, &b);
    let mut c = vec![0.0f32; m * n];
    sgemm_simple(m, &a, &packed_b, &mut c);

    for i in 0..m * n {
        assert!(
            (c[i] - expected[i]).abs() < 1e-3,
            "index {}: got {}, expected {}",
            i,
            c[i],
            expected[i]
        );
    }
}

#[test]
fn beta_accumulate() {
    let (m, k, n) = (2, 3, 2);
    let a = vec![1.0f32; m * k];
    let b = vec![1.0f32; k * n];
    let packed_b = PackedMatrix::new(k, n, &b);

    // First pass: C = A*B (each element = k = 3.0)
    let mut c = vec![0.0f32; m * n];
    sgemm_simple(m, &a, &packed_b, &mut c);
    for &v in &c {
        assert!((v - 3.0).abs() < 1e-4);
    }

    // Second pass: C = 1.0*C + A*B (each element = 3 + 3 = 6.0)
    sgemm(m, &a, &packed_b, 1.0, &mut c);
    for &v in &c {
        assert!((v - 6.0).abs() < 1e-4);
    }
}
