use anyhow::{anyhow, Result};
use openvino::{CompiledModel, Core, DeviceType, InferRequest, Shape};
use rand::RngExt;
use std::path::PathBuf;
use std::time::Instant;

fn bucket_size(n: usize) -> usize {
    let min = 64;
    if n <= min { return min; }
    n.next_power_of_two()
}

struct OvModel {
    _core: Core,
    _compiled_model: CompiledModel,
    infer_request: InferRequest,
}

fn load_model(assets: &PathBuf) -> Result<OvModel> {
    let mut core = Core::new().map_err(|e| anyhow!("ov core: {e:?}"))?;

    let temp_dir = tempfile::tempdir()?;
    let xml_path = temp_dir.path().join("model.xml");
    {
        let compressed = std::fs::read(assets.join("xtr-ov-int4.xml.zst"))?;
        let xml_bytes = zstd::stream::decode_all(std::io::Cursor::new(compressed))?;
        std::fs::write(&xml_path, xml_bytes)?;
    }
    let bin_path = assets.join("xtr-ov-int4.bin");

    let model = core.read_model_from_file(
        xml_path.to_str().unwrap(),
        bin_path.to_str().unwrap(),
    ).map_err(|e| anyhow!("read model: {e:?}"))?;

    let mut compiled_model = core.compile_model(&model, DeviceType::CPU)
        .map_err(|e| anyhow!("compile: {e:?}"))?;

    let infer_request = compiled_model.create_infer_request()
        .map_err(|e| anyhow!("infer request: {e:?}"))?;

    // Keep temp_dir alive until model is loaded — XML is read during read_model_from_file
    drop(temp_dir);

    Ok(OvModel { _core: core, _compiled_model: compiled_model, infer_request })
}

fn run_inference(ir: &mut InferRequest, seq_len: usize) -> Result<Vec<f32>> {
    let padded_len = bucket_size(seq_len);
    let mut input_data: Vec<i64> = (1..=seq_len as i64).collect();
    input_data.resize(padded_len, 0);

    let ov_shape = Shape::new(&[1i64, padded_len as i64])
        .map_err(|e| anyhow!("shape: {e:?}"))?;
    let mut ov_input = openvino::Tensor::new(openvino::ElementType::I64, &ov_shape)
        .map_err(|e| anyhow!("tensor: {e:?}"))?;
    ov_input.get_data_mut::<i64>()
        .map_err(|e| anyhow!("data: {e:?}"))?
        .copy_from_slice(&input_data);

    ir.set_input_tensor(&ov_input)
        .map_err(|e| anyhow!("set input: {e:?}"))?;
    ir.infer()
        .map_err(|e| anyhow!("infer: {e:?}"))?;

    let ov_output = ir.get_output_tensor()
        .map_err(|e| anyhow!("output: {e:?}"))?;
    let output_shape = ov_output.get_shape()
        .map_err(|e| anyhow!("output shape: {e:?}"))?;
    let dims: Vec<usize> = output_shape.get_dimensions().iter().map(|&d| d as usize).collect();
    let embed_dim = *dims.last().ok_or_else(|| anyhow!("empty dims"))?;

    let all_data: Vec<f32> = ov_output.get_data::<f32>()
        .map_err(|e| anyhow!("output data: {e:?}"))?
        .to_vec();
    std::mem::forget(ov_output);

    // Slice off padding
    let unpadded: Vec<f32> = all_data[..seq_len * embed_dim].to_vec();
    Ok(unpadded)
}

/// Test 1: Shape variation — random seq_len each iteration
fn test_shape_variation(model: &mut OvModel) -> Result<()> {
    eprintln!("\n=== Test 1: Shape Variation (200 iterations) ===");
    eprintln!("  Starting warmup...");
    let mut rng = rand::rng();

    // Warmup
    eprintln!("  Running warmup inference...");
    let _ = run_inference(&mut model.infer_request, 64)?;
    eprintln!("  Warmup complete.");

    let mut prev_len = 64;
    for i in 0..200 {
        let seq_len: usize = rng.random_range(16..=512);
        let result = run_inference(&mut model.infer_request, seq_len);
        match result {
            Ok(data) => {
                // Validate: output should have seq_len * embed_dim elements
                // embed_dim is 128 for XTR
                if data.len() != seq_len * 128 {
                    eprintln!("  FAIL iter {i}: expected {} floats, got {} (seq_len={seq_len})",
                        seq_len * 128, data.len());
                    return Err(anyhow!("shape mismatch"));
                }
                // Check for NaN
                let nan_count = data.iter().filter(|x| x.is_nan()).count();
                if nan_count > 0 {
                    eprintln!("  FAIL iter {i}: {nan_count} NaN values (seq_len={seq_len}, prev={prev_len})");
                    return Err(anyhow!("NaN in output"));
                }
            }
            Err(e) => {
                eprintln!("  FAIL iter {i}: {e} (seq_len={seq_len}, prev={prev_len})");
                return Err(e);
            }
        }
        prev_len = seq_len;
        if (i + 1) % 50 == 0 {
            eprintln!("  ... {}/200 OK", i + 1);
        }
    }
    eprintln!("  PASS");
    Ok(())
}

/// Test 2: Output tensor lifetime — compare natural drop vs mem::forget
fn test_output_tensor_lifetime(model: &mut OvModel) -> Result<()> {
    eprintln!("\n=== Test 2: Output Tensor Lifetime (50+50 iterations) ===");

    // Warmup
    let _ = run_inference(&mut model.infer_request, 64)?;

    // Phase A: with mem::forget (our run_inference already does this)
    eprintln!("  Phase A: mem::forget on output tensor...");
    for i in 0..50 {
        let result = run_inference(&mut model.infer_request, 64);
        if let Err(e) = result {
            eprintln!("  FAIL phase A iter {i}: {e}");
            return Err(e);
        }
    }
    eprintln!("  Phase A: 50/50 OK");

    // Phase B: natural drop (no mem::forget)
    eprintln!("  Phase B: natural drop on output tensor...");
    for i in 0..50 {
        let padded_len = bucket_size(64);
        let mut input_data: Vec<i64> = (1..=64i64).collect();
        input_data.resize(padded_len, 0);

        let ov_shape = Shape::new(&[1i64, padded_len as i64])
            .map_err(|e| anyhow!("shape: {e:?}"))?;
        let mut ov_input = openvino::Tensor::new(openvino::ElementType::I64, &ov_shape)
            .map_err(|e| anyhow!("tensor: {e:?}"))?;
        ov_input.get_data_mut::<i64>()
            .map_err(|e| anyhow!("data: {e:?}"))?
            .copy_from_slice(&input_data);

        model.infer_request.set_input_tensor(&ov_input)
            .map_err(|e| anyhow!("set input: {e:?}"))?;
        model.infer_request.infer()
            .map_err(|e| anyhow!("infer iter {i}: {e:?}"))?;

        let ov_output = model.infer_request.get_output_tensor()
            .map_err(|e| anyhow!("output iter {i}: {e:?}"))?;
        let data: Vec<f32> = ov_output.get_data::<f32>()
            .map_err(|e| anyhow!("output data: {e:?}"))?
            .to_vec();
        // Natural drop — ov_output drops here, calling ov_tensor_free

        let nan_count = data.iter().filter(|x| x.is_nan()).count();
        if nan_count > 0 {
            eprintln!("  FAIL phase B iter {i}: {nan_count} NaN values after natural drop");
            return Err(anyhow!("NaN after natural drop"));
        }
    }
    eprintln!("  Phase B: 50/50 OK");
    eprintln!("  PASS");
    Ok(())
}

/// Test 3: Core/CompiledModel lifetime
fn test_core_lifetime(assets: &PathBuf) -> Result<()> {
    eprintln!("\n=== Test 3: Core/CompiledModel Lifetime (100+100 iterations) ===");

    // Phase A: drop Core+CompiledModel immediately (old behavior)
    eprintln!("  Phase A: drop Core+CompiledModel after creating IR...");
    {
        let mut core = Core::new().map_err(|e| anyhow!("core: {e:?}"))?;
        let temp_dir = tempfile::tempdir()?;
        let xml_path = temp_dir.path().join("model.xml");
        {
            let compressed = std::fs::read(assets.join("xtr-ov-int4.xml.zst"))?;
            let xml_bytes = zstd::stream::decode_all(std::io::Cursor::new(compressed))?;
            std::fs::write(&xml_path, xml_bytes)?;
        }
        let bin_path = assets.join("xtr-ov-int4.bin");
        let model = core.read_model_from_file(
            xml_path.to_str().unwrap(),
            bin_path.to_str().unwrap(),
        ).map_err(|e| anyhow!("read model: {e:?}"))?;
        let mut compiled = core.compile_model(&model, DeviceType::CPU)
            .map_err(|e| anyhow!("compile: {e:?}"))?;
        let mut ir = compiled.create_infer_request()
            .map_err(|e| anyhow!("infer request: {e:?}"))?;

        // Drop Core and CompiledModel
        drop(compiled);
        drop(core);

        for i in 0..100 {
            let result = run_inference_with_ir(&mut ir, 64);
            match result {
                Ok(_) => {}
                Err(e) => {
                    eprintln!("  FAIL phase A iter {i}: {e}");
                    return Err(e);
                }
            }
            if (i + 1) % 25 == 0 {
                eprintln!("  ... {}/100 OK", i + 1);
            }
        }
        eprintln!("  Phase A: 100/100 OK");
    }

    // Phase B: keep Core+CompiledModel alive (new behavior)
    eprintln!("  Phase B: keep Core+CompiledModel alive...");
    {
        let mut model = load_model(assets)?;
        let _ = run_inference(&mut model.infer_request, 64)?; // warmup

        for i in 0..100 {
            let result = run_inference(&mut model.infer_request, 64);
            match result {
                Ok(_) => {}
                Err(e) => {
                    eprintln!("  FAIL phase B iter {i}: {e}");
                    return Err(e);
                }
            }
            if (i + 1) % 25 == 0 {
                eprintln!("  ... {}/100 OK", i + 1);
            }
        }
        eprintln!("  Phase B: 100/100 OK");
    }
    eprintln!("  PASS");
    Ok(())
}

fn run_inference_with_ir(ir: &mut InferRequest, seq_len: usize) -> Result<Vec<f32>> {
    let padded_len = bucket_size(seq_len);
    let mut input_data: Vec<i64> = (1..=seq_len as i64).collect();
    input_data.resize(padded_len, 0);

    let ov_shape = Shape::new(&[1i64, padded_len as i64])
        .map_err(|e| anyhow!("shape: {e:?}"))?;
    let mut ov_input = openvino::Tensor::new(openvino::ElementType::I64, &ov_shape)
        .map_err(|e| anyhow!("tensor: {e:?}"))?;
    ov_input.get_data_mut::<i64>()
        .map_err(|e| anyhow!("data: {e:?}"))?
        .copy_from_slice(&input_data);

    ir.set_input_tensor(&ov_input)
        .map_err(|e| anyhow!("set input: {e:?}"))?;
    ir.infer()
        .map_err(|e| anyhow!("infer: {e:?}"))?;

    let ov_output = ir.get_output_tensor()
        .map_err(|e| anyhow!("output: {e:?}"))?;
    let output_shape = ov_output.get_shape()
        .map_err(|e| anyhow!("output shape: {e:?}"))?;
    let dims: Vec<usize> = output_shape.get_dimensions().iter().map(|&d| d as usize).collect();
    let embed_dim = *dims.last().ok_or_else(|| anyhow!("empty dims"))?;

    let all_data: Vec<f32> = ov_output.get_data::<f32>()
        .map_err(|e| anyhow!("output data: {e:?}"))?
        .to_vec();
    std::mem::forget(ov_output);

    let unpadded: Vec<f32> = all_data[..seq_len * embed_dim].to_vec();
    Ok(unpadded)
}

/// Test 4: Long-running leak detection
fn test_leak_detection(model: &mut OvModel) -> Result<()> {
    eprintln!("\n=== Test 4: Leak Detection (1000 iterations) ===");
    let mut rng = rand::rng();

    // Warmup — run enough iterations to trigger all lazy allocations
    eprintln!("  Warming up...");
    for _ in 0..50 {
        let seq_len: usize = rng.random_range(16..=512);
        let _ = run_inference(&mut model.infer_request, seq_len)?;
    }

    let rss_before = get_rss_mb();
    eprintln!("  RSS after warmup: {rss_before:.1} MB");

    let t0 = Instant::now();
    for i in 0..1000 {
        let seq_len: usize = rng.random_range(16..=512);
        let result = run_inference(&mut model.infer_request, seq_len);
        if let Err(e) = result {
            eprintln!("  FAIL iter {i}: {e} (seq_len={seq_len})");
            return Err(e);
        }
        if (i + 1) % 100 == 0 {
            let rss = get_rss_mb();
            eprintln!("  ... {}/1000 OK (RSS: {rss:.1} MB, elapsed: {:.1?})", i + 1, t0.elapsed());
        }
    }

    let rss_after = get_rss_mb();
    let growth = rss_after - rss_before;
    eprintln!("  RSS after: {rss_after:.1} MB (growth: {growth:+.1} MB)");

    if growth > 50.0 {
        eprintln!("  FAIL: RSS grew by {growth:.1} MB (threshold: 50 MB)");
        return Err(anyhow!("memory leak detected"));
    }
    eprintln!("  PASS");
    Ok(())
}

fn get_rss_mb() -> f64 {
    let output = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok();
    output
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<f64>().ok())
        .map(|kb| kb / 1024.0)
        .unwrap_or(0.0)
}

/// Test if repeatedly dropping and recreating Core causes crashes
fn test_drop_recreate(assets: &PathBuf) -> Result<()> {
    eprintln!("\n=== Test 5: Drop/Recreate Core (20 cycles) ===");

    for i in 1..=20 {
        eprintln!("  Cycle {}: creating Core+Model...", i);
        let model = load_model(assets)?;
        eprintln!("  Cycle {}: dropping...", i);
        drop(model);
    }

    eprintln!("  PASS");
    Ok(())
}

fn main() -> Result<()> {
    let assets = PathBuf::from(std::env::args().nth(1).unwrap_or_else(|| "assets".into()));
    eprintln!("assets dir: {}", assets.display());

    if !assets.join("xtr-ov-int4.bin").exists() {
        return Err(anyhow!("missing xtr-ov-int4.bin in {}", assets.display()));
    }

    let only = std::env::args().nth(2);

    eprintln!("Loading model...");
    // Create model once — single Core per process
    let mut model = load_model(&assets)?;
    eprintln!("Model loaded successfully.");

    let mut passed = 0;
    let mut failed = 0;

    // Tests that use the shared model
    if only.as_deref() == Some("shape") || only.is_none() {
        match test_shape_variation(&mut model) {
            Ok(()) => passed += 1,
            Err(e) => {
                eprintln!("  FAILED: {e}");
                failed += 1;
            }
        }
    }

    if only.as_deref() == Some("output") || only.is_none() {
        match test_output_tensor_lifetime(&mut model) {
            Ok(()) => passed += 1,
            Err(e) => {
                eprintln!("  FAILED: {e}");
                failed += 1;
            }
        }
    }

    // Test 3 creates its own Core instances (that's what it tests)
    if only.as_deref() == Some("core") || only.is_none() {
        match test_core_lifetime(&assets) {
            Ok(()) => passed += 1,
            Err(e) => {
                eprintln!("  FAILED: {e}");
                failed += 1;
            }
        }
    }

    if only.as_deref() == Some("leak") || only.is_none() {
        match test_leak_detection(&mut model) {
            Ok(()) => passed += 1,
            Err(e) => {
                eprintln!("  FAILED: {e}");
                failed += 1;
            }
        }
    }

    // Drop the shared model before test 5
    drop(model);

    if only.as_deref() == Some("recreate") || only.is_none() {
        match test_drop_recreate(&assets) {
            Ok(()) => passed += 1,
            Err(e) => {
                eprintln!("  FAILED: {e}");
                failed += 1;
            }
        }
    }

    eprintln!("\n=== Results: {passed} passed, {failed} failed ===");
    if failed > 0 {
        std::process::exit(1);
    }
    Ok(())
}
