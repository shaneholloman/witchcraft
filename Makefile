env/pyvenv.cfg:
	uv venv env

env/bin/transformers: env/pyvenv.cfg
	(source env/*/activate; uv pip install -r requirements.txt)

assets:
	mkdir -p assets

assets/config.json assets/tokenizer.json xtr.safetensors: assets env/bin/transformers
	(source env/*/activate; python downloadweights.py)

assets/xtr.gguf:  assets xtr.safetensors
	cargo run -p quantize-tool xtr.safetensors assets/xtr.gguf

assets/xtr-ov-int4.bin assets/xtr-ov-int4.xml:
	python quantize-int4.py

download: assets assets/config.json assets/tokenizer.json assets/xtr.gguf assets/xtr-ov-int4.bin assets/xtr-ov-int4.xml

build: download
	RUSTFLAGS='-C target-feature=+neon' cargo build --release --target aarch64-apple-darwin --features t5-quantized,metal,accelerate,napi
	ln -vf target/aarch64-apple-darwin/release/libwarp.dylib target/release/warp.node

module:
	cargo build --release --target aarch64-apple-darwin --features t5-quantized,metal,accelerate
	cargo build --release --target x86_64-apple-darwin --features t5-quantized,accelerate
	lipo -create target/aarch64-apple-darwin/release/libwarp.dylib target/x86_64-apple-darwin/release/libwarp.dylib -output target/release/warp-macos-universal.node

winmodule:
	cargo xwin build --release --target x86_64-pc-windows-msvc
	ln -vf target/x86_64-pc-windows-msvc/release/warp.dll target/release/warp-windows.node

win: download
	RUSTFLAGS='-C target-feature=+avx2' cargo xwin build --release --target x86_64-pc-windows-msvc --features t5-quantized,napi

macintel: download
	RUSTFLAGS='-C target-cpu=haswell' cargo build --release --target x86_64-apple-darwin --features t5-quantized,accelerate,hybrid-dequant

macintelasan: download
	rustup override set nightly
	RUSTFLAGS="-Z sanitizer=address -C target-feature=+avx2,+fma" cargo build -Z build-std --release --target x86_64-apple-darwin --features t5-openvino,accelerate

run: build
	node index.js

mcp: download
	cargo build --release --target aarch64-apple-darwin --features t5-quantized,metal,accelerate,napi
	ln -vf target/aarch64-apple-darwin/release/libwarp.dylib target/release/warp.node
	yarn napi build --release --features metal
	yarn tsc
	cmcp "node dist/index.js" tools/call name=search 'arguments:={"q": "teenagers and acne" }'

test: download
	RUST_LOG=debug cargo llvm-cov nextest --release --features napi,t5-quantized,metal,accelerate --lcov --output-path lcov.info # --no-capture
	genhtml lcov.info

nfcorpus: download
	cargo run --release --features metal,t5-quantized,progress  --bin warp-cli readcsv datasets/nfcorpus.tsv
	cargo run --release --features metal,t5-quantized,progress --bin warp-cli embed
	cargo run --release --features metal,t5-quantized,progress --bin warp-cli index
