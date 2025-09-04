#!/usr/bin/env node
/* eslint-disable @typescript-eslint/no-var-requires */
const path = require('path');
const util = require('util');
const exec = util.promisify(require('child_process').exec);

async function run(command) {
  console.log(`running: ${command} ...`);
  const { stdout, stderr } = await exec(command);
  console.log(stdout);
  console.error(stderr);
}

async function build(platform, arch) {
  if (platform == 'darwin') {
    await run(`cargo build --locked --release --target aarch64-apple-darwin --features accelerate`);
    await run(`cargo build --locked --release --target x86_64-apple-darwin --features accelerate`);
	await run(`lipo -create target/aarch64-apple-darwin/release/libwarp.dylib target/x86_64-apple-darwin/release/libwarp.dylib -output warp.node`);
  } else if (platform == 'win32') {
    var target = "";
    if (arch == "x64") {
      target = "x86_64";
    } else if (arch == "ia32") {
	  target = "i686";
    } else if (arch == "arm64") {
	  target = "aarch64";
    } else {
      console.error(`unsupported warp target ${platform} ${arch}`);
      return;
    }
	await run(`cargo build --locked --release --target ${target}-pc-windows-msvc`);
	await run(`cp target/${target}-pc-windows-msvc/release/warp.dll warp.node`);
  }
}

const platform = process.env.npm_config_platform || process.platform;
const arch = process.env.npm_config_arch || process.arch;

if (process.env.ENABLE_WARP == '1') {
  console.log(`building Warp module for ${platform} ${arch}`);
  build(platform, arch);
} else {
  console.log(`Not building Warp, set ENABLE_WARP=1 in environment to enable`);
}
