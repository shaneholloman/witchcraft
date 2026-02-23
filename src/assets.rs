//! assets.rs — generic Zstd asset loader with optional compile-time embedding.

use once_cell::sync::OnceCell;
use std::{
    io::Cursor,
    path::{Path, PathBuf},
};

#[cfg(not(feature = "embed-assets"))]
use std::fs;

/// A Zstandard-compressed asset that is decompressed at most once and
/// cached for the program’s lifetime.
pub struct Asset {
    #[cfg(feature = "embed-assets")]
    compressed: &'static [u8],

    #[cfg(not(feature = "embed-assets"))]
    path: &'static str,

    decompressed: OnceCell<Result<Vec<u8>, ()>>,
}

impl Asset {
    #[cfg(feature = "embed-assets")]
    pub const fn new_embedded(compressed: &'static [u8]) -> Self {
        Self {
            compressed,
            decompressed: OnceCell::new(),
        }
    }

    #[cfg(not(feature = "embed-assets"))]
    pub const fn new_file(path: &'static str) -> Self {
        Self {
            path,
            decompressed: OnceCell::new(),
        }
    }

    /// Decompress directly into a file without buffering the entire payload in RAM.
    pub fn decompress_to_file(&'static self, _assets: &PathBuf, dest: &Path) -> Result<(), ()> {
        let mut out = std::fs::File::create(dest)
            .map_err(|e| log::warn!("failed to create {}: {}", dest.display(), e))?;

        #[cfg(feature = "embed-assets")]
        zstd::stream::copy_decode(&mut Cursor::new(self.compressed), &mut out)
            .map_err(|e| log::warn!("failed to decompress asset: {}", e))?;

        #[cfg(not(feature = "embed-assets"))]
        {
            let src = _assets.join(self.path);
            let mut reader = std::fs::File::open(&src)
                .map_err(|e| log::warn!("failed to open {}: {}", src.display(), e))?;
            zstd::stream::copy_decode(&mut reader, &mut out)
                .map_err(|e| log::warn!("failed to decompress asset: {}", e))?;
        }

        Ok(())
    }

    /// Return the decompressed bytes; performs work only on first call.
    pub fn bytes(&'static self, _assets: &PathBuf) -> Result<&'static [u8], ()> {
        self.decompressed
            .get_or_init(|| {
                #[cfg(feature = "embed-assets")]
                let compressed: Vec<u8> = self.compressed.to_vec();

                #[cfg(not(feature = "embed-assets"))]
                let compressed: Vec<u8> = match fs::read(Path::new(&_assets.join(self.path))) {
                    Ok(c) => c,
                    Err(e) => {
                        log::warn!("failed to read warp asset {}: {}", self.path, e);
                        return Err(());
                    }
                };

                let mut cursor = Cursor::new(compressed);
                match zstd::stream::decode_all(&mut cursor) {
                    Ok(d) => Ok(d),
                    Err(e) => {
                        log::warn!("failed to decompress warp asset {}", e);
                        Err(())
                    }
                }
            })
            .as_deref()
            .map_err(|_| ())
    }
}

#[macro_export]
macro_rules! embed_zst_asset {
    ($vis:vis $name:ident, $path:literal) => {
        #[cfg(feature = "embed-assets")]
        $vis static $name: $crate::assets::Asset =
            $crate::assets::Asset::new_embedded(include_bytes!(
                concat!(env!("CARGO_MANIFEST_DIR"), "/assets/", $path)
            ));

        #[cfg(not(feature = "embed-assets"))]
        $vis static $name: $crate::assets::Asset =
            $crate::assets::Asset::new_file($path);
    };
}
