
use candle_core::{Device, Tensor};
use anyhow::{Result};

pub trait TensorPackOps {
    fn from_q4_bytes(buffer: &[u8], cols: usize, device: &Device) -> Result<Tensor>;
    fn from_f32_bytes(bytes: &[u8], cols: usize, device: &Device) -> Result<Tensor>;
    fn to_q4_bytes(&self) -> Result<Vec<u8>>;
    fn to_f32_bytes(&self) -> Result<Vec<u8>>;
    fn to_u32_bytes(&self) -> Result<Vec<u8>>;
}

impl TensorPackOps for Tensor {
    fn from_q4_bytes(bytes: &[u8], cols: usize, device: &Device) -> Result<Tensor> {
        let mut out = Vec::with_capacity(bytes.len() * 2);
        for &byte in bytes {
            let high = (byte >> 4) & 0x0f;
            let low = byte & 0x0f;
            out.push(high as f32);
            out.push(low as f32);
        }

        assert!(
            out.len() % cols == 0,
            "Unpacked data length ({}) must be divisible by cols ({})",
            out.len(),
            cols
        );
        let rows = out.len() / cols;
        Ok(Tensor::from_vec(out, &[rows, cols], device)?)
    }

    fn from_f32_bytes(bytes: &[u8], cols: usize, device: &Device) -> Result<Tensor> {
        let f32_size = size_of::<f32>();

        assert!(bytes.len() % f32_size == 0);
        let total_f32s = bytes.len() / f32_size;

        let rows = total_f32s / cols;

        let mut f32s = Vec::with_capacity(total_f32s);
        for chunk in bytes.chunks_exact(f32_size) {
            let arr: [u8; 4] = chunk.try_into().unwrap();
            f32s.push(f32::from_ne_bytes(arr));
        }

        Ok(Tensor::from_vec(f32s, &[rows, cols], &device)?)
    }

    fn to_q4_bytes(&self) -> Result<Vec<u8>> {
        let data = self.to_vec2::<f32>()?;
        let flat: Vec<f32> = data.into_iter().flatten().collect();

        assert!(
            flat.len() % 2 == 0,
            "Tensor must have an even number of elements to pack"
        );

        let mut packed = Vec::with_capacity(flat.len() / 2);
        for chunk in flat.chunks(2) {
            let high = chunk[0] as u8 & 0x0f;
            let low = chunk[1] as u8 & 0x0f;
            packed.push((high << 4) | low);
        }
        Ok(packed)
    }

    fn to_f32_bytes(&self) -> Result<Vec<u8>> {
        let floats: Vec<f32> = self.to_vec1::<f32>()?;
        let mut bytes = Vec::with_capacity(floats.len() * 4);

        for f in floats {
            bytes.extend_from_slice(&f.to_ne_bytes());
        }
        Ok(bytes)
    }

    fn to_u32_bytes(&self) -> Result<Vec<u8>> {
        let ulongs: Vec<u32> = self.to_vec1::<u32>()?;
        let mut bytes = Vec::with_capacity(ulongs.len() * 4);

        for u in ulongs {
            bytes.extend_from_slice(&u.to_ne_bytes());
        }
        Ok(bytes)
    }
}
