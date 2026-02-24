use super::t5_encoder;
use anyhow::Result;
use candle_core::{Device, Tensor};
use log::debug;
use tokenizers::Tokenizer;

fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(2)?.sqrt()?)?)
}
pub struct Embedder {
    tokenizer: Tokenizer,
    model: t5_encoder::T5EncoderModel,
}

impl Embedder {
    pub fn new(device: &Device, assets: &std::path::PathBuf) -> Result<Self> {
        let (builder, tokenizer) = t5_encoder::T5ModelBuilder::load(assets)?;
        let model = builder.build_encoder(&device, assets)?;
        Ok(Self { tokenizer, model })
    }

    pub fn embed(self: &Self, text: &str) -> Result<(Tensor, Vec<(usize, usize)>)> {
        let now = std::time::Instant::now();
        let model = &self.model;
        let device = model.device();

        let max_len: usize = 2048;
        let stride: usize = 256;

        let encoding = self.tokenizer.encode(text, true).unwrap();
        let ids = encoding.get_ids();
        let offsets = encoding.get_offsets().to_vec();

        let n_tokens = ids.len();
        let mut accum: Vec<Option<Tensor>> = vec![None; n_tokens];

        let mut start = 0;
        loop {
            let end = (start + max_len).min(n_tokens);
            let input = Tensor::new(&ids[start..end], &device)?.unsqueeze(0)?;
            let chunk = model.forward(&input)?.squeeze(0)?.to_device(&Device::Cpu)?;

            let (m, _n) = chunk.dims2()?;
            for i in 0..m {
                let global_idx = start + i;
                if global_idx >= n_tokens {
                    break;
                }
                let emb = chunk.get(i)?;
                match &accum[global_idx] {
                    None => accum[global_idx] = Some(emb.clone()),
                    Some(prev) => {
                        let sum = (prev + &emb)?;
                        accum[global_idx] = Some(sum);
                    }
                }
            }

            if end == n_tokens {
                break;
            }
            start = end.saturating_sub(stride); // overlap window
        }

        let token_embs: Vec<Tensor> = accum
            .into_iter()
            .enumerate()
            .map(|(i, maybe_t)| {
                maybe_t.unwrap_or_else(|| {
                    panic!("Missing embedding for token {} — check stride settings", i)
                })
            })
            .collect();

        let matrix = Tensor::stack(&token_embs, 0)?.unsqueeze(0)?;
        let normalized = normalize_l2(&matrix)?;
        debug!("embedder took {} ms.", now.elapsed().as_millis());
        Ok((normalized, offsets))
    }

    /*
    pub fn embed(self: &Self, text: &str) -> Result<(Tensor, Vec<(usize, usize)>)> {
        let now = std::time::Instant::now();
        let enc = self.tokenizer.encode(text, true).map_err(E::msg).unwrap();
        let offsets = enc.get_offsets().to_vec();
        let tokens = enc.get_ids().to_vec();
        let token_ids = Tensor::new(&tokens[..], self.model.device())
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let embeddings = self.model.forward(&token_ids).unwrap();
        debug!("embedder took {} ms.", now.elapsed().as_millis());
        let normalized = normalize_l2(&embeddings)?;
        Ok((normalized, offsets))
    }
    */
}
