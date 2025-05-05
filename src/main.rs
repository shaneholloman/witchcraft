use std::path::PathBuf;

//use candle_transformers::models::t5;
mod t5;

use anyhow::{Error as E, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
//use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

const DTYPE: DType = DType::F32;

struct T5ModelBuilder {
    device: Device,
    config: t5::Config,
    weights_filename: Vec<PathBuf>,
}

impl T5ModelBuilder {
    pub fn load() -> Result<(Self, Tokenizer)> {
        let device = Device::Cpu;
        //let (default_model, default_revision) = ("t5-base", "main");
        //let default_model = default_model.to_string();
        //let default_revision = default_revision.to_string();
        //let (model_id, revision) = (default_model, default_revision);
        //let repo = Repo::with_revision(model_id.clone(), RepoType::Model, revision);
        //let api = Api::new()?;
        //let repo = api.repo(repo);
        let path = PathBuf::from(r"/Users/jhansen/src/xtr-warp/xtr-base-en/model.safetensors");
        let weights_filename = vec![path];
        let config = std::fs::read_to_string("/Users/jhansen/src/xtr-warp/xtr-base-en/config.json")?;
        let config: t5::Config = serde_json::from_str(&config)?;
        let tokenizer = Tokenizer::from_file("/Users/jhansen/src/xtr-warp/xtr-base-en/tokenizer.json").map_err(E::msg)?;
        Ok((
            Self {
                device,
                config,
                weights_filename,
            },
            tokenizer,
        ))
    }

    pub fn build_encoder(&self) -> Result<t5::T5EncoderModel> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&self.weights_filename, DTYPE, &self.device)?
        };
        Ok(t5::T5EncoderModel::load(vb, &self.config)?)
    }

}

fn main() -> Result<()> {

    let (builder, tokenizer) = T5ModelBuilder::load()?;
    let model = builder.build_encoder()?;

    loop {
        let now = std::time::Instant::now();
        let tokens = tokenizer
            .encode("do buildings change size due to weather?", true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        let token_ids = Tensor::new(&tokens[..], model.device())?.unsqueeze(0)?;
        println!("ids {}", token_ids);
        let embeddings = model.forward(&token_ids)?;
        let elapsed_time = now.elapsed();
        println!("Running slow_function() took {} ms.", elapsed_time.as_millis());
        println!("embeddings {}", embeddings);
        //println!("embeddings {}", normalize_l2(&embeddings).unwrap());
    }


    Ok(())
}

pub fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
}
