use rand::prelude::*;
use rusqlite::{Connection, Statement, Result as SQLResult};
use scan_dir::ScanDir;
use serde::Deserialize;
use sha2::{Sha256, Digest};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use std::path::PathBuf;

//use candle_transformers::models::t5;
mod t5;

use anyhow::{Error as E, Result};
use candle_core::{DType, Device, Tensor, D, IndexOp};
use candle_nn::VarBuilder;
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
        let path = PathBuf::from(r"/Users/jhansen/src/xtr-warp/foo.safetensors");
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

#[derive(Debug, Deserialize)]
struct Record {
    _line: u32,
    text: String,
}

fn cdist(x1: &Tensor, x2: &Tensor) -> Result<Tensor> {
    let x1 = x1.unsqueeze(0)?;
    let x2 = x2.unsqueeze(1)?;
    Ok(x1
        .broadcast_sub(&x2)?
        .sqr()?
        .sum(D::Minus1)?
        .sqrt()?
        .transpose(D::Minus1, D::Minus2)?)
}

fn kmeans(data: &Tensor, k: usize, max_iter: u32, device: &Device) -> Result<(Tensor, Tensor)> {
    println!("kmeans {}", data);
    let (n, _) = data.dims2()?;
    println!("kmeans {}", n);
    let mut rng = rand::rng();
    let mut indices = (0..n).collect::<Vec<_>>();
    indices.shuffle(&mut rng);

    let centroid_idx = indices[..k]
        .iter()
        .copied()
        .map(|x| x as u32)
        .collect::<Vec<_>>();

    let centroid_idx_tensor = Tensor::from_slice(centroid_idx.as_slice(), (k,), device)?;
    let mut centers = data.index_select(&centroid_idx_tensor, 0)?;
    let mut cluster_assignments = Tensor::zeros((n,), DType::U32, device)?;
    for _ in 0..max_iter {
        let dist = cdist(data, &centers)?;
        cluster_assignments = dist.argmin(D::Minus1)?;
        let mut centers_vec = vec![];
        for i in 0..k {
            let mut indices = vec![];
            cluster_assignments
                .to_vec1::<u32>()?
                .iter()
                .enumerate()
                .for_each(|(j, x)| {
                    if *x == i as u32 {
                        indices.push(j as u32);
                    }
                });
            let indices = Tensor::from_slice(indices.as_slice(), (indices.len(),), device)?;
            let cluster_data = data.index_select(&indices, 0)?;
            let mean = cluster_data.mean(0)?;
            centers_vec.push(mean);
        }
        centers = Tensor::stack(centers_vec.as_slice(), 0)?;
    }
    Ok((centers, cluster_assignments))
}


fn split_tensor(tensor: &Tensor) -> Vec<Tensor> {
    let dims = tensor.dims();
    let num_rows = dims[0];

    // Collect each row as a separate Tensor of shape [128]
    (0..num_rows)

        .map(|i| {
            let row_tensor = tensor.i(i).unwrap();
            row_tensor.unsqueeze(0).unwrap()
        })

        .collect()
}

fn stack_tensors(vectors: Vec<Tensor>) -> Tensor {
    Tensor::cat(&vectors, 0).unwrap() // `0` means stacking along rows (axis 0)
}

fn compute_sha256<P: AsRef<Path>>(path: P) -> Result<String, Box<dyn std::error::Error>> {
    // Open the file
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();

    // Read file in chunks
    let mut buffer = [0u8; 0x10000];
    loop {
        let bytes_read = reader.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }

    // Finalize and return the hash as a hex string
    let result = hasher.finalize();
    Ok(format!("{:x}", result))
}


fn register_documents(db: &DB, dirname: &str) -> Result<()> {
    ScanDir::files().read(dirname, |iter| {
        for (entry, _name) in iter {
            let path = entry.path();
            let filename = path.to_str().unwrap();
            let hash = compute_sha256(path.clone()).unwrap();
            println!("{} hash is {}", filename, hash);
            db.add_doc(&filename, &hash).unwrap();
        }
    }).unwrap();

    Ok(())
}

struct Document {
    filename: String,
    state: String,
    hash: String,
}

pub struct Gatherer<'a> {
    documents: Box<dyn Iterator<Item = Document> + 'a>,
    tokenizer: Tokenizer,
    model: t5::T5EncoderModel,
}

impl<'a> Gatherer<'a> {
    fn new(query: &'a mut Query, model: t5::T5EncoderModel, tokenizer: Tokenizer) -> Self {
        let documents = Box::new(query.iter().unwrap().map(Result::unwrap));
        Self {
            documents,
            tokenizer,
            model,
        }
    }
}

impl<'a> Iterator for Gatherer<'a> {
    type Item = (String, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        let mut doc_embedding = vec![];
        match self.documents.next() {
            Some(document) => {
                let filename = document.filename;
                let hash = document.hash;

                let path = PathBuf::from(filename);
                println!("read file {:?}", path);
                let file = File::open(path).unwrap();
                let mut reader = BufReader::new(file);
                let mut buffer = [0u8; 0x10000];

                loop {
                    let bytes_read = reader.read(&mut buffer).unwrap();
                    if bytes_read == 0 {
                        break;
                    }

                    let now = std::time::Instant::now();
                    let utf8 = std::str::from_utf8(&buffer[..bytes_read]).unwrap();
                    let tokens = self.tokenizer
                        .encode(utf8, true)
                        .map_err(E::msg).unwrap()
                        .get_ids()
                        .to_vec();
                    let token_ids = Tensor::new(&tokens[..], self.model.device()).unwrap().unsqueeze(0).unwrap();
                    let embeddings = self.model.forward(&token_ids).unwrap();
                    let elapsed_time = now.elapsed();
                    println!("embedder took {} ms.", elapsed_time.as_millis());
                    let normalized = normalize_l2(&embeddings).unwrap().get(0).unwrap();

                    let split = split_tensor(&normalized);
                    doc_embedding.extend(split);
                }
                Some((hash, stack_tensors(doc_embedding)))
            }
            None => {
                None
            }
        }
    }
}

struct DB {
    connection: Connection,
}

struct Query<'connection> {
    stmt: Statement<'connection>,
}

impl DB {
    pub fn new() -> Self {
        //let connection = Connection::open_in_memory().unwrap();
        let connection = Connection::open("mydb.sqlite").unwrap();

        //unsafe {
            //let _guard = LoadExtensionGuard::new(&self.connection)?;
            //self.connection.load_extension("trusted/sqlite/extension", None)?
        //}

        println!("init");
        let query = "CREATE TABLE IF NOT EXISTS document(filename TEXT PRIMARY KEY,
            state TEXT CHECK(state IN ('new', 'indexed')),
            hash TEXT)";
        connection.execute(query, ()).unwrap();

        let query = "CREATE TABLE IF NOT EXISTS chunk(hash TEXT PRIMARY KEY, embedding TEXT)";
        connection.execute(query, ()).unwrap();
        Self { connection }
    }

    fn make_query(self: &Self) -> SQLResult<Query> {
        let stmt = self.connection.prepare("SELECT filename, state, hash FROM document WHERE state = 'new'")?;
        Ok(Query { stmt })
    }

    fn make_kmeans_query(self: &Self) -> SQLResult<Query> {
        let stmt = self.connection.prepare("SELECT document.filename,document.state,document.hash FROM document,chunk
            WHERE document.hash == chunk.hash AND
            state = 'indexed'")?;
        Ok(Query { stmt })
    }

    fn add_doc(self: &Self, filename: &str, hash: &str) -> SQLResult<()> {
        self.connection.execute("INSERT OR IGNORE INTO document VALUES(?1, 'new', ?2)", (&filename, &hash))?;
        Ok(())
    }

    fn set_embeddings(self: &Self, hash: &str, embeddings: &Tensor) -> SQLResult<()> {
        let es = format!("{}", embeddings);
        let tx = self.connection.unchecked_transaction()?;
        tx.execute("INSERT OR IGNORE INTO chunk VALUES(?1, ?2)", (&hash, &es)).unwrap();
        tx.execute("UPDATE document set state = 'indexed' WHERE hash == ?1", (&hash, )).unwrap();
        tx.commit()
    }
}

impl<'connection> Query<'connection> {
    fn iter(&mut self) -> SQLResult<impl Iterator<Item = SQLResult<Document>> + '_> {
        self.stmt.query_map([], |row| {
            Ok(Document {
                filename: row.get(0)?,
                state: row.get(1)?,
                hash: row.get(2)?,
            })
        })
    }
}

fn main() -> Result<()> {
    let (builder, tokenizer) = T5ModelBuilder::load()?;
    let model = builder.build_encoder()?;

    let db = DB::new();
    let mut query = db.make_query()?;
    let mut kmeans_query = db.make_kmeans_query()?;

    register_documents(&db, "documents");

    let embedding_iter = Gatherer::new(&mut query, model, tokenizer);
    for (hash, embeddings) in embedding_iter {
        db.set_embeddings(&hash, &embeddings);
    }

    for (document) in kmeans_query.iter()? {
        println!("iter hash {}", document?.hash);
    }

    //let matrix = gather_embeddings(&mut query, model, tokenizer)?;
    //let device = Device::Cpu;
    //let (_, idxs) = kmeans(&matrix, 16, 5, &device)?;
    //println!("idxs {}", idxs);

    Ok(())
}

pub fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(2)?.sqrt()?)?)
}
