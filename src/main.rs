use rand::prelude::*;
use rusqlite::{Connection, Statement, Result as SQLResult};
use scan_dir::ScanDir;
use serde::Deserialize;
use sha2::{Sha256, Digest};
use std::fs::File;
use std::io::{BufReader, Read};
use std::mem::size_of;
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
    for i in 0..max_iter {
        println!("i {}", i);
        //let dist = cdist(data, &centers)?;
        let sim = data.matmul(&centers.transpose(D::Minus1, D::Minus2)?)?;
        cluster_assignments = sim.argmax(D::Minus1)?;
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
            let sum = cluster_data.sum(0)?;
            let normalized = sum.broadcast_div(&sum.sqr()?.sum_keepdim(0)?.sqrt()?);
            centers_vec.push(normalized?);
        }
        centers = Tensor::stack(centers_vec.as_slice(), 0)?;
    }
    Ok((centers, cluster_assignments))
}

fn match_centroids(embeddings: &Tensor, centers: &Tensor) -> Result<()> {
    println!("match em");
    let sim = embeddings.matmul(&centers.transpose(D::Minus1, D::Minus2)?).unwrap();
    println!("sim {}", sim);
    let cluster_assignments = sim.argmax(D::Minus1)?;
    println!("got centroids {}", cluster_assignments);
    Ok(())
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

struct Chunk {
    //hash: String,
    embedding: Vec<u8>,
}

struct Embedder {
    tokenizer: Tokenizer,
    model: t5::T5EncoderModel,
}

impl Embedder {
    fn new() -> Self {
        let (builder, tokenizer) = T5ModelBuilder::load().unwrap();
        let model = builder.build_encoder().unwrap();
        Self {
            tokenizer,
            model
        }
    }
    fn embed(self: &Self, text: &str) -> Result<Tensor> {
        let tokens = self.tokenizer
            .encode(text, true)
            .map_err(E::msg).unwrap()
            .get_ids()
            .to_vec();
        let token_ids = Tensor::new(&tokens[..], self.model.device()).unwrap().unsqueeze(0).unwrap();
        let embeddings = self.model.forward(&token_ids).unwrap();
        //println!("embedder took {} ms.", now.elapsed().as_millis());
        normalize_l2(&embeddings)
    }
}

pub struct Gatherer<'a> {
    documents: Box<dyn Iterator<Item = Document> + 'a>,
    embedder: &'a Embedder,
}

impl<'a> Gatherer<'a> {
    fn new(query: &'a mut Query, embedder: &'a Embedder) -> Self {
        let documents = Box::new(query.iter().unwrap().map(Result::unwrap));
        Self {
            documents,
            embedder,
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
                    let embeddings = self.embedder.embed(utf8);
                    println!("embedder took {} ms.", now.elapsed().as_millis());
                    let normalized = normalize_l2(&embeddings.unwrap());

                    let split = split_tensor(&normalized.ok()?.get(0).ok()?);
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

        let query = "CREATE TABLE IF NOT EXISTS chunk(hash TEXT PRIMARY KEY, embedding BLOB NOT NULL)";
        connection.execute(query, ()).unwrap();
        Self { connection }
    }

    fn make_query(self: &Self) -> SQLResult<Query> {
        let stmt = self.connection.prepare("SELECT filename, state, hash FROM document WHERE state = 'new'")?;
        Ok(Query { stmt })
    }

    fn make_kmeans_query(self: &Self) -> SQLResult<Query> {
        let stmt = self.connection.prepare("SELECT chunk.embedding FROM document,chunk
            WHERE document.hash == chunk.hash AND
            state = 'indexed'")?;
        Ok(Query { stmt })
    }

    fn add_doc(self: &Self, filename: &str, hash: &str) -> SQLResult<()> {
        self.connection.execute("INSERT OR IGNORE INTO document VALUES(?1, 'new', ?2)", (&filename, &hash))?;
        Ok(())
    }

    fn set_embeddings(self: &Self, hash: &str, embeddings: &Vec<u8>) -> SQLResult<()> {
        let tx = self.connection.unchecked_transaction()?;
        tx.execute("INSERT OR IGNORE INTO chunk VALUES(?1, ?2)", (&hash, embeddings)).unwrap();
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

    fn iter2(&mut self) -> SQLResult<impl Iterator<Item = SQLResult<Chunk>> + '_> {
        self.stmt.query_map([], |row| {
            Ok(Chunk {
                embedding: row.get(0)?,
            })
        })
    }
}

fn vecvec_f32_to_u8_vec(data: &[Vec<f32>]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * data[0].len() * std::mem::size_of::<f32>());
    for row in data {
        for &val in row {
            bytes.extend_from_slice(&val.to_ne_bytes()); // native-endian encoding
        }
    }
    bytes
}

pub fn u8_to_tensor2d_f32(bytes: &[u8], cols: usize) -> Tensor {
    let f32_size = size_of::<f32>();

    assert!(bytes.len() % f32_size == 0);
    let total_f32s = bytes.len() / f32_size;

    let rows = total_f32s / cols;

    let mut f32s = Vec::with_capacity(total_f32s);
    for chunk in bytes.chunks_exact(f32_size) {
        let arr: [u8; 4] = chunk.try_into().unwrap();
        f32s.push(f32::from_ne_bytes(arr));
    }

    Tensor::from_vec(f32s, (rows, cols), &Device::Cpu).unwrap()
}

fn main() -> Result<()> {

    let embedder = Embedder::new();
    let db = DB::new();
    let mut query = db.make_query()?;
    let mut kmeans_query = db.make_kmeans_query()?;

    register_documents(&db, "documents").unwrap();

    let embedding_iter = Gatherer::new(&mut query, &embedder);
    for (hash, embeddings) in embedding_iter {
        //println!("for hash {} {:?}", hash, embeddings.dims2().unwrap());
        //let (b, n) = embeddings.dims2().unwrap();
        let vec = embeddings.to_vec2::<f32>();
        let bytes = vecvec_f32_to_u8_vec(&vec?);
        db.set_embeddings(&hash, &bytes).unwrap();
    }

    let mut all_embeddings = vec![];
    for chunk in kmeans_query.iter2()? {
        let t = u8_to_tensor2d_f32(&chunk?.embedding, 128);
        let split = split_tensor(&t);
        all_embeddings.extend(split);
        //println!("iter len {}", chunk?.embedding.len());
    }
    let device = Device::Cpu;
    let matrix = stack_tensors(all_embeddings);
    let now = std::time::Instant::now();
    let (centers, idxs) = kmeans(&matrix, 1024, 5, &device)?;
    let elapsed_time = now.elapsed();
    println!("kmeans took {} ms.", elapsed_time.as_millis());
    println!("idxs {}", idxs);


    let qe = embedder.embed("do children benefit from breast-feeding?")?.get(0)?;
    let _ = match_centroids(&qe, &centers).unwrap();


    //println!("idxs {}", idxs);
    //println!("v {:?}", all_embeddings);

    //let matrix = gather_embeddings(&mut query, model, tokenizer)?;
    //let (_, idxs) = kmeans(&matrix, 16, 5, &device)?;

    Ok(())
}

pub fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(2)?.sqrt()?)?)
}
