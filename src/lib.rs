use anyhow::Result;
use candle_core::Device;
use napi::bindgen_prelude::*;
use napi_derive::napi;
use once_cell::sync::Lazy;
use std::sync::Mutex;
//use std::thread;
use std::time::Duration;

use std::{
    sync::{LazyLock, mpsc},
    thread::{self, JoinHandle},
};

#[derive(Debug)]
pub struct Indexer {
    tx: mpsc::Sender<Job>,
    _handle: JoinHandle<()>,
}

static INDEXER: LazyLock<Indexer> = LazyLock::new(|| {
    Indexer::new()
});

type Job = String;

impl Indexer {
    pub fn new() -> Self {
        let (tx, rx) = mpsc::channel::<Job>();
        let handle = thread::spawn(move || {
            let db = warp::DB::new("mydb.sqlite");
            let device = Device::new_metal(0).unwrap();
            while let Ok(job) = rx.recv() {
                println!("got job {}", job);
                if job == "index" {
                    let count = warp::count_unindexed_chunks(&db).unwrap();
                    println!("count {}", count);
                    if count >= 2048 {
                        warp::index_chunks(&db, &device).unwrap();
                    }
                } else if job == "embed" { 
                    warp::embed_chunks(&db, &device).unwrap();
                }
            }
        });
        Indexer { tx, _handle: handle }
    }
    pub fn submit(&self, job: Job) {
        let _ = self.tx.send(job);
    }
}

mod warp;

#[napi(js_name = "Warp")]
pub struct Warp {
    db: warp::DB,
    device: Device,
    embedder: warp::Embedder,
}

#[napi]
impl Warp {
    #[napi(constructor)]
    pub fn new() -> Self {
        let db = warp::DB::new("mydb.sqlite");
        let device = Device::new_metal(0).unwrap();
        let embedder = warp::Embedder::new(&device);

        Self {
            db: db,
            device: device,
            embedder: embedder,
        }
    }
    #[napi]
    pub fn search(&self, q: String) -> Vec<(String, String)> {
        warp::search(&self.db, &self.embedder, &q, true).unwrap()
    }

    #[napi]
    pub fn add(&self, metadata: String, body: String) {
        println!("add {}", body);
        warp::add_doc_from_string(&self.db, &metadata, &body).unwrap();
    }

    #[napi]
    pub fn embed(&self, metadata: String, body: String) {
        INDEXER.submit("embed".to_string());
    }

    #[napi]
    pub fn index(&self, metadata: String, body: String) {
        INDEXER.submit("index".to_string());
    }
}

