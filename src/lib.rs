use napi_derive::napi;
use napi::bindgen_prelude::*;
use napi::{Env, ScopedTask};

use std::{
    sync::{mpsc, Arc, Mutex, OnceLock},
    thread::{self, JoinHandle},
};

type Job = (String, String, String);

#[derive(Debug)]
pub struct Indexer {
    tx: mpsc::Sender<Job>,
    _handle: JoinHandle<()>,
}

static INDEXER: OnceLock<Indexer> = OnceLock::new();

impl Indexer {
    pub fn new(db_name: String) -> Self {
        let (tx, rx) = mpsc::channel::<Job>();
        let db = warp::DB::new(&db_name);
        let handle = thread::spawn(move || {
            let device = warp::make_device();
            while let Ok(job) = rx.recv() {
                let (command, arg1, arg2) = job;
                if command == "add" {
                    warp::add_doc_from_string(&db, &arg1, &arg2).unwrap();
                } else if command == "index" {
                    let count = warp::count_unindexed_chunks(&db).unwrap();
                    println!("count {}", count);
                    if count >= 2048 {
                        warp::index_chunks(&db, &device).unwrap();
                    }
                } else if command == "embed" {
                    warp::embed_chunks(&db, &device).unwrap();
                }
            }
        });
        Indexer {
            tx,
            _handle: handle,
        }
    }
    pub fn init_global(db_name: String) -> &'static Self {
        INDEXER.get_or_init(|| Indexer::new(db_name))
    }

    pub fn global() -> &'static Self {
        INDEXER.get().expect("Indexer not initialized. Call `Indexer::init_global()` first.")
    }
}

mod warp;

struct WarpInner {
    embedder: warp::Embedder,
    db: warp::DB,
    cache: warp::EmbeddingsCache,
}

impl WarpInner {

    pub fn new(db_name: String) -> Self {
        let _indexer = Indexer::init_global(db_name.clone());
        let db = warp::DB::new_reader(&db_name.clone());
        let device = warp::make_device();
        let embedder = warp::Embedder::new(&device);
        let cache = warp::EmbeddingsCache::new(16);

        Self {
            embedder: embedder,
            db: db,
            cache: cache,
        }
    }

    pub fn search(&mut self, q: &String, threshold: f32, top_k: usize, sql_filter: &String) -> Vec<(f32, String, String)> {
        let filter = if !sql_filter.is_empty() {
            Some(sql_filter.as_str())
        } else {
            None
        };
        match warp::search(&self.db, &self.embedder, &mut self.cache, &q, threshold, top_k, true, filter) {
            Ok(v) => v,
            Err(e) => {
                println!("error {} querying for {}", e, &q);
                [].to_vec()
            }
        }
    }

    pub fn score(&mut self, q: &String, sentences: &Vec<String>) -> Vec<f32> {
        if sentences.len() != 0 {
            match warp::score_query_sentences(&self.embedder, &mut self.cache, &q, &sentences) {
                Ok(v) => v,
                Err(e) => {
                    println!("error {} scoring", e);
                    [].to_vec()
                }
            }
        } else {
            [].to_vec()
        }
    }
}

static INNER: OnceLock<Arc<Mutex<WarpInner>>> = OnceLock::new();
fn inner(db_name: String) -> Arc<Mutex<WarpInner>> {
    INNER.get_or_init(|| Arc::new(Mutex::new(WarpInner::new(db_name)))).clone()
}

pub struct SearchTask {
    inner: Arc<Mutex<WarpInner>>,
    q: String,
    threshold: f32,
    top_k: usize,
    sql_filter: String,
}

impl<'env> ScopedTask<'env> for SearchTask {
    type Output = Vec<(f32, String, String)>;
    type JsValue = Object<'env>;

    fn compute(&mut self) -> Result<Self::Output> {
        Ok(self.inner.lock().unwrap().search(&self.q, self.threshold, self.top_k, &self.sql_filter))
    }

    fn resolve(&mut self, env: &'env Env, out: Self::Output) -> Result<Self::JsValue> {
        // Outer array
        let mut outer: Array<'env> = env.create_array(out.len() as u32)?;
        for (i, (score, a, b)) in out.into_iter().enumerate() {
            // Triplet [number, string, string]
            let mut triplet: Array<'env> = env.create_array(3)?;
            triplet.set(0, score as f64)?; // numbers as f64
            triplet.set(1, a)?;            // String -> JS string
            triplet.set(2, b)?;
            outer.set(i as u32, triplet)?; // set the triplet into the outer array
        }
        // Return Object<'env> (Array is an Object; coerce to Object)
        outer.coerce_to_object()
    }

}

pub struct ScoreTask {
    inner: Arc<Mutex<WarpInner>>,
    q: String,
    sentences: Vec<String>,
}

impl<'env> ScopedTask<'env> for ScoreTask {
    type Output = Vec<f32>;
    type JsValue = Object<'env>;

    fn compute(&mut self) -> Result<Self::Output> {
        Ok(self.inner.lock().unwrap().score(&self.q, &self.sentences))
    }

    fn resolve(&mut self, env: &'env Env, output: Self::Output) -> Result<Self::JsValue> {
        let mut rows: Vec<f32> = Vec::with_capacity(output.len());
        for score in output {
            rows.push(score);
        }
        let arr = Array::from_vec(env, rows)?;
        // If you need to return Object<'env>, coerce the Array to Object
        arr.coerce_to_object()
    }
}

#[napi(js_name = "Warp")]
pub struct Warp {
    db_name: String,
}

#[napi]
impl Warp {
    #[napi(constructor)]
    pub fn new(db_name: String) -> Self {
        let _indexer = Indexer::init_global(db_name.clone());
        Self {
            db_name: db_name,
        }
    }
    #[napi]
    pub fn search(&mut self, q: String, threshold: f64, top_k: u32, sql_filter: String) -> AsyncTask<SearchTask> {
        AsyncTask::new(SearchTask {
            inner: inner(self.db_name.clone()),
            q: q,
            threshold: threshold as f32,
            top_k: top_k as usize,
            sql_filter: sql_filter,
        })
    }

    #[napi]
    pub fn score(&mut self, q: String, sentences: Vec<String>) -> AsyncTask<ScoreTask> {
        AsyncTask::new(ScoreTask {
            inner: inner(self.db_name.clone()),
            q: q,
            sentences: sentences,
        })
    }

    #[napi]
    pub fn add(&self, metadata: String, body: String) {
        Indexer::global().tx.send(("add".to_string(), metadata, body)).unwrap();
    }

    #[napi]
    pub fn embed(&self) {
        Indexer::global().tx.send(("embed".to_string(), "".to_string(), "".to_string())).unwrap();
    }

    #[napi]
    pub fn index(&self) {
        Indexer::global().tx.send(("index".to_string(), "".to_string(), "".to_string())).unwrap();
    }
}
