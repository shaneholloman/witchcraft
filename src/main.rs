use anyhow::Result;
use std::env;

mod warp;

fn main() -> Result<()> {

    let args: Vec<String> = env::args().collect();
    let db = warp::DB::new("mydb.sqlite");
    let device = warp::make_device();
    let embedder = warp::Embedder::new(&device);
    let mut cache = warp::EmbeddingsCache::new(1);

    if args.len() == 3 && args[1] == "scan" {

        warp::scan_documents_dir(&db, &args[2]).unwrap();

    } else if args.len() == 3 && args[1] == "readcsv" {

        warp::read_csv(&db, &args[2]).unwrap();

    } else if args.len() == 3 && args[1] == "add" {

        warp::add_doc_from_file(&db, &args[2]).unwrap();

    } else if args.len() == 2 && &args[1] == "embed" {

        warp::embed_chunks(&db, &device).unwrap();

    } else if args.len() == 2 && &args[1] == "index" {

        warp::index_chunks(&db, &device).unwrap();

    } else if args.len() >= 3 && (args[1] == "query" || args[1] == "hybrid") {

        let q = &args[2..].join(" ");
        let use_fulltext = args[1] == "hybrid";
        let results = warp::search(&db, &embedder, &mut cache, &q, 0.75, 10, use_fulltext, None).unwrap();
        for (score, filename, body) in results {
            println!("{} : {} : {}", score, filename, body);
        }

    } else if args.len() >= 4 && (args[1] == "querycsv" || args[1] == "hybridcsv" || args[1] == "fulltextcsv") {

        let use_fulltext = args[1] == "hybridcsv" || args[1] == "fulltextcsv";
        let use_semantic = args[1] != "fulltextcsv";
        let csvname = &args[2];
        let outputname = &args[3];
        warp::bulk_search(&db, &embedder, &csvname, &outputname, use_fulltext, use_semantic).unwrap();

    } else if args.len() >= 4 && &args[1] == "score" {

        let sentences: Vec<String> = std::env::args().skip(3).collect();
        let scores = warp::score_query_sentences(&embedder, &mut cache, &args[2], &sentences).unwrap();
        for (i, score) in scores.iter().enumerate() {
            println!("`{}': score={}", args[3 + i], *score);
        }

    } else {
       eprintln!("\n*** Usage: {} scan | readcsv <file> | embed | index | query <text> | hybrid <text> | querycsv <file> <results-file> ***\n", args[0]);
    };
    Ok(())
}
