use anyhow::{Context, Result, anyhow};
use pdf_extract::extract_text;
use qdrant_client::Qdrant;
use qdrant_client::qdrant::{
    CountPointsBuilder, CreateCollectionBuilder, Distance, QueryPointsBuilder, VectorParamsBuilder,
};
use rig::Embed;
use rig::integrations::cli_chatbot::ChatBotBuilder;
use rig::client::{CompletionClient, EmbeddingsClient, Nothing};
use rig::embeddings::EmbeddingsBuilder;
use rig::providers::ollama;
use rig::vector_store::InsertDocuments;
use rig_qdrant::QdrantVectorStore;
use serde::{Deserialize, Serialize};
use std::path::Path;

const QDRANT_URL: &str = "http://localhost:6334";
const COLLECTION_NAME: &str = "rag_documents";
const PDF_PATHS: [&str; 2] = [
    "documents/Moores_Law_for_Everything.pdf",
    "documents/The_Last_Question.pdf",
];
const CHUNK_SIZE: usize = 1000;
const CHUNK_OVERLAP: usize = 200;
const TOP_K_CHUNKS: usize = 5;

// Qdrant payloads must be JSON objects, so chunks are wrapped in a struct.
#[derive(Embed, Serialize, Deserialize, Debug)]
struct Document {
    source: String,
    chunk_index: usize,
    #[embed]
    content: String,
}

fn load_pdf_content<P: AsRef<Path>>(file_path: P) -> Result<String> {
    extract_text(file_path.as_ref())
        .with_context(|| format!("Failed to extract text from PDF: {:?}", file_path.as_ref()))
}

/// Splits text into chunks of at most `CHUNK_SIZE` characters on word
/// boundaries, with roughly `CHUNK_OVERLAP` characters shared between
/// neighbouring chunks so sentences spanning a boundary are not lost.
fn chunk_text(text: &str) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut chunks = Vec::new();
    let mut start = 0;

    while start < words.len() {
        let mut end = start;
        let mut len = 0;
        while end < words.len() && (len == 0 || len + words[end].len() + 1 <= CHUNK_SIZE) {
            len += words[end].len() + 1;
            end += 1;
        }
        chunks.push(words[start..end].join(" "));
        if end == words.len() {
            break;
        }

        // Step back from `end` to build the overlap, always advancing by at least one word.
        let mut next = end;
        let mut overlap = 0;
        while next > start + 1 && overlap + words[next - 1].len() + 1 <= CHUNK_OVERLAP {
            next -= 1;
            overlap += words[next].len() + 1;
        }
        start = next;
    }

    chunks
}

#[tokio::main]
async fn main() -> Result<()> {
    let ollama_client = ollama::Client::new(Nothing)?;
    let embedding_model = ollama_client.embedding_model("qwen3-embedding:4b");

    let qdrant = Qdrant::from_url(QDRANT_URL)
        .build()
        .context("Failed to create Qdrant client")?;

    let collection_exists = qdrant
        .collection_exists(COLLECTION_NAME)
        .await
        .with_context(|| format!("Failed to reach Qdrant at {QDRANT_URL}"))?;
    let point_count = if collection_exists {
        qdrant
            .count(CountPointsBuilder::new(COLLECTION_NAME).exact(true))
            .await?
            .result
            .map_or(0, |r| r.count)
    } else {
        0
    };

    let query_params = QueryPointsBuilder::new(COLLECTION_NAME)
        .with_payload(true)
        .build();

    // Only embed and ingest the PDFs when the collection is empty; otherwise
    // reuse the embeddings already persisted in Qdrant.
    let embeddings = if point_count == 0 {
        let mut builder = EmbeddingsBuilder::new(embedding_model.clone());
        for path in PDF_PATHS {
            let chunks = chunk_text(&load_pdf_content(path)?);
            println!("Split {path} into {} chunks", chunks.len());
            builder = builder.documents(chunks.into_iter().enumerate().map(|(i, content)| {
                Document {
                    source: path.to_string(),
                    chunk_index: i,
                    content,
                }
            }))?;
        }
        Some(builder.build().await?)
    } else {
        println!("Using {point_count} existing points in Qdrant collection '{COLLECTION_NAME}'");
        None
    };

    if let Some(embeddings) = &embeddings
        && !collection_exists
    {
        let dimensions = embeddings
            .first()
            .map(|(_, e)| e.first().vec.len())
            .context("No embeddings were produced")?;
        qdrant
            .create_collection(
                CreateCollectionBuilder::new(COLLECTION_NAME)
                    .vectors_config(VectorParamsBuilder::new(dimensions as u64, Distance::Cosine)),
            )
            .await?;
    }

    let vector_store = QdrantVectorStore::new(qdrant, embedding_model, query_params);

    if let Some(embeddings) = embeddings {
        vector_store
            .insert_documents(embeddings)
            .await
            .map_err(|err| anyhow!("Failed to insert documents into Qdrant: {err}"))?;
    }

    let rag_agent = ollama_client
        .agent("qwen3.5:9b")
        .preamble(
            "You are a helpful assistant that answers questions using the provided PDF excerpts. \
             Each excerpt includes its source file and chunk index.",
        )
        .dynamic_context(TOP_K_CHUNKS, vector_store)
        .build();

    let chatbot = ChatBotBuilder::new().agent(rag_agent).build();
    chatbot.run().await?;

    Ok(())
}
