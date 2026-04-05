import { pipeline } from "@xenova/transformers";

let extractor: any = null;

async function getExtractor() {
  if (!extractor) {
    extractor = await pipeline(
      "feature-extraction",
      "Xenova/all-MiniLM-L6-v2"
    );
  }
  return extractor;
}

export async function getEmbedding(text: string): Promise<number[]> {
  try {
    const model = await getExtractor();

    const output = await model(text);

    if (!output || !output.data || !output.dims) {
      throw new Error("Invalid model output");
    }

    // omkar bro this part is for converting the output tensor to a 2D array of token embeddings
    const { data, dims } = output;

    // brother this is the shape of the output tensor: [batch_size, num_tokens, embedding_dim]
    const [batch, tokens, dim] = dims;

    if (!tokens || !dim) {
      throw new Error("Invalid tensor shape");
    }

    // brother this part is for slicing the flat data array into token embeddings
    const tokenEmbeddings: number[][] = [];

    // brother here we slice the flat data array into token embeddings and push them into the tokenEmbeddings array
    for (let i = 0; i < tokens; i++) {
      const start = i * dim;
      const end = start + dim;
      tokenEmbeddings.push(Array.from(data.slice(start, end)));
    }

    const embedding = meanPooling(tokenEmbeddings);
    return normalize(embedding);

  } catch (err: any) {
    console.error("Embedding error:", err.message);
    throw new Error("Failed to generate embedding");
  }
}

function meanPooling(vectors: number[][]): number[] {
  const dim = vectors[0].length;
  const mean = new Array(dim).fill(0);

  for (const vec of vectors) {
    for (let i = 0; i < dim; i++) {
      mean[i] += vec[i];
    }
  }

  for (let i = 0; i < dim; i++) {
    mean[i] /= vectors.length;
  }

  return mean;
}

function normalize(vec: number[]): number[] {
  const norm = Math.sqrt(vec.reduce((s, v) => s + v * v, 0));
  return vec.map(v => v / norm);
}