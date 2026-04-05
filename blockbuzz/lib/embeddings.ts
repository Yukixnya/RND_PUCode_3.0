import "dotenv/config";

const HF_API_URL =
  "https://router.huggingface.co/hf-inference/models/BAAI/bge-small-en-v1.5";

export async function getEmbedding(text: string): Promise<number[]> {
  try {
    const res = await fetch(HF_API_URL, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${process.env.HF_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        inputs: text,
      }),
    });

    if (!res.ok) {
      throw new Error(`HF API error: ${await res.text()}`);
    }

    const data = await res.json();

    if (!Array.isArray(data)) {
      throw new Error("Invalid embedding response");
    }

    return data;
  } catch (err: any) {
    console.error("Embedding error:", err.message);
    throw new Error("Failed to generate embedding");
  }
}