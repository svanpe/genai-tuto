import chromadb
import ollama




client = chromadb.PersistentClient(path="./chromadb/en")
client.list_collections()
collection = client.get_collection(name=client.list_collections()[0].name)

# an example prompt

prompt = """Can you summarize the following characteristics of the instruments in the form of list :
            * name of the instrument {source}
            * identification (ISIN)
            * risk class
            * horizon
            * performances 
        """
# generate an embedding for the prompt and retrieve the most relevant doc
response = ollama.embeddings(
    prompt=prompt,
    model="llama3"
)
results = collection.query(
    query_embeddings=[response["embedding"]],
    n_results=1
)
data = results['documents'][0][0]

print(f"Using this data: {data}. Respond to this prompt: {prompt}")

# generate a response combining the prompt and data we retrieved in step 2
output = ollama.generate(
    model="llama3",
    prompt=f"Using this data: {data}. Respond to this prompt: {prompt}"
)

print(output['response'])

