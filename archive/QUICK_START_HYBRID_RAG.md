# Quick Start Guide: Hybrid Graph + Vector RAG

## Prerequisites

1. **Install required packages:**

```bash
pip install openai tiktoken neo4j spacy langchain langchain-openai langchain-huggingface
python -m spacy download en_core_web_sm
```

2. **Set environment variables:**

```bash
# .env file
NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASS=your_password
OPENAI_API_KEY=sk-...
```

3. **Start Neo4j:**

```bash
# Make sure Neo4j is running on port 7687
# Version required: 5.14.0+ (for vector index support)
```

---

## Step 1: Clear Existing Graph (Optional)

If you have old data from previous runs:

```cypher
// In Neo4j Browser (http://localhost:7474)
MATCH (n) DETACH DELETE n
```

---

## Step 2: Create Knowledge Graph

```python
# Run in Streamlit or Python script
import streamlit as st
from hvac_code_chatbot import HVACGraphRAG, HVACCodeProcessor
import os
from dotenv import load_dotenv

load_dotenv()

# 1. Load documents
processor = HVACCodeProcessor()
documents = processor.load_hvac_documents("hvac_documents/")
print(f"Loaded {len(documents)} documents")

# 2. Initialize GraphRAG
graphrag = HVACGraphRAG(
    os.getenv("NEO4J_URI"),
    os.getenv("NEO4J_USER"),
    os.getenv("NEO4J_PASS")
)

# 3. Extract entities (Tier 1 - strict filtering)
entities = graphrag.create_hvac_entities(documents)
print(f"Extracted {len(entities)} high-quality entities")

# 4. Create knowledge graph
# This will:
# - Create Document node
# - Create Chunk nodes with embeddings
# - Create Entity nodes (Component, Code, Brand, etc.)
# - Create semantic relationships
# - Create MENTIONS relationships (chunk-to-entity)
# - Create vector index on Chunk.embedding
graphrag.create_knowledge_graph(entities, documents)
```

**Expected Console Output:**

```
Loaded 1 documents
Extracted 127 high-quality entities (Tier 1 - graph nodes)
Creating document chunks with embeddings...
Created 143 chunks for document hvac_code.pdf
Created 143 chunk nodes with embeddings
Creating MENTIONS relationships (chunk-to-entity)...
Created 387 MENTIONS relationships (chunk-to-entity)
Creating vector index on Chunk nodes...
Vector index created: 1536D cosine similarity
✓ Vector index created successfully
```

---

## Step 3: Verify Graph Structure

```cypher
// In Neo4j Browser

// 1. Check node counts
MATCH (n) RETURN labels(n)[0] as label, count(*) as count
ORDER BY count DESC

// Expected output:
// Chunk:     143
// Component: 89
// Code:      23
// Document:  1
// Brand:     8
// ...

// 2. Check relationship distribution
MATCH ()-[r]->() RETURN type(r), count(*)
ORDER BY count(*) DESC

// Expected output:
// MENTIONS:          387  (chunk-to-entity, high confidence)
// PART_OF:           143  (chunk-to-document)
// COMPLIES_WITH:     67   (semantic relationships)
// MANUFACTURED_BY:   12
// ...

// 3. Verify NO MENTIONED_IN relationships
MATCH ()-[r:MENTIONED_IN]->() RETURN count(r)
// Should return: 0

// 4. Check vector index
SHOW INDEXES
// Should see: chunk_embedding_index (ONLINE)

// 5. Test vector query
CALL db.index.vector.queryNodes('chunk_embedding_index', 3,
    [0.1, 0.2, 0.3, ...] // Use actual embedding
)
YIELD node, score
RETURN node.chunk_index, score LIMIT 3
```

---

## Step 4: Test Queries

### Test 1: Graph Query Only

```python
# Test natural language -> Cypher conversion
graph_results = graphrag.query_graph_with_natural_language(
    "What codes apply to air handlers?"
)

print(f"Entities found: {graph_results['entities_found']}")
print(f"Graph results: {len(graph_results['graph_results'])}")
print(f"Cypher queries used: {graph_results['cypher_queries']}")

# Expected output:
# Entities found: ['air handlers']
# Graph results: 5
# Cypher queries used: ['Code Compliance Query']
```

### Test 2: Vector Similarity Only

```python
# Test cosine similarity search
vector_results = graphrag.query_vector_similarity(
    "What codes apply to air handlers?",
    top_k=5
)

print(f"Vector results: {len(vector_results)}")
for i, result in enumerate(vector_results, 1):
    print(f"{i}. Similarity: {result['similarity_score']:.3f} | Chunk {result['chunk_index']}")

# Expected output:
# Vector results: 5
# 1. Similarity: 0.873 | Chunk 23
# 2. Similarity: 0.821 | Chunk 45
# ...
```

### Test 3: Hybrid Search

```python
# Test combined graph + vector search
hybrid_results = graphrag.hybrid_search("What codes apply to air handlers?")

print(f"Graph results: {hybrid_results['num_graph_results']}")
print(f"Vector results: {hybrid_results['num_vector_results']}")
print(f"Graph queries: {hybrid_results['graph_queries_used']}")
print(f"Fusion strategy: {hybrid_results['fusion_strategy']}")

# Expected output:
# Graph results: 5
# Vector results: 5
# Graph queries: ['Code Compliance Query']
# Fusion strategy: graph_prioritized
```

### Test 4: End-to-End Query

```python
from hvac_code_chatbot import HVACChatbot

# Initialize chatbot
chatbot = HVACChatbot(
    os.getenv("NEO4J_URI"),
    os.getenv("NEO4J_USER"),
    os.getenv("NEO4J_PASS"),
    os.getenv("OPENAI_API_KEY")
)

# Ask question (uses hybrid search by default)
answer = chatbot.query_hvac_system(
    "What are the Atlanta code requirements for Carrier air handlers?"
)

print(answer)

# Expected output includes:
# - **KNOWLEDGE GRAPH FACTS (Verified Relationships):**
# - **RELEVANT DOCUMENT SECTIONS (Cosine Similarity):**
# - **Search Metadata:**
```

---

## Step 5: Test All 6 Intent Patterns

```python
test_cases = [
    {
        "question": "What codes apply to air handlers?",
        "expected_intent": "Code Compliance Query",
        "expected_graph_results": ">= 1"
    },
    {
        "question": "How do I fix low airflow?",
        "expected_intent": "Problem-Solution Query",
        "expected_graph_results": ">= 1"
    },
    {
        "question": "What Carrier equipment is mentioned?",
        "expected_intent": "Manufacturer Query",
        "expected_graph_results": ">= 1"
    },
    {
        "question": "What are the SEER ratings?",
        "expected_intent": "Specification Query",
        "expected_graph_results": ">= 1"
    },
    {
        "question": "What components are in the air handler?",
        "expected_intent": "System Hierarchy Query",
        "expected_graph_results": ">= 1"
    },
    {
        "question": "What equipment is installed in Atlanta?",
        "expected_intent": "Location Query",
        "expected_graph_results": ">= 1"
    }
]

for test in test_cases:
    hybrid_results = graphrag.hybrid_search(test["question"])

    print(f"\n{'='*60}")
    print(f"Question: {test['question']}")
    print(f"Expected Intent: {test['expected_intent']}")
    print(f"Actual Queries: {hybrid_results['graph_queries_used']}")
    print(f"Graph Results: {hybrid_results['num_graph_results']}")
    print(f"Vector Results: {hybrid_results['num_vector_results']}")

    # Verify
    if test['expected_intent'] in hybrid_results['graph_queries_used']:
        print("✓ Intent detection PASSED")
    else:
        print("✗ Intent detection FAILED")
```

---

## Common Issues & Solutions

### Issue 1: ImportError for OpenAI

**Error:** `ImportError: cannot import name 'OpenAI' from 'openai'`

**Solution:**

```bash
pip install --upgrade openai
# Make sure version >= 1.0.0
```

### Issue 2: Vector index not found

**Error:** `No such index: chunk_embedding_index`

**Solution:**

```cypher
// Manually create in Neo4j Browser
CREATE VECTOR INDEX chunk_embedding_index IF NOT EXISTS
FOR (c:Chunk)
ON c.embedding
OPTIONS {
    indexConfig: {
        `vector.dimensions`: 1536,
        `vector.similarity_function`: 'cosine'
    }
}

// Wait 10-30 seconds, then verify
SHOW INDEXES
```

### Issue 3: All embeddings are zero

**Error:** Similarity scores all identical (0.0 or 1.0)

**Solution:**

```python
# Test OpenAI API key
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.embeddings.create(
    model="text-embedding-3-small",
    input="test"
)
print(f"Embedding length: {len(response.data[0].embedding)}")
# Should print: 1536

# Check first few values
print(response.data[0].embedding[:5])
# Should be non-zero floats
```

### Issue 4: No graph results

**Error:** `hybrid_results['num_graph_results'] == 0` for all questions

**Solution:**

```cypher
// Check if semantic relationships exist
MATCH ()-[r:COMPLIES_WITH]->() RETURN count(r)
// If 0, relationships weren't created

// Check entity confidence scores
MATCH (n:Component) RETURN n.text, n.confidence
ORDER BY n.confidence DESC LIMIT 10
// If all < 0.6, entities were filtered out

// Lower threshold temporarily for testing
// In Python: change confidence < 0.6 to confidence < 0.5
```

### Issue 5: Memory error during chunking

**Error:** `MemoryError` when processing large PDFs

**Solution:**

```python
# Reduce chunk size in chunk_document_with_embeddings()
# Change from 1000 to 500 tokens max:

if current_tokens + para_tokens > 500:  # Was 1000
    # Save chunk
```

---

## Performance Benchmarks

Expected performance on a document with ~10,000 tokens (20-30 pages):

| Operation             | Time          | Output                 |
| --------------------- | ------------- | ---------------------- |
| Entity extraction     | 5-10 sec      | 100-200 entities       |
| Chunking              | 2-5 sec       | 50-200 chunks          |
| Embedding generation  | 10-20 sec     | 50-200 embeddings      |
| Graph creation        | 3-8 sec       | Nodes + relationships  |
| Vector index creation | 1-2 sec       | Index ready            |
| **Total setup**       | **20-45 sec** | Complete graph         |
| **Hybrid query**      | **< 500 ms**  | Graph + vector results |

---

## Next Steps

1. **Add more documents:**

   - Upload additional HVAC PDFs to `hvac_documents/`
   - Re-run entity extraction and graph creation
   - System automatically handles multi-document graphs

2. **Customize intent patterns:**

   - Edit `query_graph_with_natural_language()` method
   - Add new keywords for intent detection
   - Create custom Cypher queries

3. **Tune fusion weights:**

   - Adjust graph vs. vector prioritization
   - Track user feedback on answer quality
   - Implement weighted scoring

4. **Deploy to production:**
   - Use persistent Neo4j instance (not Docker ephemeral)
   - Scale OpenAI embedding calls (batch API)
   - Add caching for common queries

---

## Monitoring & Maintenance

### Daily Checks

```cypher
// 1. Check graph health
MATCH (n) RETURN labels(n)[0] as label, count(*)
ORDER BY count DESC

// 2. Check relationship balance
MATCH ()-[r]->() RETURN type(r), count(*)
ORDER BY count DESC

// 3. Verify vector index is online
SHOW INDEXES
```

### Weekly Checks

```python
# Test all 6 intent patterns still work
# Run test_cases loop from Step 5
# Verify > 0 graph results for each

# Check embedding quality
test_embedding = graphrag._get_embedding("air handler")
non_zero = sum(1 for x in test_embedding if abs(x) > 0.01)
print(f"Non-zero dimensions: {non_zero} / 1536")
# Should be: > 1400
```

### Monthly Checks

```cypher
// Check for orphaned nodes (no relationships)
MATCH (n)
WHERE NOT (n)--()
RETURN labels(n)[0] as label, count(*)
ORDER BY count DESC
// Clean up if > 10% of nodes

// Check average chunk similarity
MATCH (c:Chunk)
WITH collect(c.embedding) as embeddings
RETURN size(embeddings)
// Should match chunk count
```

---

## Success Criteria

Your hybrid Graph + Vector RAG system is working correctly if:

✅ **Graph size:** 100-300 total nodes (not thousands)  
✅ **Chunk coverage:** 50-200 chunks with embeddings  
✅ **Relationships:** 400-600 total (no MENTIONED_IN)  
✅ **Vector index:** ONLINE status, 1536 dimensions  
✅ **Hybrid search:** Returns both graph + vector results  
✅ **Query speed:** < 500ms for hybrid search  
✅ **Answer quality:** Graph facts + semantic context

**If all criteria met: System is production-ready! 🎉**
