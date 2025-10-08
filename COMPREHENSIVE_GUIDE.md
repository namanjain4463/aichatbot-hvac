# 📚 HVAC Chatbot - Complete Guide

## Hybrid Graph + Vector RAG System

**Version:** 2.0  
**Last Updated:** October 8, 2025  
**System Type:** Hybrid Knowledge Graph + Vector Embeddings RAG

---

# 📖 Table of Contents

1. [Quick Start](#quick-start)
2. [System Architecture](#system-architecture)
3. [Installation & Setup](#installation--setup)
4. [Knowledge Graph Schema](#knowledge-graph-schema)
5. [Usage Guide](#usage-guide)
6. [Query Patterns & Examples](#query-patterns--examples)
7. [Optimization Guides](#optimization-guides)
8. [Testing & Validation](#testing--validation)
9. [Troubleshooting](#troubleshooting)
10. [Neo4j Validation Queries](#neo4j-validation-queries)
11. [Quick Reference](#quick-reference)

---

# 🚀 Quick Start

## Prerequisites

**1. Install Python Packages:**

```bash
pip install openai tiktoken neo4j spacy langchain langchain-openai langchain-huggingface streamlit
python -m spacy download en_core_web_sm
```

**2. Set Environment Variables:**

Create a `.env` file:

```bash
NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASS=your_password
OPENAI_API_KEY=sk-your-key-here
```

**3. Start Neo4j:**

- Version required: 5.14.0+ (for vector index support)
- Default port: 7687 (Bolt protocol)
- Browser UI: http://localhost:7474

## 5-Minute Setup

```python
from hvac_code_chatbot import HVACGraphRAG, HVACCodeProcessor
import os
from dotenv import load_dotenv

load_dotenv()

# 1. Load your PDF documents
processor = HVACCodeProcessor()
documents = processor.load_hvac_documents("hvac_documents/")

# 2. Initialize the system
graphrag = HVACGraphRAG(
    os.getenv("NEO4J_URI"),
    os.getenv("NEO4J_USER"),
    os.getenv("NEO4J_PASS")
)

# 3. Create knowledge graph (one-time setup)
entities = graphrag.create_hvac_entities(documents)
graphrag.create_knowledge_graph(entities, documents)

# 4. Start querying!
answer = graphrag.query("What codes apply to air handlers?")
print(answer)
```

**Run the Web UI:**

```bash
streamlit run app.py
```

---

# 🏗️ System Architecture

## Overview

This is a **Hybrid Graph + Vector RAG** system that combines:

- **Knowledge Graph** (Neo4j) for structured semantic relationships
- **Vector Embeddings** (1536D) for semantic similarity search
- **LLM** (OpenAI GPT) for natural language generation

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    USER QUESTION                             │
│                  (Natural Language)                          │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│              HYBRID PARALLEL SEARCH                          │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────────┐   ┌─────────────────────────┐    │
│  │   GRAPH QUERY        │   │  VECTOR SIMILARITY      │    │
│  │   (Cypher)           │   │  (Cosine on Embeddings) │    │
│  ├──────────────────────┤   ├─────────────────────────┤    │
│  │ • Intent Detection   │   │ • Question Embedding    │    │
│  │ • 6 Cypher Patterns  │   │ • Chunk Similarity      │    │
│  │ • Relationship       │   │ • Top-K Results         │    │
│  │   Traversal          │   │ • Cosine Distance       │    │
│  └──────────┬───────────┘   └────────┬────────────────┘    │
│             │                        │                      │
│             └────────┬───────────────┘                      │
│                      │                                       │
└──────────────────────┼───────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              RESULT FUSION & RANKING                         │
├─────────────────────────────────────────────────────────────┤
│  • Graph Results (Prioritized - Verified Relationships)      │
│  • Vector Results (Supporting - Relevant Document Chunks)    │
│  • Metadata Tracking (Which method found what)               │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                LLM RESPONSE GENERATION                       │
│         (OpenAI GPT with Combined Context)                   │
└─────────────────────────────────────────────────────────────┘
```

## Key Features

✅ **Two-Tier Entity Extraction**

- Tier 1 (≥0.6 confidence): High-quality graph nodes
- Tier 2 (≥0.3 confidence): Chunk embeddings for full coverage

✅ **Hybrid Search**

- Graph: Semantic relationships (COMPLIES_WITH, HAS_SPECIFICATION, etc.)
- Vector: Cosine similarity on 1536D embeddings

✅ **6 Intent Detection Patterns**

- Code compliance
- Problem-solution
- Specifications
- Brand/manufacturer
- System hierarchy
- Location-based

✅ **Intelligent Result Fusion**

- Prioritizes graph relationships
- Enriches with vector similarity chunks
- Tracks source provenance

---

# ⚙️ Installation & Setup

## Step 1: Install Dependencies

```bash
# Core packages
pip install openai==1.3.0
pip install tiktoken==0.5.1
pip install neo4j==5.14.0
pip install spacy==3.7.2
pip install langchain==0.1.0
pip install langchain-openai==0.0.2
pip install langchain-huggingface==0.0.1
pip install streamlit==1.28.0

# Download spaCy model
python -m spacy download en_core_web_sm
```

## Step 2: Configure Neo4j

**Option A: Neo4j Desktop**

1. Download from https://neo4j.com/download/
2. Create a new database
3. Set password
4. Start database

**Option B: Neo4j Docker**

```bash
docker run \
    --name neo4j \
    -p7474:7474 -p7687:7687 \
    -d \
    -e NEO4J_AUTH=neo4j/your_password \
    neo4j:5.14.0
```

## Step 3: Set Environment Variables

Create `.env` file in project root:

```bash
# Neo4j Connection
NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASS=your_password

# OpenAI API
OPENAI_API_KEY=sk-your-openai-api-key

# Optional Settings
CHUNK_SIZE=800
OVERLAP=100
SIMILARITY_THRESHOLD=0.75
```

## Step 4: Prepare Documents

```bash
# Create documents directory
mkdir hvac_documents

# Place your PDF files here
cp your_hvac_code.pdf hvac_documents/
```

## Step 5: Initialize System

```python
from hvac_code_chatbot import HVACGraphRAG, HVACCodeProcessor
import os
from dotenv import load_dotenv

load_dotenv()

# Load documents
processor = HVACCodeProcessor()
documents = processor.load_hvac_documents("hvac_documents/")

# Initialize GraphRAG
graphrag = HVACGraphRAG(
    os.getenv("NEO4J_URI"),
    os.getenv("NEO4J_USER"),
    os.getenv("NEO4J_PASS")
)

# Create knowledge graph
entities = graphrag.create_hvac_entities(documents)
graphrag.create_knowledge_graph(entities, documents)

print("✅ System ready!")
```

---

# 🗺️ Knowledge Graph Schema

## Node Types (9 Labels)

### Primary Nodes

**1. :Document**

- `source` (String): File path
- `type` (String): "PDF", "TXT", etc.
- `location` (String): Storage location
- `title` (String): Document title

**2. :Chunk**

- `text` (String): Chunk content
- `chunk_index` (Integer): Position in document
- `embedding` (List[Float], 1536D): Vector embedding
- `token_count` (Integer): Number of tokens

**3. :Component**

- `text` (String): Component name
- `confidence` (Float): Extraction confidence
- `hvac_type` (String): Equipment category
- `importance` (Float): Relevance score

**4. :Code**

- `text` (String): Code section reference
- `code_ref` (String): Standardized reference
- `confidence` (Float): Extraction confidence

**5. :Brand**

- `text` (String): Manufacturer name
- `confidence` (Float): Extraction confidence

**6. :Problem**

- `text` (String): Issue description
- `confidence` (Float): Extraction confidence

**7. :Solution**

- `text` (String): Resolution steps
- `confidence` (Float): Extraction confidence

**8. :Location**

- `text` (String): Geographic location
- `atlanta_specific` (Boolean): Local relevance

**9. :Specification**

- `text` (String): Spec value
- `spec_type` (String): "SEER", "BTU", "CFM", etc.
- `confidence` (Float): Extraction confidence

## Relationship Types (10 Types)

### Structural Relationships

**1. :PART_OF** (Component → Component/System)

- Confidence: 0.95
- Hierarchy relationships

**2. :MENTIONS** (Entity → Chunk)

- Confidence: 0.85
- Bridge between graph and vector search

### Semantic Relationships

**3. :COMPLIES_WITH** (Component → Code)

- Confidence: 0.90
- Code compliance mapping

**4. :HAS_SPECIFICATION** (Component → Specification)

- Confidence: 0.85
- Technical specifications

**5. :MANUFACTURED_BY** (Component → Brand)

- Confidence: 0.85
- Equipment manufacturers

**6. :RESOLVED_BY** (Problem → Solution)

- Confidence: 0.90
- Troubleshooting relationships

**7. :INSTALLED_IN** (Component → Location)

- Confidence: 0.80
- Geographic locations

**8. :BELONGS_TO** (Component → Category)

- Confidence: 0.90
- Equipment categorization

**9. :FOLLOWS** (Code → Standard)

- Confidence: 0.95
- Standards compliance

**10. :RELATED_TO** (Code → Code)

- Confidence: 0.75
- Cross-references

## Visual Schema

```
:Document
    │
    │ :PART_OF
    ▼
:Chunk (embeddings: 1536D)
    │
    │ :MENTIONS (bridge)
    ▼
:Component ──:COMPLIES_WITH──► :Code ──:FOLLOWS──► :Standard
    │
    ├──:HAS_SPECIFICATION──► :Specification
    ├──:MANUFACTURED_BY──► :Brand
    ├──:INSTALLED_IN──► :Location
    └──:BELONGS_TO──► :Category

:Problem ──:RESOLVED_BY──► :Solution
```

---

# 📘 Usage Guide

## Basic Queries

```python
# Simple query
answer = graphrag.query("What codes apply to air handlers?")

# Query with context
answer = graphrag.query(
    "How do I fix low airflow issues?",
    include_sources=True
)

# Query specific document
answer = graphrag.query(
    "What does Section 403.2 say?",
    document_filter="IMC_2021.pdf"
)
```

## Web Interface (Streamlit)

```bash
# Start the app
streamlit run app.py
```

**Features:**

- Document upload
- Real-time graph creation
- Interactive Q&A
- Source citations
- Response time tracking

## Python API

```python
# Initialize
from hvac_code_chatbot import HVACGraphRAG

graphrag = HVACGraphRAG(uri, user, password)

# Load documents
from hvac_code_chatbot import HVACCodeProcessor
processor = HVACCodeProcessor()
docs = processor.load_hvac_documents("hvac_documents/")

# Create graph
entities = graphrag.create_hvac_entities(docs)
graphrag.create_knowledge_graph(entities, docs)

# Query
answer = graphrag.query("Your question here")
print(answer)

# Access graph directly
with graphrag.driver.session() as session:
    result = session.run("MATCH (n:Component) RETURN n LIMIT 10")
    for record in result:
        print(record['n']['text'])
```

---

# 🔍 Query Patterns & Examples

## Pattern 1: Code Compliance Queries

**Intent:** Find code requirements

**Example Queries:**

- "What codes apply to air handlers?"
- "What are the ASHRAE requirements for ventilation?"
- "What IMC sections govern furnace installation?"

**Graph Pattern:**

```cypher
MATCH (comp:Component)-[r:COMPLIES_WITH]->(code:Code)
WHERE comp.text =~ '(?i).*air handler.*'
RETURN comp.text, code.text, r.confidence
```

**Expected Results:**

- Component → Code relationships
- Code section references
- Compliance requirements

---

## Pattern 2: Problem-Solution Queries

**Intent:** Troubleshooting

**Example Queries:**

- "How do I fix low airflow issues?"
- "What causes heating system failures?"
- "How to troubleshoot air conditioning problems?"

**Graph Pattern:**

```cypher
MATCH (prob:Problem)-[r:RESOLVED_BY]->(sol:Solution)
WHERE prob.text =~ '(?i).*low airflow.*'
RETURN prob.text, sol.text, r.confidence
```

**Expected Results:**

- Problem descriptions
- Solution steps
- Troubleshooting guides

---

## Pattern 3: Specification Queries

**Intent:** Technical specs

**Example Queries:**

- "What is the SEER rating for heat pumps?"
- "What is the BTU capacity of the furnace?"
- "What are the efficiency ratings mentioned?"

**Graph Pattern:**

```cypher
MATCH (comp:Component)-[r:HAS_SPECIFICATION]->(spec:Specification)
WHERE spec.spec_type = 'SEER'
RETURN comp.text, spec.text, spec.spec_type
```

**Expected Results:**

- SEER ratings
- BTU capacities
- CFM ratings
- Efficiency metrics

---

## Pattern 4: Brand/Manufacturer Queries

**Intent:** Equipment manufacturers

**Example Queries:**

- "What Carrier equipment is mentioned?"
- "Who manufactures the air conditioning unit?"
- "What Trane products are discussed?"

**Graph Pattern:**

```cypher
MATCH (comp:Component)-[r:MANUFACTURED_BY]->(brand:Brand)
WHERE brand.text =~ '(?i).*carrier.*'
RETURN comp.text, brand.text
```

**Expected Results:**

- Equipment by manufacturer
- Brand associations
- Product lines

---

## Pattern 5: System Hierarchy Queries

**Intent:** Component relationships

**Example Queries:**

- "What components are part of the HVAC system?"
- "What is connected to the air handler?"
- "What parts make up the heating system?"

**Graph Pattern:**

```cypher
MATCH (comp:Component)-[r:PART_OF]->(system:Component)
RETURN comp.text AS part, system.text AS system
```

**Expected Results:**

- Component hierarchy
- System structure
- Part relationships

---

## Pattern 6: Location Queries

**Intent:** Geographic information

**Example Queries:**

- "What equipment is in Atlanta?"
- "Where is the furnace installed?"
- "What are the local code requirements?"

**Graph Pattern:**

```cypher
MATCH (comp:Component)-[r:INSTALLED_IN]->(loc:Location)
WHERE loc.text =~ '(?i).*atlanta.*'
RETURN comp.text, loc.text
```

**Expected Results:**

- Location-specific installations
- Local code requirements
- Geographic references

---

# ⚡ Optimization Guides

## Extracting More Relationships

### Problem: Few COMPLIES_WITH Relationships

**Why:** Pattern matching may not capture your document's specific language.

**Solution:** Enhanced relationship extraction patterns

**Add these patterns to your system:**

```python
def extract_compliance_relationships_enhanced(self, session, chunks):
    """
    Extract COMPLIES_WITH relationships from code language patterns
    """
    import re

    compliance_patterns = [
        # Standard patterns
        (r'(\w+(?:\s+\w+){0,2})\s+(?:must comply with|complies with|shall comply with)\s+((?:Section|IMC|IECC|ASHRAE)\s*\d+(?:\.\d+)*)', 'forward'),

        # Code-first patterns
        (r'((?:Section|IMC|IECC|ASHRAE)\s*\d+(?:\.\d+)*)\s+(?:applies to|governs|regulates|covers)\s+(\w+(?:\s+\w+){0,2})', 'reverse'),

        # Requirements patterns
        (r'(\w+(?:\s+\w+){0,2})\s+(?:shall meet|must meet|shall satisfy)\s+(?:the )?(?:requirements of|provisions of)\s+((?:Section|IMC|IECC|ASHRAE)\s*\d+(?:\.\d+)*)', 'forward'),

        # Subject-to patterns
        (r'(\w+(?:\s+\w+){0,2})\s+(?:are subject to|is subject to)\s+((?:Section|IMC|IECC|ASHRAE)\s*\d+(?:\.\d+)*)', 'forward'),

        # Installation patterns
        (r'(\w+(?:\s+\w+){0,2})\s+(?:shall be installed|shall be constructed)\s+(?:in accordance with|per|according to)\s+((?:Section|IMC|IECC|ASHRAE)\s*\d+(?:\.\d+)*)', 'forward'),
    ]

    relationships_created = 0

    for chunk in chunks:
        for pattern, direction in compliance_patterns:
            matches = re.finditer(pattern, chunk['text'], re.IGNORECASE)

            for match in matches:
                if direction == 'forward':
                    component = match.group(1).strip()
                    code = match.group(2).strip()
                else:
                    code = match.group(1).strip()
                    component = match.group(2).strip()

                # Create relationship
                session.run("""
                    MERGE (comp:Component {text: $component})
                    MERGE (code:Code {text: $code})
                    MERGE (comp)-[r:COMPLIES_WITH]->(code)
                    SET r.confidence = 0.8,
                        r.source = 'pattern_match'
                """, component=component, code=code)

                relationships_created += 1

    return relationships_created
```

### Specification Extraction

```python
def extract_specification_relationships_enhanced(self, session, chunks):
    """
    Extract HAS_SPECIFICATION relationships
    """
    spec_patterns = [
        # Efficiency ratings
        (r'(\w+(?:\s+\w+){0,2})\s+(?:with|rated at)\s+(\d+(?:\.\d+)?\s*SEER)', 'SEER'),
        (r'(\w+(?:\s+\w+){0,2})\s+(?:with|rated at)\s+(\d+(?:\.\d+)?\s*AFUE)', 'AFUE'),

        # Capacity
        (r'(\w+(?:\s+\w+){0,2})\s+(?:with|rated at)\s+(\d+(?:,\d{3})*\s*BTU)', 'BTU'),

        # Airflow
        (r'(\w+(?:\s+\w+){0,2})\s+(?:providing)\s+(\d+(?:,\d{3})*\s*CFM)', 'CFM'),
    ]

    # Implementation similar to above
```

## Optimal Query Patterns for Code Documents

### Strategy 1: Use MENTIONS as Bridge

Since code documents have many MENTIONS relationships, leverage them:

```cypher
// Find codes mentioned with components (implicit compliance)
MATCH (comp:Component)-[:MENTIONS]->(chunk:Chunk)
      <-[:MENTIONS]-(code:Code)
WHERE comp.text =~ '(?i).*air handler.*'
  AND chunk.text =~ '(?i).*(comply|shall meet|subject to).*'
RETURN comp.text, code.text, chunk.text AS evidence
```

### Strategy 2: Co-occurrence Queries

```cypher
// Find components and codes that appear together
MATCH (comp:Component)-[:MENTIONS]->(chunk:Chunk)
      <-[:MENTIONS]-(code:Code)
RETURN comp.text, code.text,
       count(chunk) AS co_occurrences
ORDER BY co_occurrences DESC
LIMIT 20
```

### Strategy 3: Multi-hop Queries

```cypher
// Find related components through shared codes
MATCH (comp1:Component)-[:MENTIONS]->(:Chunk)
      <-[:MENTIONS]-(code:Code)
      -[:MENTIONS]->(:Chunk)<-[:MENTIONS]-(comp2:Component)
WHERE comp1.text =~ '(?i).*furnace.*'
  AND comp1 <> comp2
RETURN comp1.text, code.text, comp2.text AS related_component
LIMIT 15
```

### Strategy 4: Create Indexes

```cypher
// Speed up text searches
CREATE INDEX entity_text_index IF NOT EXISTS
FOR (e:Component) ON (e.text);

CREATE INDEX code_text_index IF NOT EXISTS
FOR (c:Code) ON (c.text);

CREATE INDEX chunk_index IF NOT EXISTS
FOR (c:Chunk) ON (c.chunk_index);
```

---

# 🧪 Testing & Validation

## Quick 5-Minute Test

```python
# 1. Load system
graphrag = HVACGraphRAG(uri, user, password)

# 2. Run basic queries
test_queries = [
    "What codes apply to furnaces?",
    "How to fix low airflow?",
    "What is the SEER rating?"
]

for query in test_queries:
    answer = graphrag.query(query)
    print(f"Q: {query}")
    print(f"A: {answer}\n")
```

## Comprehensive Testing

### Test Set 1: Code Compliance

```
✓ "What codes apply to air handlers?"
✓ "What are the ASHRAE requirements for ventilation?"
✓ "What IMC sections govern furnace installation?"
✓ "What are the IECC energy efficiency requirements?"
✓ "Show me code compliance for boilers"
```

### Test Set 2: Problem-Solution

```
✓ "How do I fix low airflow issues?"
✓ "What causes heating system failures?"
✓ "How to troubleshoot air conditioning problems?"
✓ "What are common furnace repair issues?"
✓ "How to solve ductwork leakage?"
```

### Test Set 3: Specifications

```
✓ "What is the SEER rating for heat pumps?"
✓ "What is the BTU capacity of the furnace?"
✓ "What are the efficiency ratings mentioned?"
✓ "What is the CFM rating for air handlers?"
✓ "What are the performance specifications?"
```

### Test Set 4: Edge Cases

```
✓ "" (empty string)
✓ "What?" (ambiguous)
✓ "What is Python?" (non-HVAC)
✓ Very long queries (500+ words)
✓ Special characters: "12\" ducts"
```

## Neo4j Validation

```cypher
// 1. Count nodes by type
MATCH (n)
RETURN labels(n)[0] AS type, count(*) AS count
ORDER BY count DESC;

// Expected: Component ~50-200, Chunk ~100-300, Code ~30-100

// 2. Count relationships
MATCH ()-[r]->()
RETURN type(r) AS relationship, count(*) AS count
ORDER BY count DESC;

// Expected: MENTIONS dominant (50-70%), others distributed

// 3. Check embeddings
MATCH (c:Chunk)
WHERE c.embedding IS NOT NULL
RETURN count(*) AS chunks_with_embeddings;

// Expected: 100% of chunks have embeddings

// 4. Verify vector index
SHOW INDEXES
YIELD name, type, labelsOrTypes, properties
WHERE type = 'VECTOR';

// Expected: 1 vector index on Chunk.embedding, 1536 dimensions
```

## Performance Benchmarks

| Query Type          | Target Time | Expected Quality |
| ------------------- | ----------- | ---------------- |
| Exact code lookup   | < 100ms     | ⭐⭐⭐⭐⭐       |
| Component search    | < 200ms     | ⭐⭐⭐⭐         |
| Multi-hop queries   | < 500ms     | ⭐⭐⭐⭐         |
| Hybrid vector+graph | < 1s        | ⭐⭐⭐⭐⭐       |

---

# 🔧 Troubleshooting

## Issue 1: "Response appears incomplete, retrying..."

**Symptom:** Chatbot keeps retrying responses

**Cause:** `is_response_incomplete()` function too strict

**Solution:** Already fixed in latest version. Function now only flags truly incomplete responses (ending with ellipsis, conjunctions, or < 5 words without punctuation).

**Verify Fix:**

```python
# Check hvac_code_chatbot.py line ~2560
# Should have 4 lenient checks, not 3 strict ones
```

---

## Issue 2: No COMPLIES_WITH Relationships

**Symptom:** Graph query returns 0 compliance relationships

**Cause:** Pattern matching doesn't match your document's language

**Solution:** Add enhanced extraction patterns (see Optimization section above)

**Verify:**

```cypher
MATCH ()-[r:COMPLIES_WITH]->()
RETURN count(r);
// Should return > 0 after adding patterns
```

---

## Issue 3: Slow Query Performance

**Symptom:** Queries take > 5 seconds

**Cause:** Missing indexes on text fields

**Solution:**

```cypher
// Create text indexes
CREATE INDEX entity_text_index IF NOT EXISTS
FOR (e:Component) ON (e.text);

CREATE INDEX code_text_index IF NOT EXISTS
FOR (c:Code) ON (c.text);

CREATE INDEX chunk_index IF NOT EXISTS
FOR (c:Chunk) ON (c.chunk_index);
```

**Verify:**

```cypher
SHOW INDEXES;
// Should show 4+ indexes including vector index
```

---

## Issue 4: Vector Search Not Working

**Symptom:** Only graph results returned, no vector similarity

**Cause:** Vector index not created or embeddings missing

**Solution:**

```cypher
// 1. Check if chunks have embeddings
MATCH (c:Chunk)
WHERE c.embedding IS NULL
RETURN count(*);
// Should return 0

// 2. Recreate vector index if needed
DROP INDEX chunk_embedding_index IF EXISTS;

CALL db.index.vector.createNodeIndex(
  'chunk_embedding_index',
  'Chunk',
  'embedding',
  1536,
  'cosine'
);
```

---

## Issue 5: AttributeError for KNOWN_MANUFACTURERS

**Symptom:** `'HVACGraphRAG' object has no attribute 'KNOWN_MANUFACTURERS'`

**Cause:** Trying to access global constant as instance attribute

**Solution:** Already fixed. Global constants accessed directly, not via `self.`

---

## Issue 6: Transaction Timeout

**Symptom:** Neo4j transaction timeout during graph creation

**Cause:** Large batch operations

**Solution:**

```python
# Increase timeout in neo4j.conf
dbms.transaction.timeout=300s

# Or process in smaller batches
batch_size = 100
for i in range(0, len(entities), batch_size):
    batch = entities[i:i+batch_size]
    # Process batch
```

---

# 🗄️ Neo4j Validation Queries

## Basic Structure Validation

```cypher
// 1. Node count by type
MATCH (n)
RETURN labels(n)[0] AS node_type, count(*) AS count
ORDER BY count DESC;

// 2. Relationship count by type
MATCH ()-[r]->()
RETURN type(r) AS relationship_type, count(*) AS count
ORDER BY count DESC;

// 3. Total counts
MATCH (n)
WITH count(n) AS total_nodes
MATCH ()-[r]->()
WITH total_nodes, count(r) AS total_relationships
RETURN total_nodes, total_relationships,
       total_relationships * 1.0 / total_nodes AS avg_relationships_per_node;
```

## Data Quality Checks

```cypher
// 1. Orphaned nodes (no relationships)
MATCH (n)
WHERE NOT (n)--()
RETURN labels(n)[0] AS type, n.text AS text
LIMIT 20;

// 2. Chunks without embeddings
MATCH (c:Chunk)
WHERE c.embedding IS NULL
RETURN count(*) AS chunks_without_embeddings;

// 3. Entities without MENTIONS
MATCH (e)
WHERE NOT e:Chunk AND NOT e:Document
  AND NOT (e)-[:MENTIONS]->()
RETURN labels(e)[0] AS entity_type, e.text
LIMIT 20;

// 4. Confidence distribution
MATCH (n)
WHERE n.confidence IS NOT NULL
RETURN labels(n)[0] AS type,
       avg(n.confidence) AS avg_confidence,
       min(n.confidence) AS min_confidence,
       max(n.confidence) AS max_confidence
ORDER BY avg_confidence DESC;
```

## Graph Metrics

```cypher
// 1. Most connected nodes
MATCH (n)
WITH n, size((n)--()) AS degree
ORDER BY degree DESC
LIMIT 10
RETURN labels(n)[0] AS type, n.text, degree;

// 2. Relationship density by type
MATCH (n)
WHERE NOT n:Chunk AND NOT n:Document
WITH labels(n)[0] AS type, count(n) AS node_count
MATCH ()-[r]->()
WHERE type(r) IN ['COMPLIES_WITH', 'HAS_SPECIFICATION', 'MANUFACTURED_BY']
WITH type, node_count, type(r) AS rel_type, count(r) AS rel_count
RETURN type, rel_type, rel_count,
       rel_count * 1.0 / node_count AS density
ORDER BY density DESC;

// 3. Average path length (sample)
MATCH (a:Component), (b:Component)
WHERE id(a) < id(b)
WITH a, b LIMIT 100
MATCH path = shortestPath((a)-[*..5]-(b))
RETURN avg(length(path)) AS avg_path_length;
```

## Vector Index Validation

```cypher
// 1. Check vector index exists
SHOW INDEXES
YIELD name, type, labelsOrTypes, properties
WHERE type = 'VECTOR'
RETURN name, labelsOrTypes, properties;

// 2. Test vector similarity
MATCH (c:Chunk)
WHERE c.embedding IS NOT NULL
WITH c LIMIT 1
CALL db.index.vector.queryNodes(
  'chunk_embedding_index',
  10,
  c.embedding
) YIELD node, score
RETURN node.text AS similar_chunk, score
ORDER BY score DESC;
```

## Document Coverage

```cypher
// 1. Chunks per document
MATCH (d:Document)<-[:PART_OF]-(c:Chunk)
RETURN d.source AS document,
       count(c) AS chunk_count,
       avg(c.token_count) AS avg_tokens_per_chunk
ORDER BY chunk_count DESC;

// 2. Entities per document
MATCH (d:Document)<-[:MENTIONED_IN]-(e)
WHERE NOT e:Chunk
RETURN d.source AS document,
       count(DISTINCT e) AS entity_count,
       collect(DISTINCT labels(e)[0]) AS entity_types;

// 3. Coverage by entity type
MATCH (d:Document)<-[:MENTIONED_IN]-(e)
WHERE NOT e:Chunk
RETURN d.source AS document,
       labels(e)[0] AS entity_type,
       count(*) AS count
ORDER BY document, count DESC;
```

---

# 📋 Quick Reference

## Common Commands

```bash
# Start Neo4j
neo4j start

# Run Streamlit app
streamlit run app.py

# Clear graph database
# In Neo4j Browser:
MATCH (n) DETACH DELETE n

# Install dependencies
pip install -r requirements.txt
```

## Quick Cypher Queries

```cypher
// View all node types
MATCH (n) RETURN DISTINCT labels(n), count(*);

// View all relationships
MATCH ()-[r]->() RETURN DISTINCT type(r), count(*);

// Find components
MATCH (c:Component) RETURN c.text LIMIT 20;

// Find codes
MATCH (c:Code) RETURN c.text LIMIT 20;

// Find compliance relationships
MATCH (comp:Component)-[r:COMPLIES_WITH]->(code:Code)
RETURN comp.text, code.text LIMIT 10;
```

## Python Quick Start

```python
# Minimal example
from hvac_code_chatbot import HVACGraphRAG
from dotenv import load_dotenv
import os

load_dotenv()
graphrag = HVACGraphRAG(
    os.getenv("NEO4J_URI"),
    os.getenv("NEO4J_USER"),
    os.getenv("NEO4J_PASS")
)

# Query
answer = graphrag.query("What codes apply to air handlers?")
print(answer)
```

## Environment Variables

```bash
# Required
NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASS=your_password
OPENAI_API_KEY=sk-your-key

# Optional
CHUNK_SIZE=800
OVERLAP=100
SIMILARITY_THRESHOLD=0.75
TEMPERATURE=0.1
```

## File Structure

```
aichatbot-hvac/
├── hvac_code_chatbot.py      # Main system
├── load_hvac_documents.py    # Document processor
├── setup_hvac.py             # Setup utilities
├── app.py                    # Streamlit UI
├── requirements.txt          # Dependencies
├── .env                      # Environment variables
├── hvac_documents/           # PDF storage
│   └── your_pdfs_here.pdf
└── COMPREHENSIVE_GUIDE.md    # This file
```

---

# 📞 Support & Additional Resources

## Key Documentation Files

- **This Guide:** Complete reference
- **TROUBLESHOOTING_INCOMPLETE_RESPONSES.md:** Response retry issues
- **ENHANCED_RELATIONSHIP_EXTRACTION.md:** Add more relationships
- **OPTIMAL_QUERY_PATTERNS.md:** Query optimization for code documents

## Neo4j Resources

- Neo4j Browser: http://localhost:7474
- Neo4j Cypher Manual: https://neo4j.com/docs/cypher-manual/
- Vector Index Guide: https://neo4j.com/docs/cypher-manual/current/indexes-for-vector-search/

## OpenAI Resources

- API Documentation: https://platform.openai.com/docs
- Embeddings Guide: https://platform.openai.com/docs/guides/embeddings

---

**Last Updated:** October 8, 2025  
**Version:** 2.0  
**System Status:** ✅ Production Ready

---

_This guide consolidates all documentation into one comprehensive resource. For specific topics, refer to the Table of Contents at the top._
