# 📋 HVAC Chatbot - Quick Reference Cheat Sheet

**Version:** 2.0 | **Last Updated:** October 8, 2025

---

## ⚡ Quick Start (3 Steps)

```bash
# 1. Install & Configure
pip install openai tiktoken neo4j spacy langchain streamlit
python -m spacy download en_core_web_sm

# 2. Set .env file
NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASS=your_password
OPENAI_API_KEY=sk-your-key

# 3. Run
streamlit run app.py
```

---

## 🎯 Common Commands

### Start/Stop Services

```bash
# Start Neo4j
neo4j start

# Run Streamlit App
streamlit run app.py

# Open Neo4j Browser
# Navigate to: http://localhost:7474
```

### Clear Database

```cypher
MATCH (n) DETACH DELETE n
```

### Python Quick Query

```python
from hvac_code_chatbot import HVACGraphRAG
from dotenv import load_dotenv
import os

load_dotenv()
graphrag = HVACGraphRAG(
    os.getenv("NEO4J_URI"),
    os.getenv("NEO4J_USER"),
    os.getenv("NEO4J_PASS")
)

answer = graphrag.query("What codes apply to air handlers?")
print(answer)
```

---

## 🗺️ Graph Schema At-a-Glance

### Node Types (9)

- `:Document` - PDF source files
- `:Chunk` - Text chunks with 1536D embeddings
- `:Component` - HVAC equipment
- `:Code` - Code sections (IMC, ASHRAE, IECC)
- `:Brand` - Manufacturers (Carrier, Trane, etc.)
- `:Problem` - Issues/failures
- `:Solution` - Fixes/resolutions
- `:Location` - Geographic references
- `:Specification` - Technical specs (SEER, BTU, CFM)

### Relationship Types (10)

- `:MENTIONS` - Chunk ↔ Entity (bridge for hybrid search)
- `:COMPLIES_WITH` - Component → Code
- `:HAS_SPECIFICATION` - Component → Specification
- `:MANUFACTURED_BY` - Component → Brand
- `:RESOLVED_BY` - Problem → Solution
- `:PART_OF` - Component → System
- `:INSTALLED_IN` - Component → Location
- `:BELONGS_TO` - Component → Category
- `:FOLLOWS` - Code → Standard
- `:RELATED_TO` - Code → Code

---

## 🔍 Quick Cypher Queries

### View Structure

```cypher
// Count nodes by type
MATCH (n) RETURN labels(n)[0] AS type, count(*) AS count ORDER BY count DESC;

// Count relationships by type
MATCH ()-[r]->() RETURN type(r) AS rel_type, count(*) AS count ORDER BY count DESC;

// View all indexes
SHOW INDEXES;
```

### Find Data

```cypher
// Find components
MATCH (c:Component) RETURN c.text LIMIT 20;

// Find codes
MATCH (c:Code) RETURN c.text LIMIT 20;

// Find compliance relationships
MATCH (comp:Component)-[r:COMPLIES_WITH]->(code:Code)
RETURN comp.text, code.text, r.confidence LIMIT 10;

// Find most connected nodes
MATCH (n)
WITH n, size((n)--()) AS degree
ORDER BY degree DESC LIMIT 10
RETURN labels(n)[0] AS type, n.text, degree;
```

### Check Health

```cypher
// Chunks with embeddings
MATCH (c:Chunk) WHERE c.embedding IS NOT NULL
RETURN count(*) AS chunks_with_embeddings;

// Orphaned nodes
MATCH (n) WHERE NOT (n)--()
RETURN labels(n)[0] AS type, count(*) AS count;

// Check vector index
SHOW INDEXES YIELD name, type WHERE type = 'VECTOR' RETURN name;
```

---

## 🎯 Query Patterns

### Pattern 1: Code Compliance

```cypher
// Direct relationship
MATCH (comp:Component)-[r:COMPLIES_WITH]->(code:Code)
WHERE comp.text =~ '(?i).*air handler.*'
RETURN comp.text, code.text, r.confidence;

// Via MENTIONS bridge (if no direct relationship)
MATCH (comp:Component)-[:MENTIONS]->(chunk:Chunk)<-[:MENTIONS]-(code:Code)
WHERE comp.text =~ '(?i).*air handler.*'
RETURN comp.text, code.text, count(chunk) AS evidence;
```

### Pattern 2: Specifications

```cypher
MATCH (comp:Component)-[r:HAS_SPECIFICATION]->(spec:Specification)
WHERE spec.spec_type = 'SEER'
RETURN comp.text, spec.text, spec.spec_type;
```

### Pattern 3: Hierarchy

```cypher
MATCH (part:Component)-[r:PART_OF]->(system:Component)
RETURN part.text, system.text;
```

### Pattern 4: Co-occurrence

```cypher
// Find entities appearing together
MATCH (e1)-[:MENTIONS]->(chunk:Chunk)<-[:MENTIONS]-(e2)
WHERE e1.text =~ '(?i).*furnace.*'
RETURN labels(e2)[0] AS type, e2.text, count(chunk) AS co_occurrences
ORDER BY co_occurrences DESC LIMIT 10;
```

---

## 🔧 Quick Fixes

### Issue: No COMPLIES_WITH Relationships

```python
# Add enhanced extraction patterns (in hvac_code_chatbot.py)
compliance_patterns = [
    (r'(\w+(?:\s+\w+){0,2})\s+(?:must comply with|shall comply with)\s+(Section\s*\d+)', 'forward'),
    (r'(Section\s*\d+)\s+(?:applies to|governs)\s+(\w+(?:\s+\w+){0,2})', 'reverse'),
    (r'(\w+(?:\s+\w+){0,2})\s+(?:shall meet requirements of)\s+(Section\s*\d+)', 'forward'),
]
```

### Issue: Slow Queries

```cypher
// Create indexes
CREATE INDEX entity_text_index IF NOT EXISTS FOR (e:Component) ON (e.text);
CREATE INDEX code_text_index IF NOT EXISTS FOR (c:Code) ON (c.text);
CREATE INDEX chunk_index IF NOT EXISTS FOR (c:Chunk) ON (c.chunk_index);
```

### Issue: Vector Search Not Working

```cypher
// Recreate vector index
DROP INDEX chunk_embedding_index IF EXISTS;

CALL db.index.vector.createNodeIndex(
  'chunk_embedding_index',
  'Chunk',
  'embedding',
  1536,
  'cosine'
);
```

### Issue: "Response appears incomplete"

Already fixed in latest version. Check `is_response_incomplete()` function has 4 lenient checks.

---

## 📊 Validation Checklist

### ✅ Graph Created Successfully

```cypher
// Expected counts (approximate)
MATCH (n:Component) RETURN count(n);  // 50-200
MATCH (n:Chunk) RETURN count(n);      // 100-300
MATCH (n:Code) RETURN count(n);       // 30-100
MATCH ()-[r:MENTIONS]->() RETURN count(r);  // 50-70% of all relationships
```

### ✅ Embeddings Working

```cypher
// Should return same count as total chunks
MATCH (c:Chunk) WHERE c.embedding IS NOT NULL RETURN count(c);
```

### ✅ Vector Index Active

```cypher
// Should show 1 vector index with 1536 dimensions
SHOW INDEXES YIELD name, type, properties WHERE type = 'VECTOR';
```

### ✅ Hybrid Search Working

Test query: "What codes apply to air handlers?"

- Should return both graph relationships AND vector chunks
- Response time: < 5 seconds

---

## 🧪 Quick Test Queries

### Smoke Test (5 queries, 2 minutes)

```
1. "What codes apply to furnaces?"
2. "How to fix low airflow?"
3. "What is the SEER rating?"
4. "Who manufactures air handlers?"
5. "What components are in the system?"
```

**All pass?** ✅ System working!

### Edge Case Test

```
1. "" (empty query)
2. "What is Python?" (non-HVAC)
3. "What is the CFM for 12\" ducts?" (special chars)
```

**No crashes?** ✅ Error handling working!

---

## 📁 File Structure

```
aichatbot-hvac/
├── hvac_code_chatbot.py          # Main system (2900+ lines)
├── load_hvac_documents.py        # PDF processor
├── setup_hvac.py                 # Setup utilities
├── app.py                        # Streamlit UI
├── requirements.txt              # Dependencies
├── .env                          # Config (create this)
├── hvac_documents/               # Put PDFs here
│   └── your_hvac_code.pdf
├── COMPREHENSIVE_GUIDE.md        # Full documentation
└── CHEAT_SHEET.md               # This file
```

---

## 🎯 6 Intent Patterns

| Pattern              | Example Query              | Graph Query                      |
| -------------------- | -------------------------- | -------------------------------- |
| **Code Compliance**  | "What codes apply to X?"   | `COMPLIES_WITH` relationship     |
| **Problem-Solution** | "How to fix Y?"            | `RESOLVED_BY` relationship       |
| **Specifications**   | "What is the SEER rating?" | `HAS_SPECIFICATION` relationship |
| **Brand**            | "What Carrier equipment?"  | `MANUFACTURED_BY` relationship   |
| **Hierarchy**        | "What parts are in X?"     | `PART_OF` relationship           |
| **Location**         | "What's in Atlanta?"       | `INSTALLED_IN` relationship      |

---

## ⚙️ Environment Variables

```bash
# Required
NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASS=your_password
OPENAI_API_KEY=sk-your-key-here

# Optional (defaults shown)
CHUNK_SIZE=800
OVERLAP=100
SIMILARITY_THRESHOLD=0.75
TEMPERATURE=0.1
```

---

## 🚨 Common Errors & Solutions

| Error                                 | Cause              | Solution                        |
| ------------------------------------- | ------------------ | ------------------------------- |
| "Response appears incomplete"         | Fixed in v2.0      | Update to latest version        |
| "Connection refused 7687"             | Neo4j not running  | `neo4j start`                   |
| "No module named 'neo4j'"             | Missing dependency | `pip install neo4j`             |
| "Vector index not found"              | Index not created  | Run vector index creation query |
| "AttributeError: KNOWN_MANUFACTURERS" | Old bug            | Fixed in v2.0                   |
| Query timeout                         | Large graph        | Increase timeout or add indexes |

---

## 📈 Performance Targets

| Metric            | Target  | Good     | Needs Work |
| ----------------- | ------- | -------- | ---------- |
| Exact code lookup | < 100ms | < 200ms  | > 500ms    |
| Component search  | < 200ms | < 500ms  | > 1s       |
| Hybrid query      | < 1s    | < 2s     | > 5s       |
| Graph creation    | < 5 min | < 10 min | > 15 min   |

---

## 💡 Pro Tips

1. **Always create indexes** after building graph
2. **Use MENTIONS as bridge** when direct relationships don't exist
3. **Check Neo4j Browser** to visualize your graph
4. **Start with small PDFs** (< 50 pages) for testing
5. **Keep OpenAI API key** in .env, never commit it
6. **Clear graph between runs** if testing different documents
7. **Monitor response times** to identify slow queries

---

## 📞 Need More Detail?

- **Full Guide:** `COMPREHENSIVE_GUIDE.md` (9,500+ lines)
- **Troubleshooting:** `TROUBLESHOOTING_INCOMPLETE_RESPONSES.md`
- **Optimization:** `ENHANCED_RELATIONSHIP_EXTRACTION.md`
- **Query Patterns:** `OPTIMAL_QUERY_PATTERNS.md`

---

## 🎓 Quick Learning Path

**Day 1:** Install, setup, run first query  
**Day 2:** Understand graph schema, explore Neo4j Browser  
**Day 3:** Test all 6 intent patterns  
**Day 4:** Add custom extraction patterns  
**Day 5:** Optimize queries, create indexes

---

**🎯 Everything you need on one page!**

_Print this and keep it handy while working with the system._

---

**Last Updated:** October 8, 2025  
**Version:** 2.0  
**Status:** ✅ Production Ready
