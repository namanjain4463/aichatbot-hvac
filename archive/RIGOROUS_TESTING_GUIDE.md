# Rigorous Testing Guide: Hybrid Graph + Vector RAG System

## 🎯 Testing Objectives

1. **Verify hybrid search works** (graph + vector in parallel)
2. **Test all 6 Cypher intent patterns** (code, problem-solution, specs, brands, hierarchy, location)
3. **Validate vector similarity search** (cosine on embeddings)
4. **Test edge cases** (empty results, malformed queries, missing data)
5. **Performance testing** (query speed, result quality)
6. **Data quality checks** (graph structure, embeddings, relationships)

---

## 📋 Pre-Test Checklist

### 1. System Health Check

```bash
# Check Neo4j is running
# Open Neo4j Browser: http://localhost:7474
# Login with: neo4j / your_password
```

### 2. Verify Environment

```python
# In Neo4j Browser, run:
CALL db.ping()
// Should return: {success: true}

// Check Neo4j version (need 5.14.0+)
CALL dbms.components()
YIELD name, versions, edition
RETURN name, versions[0] as version, edition
```

### 3. Database State Before Upload

```cypher
// Clear any existing data (if needed)
MATCH (n) RETURN count(n) as node_count;
MATCH ()-[r]->() RETURN count(r) as relationship_count;

// Expected: 0 nodes, 0 relationships (clean start)
// OR: Existing data from previous runs
```

---

## 🧪 Test Suite 1: Knowledge Graph Creation

### Test 1.1: Upload PDF and Create Graph

**Action:**

1. Start Streamlit: `streamlit run .\hvac_code_chatbot.py`
2. Upload your HVAC PDF file
3. Click "Create Knowledge Graph"

**Expected Outcomes:**

- ✅ Progress indicators appear (9 steps with checkmarks)
- ✅ "Knowledge graph created successfully!" message
- ✅ No timeout errors (transaction fix working)
- ✅ No attribute errors (driver fix working)

**Validation Queries (in Neo4j Browser):**

```cypher
// 1. Count nodes by type
MATCH (n)
RETURN labels(n)[0] as node_type, count(*) as count
ORDER BY count DESC;

// Expected output:
// node_type      | count
// ----------------|-------
// Chunk          | 50-200   (depending on PDF size)
// Component      | 50-150   (HVAC components)
// Code           | 10-50    (Code sections)
// Problem        | 10-30    (Issues)
// Solution       | 10-30    (Fixes)
// Brand          | 5-20     (Manufacturers)
// Location       | 1-10     (Places like Atlanta)
// Specification  | 10-30    (Technical specs)

// 2. Count relationships by type
MATCH ()-[r]->()
RETURN type(r) as relationship_type, count(*) as count
ORDER BY count DESC;

// Expected output:
// relationship_type    | count
// ---------------------|-------
// MENTIONS             | 200-500  (chunk-to-entity)
// PART_OF              | 50-150   (chunk-to-document)
// COMPLIES_WITH        | 20-80    (component-to-code)
// RESOLVED_BY          | 10-30    (problem-to-solution)
// HAS_SPECIFICATION    | 10-40    (component-to-spec)
// MANUFACTURED_BY      | 10-30    (component-to-brand)
// INSTALLED_IN         | 5-20     (component-to-location)

// 3. Check vector index exists
SHOW INDEXES;

// Expected: chunk_embedding_index (ONLINE, VECTOR, Chunk.embedding)

// 4. Sample chunk with embedding
MATCH (c:Chunk)
RETURN c.text, c.chunk_index, c.token_count, size(c.embedding) as embedding_dim
LIMIT 3;

// Expected: embedding_dim = 1536 (text-embedding-3-small)

// 5. Sample MENTIONS relationships
MATCH (c:Chunk)-[r:MENTIONS]->(e)
RETURN c.chunk_index, type(r), labels(e)[0], e.text, r.confidence
ORDER BY r.confidence DESC
LIMIT 10;

// Expected: confidence >= 0.85 (our threshold)

// 6. Check document node
MATCH (d:Document)
RETURN d.source, d.total_chunks, d.created_at;

// Expected: Your PDF filename, chunk count, timestamp
```

**Edge Case 1.1a: Very Small PDF (< 1 page)**

- Upload a minimal PDF
- Expected: At least 1-2 chunks, fewer entities
- Validation: Graph should still be created without errors

**Edge Case 1.1b: Very Large PDF (> 100 pages)**

- Upload a large PDF
- Expected: May take 2-5 minutes, but no timeout (batching working)
- Validation: Progress updates appear for each batch

**Edge Case 1.1c: PDF with No HVAC Content**

- Upload a non-HVAC PDF (e.g., recipe book)
- Expected: Chunks created, but few/no HVAC entities
- Validation: Query should return "no relevant information found"

---

## 🧪 Test Suite 2: Hybrid Search - Intent Patterns

### Test 2.1: Code Compliance Queries (Intent 1)

**Query 1:** "What codes apply to air handlers?"

```
Expected Results:
✅ Graph results: Component-to-Code relationships (COMPLIES_WITH)
✅ Vector results: Chunks mentioning air handlers + codes
✅ Combined context includes both
✅ Answer cites specific code sections
```

**Query 2:** "What are the ASHRAE requirements for ventilation?"

```
Expected Results:
✅ Graph facts about ASHRAE codes
✅ Vector chunks with ventilation requirements
✅ Specific section numbers (e.g., ASHRAE 62.1)
```

**Edge Case 2.1a:** "What codes apply to xyz nonsense?"

```
Expected:
⚠️ Graph: No results (no matching entities)
⚠️ Vector: May find some chunks (semantic similarity)
✅ Answer: "No specific code information found for xyz nonsense"
```

**Validation Query:**

```cypher
// Check code compliance relationships
MATCH (c:Component)-[r:COMPLIES_WITH]->(code:Code)
WHERE toLower(c.text) CONTAINS 'air handler'
RETURN c.text, code.text, r.confidence, r.compliance_type
ORDER BY r.confidence DESC;

// Should return results if graph has this data
```

---

### Test 2.2: Problem-Solution Queries (Intent 2)

**Query 3:** "How do I fix low airflow issues?"

```
Expected Results:
✅ Graph results: Problem-to-Solution relationships (RESOLVED_BY)
✅ Vector results: Chunks about airflow problems
✅ Specific solutions (e.g., "clean filters", "check ducts")
```

**Query 4:** "What causes heating system failures?"

```
Expected Results:
✅ Problem entities related to heating
✅ Related solution entities
✅ Troubleshooting steps
```

**Edge Case 2.2a:** "How do I fix a problem that doesn't exist?"

```
Expected:
⚠️ Graph: No problem-solution matches
⚠️ Vector: May find general troubleshooting chunks
✅ Answer: Generic advice or "no specific solution found"
```

**Validation Query:**

```cypher
// Check problem-solution relationships
MATCH (p:Problem)-[r:RESOLVED_BY]->(s:Solution)
WHERE toLower(p.text) CONTAINS 'airflow'
   OR toLower(p.text) CONTAINS 'flow'
RETURN p.text, s.text, r.confidence
ORDER BY r.confidence DESC;
```

---

### Test 2.3: Specification Queries (Intent 3)

**Query 5:** "What are the efficiency ratings for heat pumps?"

```
Expected Results:
✅ Graph: Component-to-Specification relationships (HAS_SPECIFICATION)
✅ SEER/HSPF ratings
✅ Performance specifications
```

**Query 6:** "What is the BTU capacity of the furnace?"

```
Expected Results:
✅ BTU specifications from graph
✅ Supporting chunks with capacity info
```

**Edge Case 2.3a:** "What is the specification for quantum HVAC?"

```
Expected:
⚠️ No graph results (entity doesn't exist)
⚠️ Vector may return generic spec chunks
✅ Answer: "No specifications found for quantum HVAC"
```

**Validation Query:**

```cypher
// Check specification relationships
MATCH (c:Component)-[r:HAS_SPECIFICATION]->(s:Specification)
WHERE toLower(c.text) CONTAINS 'heat pump'
   OR toLower(c.text) CONTAINS 'furnace'
RETURN c.text, s.text, r.confidence, r.spec_type
ORDER BY r.confidence DESC;
```

---

### Test 2.4: Brand/Manufacturer Queries (Intent 4)

**Query 7:** "What Carrier equipment is mentioned?"

```
Expected Results:
✅ Graph: Brand relationships (MANUFACTURED_BY)
✅ Carrier components
✅ Model numbers if available
```

**Query 8:** "Who manufactures the air conditioning unit?"

```
Expected Results:
✅ Brand entities
✅ Component-to-Brand relationships
```

**Edge Case 2.4a:** "What equipment does FakeCompany make?"

```
Expected:
⚠️ No graph results (brand not in data)
⚠️ Vector search may return unrelated chunks
✅ Answer: "No information found about FakeCompany"
```

**Validation Query:**

```cypher
// Check brand relationships
MATCH (c:Component)-[r:MANUFACTURED_BY]->(b:Brand)
WHERE toLower(b.text) CONTAINS 'carrier'
RETURN c.text, b.text, r.confidence
ORDER BY r.confidence DESC;
```

---

### Test 2.5: System Hierarchy Queries (Intent 5)

**Query 9:** "What components are part of the HVAC system?"

```
Expected Results:
✅ Graph: PART_OF relationships
✅ Component hierarchy
✅ System-level organization
```

**Query 10:** "What is connected to the air handler?"

```
Expected Results:
✅ Related components
✅ Hierarchical relationships
```

**Validation Query:**

```cypher
// Check hierarchy relationships
MATCH (c:Component)-[r:PART_OF]->(system:Component)
RETURN c.text, system.text, r.confidence
ORDER BY r.confidence DESC
LIMIT 10;
```

---

### Test 2.6: Location Queries (Intent 6)

**Query 11:** "What equipment is in Atlanta?"

```
Expected Results:
✅ Graph: INSTALLED_IN relationships
✅ Location-specific components
✅ Geographic information
```

**Query 12:** "Where is the furnace installed?"

```
Expected Results:
✅ Location entities
✅ Installation information
```

**Edge Case 2.6a:** "What equipment is in Mars?"

```
Expected:
⚠️ No location match
⚠️ Answer: "No equipment found in Mars"
```

**Validation Query:**

```cypher
// Check location relationships
MATCH (c)-[r:INSTALLED_IN]->(l:Location)
WHERE toLower(l.text) CONTAINS 'atlanta'
RETURN c.text, l.text, r.confidence
ORDER BY r.confidence DESC;
```

---

## 🧪 Test Suite 3: Vector Search Functionality

### Test 3.1: Pure Vector Search

**Query 13:** "Explain how ductwork design affects efficiency"

```
Expected Results:
✅ Vector search finds relevant chunks
✅ Even if no specific graph relationships exist
✅ Semantic similarity working (embedding-based)
```

**Query 14:** "What is the purpose of air filters?"

```
Expected Results:
✅ Chunks about filters
✅ Cosine similarity > 0.7 (good match)
```

**Validation Query:**

```cypher
// Manual vector search test
MATCH (c:Chunk)
WHERE c.text CONTAINS 'ductwork' OR c.text CONTAINS 'filter'
RETURN c.text, c.chunk_index, size(c.embedding) as dim
LIMIT 5;

// Check if embeddings exist
MATCH (c:Chunk)
WHERE c.embedding IS NULL
RETURN count(c) as chunks_without_embeddings;
// Expected: 0 (all chunks should have embeddings)
```

---

### Test 3.2: Hybrid Result Fusion

**Query 15:** "What are SEER requirements for air conditioners?"

```
Expected Results:
✅ Graph facts prioritized (code requirements)
✅ Vector chunks as supporting context
✅ Both sources visible in answer
✅ Graph facts appear first in context
```

**Check in Response:**

- Look for "**KNOWLEDGE GRAPH FACTS**" section
- Look for "**RELEVANT CHUNKS**" section
- Graph facts should appear before vector chunks

---

## 🧪 Test Suite 4: Edge Cases & Error Handling

### Test 4.1: Empty/Null Queries

**Query 16:** "" (empty string)

```
Expected:
✅ No crash
✅ Graceful error message
✅ "Please provide a question"
```

**Query 17:** " " (whitespace only)

```
Expected:
✅ Handled gracefully
✅ No database queries executed
```

---

### Test 4.2: Very Long Queries

**Query 18:** (500+ word question)

```
"What are the detailed requirements specifications for commercial HVAC systems
including but not limited to air handlers, heat pumps, furnaces, boilers, and
all associated ductwork, ventilation systems, filtration equipment, and control
mechanisms that must comply with ASHRAE standards, IECC energy codes, IMC
mechanical codes, and all other applicable regulations for buildings in Atlanta,
Georgia, United States of America, with particular attention to efficiency
ratings, BTU capacities, SEER ratings, installation procedures, maintenance
schedules, troubleshooting protocols, and any manufacturer-specific
recommendations from brands like Carrier, Trane, Lennox, York, and Rheem?"

Expected:
✅ Query processed (truncated if needed)
✅ Embeddings generated successfully
✅ Results returned (may be slower)
```

---

### Test 4.3: Special Characters

**Query 19:** "What is the CFM rating for 12\" ducts?"

```
Expected:
✅ Special characters handled (quotes, inches symbol)
✅ No Cypher injection
✅ Results found
```

**Query 20:** "Air handler model #1234-ABC/XYZ"

```
Expected:
✅ Special chars (#, -, /) handled
✅ Entity matching works
```

---

### Test 4.4: Ambiguous Queries

**Query 21:** "Tell me about it"

```
Expected:
⚠️ No clear intent
⚠️ May return generic results
✅ No crash, some answer provided
```

**Query 22:** "What?"

```
Expected:
✅ Graceful handling
✅ Request for clarification
```

---

### Test 4.5: Non-HVAC Queries

**Query 23:** "What is the capital of France?"

```
Expected:
⚠️ No graph results (no HVAC entities)
⚠️ No relevant vector chunks
✅ Answer: "No HVAC-related information found" or generic LLM response
```

**Query 24:** "How do I bake a cake?"

```
Expected:
⚠️ No relevant data
✅ Polite response about system purpose
```

---

## 🧪 Test Suite 5: Performance & Quality

### Test 5.1: Query Response Time

**Benchmark Queries:**

```
Query 1: "What codes apply to furnaces?"
Query 2: "How do I fix low airflow?"
Query 3: "What is the SEER rating?"

Expected Response Times:
✅ < 3 seconds: Graph query + vector search
✅ < 2 seconds: LLM response generation
✅ Total < 5 seconds: End-to-end
```

**Measure in Streamlit:**

- Watch for delays in UI
- Check if "Thinking..." indicator appears
- Time from submit to answer

---

### Test 5.2: Result Quality

**Quality Metrics:**

1. **Relevance Score:**

   - Answer addresses the question: ✅ / ⚠️ / ❌
   - Citations are accurate: ✅ / ⚠️ / ❌
   - No hallucinations: ✅ / ⚠️ / ❌

2. **Graph vs Vector Balance:**

   - Uses graph facts when available: ✅
   - Falls back to vector when no graph data: ✅
   - Combines both appropriately: ✅

3. **Answer Completeness:**
   - Answers the question fully: ✅
   - Provides context: ✅
   - Cites sources: ✅

---

### Test 5.3: Consistency Test

**Query 25:** "What codes apply to air handlers?" (repeat 3 times)

```
Expected:
✅ Same graph facts returned each time
✅ Vector chunks may vary slightly (top-k selection)
✅ Core answer is consistent
```

---

## 🧪 Test Suite 6: Data Quality Checks

### Test 6.1: Entity Extraction Quality

**Validation Queries:**

```cypher
// 1. Check entity confidence distribution
MATCH (e)
WHERE e.confidence IS NOT NULL
RETURN labels(e)[0] as entity_type,
       avg(e.confidence) as avg_confidence,
       min(e.confidence) as min_confidence,
       max(e.confidence) as max_confidence,
       count(*) as count
ORDER BY avg_confidence DESC;

// Expected:
// - Tier 1 (graph) entities: avg_confidence >= 0.6
// - All entities: min_confidence >= 0.3

// 2. Check for duplicate entities
MATCH (e)
WHERE e.text IS NOT NULL
WITH e.text as entity_text, labels(e)[0] as type, count(*) as occurrences
WHERE occurrences > 1
RETURN entity_text, type, occurrences
ORDER BY occurrences DESC
LIMIT 10;

// Expected: Few duplicates (deduplication working)

// 3. Check chunk coverage
MATCH (c:Chunk)
RETURN min(c.chunk_index) as first_chunk,
       max(c.chunk_index) as last_chunk,
       count(*) as total_chunks;

// Expected: Continuous sequence (0, 1, 2, ...)

// 4. Check MENTIONS relationship quality
MATCH (chunk:Chunk)-[r:MENTIONS]->(entity)
RETURN labels(entity)[0] as entity_type,
       avg(r.confidence) as avg_confidence,
       count(*) as mention_count
ORDER BY mention_count DESC;

// Expected: confidence >= 0.85 (our threshold)

// 5. Check token counts
MATCH (c:Chunk)
RETURN avg(c.token_count) as avg_tokens,
       min(c.token_count) as min_tokens,
       max(c.token_count) as max_tokens;

// Expected: avg 500-800, min > 50, max < 1200
```

---

### Test 6.2: Relationship Quality

```cypher
// 1. Check relationship confidence scores
MATCH ()-[r]->()
WHERE r.confidence IS NOT NULL
RETURN type(r) as relationship_type,
       avg(r.confidence) as avg_confidence,
       count(*) as count
ORDER BY avg_confidence DESC;

// Expected: All >= 0.3 (minimum threshold)

// 2. Check for orphaned nodes (no relationships)
MATCH (n)
WHERE NOT (n)--()
RETURN labels(n)[0] as node_type, count(*) as orphan_count;

// Expected: Only Document and some Chunks might be orphans

// 3. Check bidirectional relationships (shouldn't exist)
MATCH (a)-[r1]->(b)-[r2]->(a)
WHERE type(r1) = type(r2)
RETURN type(r1) as relationship_type, count(*) as circular_count;

// Expected: 0 (no circular relationships of same type)
```

---

## 🧪 Test Suite 7: System Integration

### Test 7.1: Multiple PDFs (if supported)

**Action:**

1. Upload first PDF → Create graph
2. Upload second PDF → Create graph

**Expected:**

```
Option A: Graphs merged (cumulative)
Option B: Graph replaced (current implementation)

Check which behavior occurs and verify data consistency
```

---

### Test 7.2: System Restart Persistence

**Action:**

1. Create knowledge graph
2. Close Streamlit
3. Restart Streamlit
4. Query without re-creating graph

**Expected:**
✅ Graph persists in Neo4j
✅ Queries work immediately
✅ No need to recreate

---

## 📊 Test Results Template

```markdown
## Test Run: [Date] [Time]

### Environment

- Neo4j Version: \_\_\_
- Python Version: \_\_\_
- PDF Size: \_\_\_ pages
- PDF Topic: \_\_\_

### Test Results Summary

| Test Suite         | Pass | Fail | Notes |
| ------------------ | ---- | ---- | ----- |
| 1. Graph Creation  | ✅   | ❌   |       |
| 2. Intent Patterns | ✅   | ❌   |       |
| 3. Vector Search   | ✅   | ❌   |       |
| 4. Edge Cases      | ✅   | ❌   |       |
| 5. Performance     | ✅   | ❌   |       |
| 6. Data Quality    | ✅   | ❌   |       |
| 7. Integration     | ✅   | ❌   |       |

### Graph Statistics

- Total Nodes: \_\_\_
- Total Relationships: \_\_\_
- Chunk Count: \_\_\_
- Entity Count: \_\_\_
- Vector Index: ONLINE / OFFLINE

### Query Performance

- Avg Response Time: \_\_\_ seconds
- Graph Query Time: \_\_\_ seconds
- Vector Search Time: \_\_\_ seconds
- LLM Generation Time: \_\_\_ seconds

### Issues Found

1. [Issue description]
2. [Issue description]

### Recommendations

1. [Recommendation]
2. [Recommendation]
```

---

## 🎯 Success Criteria

✅ **Minimum Requirements:**

- Knowledge graph created without errors
- At least 3 intent patterns return results
- Vector search returns relevant chunks
- No attribute/timeout errors
- Queries complete in < 10 seconds

✅ **Optimal Performance:**

- All 6 intent patterns work
- Graph + Vector fusion visible in answers
- Response time < 5 seconds
- No duplicate entities
- Relationship confidence > 0.85

✅ **Production Ready:**

- All edge cases handled gracefully
- No crashes on malformed input
- Consistent results across queries
- High-quality answers with citations
- Graph structure optimized

---

## 🐛 Common Issues & Solutions

### Issue 1: "No graph results found"

**Cause:** PDF may not have entities matching query
**Solution:** Try broader queries, check entity extraction

### Issue 2: Vector search returns poor results

**Cause:** Embedding quality or query formulation
**Solution:** Rephrase query, check chunk text quality

### Issue 3: Slow queries (> 10 seconds)

**Cause:** Large graph, complex queries
**Solution:** Add indexes, optimize Cypher queries

### Issue 4: Attribute errors still occurring

**Cause:** Missed a `self.driver` reference
**Solution:** Search code for remaining issues

---

## 📝 Next Steps After Testing

Based on test results:

1. **If all tests pass:** System is production-ready ✅
2. **If some tests fail:** Debug specific intent patterns
3. **If edge cases fail:** Improve error handling
4. **If performance poor:** Optimize queries/indexing

Good luck with testing! 🚀
