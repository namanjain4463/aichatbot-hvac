# Quick Testing Checklist ✅

## Pre-Test Setup

- [ ] Neo4j is running (http://localhost:7474)
- [ ] Streamlit app started (`streamlit run .\hvac_code_chatbot.py`)
- [ ] PDF file ready to upload
- [ ] Neo4j Browser open for validation queries

---

## Phase 1: Graph Creation (10 min)

- [ ] Upload PDF successfully
- [ ] Click "Create Knowledge Graph"
- [ ] All 9 steps complete with checkmarks (✓✓✓✓✓✓✓✓✓)
- [ ] "Knowledge graph created successfully!" message appears
- [ ] No timeout errors
- [ ] No attribute errors

**Validation (in Neo4j Browser):**

```cypher
// Quick stats
MATCH (n) RETURN labels(n)[0] as type, count(*) ORDER BY count DESC;
MATCH ()-[r]->() RETURN type(r), count(*) ORDER BY count DESC;
SHOW INDEXES;
```

**Expected Results:**

- [ ] Chunk nodes: 50-200
- [ ] Entity nodes: 50-150 (Component, Code, Problem, etc.)
- [ ] MENTIONS relationships: 200-500
- [ ] Vector index: chunk_embedding_index (ONLINE)

---

## Phase 2: Basic Queries (5 min)

### Query 1: Code Compliance

- [ ] **Query:** "What codes apply to air handlers?"
- [ ] Graph results appear
- [ ] Vector chunks appear
- [ ] Answer cites specific codes
- [ ] Response time < 5 seconds

### Query 2: Problem-Solution

- [ ] **Query:** "How do I fix low airflow?"
- [ ] Problem-solution relationships found
- [ ] Practical solutions provided
- [ ] Response time < 5 seconds

### Query 3: Specifications

- [ ] **Query:** "What is the SEER rating for heat pumps?"
- [ ] Specification entities found
- [ ] Technical details in answer
- [ ] Response time < 5 seconds

---

## Phase 3: Intent Pattern Coverage (10 min)

- [ ] **Intent 1 (Code):** "What are ASHRAE requirements?" → Code entities
- [ ] **Intent 2 (Problem):** "What causes heating failures?" → Problem/Solution
- [ ] **Intent 3 (Specs):** "What is the BTU capacity?" → Specifications
- [ ] **Intent 4 (Brand):** "What Carrier equipment is mentioned?" → Brands
- [ ] **Intent 5 (Hierarchy):** "What components are in the HVAC system?" → PART_OF
- [ ] **Intent 6 (Location):** "What equipment is in Atlanta?" → Locations

**Score: \_\_\_ / 6 intent patterns working**

---

## Phase 4: Edge Cases (10 min)

### Empty/Invalid Queries

- [ ] **Query:** "" (empty) → No crash, graceful error
- [ ] **Query:** " " (whitespace) → Handled gracefully
- [ ] **Query:** "xyz nonsense" → No results, polite message

### Special Characters

- [ ] **Query:** "12\" duct CFM rating?" → Special chars handled
- [ ] **Query:** "Model #1234-ABC/XYZ" → No parsing errors

### Non-HVAC Queries

- [ ] **Query:** "What is the capital of France?" → No HVAC data returned
- [ ] **Query:** "How do I bake a cake?" → Appropriate response

### Very Long Query

- [ ] **Query:** (500+ words) → Processed without error

---

## Phase 5: Hybrid Search Validation (5 min)

### Check Result Fusion

- [ ] **Query:** "What codes apply to furnaces?"
- [ ] Answer shows "**KNOWLEDGE GRAPH FACTS**" section
- [ ] Answer shows "**RELEVANT CHUNKS**" section
- [ ] Graph facts appear BEFORE vector chunks
- [ ] Both sources contribute to answer

---

## Phase 6: Data Quality (10 min)

### Entity Quality

```cypher
// In Neo4j Browser:

// 1. Confidence distribution
MATCH (e) WHERE e.confidence IS NOT NULL
RETURN labels(e)[0], avg(e.confidence), min(e.confidence)
ORDER BY avg(e.confidence) DESC;
```

- [ ] Tier 1 entities: confidence >= 0.6
- [ ] Tier 2 entities: confidence >= 0.3

### Relationship Quality

```cypher
// 2. MENTIONS confidence
MATCH ()-[r:MENTIONS]->()
RETURN avg(r.confidence), min(r.confidence), count(r);
```

- [ ] MENTIONS avg confidence >= 0.85
- [ ] MENTIONS count: 200-500

### Chunk Quality

```cypher
// 3. Chunk tokens
MATCH (c:Chunk)
RETURN avg(c.token_count), min(c.token_count), max(c.token_count);
```

- [ ] Avg tokens: 500-800
- [ ] Min tokens: > 50
- [ ] Max tokens: < 1200

### Embedding Quality

```cypher
// 4. Embeddings exist
MATCH (c:Chunk) WHERE c.embedding IS NULL
RETURN count(c);
```

- [ ] Result: 0 (all chunks have embeddings)

```cypher
// 5. Embedding dimension
MATCH (c:Chunk)
RETURN size(c.embedding) as dim LIMIT 1;
```

- [ ] Dimension: 1536 (text-embedding-3-small)

---

## Phase 7: Performance (5 min)

### Response Time Benchmark

- [ ] Query 1: "What codes apply to furnaces?" → \_\_\_ seconds
- [ ] Query 2: "How do I fix low airflow?" → \_\_\_ seconds
- [ ] Query 3: "What is the SEER rating?" → \_\_\_ seconds

**Average response time: \_\_\_ seconds**

### Performance Criteria

- [ ] ✅ Excellent: < 3 seconds
- [ ] ⚠️ Acceptable: 3-5 seconds
- [ ] ❌ Needs improvement: > 5 seconds

---

## Phase 8: Consistency (5 min)

### Repeat Same Query 3 Times

**Query:** "What codes apply to air handlers?"

- [ ] **Run 1:** Results recorded
- [ ] **Run 2:** Same graph facts as Run 1
- [ ] **Run 3:** Same graph facts as Run 1
- [ ] Vector chunks may vary slightly (expected)
- [ ] Core answer is consistent across all runs

---

## Final Score

### Critical Tests (Must Pass)

- [ ] Graph creation works without errors (CRITICAL)
- [ ] At least 3 intent patterns work (CRITICAL)
- [ ] Vector search returns results (CRITICAL)
- [ ] No crashes on edge cases (CRITICAL)
- [ ] Response time < 10 seconds (CRITICAL)

**Critical tests passed: \_\_\_ / 5**

### Quality Tests (Should Pass)

- [ ] All 6 intent patterns work
- [ ] Graph + vector fusion visible
- [ ] High entity confidence (>0.6)
- [ ] High MENTIONS confidence (>0.85)
- [ ] Consistent results
- [ ] Response time < 5 seconds

**Quality tests passed: \_\_\_ / 6**

---

## Overall Assessment

**Total Score: \_\_\_ / 11 critical + quality tests**

### Rating:

- 11/11: 🌟 Production Ready
- 9-10/11: ✅ Excellent - minor tweaks needed
- 7-8/11: ⚠️ Good - some improvements needed
- 5-6/11: ⚠️ Acceptable - debugging required
- < 5/11: ❌ Needs significant work

---

## Issues Discovered

1. ***
2. ***
3. ***

## Next Steps

1. ***
2. ***
3. ***

---

## Test Completion

- **Tester:** ******\_\_\_******
- **Date:** ******\_\_\_******
- **Duration:** \_\_\_ minutes
- **PDF Used:** ******\_\_\_******
- **Overall Result:** PASS / FAIL / PARTIAL
