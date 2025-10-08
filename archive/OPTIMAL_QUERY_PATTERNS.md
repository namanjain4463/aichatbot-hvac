# 3. How to Query This Graph Structure Effectively

## 🎯 Your Graph's Unique Structure

Your graph is **code-document focused** with a **two-tier architecture**:

```
Tier 1 (Entities):                    Tier 2 (Content):
┌─────────────────┐                   ┌─────────────────┐
│ 67 Code nodes   │◄──────────────────│ 118 Chunks      │
│ 53 Components   │    307 MENTIONS   │                 │
│ 62 Specs        │─────────────────►│ (Your PDF text) │
└─────────────────┘                   └─────────────────┘
         │
         │ COMPLIES_WITH
         │ HAS_SPECIFICATION
         │ PART_OF
         ▼
```

**Key Insight:** The MENTIONS relationships are your **bridge** between semantic search (chunks) and graph queries (entities).

---

## 🔍 Query Pattern Library

### Pattern 1: **"What codes apply to X?"** (Component → Code)

#### Option A: Direct Graph Query (if COMPLIES_WITH exists)

```cypher
// Find codes that apply to air handlers
MATCH (comp:Component)-[r:COMPLIES_WITH]->(code:Code)
WHERE comp.text =~ '(?i).*air handler.*'
RETURN comp.text AS component,
       code.text AS code_section,
       r.confidence AS confidence
ORDER BY r.confidence DESC;
```

#### Option B: Via MENTIONS Bridge (works with your current graph)

```cypher
// Find codes mentioned in same chunks as air handlers
MATCH (comp:Component)-[:MENTIONS]->(chunk:Chunk)<-[:MENTIONS]-(code:Code)
WHERE comp.text =~ '(?i).*air handler.*'
RETURN comp.text AS component,
       code.text AS code_section,
       count(chunk) AS co_occurrences,
       collect(DISTINCT chunk.chunk_index) AS chunk_locations
ORDER BY co_occurrences DESC
LIMIT 10;
```

**Why this works:** If "air handler" and "Section 403.2" appear in the same chunk, they're likely related!

---

### Pattern 2: **"What does Section X require?"** (Code → Requirements)

#### Graph + Text Hybrid Query

```cypher
// Find all content related to Section 403.2
MATCH (code:Code)-[:MENTIONS]->(chunk:Chunk)
WHERE code.text =~ '(?i).*403\.2.*'
RETURN code.text AS code_section,
       collect(chunk.text) AS requirement_text,
       collect(chunk.chunk_index) AS chunk_indices
ORDER BY chunk_indices;
```

#### With Component Context

```cypher
// Section 403.2 + what components it affects
MATCH (code:Code)-[:MENTIONS]->(chunk:Chunk)<-[:MENTIONS]-(comp:Component)
WHERE code.text =~ '(?i).*403\.2.*'
RETURN code.text AS code_section,
       collect(DISTINCT comp.text) AS affected_components,
       collect(DISTINCT chunk.text) AS requirement_details;
```

---

### Pattern 3: **"What equipment is regulated by Code Y?"** (Reverse Query)

```cypher
// Find all components associated with IMC codes
MATCH (comp:Component)-[:MENTIONS]->(chunk:Chunk)<-[:MENTIONS]-(code:Code)
WHERE code.text =~ '(?i).*IMC.*'
RETURN code.text AS code_section,
       collect(DISTINCT comp.text) AS regulated_components,
       count(DISTINCT chunk) AS evidence_strength
ORDER BY evidence_strength DESC;
```

---

### Pattern 4: **"What are the specs for X?"** (Component → Specifications)

```cypher
// Find specifications for furnaces
MATCH (comp:Component)-[r:HAS_SPECIFICATION]->(spec:Specification)
WHERE comp.text =~ '(?i).*furnace.*'
RETURN comp.text AS component,
       spec.text AS specification,
       spec.spec_type AS spec_category,
       r.confidence AS confidence
ORDER BY confidence DESC;
```

**If HAS_SPECIFICATION is sparse, use MENTIONS bridge:**

```cypher
// Find specs mentioned with furnaces
MATCH (comp:Component)-[:MENTIONS]->(chunk:Chunk)
WHERE comp.text =~ '(?i).*furnace.*'
  AND chunk.text =~ '(?i).*(BTU|AFUE|SEER|CFM|\d+\s*ton).*'
RETURN comp.text AS component,
       chunk.text AS specification_text,
       chunk.chunk_index AS location;
```

---

### Pattern 5: **"Multi-hop Relationship Queries"**

#### Find code compliance chain: Component → Code → Related Components

```cypher
// What OTHER components share codes with air handlers?
MATCH (comp1:Component)-[:MENTIONS]->(chunk1:Chunk)<-[:MENTIONS]-(code:Code)
      -[:MENTIONS]->(chunk2:Chunk)<-[:MENTIONS]-(comp2:Component)
WHERE comp1.text =~ '(?i).*air handler.*'
  AND comp1 <> comp2
RETURN comp1.text AS component_1,
       code.text AS shared_code,
       comp2.text AS related_component,
       count(DISTINCT chunk2) AS relationship_strength
ORDER BY relationship_strength DESC
LIMIT 15;
```

---

## 🚀 Optimized Hybrid Search Patterns

Your system has **vector embeddings** (1536 dims) AND **graph structure**. Combine them!

### Hybrid Pattern: Semantic Search → Graph Enrichment

```python
# In your query_graph_with_natural_language method:

def query_hybrid_code_focused(self, user_question):
    """
    1. Vector search for relevant chunks (semantic)
    2. Find entities mentioned in those chunks (graph)
    3. Expand via relationships (graph)
    4. Return enriched context (hybrid)
    """

    # Step 1: Semantic search for chunks
    question_embedding = self.create_embedding(user_question)

    with self.driver.session() as session:
        # Find top 5 most relevant chunks
        vector_results = session.run("""
            MATCH (chunk:Chunk)
            WHERE chunk.embedding IS NOT NULL
            WITH chunk,
                 gds.similarity.cosine(chunk.embedding, $embedding) AS similarity
            WHERE similarity > 0.75
            ORDER BY similarity DESC
            LIMIT 5
            RETURN chunk.chunk_index AS chunk_index,
                   chunk.text AS chunk_text,
                   similarity
        """, embedding=question_embedding)

        top_chunks = [dict(record) for record in vector_results]
        chunk_indices = [c['chunk_index'] for c in top_chunks]

        # Step 2: Find entities mentioned in those chunks
        graph_expansion = session.run("""
            MATCH (chunk:Chunk)<-[:MENTIONS]-(entity)
            WHERE chunk.chunk_index IN $chunk_indices

            // Expand to find related entities
            OPTIONAL MATCH (entity)-[r]->(related)
            WHERE type(r) IN ['COMPLIES_WITH', 'HAS_SPECIFICATION', 'PART_OF']

            RETURN DISTINCT
                   labels(entity)[0] AS entity_type,
                   entity.text AS entity_text,
                   collect(DISTINCT {
                       rel_type: type(r),
                       target: related.text,
                       target_type: labels(related)[0]
                   }) AS relationships
        """, chunk_indices=chunk_indices)

        entities = [dict(record) for record in graph_expansion]

    # Step 3: Combine for context
    context = self._build_hybrid_context(top_chunks, entities)

    # Step 4: Send to LLM
    return self._generate_answer_with_context(user_question, context)
```

---

## 📊 Performance Optimization Queries

### Create Indexes (Run Once)

```cypher
// Index on entity text for faster regex matching
CREATE INDEX entity_text_index IF NOT EXISTS
FOR (e:Component) ON (e.text);

CREATE INDEX code_text_index IF NOT EXISTS
FOR (c:Code) ON (c.text);

// Index on chunk indices
CREATE INDEX chunk_index IF NOT EXISTS
FOR (c:Chunk) ON (c.chunk_index);

// Index on relationship confidence
CREATE INDEX rel_confidence IF NOT EXISTS
FOR ()-[r:COMPLIES_WITH]-() ON (r.confidence);
```

### Check Query Performance

```cypher
// Profile a query to see execution plan
PROFILE
MATCH (comp:Component)-[:MENTIONS]->(chunk:Chunk)<-[:MENTIONS]-(code:Code)
WHERE comp.text =~ '(?i).*furnace.*'
RETURN comp.text, code.text, count(chunk) AS strength
ORDER BY strength DESC;
```

Look for:

- ✅ "NodeIndexSeek" (using indexes - GOOD)
- ❌ "AllNodesScan" (scanning everything - BAD)

---

## 🎯 Query Strategy for Code Documents

Your PDF is **regulatory text**, so use these strategies:

### Strategy 1: **Exact Code Section Lookups**

```cypher
// Fast: Direct code node lookup
MATCH (code:Code {text: "Section 403.2"})
      -[:MENTIONS]->(chunk:Chunk)
RETURN chunk.text AS requirement_text;
```

### Strategy 2: **Fuzzy Code References**

```cypher
// Flexible: Regex for variations
MATCH (code:Code)-[:MENTIONS]->(chunk:Chunk)
WHERE code.text =~ '(?i).*(403\.2|IMC 403|Section 403).*'
RETURN DISTINCT chunk.text;
```

### Strategy 3: **Component-Centric Queries**

```cypher
// "Tell me everything about air handlers"
MATCH (comp:Component)-[:MENTIONS]->(chunk:Chunk)
WHERE comp.text =~ '(?i).*air handler.*'

OPTIONAL MATCH (comp)-[:MENTIONS]->(same_chunk:Chunk)
              <-[:MENTIONS]-(code:Code)

OPTIONAL MATCH (comp)-[:HAS_SPECIFICATION]->(spec:Specification)

RETURN comp.text AS component,
       collect(DISTINCT chunk.text) AS context_chunks,
       collect(DISTINCT code.text) AS applicable_codes,
       collect(DISTINCT spec.text) AS specifications;
```

### Strategy 4: **Cross-Reference Queries**

```cypher
// "What components and codes appear together?"
MATCH (comp:Component)-[:MENTIONS]->(chunk:Chunk)
      <-[:MENTIONS]-(code:Code)
RETURN comp.text AS component,
       code.text AS code_section,
       count(chunk) AS co_occurrence_count
ORDER BY co_occurrence_count DESC
LIMIT 20;
```

---

## 🧪 Test Your Queries

Try these on your actual graph:

```cypher
// 1. Find most connected components
MATCH (comp:Component)-[r:MENTIONS]->()
RETURN comp.text, count(r) AS connection_count
ORDER BY connection_count DESC
LIMIT 10;

// 2. Find codes with most component associations
MATCH (code:Code)-[:MENTIONS]->(chunk:Chunk)
      <-[:MENTIONS]-(comp:Component)
RETURN code.text AS code_section,
       count(DISTINCT comp) AS component_count,
       collect(DISTINCT comp.text) AS components
ORDER BY component_count DESC
LIMIT 10;

// 3. Find isolated entities (might need better extraction)
MATCH (entity)
WHERE NOT (entity)-[:MENTIONS]->(:Chunk)
RETURN labels(entity)[0] AS entity_type,
       entity.text AS text,
       count(*) AS isolated_count;
```

---

## 💡 Query Patterns Specific to Your Graph

Since you have **307 MENTIONS** relationships (71% of all relationships), **leverage them**!

### Anti-Pattern ❌

```cypher
// DON'T: Ignore chunks, query entities directly
MATCH (comp:Component)-[r:COMPLIES_WITH]->(code:Code)
RETURN comp.text, code.text;
// Returns 0 results (you have 0 COMPLIES_WITH)
```

### Good Pattern ✅

```cypher
// DO: Use MENTIONS as bridge to find implicit relationships
MATCH (comp:Component)-[:MENTIONS]->(chunk:Chunk)
      <-[:MENTIONS]-(code:Code)
WHERE chunk.text =~ '(?i).*(comply|complies|shall meet|subject to).*'
RETURN comp.text AS component,
       code.text AS code_section,
       chunk.text AS compliance_text,
       chunk.chunk_index AS evidence_location;
```

**Why?** Your COMPLIES_WITH relationships might not exist explicitly, but the **text evidence** in chunks proves the compliance relationship!

---

## 🔧 Integration with Your Chatbot

Modify `query_graph_with_natural_language` to use optimal patterns:

```python
def query_graph_with_natural_language(self, question):
    # Detect intent
    if re.search(r'(code|section|imc|ashrae|iecc)\s+\d+', question, re.I):
        intent = 'code_lookup'
    elif re.search(r'(spec|rating|capacity|efficiency|btu|seer)', question, re.I):
        intent = 'specification_query'
    elif re.search(r'(component|equipment|system|furnace|handler)', question, re.I):
        intent = 'component_query'
    else:
        intent = 'general'

    # Route to optimal query pattern
    if intent == 'code_lookup':
        return self._code_section_query(question)
    elif intent == 'specification_query':
        return self._specification_query(question)
    elif intent == 'component_query':
        return self._component_centric_query(question)
    else:
        return self._hybrid_search(question)

def _code_section_query(self, question):
    """Optimized for 'What does Section X say?' queries"""
    # Extract code reference
    code_match = re.search(r'((?:Section|IMC|IECC|ASHRAE)\s*\d+(?:\.\d+)*)', question, re.I)
    if code_match:
        code_ref = code_match.group(1)

        with self.driver.session() as session:
            result = session.run("""
                MATCH (code:Code)-[:MENTIONS]->(chunk:Chunk)
                WHERE code.text =~ $code_pattern
                OPTIONAL MATCH (chunk)<-[:MENTIONS]-(comp:Component)
                RETURN code.text AS code_section,
                       collect(DISTINCT chunk.text) AS requirements,
                       collect(DISTINCT comp.text) AS affected_components
                LIMIT 1
            """, code_pattern=f'(?i).*{re.escape(code_ref)}.*')

            return self._format_code_response(result.single())

def _component_centric_query(self, question):
    """Optimized for 'Tell me about X equipment' queries"""
    # Extract component name
    # Use MENTIONS bridge to find all related info
    pass  # Implement using Pattern 3 from above
```

---

## 📈 Expected Query Performance

With proper indexes and these patterns:

| Query Type          | Response Time | Result Quality |
| ------------------- | ------------- | -------------- |
| Exact code lookup   | < 100ms       | ⭐⭐⭐⭐⭐     |
| Component search    | < 200ms       | ⭐⭐⭐⭐       |
| Multi-hop queries   | < 500ms       | ⭐⭐⭐⭐       |
| Hybrid vector+graph | < 1s          | ⭐⭐⭐⭐⭐     |

---

## 🎓 Summary

**Your graph is optimized for code documents!**

✅ **Use MENTIONS as your primary bridge** (you have 307 of them)
✅ **Combine vector search (chunks) + graph queries (entities)** for best results
✅ **Create indexes** on entity text and chunk indices
✅ **Use regex patterns** for flexible code section matching
✅ **Leverage co-occurrence** (entities in same chunks are likely related)

**Next Steps:**

1. Add enhanced relationship extraction (see previous doc)
2. Create indexes (run the CREATE INDEX queries above)
3. Test query patterns on your actual data
4. Integrate optimal patterns into chatbot code

Your graph structure is **excellent for code documents** - now optimize your queries to take full advantage! 🚀
