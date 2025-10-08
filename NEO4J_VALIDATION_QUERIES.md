# Neo4j Validation Queries Cheat Sheet

## 🚀 Quick Reference for Testing Hybrid Graph + Vector RAG

Copy-paste these queries into Neo4j Browser (http://localhost:7474) to validate your knowledge graph.

---

## 📊 1. BASIC STATISTICS

### Count All Nodes

```cypher
MATCH (n)
RETURN count(n) as total_nodes;
```

### Count Nodes by Type

```cypher
MATCH (n)
RETURN labels(n)[0] as node_type, count(*) as count
ORDER BY count DESC;
```

**Expected Output:**

- Chunk: 50-200
- Component: 50-150
- Code: 10-50
- Problem: 10-30
- Solution: 10-30
- Brand: 5-20
- Location: 1-10
- Specification: 10-30

### Count All Relationships

```cypher
MATCH ()-[r]->()
RETURN count(r) as total_relationships;
```

### Count Relationships by Type

```cypher
MATCH ()-[r]->()
RETURN type(r) as relationship_type, count(*) as count
ORDER BY count DESC;
```

**Expected Output:**

- MENTIONS: 200-500
- PART_OF: 50-150
- COMPLIES_WITH: 20-80
- RESOLVED_BY: 10-30
- HAS_SPECIFICATION: 10-40
- MANUFACTURED_BY: 10-30
- INSTALLED_IN: 5-20

---

## 🔍 2. DATA QUALITY CHECKS

### Check for Nodes Without Labels

```cypher
MATCH (n)
WHERE size(labels(n)) = 0
RETURN count(n) as unlabeled_nodes;
```

**Expected:** 0 (all nodes should have labels)

### Check for Orphaned Nodes (No Relationships)

```cypher
MATCH (n)
WHERE NOT (n)--()
RETURN labels(n)[0] as node_type, count(*) as orphan_count
ORDER BY orphan_count DESC;
```

**Expected:** Only Document and possibly some Chunks

### Check Entity Confidence Scores

```cypher
MATCH (e)
WHERE e.confidence IS NOT NULL
RETURN labels(e)[0] as entity_type,
       avg(e.confidence) as avg_confidence,
       min(e.confidence) as min_confidence,
       max(e.confidence) as max_confidence,
       count(*) as count
ORDER BY avg_confidence DESC;
```

**Expected:**

- Tier 1 (graph) entities: avg >= 0.6
- Tier 2 (chunks) entities: avg >= 0.3

### Check Relationship Confidence Scores

```cypher
MATCH ()-[r]->()
WHERE r.confidence IS NOT NULL
RETURN type(r) as relationship_type,
       avg(r.confidence) as avg_confidence,
       min(r.confidence) as min_confidence,
       count(*) as count
ORDER BY avg_confidence DESC;
```

**Expected:** MENTIONS avg >= 0.85

### Check for Duplicate Entities

```cypher
MATCH (e)
WHERE e.text IS NOT NULL
WITH e.text as entity_text, labels(e)[0] as type, collect(e) as nodes
WHERE size(nodes) > 1
RETURN entity_text, type, size(nodes) as duplicate_count
ORDER BY duplicate_count DESC
LIMIT 20;
```

**Expected:** Few duplicates (deduplication should be working)

---

## 📦 3. CHUNK VALIDATION

### Count Chunks

```cypher
MATCH (c:Chunk)
RETURN count(c) as total_chunks;
```

### Check Chunk Index Sequence

```cypher
MATCH (c:Chunk)
RETURN min(c.chunk_index) as first_chunk,
       max(c.chunk_index) as last_chunk,
       count(*) as total_chunks;
```

**Expected:** Continuous sequence from 0 to N

### Check Chunk Token Counts

```cypher
MATCH (c:Chunk)
RETURN avg(c.token_count) as avg_tokens,
       min(c.token_count) as min_tokens,
       max(c.token_count) as max_tokens;
```

**Expected:** avg 500-800, min > 50, max < 1200

### Sample Chunks

```cypher
MATCH (c:Chunk)
RETURN c.chunk_index, c.text, c.token_count, c.source
ORDER BY c.chunk_index
LIMIT 5;
```

### Check for Chunks Without Text

```cypher
MATCH (c:Chunk)
WHERE c.text IS NULL OR c.text = ""
RETURN count(c) as empty_chunks;
```

**Expected:** 0

---

## 🎯 4. VECTOR INDEX VALIDATION

### List All Indexes

```cypher
SHOW INDEXES;
```

**Expected:** chunk_embedding_index (ONLINE, VECTOR)

### Check Vector Index Details

```cypher
SHOW INDEXES
YIELD name, type, state, populationPercent, entityType, labelsOrTypes, properties
WHERE type = "VECTOR"
RETURN name, state, populationPercent, labelsOrTypes, properties;
```

**Expected:**

- name: chunk_embedding_index
- state: ONLINE
- populationPercent: 100.0
- labelsOrTypes: ["Chunk"]
- properties: ["embedding"]

### Check Embeddings Exist

```cypher
MATCH (c:Chunk)
WHERE c.embedding IS NULL
RETURN count(c) as chunks_without_embeddings;
```

**Expected:** 0 (all chunks should have embeddings)

### Check Embedding Dimensions

```cypher
MATCH (c:Chunk)
RETURN size(c.embedding) as embedding_dimension
LIMIT 1;
```

**Expected:** 1536 (text-embedding-3-small)

### Sample Embedding

```cypher
MATCH (c:Chunk)
RETURN c.chunk_index,
       c.text as text_preview,
       size(c.embedding) as dim,
       c.embedding[0..5] as first_5_values
LIMIT 3;
```

### Test Vector Search Manually

```cypher
// Note: You need an actual embedding vector for this
// This is a placeholder - use Python to generate a real embedding
CALL db.index.vector.queryNodes(
    'chunk_embedding_index',
    5,
    [0.1, -0.2, 0.3, ...] // Replace with real 1536-dim embedding
)
YIELD node, score
RETURN node.text, node.chunk_index, score
ORDER BY score DESC;
```

---

## 🔗 5. RELATIONSHIP VALIDATION

### Check MENTIONS Relationships

```cypher
MATCH (c:Chunk)-[r:MENTIONS]->(e)
RETURN labels(e)[0] as entity_type,
       avg(r.confidence) as avg_confidence,
       count(*) as mention_count
ORDER BY mention_count DESC;
```

**Expected:** confidence >= 0.85

### Sample MENTIONS Relationships

```cypher
MATCH (c:Chunk)-[r:MENTIONS]->(e)
RETURN c.chunk_index,
       labels(e)[0] as entity_type,
       e.text as entity,
       r.confidence
ORDER BY r.confidence DESC
LIMIT 10;
```

### Check COMPLIES_WITH Relationships

```cypher
MATCH (comp)-[r:COMPLIES_WITH]->(code:Code)
RETURN labels(comp)[0] as component_type,
       comp.text as component,
       code.text as code_section,
       r.confidence,
       r.compliance_type
ORDER BY r.confidence DESC
LIMIT 10;
```

### Check RESOLVED_BY Relationships

```cypher
MATCH (p:Problem)-[r:RESOLVED_BY]->(s:Solution)
RETURN p.text as problem,
       s.text as solution,
       r.confidence
ORDER BY r.confidence DESC
LIMIT 10;
```

### Check HAS_SPECIFICATION Relationships

```cypher
MATCH (c:Component)-[r:HAS_SPECIFICATION]->(s:Specification)
RETURN c.text as component,
       s.text as specification,
       r.confidence,
       r.spec_type
ORDER BY r.confidence DESC
LIMIT 10;
```

### Check MANUFACTURED_BY Relationships

```cypher
MATCH (c:Component)-[r:MANUFACTURED_BY]->(b:Brand)
RETURN c.text as component,
       b.text as brand,
       r.confidence
ORDER BY r.confidence DESC
LIMIT 10;
```

### Check PART_OF Relationships

```cypher
MATCH (c:Component)-[r:PART_OF]->(system:Component)
RETURN c.text as component,
       system.text as system,
       r.confidence
ORDER BY r.confidence DESC
LIMIT 10;
```

### Check INSTALLED_IN Relationships

```cypher
MATCH (c)-[r:INSTALLED_IN]->(l:Location)
RETURN c.text as component,
       l.text as location,
       r.confidence
ORDER BY r.confidence DESC
LIMIT 10;
```

### Check for Circular Relationships

```cypher
MATCH (a)-[r1]->(b)-[r2]->(a)
WHERE type(r1) = type(r2)
RETURN type(r1) as relationship_type,
       a.text as node_a,
       b.text as node_b,
       count(*) as circular_count;
```

**Expected:** 0 (no circular relationships of same type)

---

## 📄 6. DOCUMENT VALIDATION

### Check Document Node

```cypher
MATCH (d:Document)
RETURN d.source,
       d.total_chunks,
       d.created_at,
       d.file_size,
       d.processing_time;
```

### Check Chunk-to-Document Relationships

```cypher
MATCH (c:Chunk)-[r:PART_OF]->(d:Document)
RETURN d.source,
       count(c) as chunk_count,
       min(c.chunk_index) as first_chunk,
       max(c.chunk_index) as last_chunk;
```

---

## 🎨 7. ENTITY-SPECIFIC QUERIES

### Find All Components

```cypher
MATCH (c:Component)
RETURN c.text, c.hvac_type, c.confidence
ORDER BY c.confidence DESC
LIMIT 20;
```

### Find All Codes

```cypher
MATCH (code:Code)
RETURN code.text, code.code_type, code.confidence
ORDER BY code.confidence DESC;
```

### Find All Problems

```cypher
MATCH (p:Problem)
RETURN p.text, p.severity, p.confidence
ORDER BY p.confidence DESC;
```

### Find All Solutions

```cypher
MATCH (s:Solution)
RETURN s.text, s.solution_type, s.confidence
ORDER BY s.confidence DESC;
```

### Find All Brands

```cypher
MATCH (b:Brand)
RETURN b.text, b.confidence
ORDER BY b.confidence DESC;
```

### Find All Locations

```cypher
MATCH (l:Location)
RETURN l.text, l.location_type, l.confidence
ORDER BY l.confidence DESC;
```

### Find All Specifications

```cypher
MATCH (s:Specification)
RETURN s.text, s.spec_type, s.confidence
ORDER BY s.confidence DESC
LIMIT 20;
```

---

## 🔎 8. SEARCH & PATTERN QUERIES

### Search by Entity Text

```cypher
// Find entities containing "air handler"
MATCH (e)
WHERE toLower(e.text) CONTAINS 'air handler'
RETURN labels(e)[0] as type, e.text, e.confidence
ORDER BY e.confidence DESC;
```

### Find Entities with High Confidence

```cypher
MATCH (e)
WHERE e.confidence > 0.9
RETURN labels(e)[0] as type, e.text, e.confidence
ORDER BY e.confidence DESC
LIMIT 20;
```

### Find Complex Patterns

```cypher
// Find components with codes AND specifications
MATCH (c:Component)-[r1:COMPLIES_WITH]->(code:Code),
      (c)-[r2:HAS_SPECIFICATION]->(spec:Specification)
RETURN c.text as component,
       code.text as code_section,
       spec.text as specification,
       r1.confidence as code_conf,
       r2.confidence as spec_conf
ORDER BY r1.confidence DESC
LIMIT 10;
```

### Find Problem-Solution Chains

```cypher
MATCH (p:Problem)-[r:RESOLVED_BY]->(s:Solution)
RETURN p.text as problem,
       s.text as solution,
       r.confidence
ORDER BY r.confidence DESC
LIMIT 10;
```

### Find Brand Equipment with Locations

```cypher
MATCH (c:Component)-[r1:MANUFACTURED_BY]->(b:Brand),
      (c)-[r2:INSTALLED_IN]->(l:Location)
RETURN c.text as component,
       b.text as brand,
       l.text as location
ORDER BY r1.confidence DESC
LIMIT 10;
```

---

## 🗑️ 9. CLEANUP QUERIES (USE WITH CAUTION)

### Delete All Nodes and Relationships

```cypher
// WARNING: This deletes EVERYTHING!
MATCH (n)
DETACH DELETE n;
```

### Delete Only Chunks and Their Relationships

```cypher
MATCH (c:Chunk)
DETACH DELETE c;
```

### Delete Only Entities (Keep Chunks)

```cypher
MATCH (e)
WHERE NOT e:Chunk AND NOT e:Document
DETACH DELETE e;
```

### Delete Specific Relationship Type

```cypher
MATCH ()-[r:MENTIONS]->()
DELETE r;
```

### Delete Vector Index

```cypher
DROP INDEX chunk_embedding_index;
```

---

## 📈 10. PERFORMANCE QUERIES

### Find Nodes with Most Relationships

```cypher
MATCH (n)
RETURN labels(n)[0] as node_type,
       n.text,
       size((n)--()) as relationship_count
ORDER BY relationship_count DESC
LIMIT 10;
```

### Find Largest Chunks

```cypher
MATCH (c:Chunk)
RETURN c.chunk_index,
       c.token_count,
       length(c.text) as char_count
ORDER BY c.token_count DESC
LIMIT 10;
```

### Database Size and Stats

```cypher
CALL apoc.meta.stats()
YIELD nodeCount, relCount, labelCount, relTypeCount
RETURN nodeCount, relCount, labelCount, relTypeCount;
```

### Check Memory Usage (if APOC available)

```cypher
CALL dbms.queryJmx("org.neo4j:instance=kernel#0,name=Page cache")
YIELD attributes
RETURN attributes.BytesRead, attributes.BytesWritten;
```

---

## 🎯 11. HYBRID SEARCH SIMULATION

### Simulate Graph Query (Code Intent)

```cypher
// What codes apply to air handlers?
MATCH (comp)-[r:COMPLIES_WITH]->(code:Code)
WHERE toLower(comp.text) CONTAINS 'air handler'
RETURN comp.text as component,
       code.text as code_section,
       r.confidence
ORDER BY r.confidence DESC
LIMIT 5;
```

### Simulate Graph Query (Problem-Solution Intent)

```cypher
// How to fix low airflow?
MATCH (p:Problem)-[r:RESOLVED_BY]->(s:Solution)
WHERE toLower(p.text) CONTAINS 'airflow'
   OR toLower(p.text) CONTAINS 'flow'
RETURN p.text as problem,
       s.text as solution,
       r.confidence
ORDER BY r.confidence DESC
LIMIT 5;
```

### Simulate Chunk Search (Text Contains)

```cypher
// Find chunks about "SEER rating"
MATCH (c:Chunk)
WHERE toLower(c.text) CONTAINS 'seer'
RETURN c.chunk_index,
       c.text,
       c.token_count
ORDER BY c.chunk_index
LIMIT 5;
```

---

## 💡 QUICK TIPS

1. **Use `LIMIT`** to avoid overwhelming results
2. **Check indexes first** - they're critical for performance
3. **Validate embeddings** - all chunks should have 1536-dim vectors
4. **Test relationships** - verify confidence thresholds
5. **Look for patterns** - complex queries show system intelligence
6. **Monitor performance** - slow queries indicate indexing issues

---

## 🚨 TROUBLESHOOTING QUERIES

### No Graph Results?

```cypher
// Check if entities exist
MATCH (e) WHERE e:Component OR e:Code OR e:Problem
RETURN labels(e)[0] as type, count(*) as count;

// If zero, entity extraction may have failed
```

### No Vector Results?

```cypher
// Check embeddings
MATCH (c:Chunk) WHERE c.embedding IS NULL
RETURN count(c);

// Check index
SHOW INDEXES WHERE type = "VECTOR";

// If index missing or embeddings null, vector search won't work
```

### Slow Queries?

```cypher
// Check database size
MATCH (n) RETURN count(n);
MATCH ()-[r]->() RETURN count(r);

// Large graphs (>10K nodes) may need additional indexes
```

---

## 📋 VALIDATION CHECKLIST

Run these in order for complete validation:

```cypher
// 1. Node count
MATCH (n) RETURN count(n);

// 2. Relationship count
MATCH ()-[r]->() RETURN count(r);

// 3. Index check
SHOW INDEXES;

// 4. Embedding check
MATCH (c:Chunk) WHERE c.embedding IS NULL RETURN count(c);

// 5. Confidence check
MATCH ()-[r:MENTIONS]->() RETURN avg(r.confidence), min(r.confidence);

// 6. Sample entities
MATCH (e) WHERE e.confidence > 0.8 RETURN labels(e)[0], e.text LIMIT 10;
```

If all 6 pass → Graph is healthy ✅

---

Copy these queries into Neo4j Browser and validate your knowledge graph! 🚀
