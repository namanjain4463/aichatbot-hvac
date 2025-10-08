# 🔧 Database Inspector Guide

**New Feature Added:** October 8, 2025  
**Location:** Streamlit UI Sidebar

---

## 📋 Overview

The **Database Inspector** feature allows you to execute Cypher queries directly from the Streamlit UI to manually inspect and validate your Neo4j knowledge graph.

---

## 🎯 Features

### 1. **Quick Query Templates** (10 Built-in Queries)

Pre-built queries for common database inspection tasks:

| Template                   | Purpose                         | Returns                           |
| -------------------------- | ------------------------------- | --------------------------------- |
| **View all node types**    | Count nodes by label            | Node types and counts             |
| **View all relationships** | Count relationships by type     | Relationship types and counts     |
| **View Components**        | List HVAC components            | Component names and confidence    |
| **View Codes**             | List code sections              | Code section references           |
| **View Compliance**        | COMPLIES_WITH relationships     | Component-Code mappings           |
| **View Specifications**    | HAS_SPECIFICATION relationships | Component specs (SEER, BTU, etc.) |
| **View Chunks**            | List document chunks            | Chunk previews and token counts   |
| **Check Embeddings**       | Verify embeddings exist         | Count of chunks with embeddings   |
| **View Indexes**           | List all database indexes       | Index details                     |
| **Graph Statistics**       | Overall graph metrics           | Node/relationship statistics      |

### 2. **Custom Query Editor**

- Write and execute any Cypher query
- Syntax highlighting
- Multi-line text area
- Edit template queries before execution

### 3. **Results Display**

- **Table View:** Results displayed in sortable DataFrame
- **Result Count:** Shows number of records returned
- **CSV Export:** Download results as CSV file
- **Copy Query:** Copy Cypher query to clipboard

### 4. **Error Handling**

- Clear error messages for syntax errors
- Helpful tips for query debugging
- Connection status indicators

---

## 🚀 How to Use

### Access the Feature

1. **Run Streamlit App:**

   ```bash
   streamlit run hvac_code_chatbot.py
   ```

2. **Look in Sidebar:**
   - Scroll to **"🔧 Database Inspector"** section

### Execute a Template Query

1. **Select Template:**

   - Click dropdown: "Choose a query template"
   - Select from 10 pre-built options

2. **Execute:**

   - Click **▶️ Execute Query** button
   - View results in table format

3. **Export (Optional):**
   - Click **📥 Download as CSV** to export results

### Write a Custom Query

1. **Select "Custom Query"** from dropdown

2. **Write Cypher:**

   ```cypher
   MATCH (c:Component)-[r:COMPLIES_WITH]->(code:Code)
   WHERE c.text =~ '(?i).*air handler.*'
   RETURN c.text AS component, code.text AS code_section, r.confidence
   ORDER BY r.confidence DESC
   LIMIT 10
   ```

3. **Execute:**

   - Click **▶️ Execute Query**
   - Results appear below

4. **Copy Query:**
   - Click **📋 Copy Query** to display in copyable format

---

## 📊 Example Queries

### 1. Find Most Connected Nodes

```cypher
MATCH (n)
WITH n, size((n)--()) AS degree
ORDER BY degree DESC
LIMIT 10
RETURN labels(n)[0] AS type, n.text AS name, degree
```

**Use Case:** Identify central entities in your graph

---

### 2. Check Data Quality

```cypher
MATCH (c:Chunk)
WHERE c.embedding IS NULL
RETURN count(c) AS chunks_without_embeddings
```

**Use Case:** Verify all chunks have embeddings

---

### 3. Find Code Sections Without Components

```cypher
MATCH (code:Code)
WHERE NOT (code)<-[:COMPLIES_WITH]-()
RETURN code.text AS orphaned_code
LIMIT 20
```

**Use Case:** Find codes not linked to any components

---

### 4. Analyze Relationship Distribution

```cypher
MATCH ()-[r]->()
WITH type(r) AS rel_type, count(*) AS count
WITH sum(count) AS total, collect({type: rel_type, count: count}) AS rels
UNWIND rels AS rel
RETURN rel.type AS relationship,
       rel.count AS count,
       round(rel.count * 100.0 / total, 2) AS percentage
ORDER BY count DESC
```

**Use Case:** Understand relationship distribution (why MENTIONS is 71%)

---

### 5. Find Components with High Confidence

```cypher
MATCH (c:Component)
WHERE c.confidence >= 0.8
RETURN c.text AS component, c.confidence
ORDER BY c.confidence DESC
LIMIT 20
```

**Use Case:** Find reliably extracted entities

---

### 6. Check Vector Index Status

```cypher
SHOW INDEXES
YIELD name, type, labelsOrTypes, properties, state
WHERE type = 'VECTOR'
RETURN name, labelsOrTypes, properties, state
```

**Use Case:** Verify vector index is online

---

### 7. Find Multi-Hop Relationships

```cypher
MATCH path = (comp:Component)-[:COMPLIES_WITH]->(code:Code)
              -[:FOLLOWS]->(standard:Standard)
WHERE comp.text =~ '(?i).*furnace.*'
RETURN comp.text AS component,
       code.text AS code_section,
       standard.name AS standard
LIMIT 10
```

**Use Case:** Trace compliance chain (Component → Code → Standard)

---

### 8. Find Co-occurring Entities

```cypher
MATCH (e1)-[:MENTIONS]->(chunk:Chunk)<-[:MENTIONS]-(e2)
WHERE e1.text =~ '(?i).*air handler.*'
  AND labels(e2)[0] = 'Code'
RETURN e2.text AS code, count(chunk) AS co_occurrences
ORDER BY co_occurrences DESC
LIMIT 10
```

**Use Case:** Find implicit relationships via MENTIONS bridge

---

### 9. Analyze Chunk Distribution

```cypher
MATCH (d:Document)<-[:PART_OF]-(c:Chunk)
RETURN d.source AS document,
       count(c) AS chunks,
       avg(c.token_count) AS avg_tokens,
       sum(c.token_count) AS total_tokens
ORDER BY chunks DESC
```

**Use Case:** Understand document chunking

---

### 10. Find Orphaned Entities

```cypher
MATCH (e)
WHERE NOT e:Chunk AND NOT e:Document
  AND NOT (e)-[:MENTIONS]->()
  AND NOT ()-[:MENTIONS]->(e)
RETURN labels(e)[0] AS type, e.text AS text, count(*) AS count
```

**Use Case:** Find entities not connected to chunks (potential extraction issues)

---

## 💡 Pro Tips

### Tip 1: Start with Templates

- Use template queries to learn Cypher syntax
- Edit templates to customize for your needs

### Tip 2: Use LIMIT

- Always add `LIMIT` to prevent overwhelming results
- Start with `LIMIT 10`, increase as needed

### Tip 3: Export Important Results

- Download query results as CSV
- Use for reporting or further analysis

### Tip 4: Check Graph Health Regularly

- Run "Check Embeddings" after graph creation
- Run "Graph Statistics" to monitor growth
- Run "View Indexes" to ensure performance

### Tip 5: Use Regex for Flexible Matching

```cypher
WHERE c.text =~ '(?i).*air handler.*'
-- (?i) = case insensitive
-- .* = wildcard
```

### Tip 6: Profile Slow Queries

```cypher
PROFILE
MATCH (c:Component)-[:COMPLIES_WITH]->(code:Code)
RETURN c.text, code.text
```

**Use Case:** Identify performance bottlenecks

### Tip 7: Combine Filters

```cypher
MATCH (c:Component)
WHERE c.confidence >= 0.8
  AND c.text =~ '(?i).*(furnace|boiler).*'
RETURN c.text, c.confidence
```

---

## 🚨 Common Errors & Solutions

### Error 1: "Query Error: Invalid input"

**Cause:** Cypher syntax error

**Solution:**

- Check for missing parentheses, brackets
- Verify MATCH, WHERE, RETURN keywords are correct
- Try a template query first

---

### Error 2: "Neo4j not available"

**Cause:** Database connection issue

**Solution:**

```bash
# Check Neo4j is running
neo4j status

# Start Neo4j if stopped
neo4j start

# Verify .env file has correct credentials
NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASS=your_password
```

---

### Error 3: "Query returned no results"

**Cause:** No matching data or graph is empty

**Solution:**

1. Check if graph was created: Run "View all node types"
2. Verify query syntax is correct
3. Try broader search criteria (remove WHERE filters)

---

### Error 4: "Memory limit exceeded"

**Cause:** Query returns too many results

**Solution:**

- Add `LIMIT` clause
- Use `count(*)` instead of returning all nodes
- Add more specific WHERE filters

---

## 🎓 Learning Cypher

### Basic Pattern Matching

```cypher
-- Match all nodes
MATCH (n)
RETURN n
LIMIT 10

-- Match specific node type
MATCH (c:Component)
RETURN c

-- Match with relationship
MATCH (a)-[r]->(b)
RETURN a, r, b
LIMIT 10

-- Match specific relationship type
MATCH (a)-[:COMPLIES_WITH]->(b)
RETURN a.text, b.text
```

### Filtering with WHERE

```cypher
-- Property filter
MATCH (c:Component)
WHERE c.confidence > 0.8
RETURN c.text

-- Pattern filter
MATCH (c:Component)
WHERE c.text =~ '(?i).*air handler.*'
RETURN c.text

-- Multiple conditions
MATCH (c:Component)
WHERE c.confidence > 0.7
  AND c.text CONTAINS 'furnace'
RETURN c.text
```

### Aggregation

```cypher
-- Count nodes
MATCH (n:Component)
RETURN count(n)

-- Group and count
MATCH (n)
RETURN labels(n)[0] AS type, count(*) AS count

-- Average, min, max
MATCH (c:Component)
RETURN avg(c.confidence) AS avg_conf,
       min(c.confidence) AS min_conf,
       max(c.confidence) AS max_conf
```

### Advanced Patterns

```cypher
-- Variable length paths
MATCH path = (a)-[*1..3]-(b)
WHERE a.text = 'Air Handler'
RETURN path
LIMIT 10

-- Optional matches
MATCH (c:Component)
OPTIONAL MATCH (c)-[r:HAS_SPECIFICATION]->(s)
RETURN c.text, s.text

-- Collect aggregation
MATCH (c:Component)-[:COMPLIES_WITH]->(code:Code)
RETURN c.text, collect(code.text) AS codes
```

---

## 📚 Resources

### Official Documentation

- **Neo4j Cypher Manual:** https://neo4j.com/docs/cypher-manual/
- **Cypher Cheat Sheet:** https://neo4j.com/docs/cypher-cheat-sheet/

### Quick References

- **COMPREHENSIVE_GUIDE.md:** Full system documentation
- **CHEAT_SHEET.md:** Quick reference guide
- **NEO4J_VALIDATION_QUERIES.md:** Validation queries

### Neo4j Browser

- Open: http://localhost:7474
- Same queries work in browser
- Visual graph exploration

---

## 🎯 Use Cases

### Use Case 1: Validation After Graph Creation

**Scenario:** Just created knowledge graph, want to verify it's correct

**Queries to Run:**

1. "View all node types" - Check expected node types exist
2. "Check Embeddings" - Verify all chunks have embeddings
3. "Graph Statistics" - Check node/relationship counts
4. "View Compliance" - Verify COMPLIES_WITH relationships exist

---

### Use Case 2: Debugging Missing Relationships

**Scenario:** Expected COMPLIES_WITH relationships but got 0

**Queries to Run:**

```cypher
-- Check if components exist
MATCH (c:Component) RETURN count(c);

-- Check if codes exist
MATCH (c:Code) RETURN count(c);

-- Check MENTIONS bridge
MATCH (comp:Component)-[:MENTIONS]->(chunk:Chunk)
      <-[:MENTIONS]-(code:Code)
RETURN comp.text, code.text, count(chunk) AS evidence
LIMIT 10;
```

---

### Use Case 3: Performance Analysis

**Scenario:** Queries are slow, want to find bottlenecks

**Queries to Run:**

```cypher
-- Check indexes exist
SHOW INDEXES;

-- Profile a slow query
PROFILE
MATCH (c:Component)-[:COMPLIES_WITH]->(code:Code)
RETURN c.text, code.text;

-- Find nodes with high degree (might slow traversals)
MATCH (n)
WITH n, size((n)--()) AS degree
WHERE degree > 50
RETURN labels(n)[0], n.text, degree;
```

---

### Use Case 4: Data Quality Audit

**Scenario:** Want to ensure graph data quality

**Queries to Run:**

1. Find orphaned nodes
2. Check confidence distribution
3. Find entities without MENTIONS
4. Verify chunk coverage
5. Check for duplicate entities

---

## 🔄 Updates & Changelog

### Version 2.0 (October 8, 2025)

- ✅ Added Database Inspector feature
- ✅ 10 pre-built query templates
- ✅ Custom query editor
- ✅ CSV export functionality
- ✅ Copy query feature
- ✅ Error handling with helpful tips

---

## 🎉 Summary

The **Database Inspector** gives you direct access to your Neo4j knowledge graph:

✅ **No need for Neo4j Browser** - Query from Streamlit  
✅ **Template Queries** - Start quickly with common tasks  
✅ **Custom Queries** - Full Cypher power at your fingertips  
✅ **Export Results** - Download as CSV for analysis  
✅ **Error Handling** - Clear messages guide you

**Perfect for:** Validation, debugging, exploration, and data quality checks!

---

**Happy Querying! 🚀**
