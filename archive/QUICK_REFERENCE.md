# 🚀 Quick Reference: Using the Strict HVAC Ontology

**For Developers and Data Scientists**

---

## 📚 Table of Contents

1. [Node Labels Reference](#node-labels-reference)
2. [Relationship Types Reference](#relationship-types-reference)
3. [Query Templates](#query-templates)
4. [Confidence Levels Guide](#confidence-levels-guide)
5. [Common Tasks](#common-tasks)

---

## 🏷️ Node Labels Reference

### **Component** (Default, ~40-50% of nodes)

**What:** HVAC equipment, systems, parts, measurements

**Examples:**

- Air Handler
- Furnace
- Boiler
- Heat Pump
- Ductwork
- Thermostat
- 80000 BTU
- 15 SEER

**Properties:**

```python
{
    "text": "Air Handler",
    "original_text": "AHU",  # if standardized
    "confidence": 1.0,
    "hvac_type": "equipment",
    "importance": "medium",
    "extracted_at": "2025-10-07T..."
}
```

---

### **Brand** (~5-10% of nodes)

**What:** HVAC manufacturers

**Examples:**

- Carrier
- Trane
- Lennox
- York
- Rheem

**How to identify:** Text matches KNOWN_MANUFACTURERS list

---

### **Code** (~20-30% of nodes)

**What:** Code sections, regulations, standards

**Examples:**

- Section 301.1
- IMC Chapter 3
- IECC 2021

**How to identify:** Contains "section", "chapter", or hvac_type == 'code'

---

### **Problem** (~5-10% of nodes)

**What:** HVAC issues and failures

**Examples:**

- Refrigerant Leak
- Low Airflow
- Frozen Coil
- Short Cycling

**How to identify:** Text matches KNOWN_PROBLEMS list

---

### **Solution** (~5-10% of nodes)

**What:** Fixes and maintenance actions

**Examples:**

- Repair Leak
- Clean Filter
- Replace Coil
- Adjust Thermostat

**How to identify:** Contains keywords: replace, repair, fix, install, adjust, clean, inspect

---

### **Location** (~3-5% of nodes)

**What:** Physical locations and zones

**Examples:**

- Atlanta
- Zone 1
- Building A
- Floor 2

**How to identify:** hvac_type == 'location' or contains location keywords

---

### **Category** (4 fixed nodes)

**What:** HVAC system categories

**Examples:**

- Heating Systems
- Cooling Systems
- Ventilation Systems
- Control Systems

**How to identify:** Created by system, not extracted

---

### **Standard** (~10-20 nodes)

**What:** Industry standards

**Examples:**

- ASHRAE 90.1
- IECC
- IMC
- IFGC

**How to identify:** Contains standard names

---

### **Document** (1-10 nodes)

**What:** Source documents

**Examples:**

- HVAC Codes.pdf
- Atlanta Building Code.pdf

**How to identify:** Created during document ingestion

---

## 🔗 Relationship Types Reference

### **:PART_OF** (Component → Component)

**Confidence:** 0.95  
**Meaning:** Component is part of a larger system  
**Example:** `(Furnace)-[:PART_OF]->(HVAC System)`

**Properties:**

```python
{
    "confidence": 0.95,
    "source": "entity_extraction",
    "relationship_type": "component_hierarchy",
    "created_at": "2025-10-07T..."
}
```

**Query Template:**

```cypher
MATCH (component:Component)-[:PART_OF]->(system:Component)
WHERE component.text = 'Furnace'
RETURN system.text
```

---

### **:COMPLIES_WITH** (Component → Code)

**Confidence:** 0.90  
**Meaning:** Component must comply with code requirement  
**Example:** `(Air Handler)-[:COMPLIES_WITH]->(Section 301.1)`

**Properties:**

```python
{
    "compliance_type": "code_requirement",
    "confidence": 0.9,
    "source": "code_reference",
    "created_at": "2025-10-07T..."
}
```

**Query Template:**

```cypher
MATCH (c:Component {text: 'Air Handler'})-[r:COMPLIES_WITH]->(code:Code)
WHERE r.confidence > 0.85
RETURN code.text, r.compliance_type
```

---

### **:FOLLOWS** (Code → Standard)

**Confidence:** 0.95  
**Meaning:** Code follows industry standard  
**Example:** `(Section 301.1)-[:FOLLOWS]->(ASHRAE)`

---

### **:HAS_SPECIFICATION** (Component → Component/Measurement)

**Confidence:** 0.85  
**Meaning:** Component has performance specification  
**Example:** `(Furnace)-[:HAS_SPECIFICATION]->(80000 BTU)`

**Properties:**

```python
{
    "spec_type": "performance_rating",
    "confidence": 0.85,
    "source": "entity_extraction",
    "created_at": "2025-10-07T..."
}
```

---

### **:INSTALLED_IN** (Component → Location)

**Confidence:** 0.80  
**Meaning:** Component is installed in location  
**Example:** `(Air Handler)-[:INSTALLED_IN]->(Atlanta)`

**Properties:**

```python
{
    "installation_type": "physical_location",
    "confidence": 0.8,
    "source": "entity_extraction",
    "created_at": "2025-10-07T..."
}
```

---

### **:BELONGS_TO** (Component → Category)

**Confidence:** 0.90  
**Meaning:** Component belongs to system category  
**Example:** `(Furnace)-[:BELONGS_TO]->(Heating Systems)`

---

### **:RESOLVED_BY** (Problem → Solution)

**Confidence:** 0.90  
**Meaning:** Problem is resolved by solution  
**Example:** `(Refrigerant Leak)-[:RESOLVED_BY]->(Repair Leak)`

**Query Template:**

```cypher
MATCH (p:Problem {text: 'Low Airflow'})-[r:RESOLVED_BY]->(s:Solution)
WHERE r.confidence > 0.8
RETURN s.text, r.confidence
ORDER BY r.confidence DESC
```

---

### **:MANUFACTURED_BY** (Component → Brand)

**Confidence:** 0.85  
**Meaning:** Component is manufactured by brand  
**Example:** `(Air Handler)-[:MANUFACTURED_BY]->(Carrier)`

**Query Template:**

```cypher
MATCH (c:Component)-[:MANUFACTURED_BY]->(b:Brand {text: 'Carrier'})
OPTIONAL MATCH (c)-[:HAS_SPECIFICATION]->(spec)
RETURN c.text, collect(spec.text) as specifications
```

---

### **:RELATED_TO** (Code → Code)

**Confidence:** 0.75  
**Meaning:** Code sections are related  
**Example:** `(Section 301.1)-[:RELATED_TO]->(Section 301.2)`

---

### **:MENTIONED_IN** (All → Document)

**Confidence:** Inherits from entity  
**Meaning:** Entity mentioned in document  
**Example:** `(Carrier)-[:MENTIONED_IN]->(HVAC Codes.pdf)`

---

## 📊 Confidence Levels Guide

### **Interpretation:**

| Range     | Level         | Meaning                            | Action                     |
| --------- | ------------- | ---------------------------------- | -------------------------- |
| 0.90-1.00 | **Very High** | Known mappings, exact matches      | Use confidently            |
| 0.80-0.89 | **High**      | Strong relationships, validated    | Use with minor caution     |
| 0.70-0.79 | **Medium**    | Likely correct, needs verification | Review before critical use |
| 0.50-0.69 | **Low**       | Uncertain, may need human review   | Use with caution           |
| 0.00-0.49 | **Very Low**  | Filtered out (not in graph)        | Not stored                 |

### **Filtering Recommendations:**

**Critical Applications (Code Compliance):**

```cypher
WHERE r.confidence >= 0.90
```

**General Queries:**

```cypher
WHERE r.confidence >= 0.75
```

**Exploratory Analysis:**

```cypher
WHERE r.confidence >= 0.60
```

---

## 🔍 Query Templates

### **1. Find Equipment by Category**

```cypher
MATCH (c:Component)-[:BELONGS_TO]->(cat:Category {name: 'Heating Systems'})
WHERE c.confidence > 0.8
RETURN c.text, c.confidence
ORDER BY c.confidence DESC
```

---

### **2. Find Code Requirements for Equipment**

```cypher
MATCH (c:Component {text: $component_name})-[r:COMPLIES_WITH]->(code:Code)
WHERE r.confidence > 0.85
RETURN code.text, r.compliance_type, r.confidence
ORDER BY r.confidence DESC
```

**Parameters:**

```python
{"component_name": "Air Handler"}
```

---

### **3. Find Solutions for Problem**

```cypher
MATCH (p:Problem)-[r:RESOLVED_BY]->(s:Solution)
WHERE toLower(p.text) CONTAINS toLower($problem_keyword)
  AND r.confidence > 0.8
RETURN p.text as problem,
       s.text as solution,
       r.confidence
ORDER BY r.confidence DESC
```

**Parameters:**

```python
{"problem_keyword": "leak"}
```

---

### **4. Find Brand Equipment with Specs**

```cypher
MATCH (c:Component)-[:MANUFACTURED_BY]->(b:Brand {text: $brand_name})
OPTIONAL MATCH (c)-[:HAS_SPECIFICATION]->(spec)
RETURN c.text as equipment,
       collect(DISTINCT spec.text) as specifications,
       c.confidence
ORDER BY c.confidence DESC
```

**Parameters:**

```python
{"brand_name": "Carrier"}
```

---

### **5. Trace Compliance Chain**

```cypher
MATCH path = (c:Component)-[:COMPLIES_WITH]->(:Code)-[:FOLLOWS]->(s:Standard)
WHERE c.text = $component_name
RETURN c.text as component,
       [n IN nodes(path) | n.text] as compliance_chain,
       [r IN relationships(path) | r.confidence] as confidences
```

**Parameters:**

```python
{"component_name": "Furnace"}
```

---

### **6. Find Location-Specific Equipment**

```cypher
MATCH (c:Component)-[:INSTALLED_IN]->(l:Location)
WHERE toLower(l.text) CONTAINS 'atlanta'
OPTIONAL MATCH (c)-[:BELONGS_TO]->(cat:Category)
RETURN c.text, cat.name, l.text
ORDER BY cat.name, c.text
```

---

### **7. Analyze Low-Confidence Relationships**

```cypher
MATCH ()-[r]->()
WHERE r.confidence < 0.7
RETURN type(r) as relationship_type,
       r.confidence,
       r.source,
       count(*) as count
ORDER BY r.confidence ASC
LIMIT 20
```

---

### **8. Find Entity Standardization Cases**

```cypher
MATCH (n)
WHERE n.original_text <> n.text
RETURN labels(n)[0] as node_type,
       n.original_text as original,
       n.text as standardized,
       n.confidence
ORDER BY n.confidence DESC
LIMIT 50
```

---

## 🛠️ Common Tasks

### **Task 1: Add New Problem-Solution Mapping**

**Location:** `hvac_code_chatbot.py`, method `create_problem_solution_relationships()`

**Edit:**

```python
problem_solution_mapping = {
    # ... existing mappings ...
    "new problem": ["solution 1", "solution 2", "solution 3"],
}
```

---

### **Task 2: Add New Known Component**

**Location:** `hvac_code_chatbot.py`, class `HVACGraphRAG`, after line 107

**Edit:**

```python
KNOWN_COMPONENTS = {
    # ... existing components ...
    "New Component Type",
}
```

---

### **Task 3: Adjust Confidence Threshold**

**For Entity Filtering:**

**Location:** `create_knowledge_graph()`, line ~595

**Edit:**

```python
if not is_valid or confidence < 0.5:  # Change 0.5 to desired threshold
    continue
```

**For Query Filtering:**

**Location:** Any Cypher query

**Edit:**

```cypher
WHERE r.confidence > 0.75  # Adjust threshold as needed
```

---

### **Task 4: Add New Relationship Type**

**Steps:**

1. Define relationship in `create_entity_relationships()`:

   ```python
   self.create_new_relationship_type(session, entities)
   ```

2. Implement method:

   ```python
   def create_new_relationship_type(self, session, entities):
       """Create new relationship type"""
       # Implementation
       session.run("""
           MATCH (source:SourceLabel {text: $source_text})
           MATCH (target:TargetLabel {text: $target_text})
           MERGE (source)-[r:NEW_RELATIONSHIP]->(target)
           SET r.confidence = $confidence,
               r.source = 'entity_extraction',
               r.created_at = $created_at
       """, ...)
   ```

3. Update documentation in `KNOWLEDGE_GRAPH_SCHEMA.md`

---

### **Task 5: Debug Graph Creation**

**Check Entity Validation:**

Add logging in `create_knowledge_graph()`:

```python
print(f"Entity: {e['text']}, Label: {node_label}, Valid: {is_valid}, Confidence: {confidence}")
```

**Check Relationship Creation:**

Add logging in relationship methods:

```python
print(f"Created {type(r)} from {source} to {target} with confidence {confidence}")
```

**Verify in Neo4j Browser:**

```cypher
// Count nodes by label
CALL db.labels() YIELD label
CALL apoc.cypher.run('MATCH (n:' + label + ') RETURN count(n) as count', {})
YIELD value
RETURN label, value.count

// Count relationships by type
CALL db.relationshipTypes() YIELD relationshipType
CALL apoc.cypher.run('MATCH ()-[r:' + relationshipType + ']->() RETURN count(r) as count', {})
YIELD value
RETURN relationshipType, value.count
```

---

## 📚 Additional Resources

- **Full Documentation:** `KNOWLEDGE_GRAPH_IMPROVEMENTS.md`
- **Schema Diagram:** `KNOWLEDGE_GRAPH_SCHEMA.md`
- **Implementation Summary:** `PHASE2_COMPLETION_SUMMARY.md`
- **File Changes:** `FILE_CHANGES_SUMMARY.md`

---

## 🎯 Best Practices

1. **Always filter by confidence** for critical queries (>= 0.85)
2. **Use strict labels** in MATCH clauses for better performance
3. **Check metadata** (source, created_at) for auditing
4. **Standardize input** before queries (use `standardize_hvac_term()`)
5. **Validate entities** before adding to graph (use `validate_entity()`)

---

**Quick Start:** Ready to query the knowledge graph! 🚀
