# 🗺️ HVAC Knowledge Graph Schema

**Visual representation of the strict ontology implemented in Phase 2**

---

## 📊 Node Types and Relationships

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        HVAC KNOWLEDGE GRAPH ONTOLOGY                        │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────┐
│  :Document  │ ◄────────────────┐
│             │                  │
│ • source    │                  │
│ • type      │                  │ :MENTIONED_IN
│ • location  │                  │ (all nodes)
│ • title     │                  │
└─────────────┘                  │
                                 │
┌─────────────┐      :PART_OF    ┌──────────────┐
│ :Component  │ ◄────────────────┤  :Component  │
│             │    (0.95)        │              │
│ • text      │                  │ • text       │
│ • confidence│ ──────────────►  │ • confidence │
│ • hvac_type │  :HAS_SPEC       │ • importance │
│ • importance│    (0.85)        └──────────────┘
└─────────────┘                         │
      │                                 │
      │ :MANUFACTURED_BY                │ :BELONGS_TO
      │      (0.85)                     │    (0.90)
      ▼                                 ▼
┌─────────────┐                  ┌──────────────┐
│   :Brand    │                  │  :Category   │
│             │                  │              │
│ • text      │                  │ • name       │
│ • confidence│                  │ • type       │
└─────────────┘                  └──────────────┘
      ▲
      │
      │ :MANUFACTURED_BY
      │      (0.85)
      │
┌─────────────┐
│ :Component  │ ─────────────► ┌──────────────┐
│             │ :COMPLIES_WITH │    :Code     │
│             │    (0.90)      │              │
└─────────────┘                │ • text       │
                               │ • code_ref   │
                               └──────────────┘
                                      │
                                      │ :FOLLOWS
                                      │  (0.95)
                                      ▼
                               ┌──────────────┐
                               │  :Standard   │
                               │              │
                               │ • name       │
                               │ • type       │
                               └──────────────┘

┌─────────────┐                ┌──────────────┐
│  :Problem   │ ─────────────► │  :Solution   │
│             │  :RESOLVED_BY  │              │
│ • text      │     (0.90)     │ • text       │
│ • confidence│                │ • confidence │
└─────────────┘                └──────────────┘

┌─────────────┐                ┌──────────────┐
│ :Component  │ ─────────────► │  :Location   │
│             │ :INSTALLED_IN  │              │
│             │    (0.80)      │ • text       │
└─────────────┘                │ • atlanta_sp │
                               └──────────────┘

┌─────────────┐                ┌──────────────┐
│    :Code    │ ─────────────► │    :Code     │
│             │  :RELATED_TO   │              │
│             │    (0.75)      │              │
└─────────────┘                └──────────────┘
```

---

## 🎯 Relationship Properties

**All relationships include:**

```python
{
    "confidence": 0.75-0.95,        # Float - reliability score
    "source": "entity_extraction",  # String - origin of relationship
    "created_at": "2025-10-07T..."  # ISO timestamp
}
```

**Type-specific properties:**

- **:PART_OF** → `relationship_type: "component_hierarchy"`
- **:COMPLIES_WITH** → `compliance_type: "code_requirement"`
- **:HAS_SPECIFICATION** → `spec_type: "performance_rating"`
- **:INSTALLED_IN** → `installation_type: "physical_location"`
- **:BELONGS_TO** → `category_type: "hvac_system"`
- **:RELATED_TO** → `relationship_type: "code_reference"`

---

## 📝 Example Graph Instance

```
┌─────────────────────┐
│    HVAC Codes.pdf   │ :Document
└─────────────────────┘
          ▲
          │ :MENTIONED_IN
          │
    ┌─────────────┐
    │   Carrier   │ :Brand
    └─────────────┘
          ▲
          │ :MANUFACTURED_BY (0.85)
          │
    ┌─────────────────────┐
    │    Air Handler      │ :Component
    │  confidence: 1.0    │
    └─────────────────────┘
          │                       │
          │ :PART_OF (0.95)       │ :COMPLIES_WITH (0.90)
          ▼                       ▼
    ┌─────────────────────┐   ┌──────────────────┐
    │   HVAC System       │   │  Section 301.1   │ :Code
    │   :Component        │   └──────────────────┘
    └─────────────────────┘           │
          │                           │ :FOLLOWS (0.95)
          │ :BELONGS_TO (0.90)        ▼
          ▼                   ┌──────────────────┐
    ┌─────────────────────┐  │   ASHRAE 90.1    │ :Standard
    │ Heating Systems     │  └──────────────────┘
    │   :Category         │
    └─────────────────────┘
          │
          │ :INSTALLED_IN (0.80)
          ▼
    ┌─────────────────────┐
    │     Atlanta         │ :Location
    │  atlanta_specific   │
    └─────────────────────┘

    ┌─────────────────────┐
    │ Refrigerant Leak    │ :Problem
    └─────────────────────┘
          │
          │ :RESOLVED_BY (0.90)
          ▼
    ┌─────────────────────┐
    │    Repair Leak      │ :Solution
    └─────────────────────┘
```

---

## 🔍 Common Query Patterns

### **1. Equipment Hierarchy**

Find all components of a system:

```cypher
MATCH (c:Component)-[:PART_OF]->(s:Component {text: 'HVAC System'})
WHERE c.confidence > 0.8
RETURN c.text, c.confidence
ORDER BY c.confidence DESC
```

### **2. Code Compliance**

Find all code requirements for equipment:

```cypher
MATCH (c:Component)-[r:COMPLIES_WITH]->(code:Code)
WHERE r.confidence > 0.85
RETURN c.text, code.text, r.compliance_type
```

### **3. Problem Diagnosis**

Find solutions for a specific problem:

```cypher
MATCH (p:Problem {text: 'Low Airflow'})-[r:RESOLVED_BY]->(s:Solution)
WHERE r.confidence > 0.8
RETURN s.text, r.confidence
ORDER BY r.confidence DESC
```

### **4. Brand Equipment**

Find all equipment by manufacturer:

```cypher
MATCH (c:Component)-[:MANUFACTURED_BY]->(b:Brand {text: 'Carrier'})
OPTIONAL MATCH (c)-[:HAS_SPECIFICATION]->(spec)
RETURN c.text, collect(spec.text) as specs
```

### **5. Location-Based Query**

Find all heating equipment in Atlanta:

```cypher
MATCH (c:Component)-[:BELONGS_TO]->(:Category {name: 'Heating Systems'})
MATCH (c)-[:INSTALLED_IN]->(l:Location)
WHERE toLower(l.text) CONTAINS 'atlanta'
RETURN c.text, l.text
```

### **6. Standard Compliance Chain**

Trace from component to standard:

```cypher
MATCH path = (c:Component)-[:COMPLIES_WITH]->(:Code)-[:FOLLOWS]->(s:Standard)
WHERE c.text = 'Air Handler'
RETURN c.text,
       [r IN relationships(path) | r.confidence] as confidences,
       s.name
```

### **7. Confidence Analysis**

Find low-confidence relationships for review:

```cypher
MATCH ()-[r]->()
WHERE r.confidence < 0.7
RETURN type(r) as rel_type,
       r.confidence,
       r.source,
       count(*) as count
ORDER BY r.confidence ASC
```

### **8. Entity Validation Check**

Find entities that were standardized:

```cypher
MATCH (n)
WHERE n.original_text <> n.text
RETURN labels(n)[0] as type,
       n.original_text as original,
       n.text as standardized,
       n.confidence
ORDER BY n.confidence DESC
```

---

## 📊 Statistics and Metrics

### **Expected Graph Size (after full document processing):**

- **Nodes:** 500-2,000

  - :Component (40-50%)
  - :Code (20-30%)
  - :Brand (5-10%)
  - :Problem (5-10%)
  - :Solution (5-10%)
  - :Location (3-5%)
  - :Category (4 fixed)
  - :Standard (10-20)
  - :Document (1-10)

- **Relationships:** 2,000-10,000
  - :MENTIONED_IN (1 per node)
  - :PART_OF (50-200)
  - :COMPLIES_WITH (200-500)
  - :BELONGS_TO (100-300)
  - :HAS_SPECIFICATION (50-150)
  - :INSTALLED_IN (50-100)
  - :RESOLVED_BY (30-80)
  - :MANUFACTURED_BY (50-150)
  - :FOLLOWS (10-30)
  - :RELATED_TO (100-300)

### **Confidence Distribution:**

- **0.90-1.00** (High): 60% of relationships

  - :PART_OF (0.95)
  - :FOLLOWS (0.95)
  - :COMPLIES_WITH (0.90)
  - :RESOLVED_BY (0.90)
  - :BELONGS_TO (0.90)

- **0.80-0.89** (Medium): 30% of relationships

  - :HAS_SPECIFICATION (0.85)
  - :MANUFACTURED_BY (0.85)
  - :INSTALLED_IN (0.80)

- **0.70-0.79** (Low): 10% of relationships
  - :RELATED_TO (0.75)

---

## 🎓 Design Principles

### **1. Semantic Clarity**

- Node labels describe what the entity IS (Brand, Component, Problem)
- Relationship types describe HOW entities relate (MANUFACTURED_BY, RESOLVED_BY)
- No generic labels (avoided :Entity, :Item, :Thing)

### **2. Confidence Tracking**

- Every relationship has confidence score
- Confidence based on extraction method and validation
- Enables filtering by reliability

### **3. Metadata Richness**

- Source tracking for auditing
- Timestamps for versioning
- Type-specific properties for context

### **4. Query Optimization**

- Explicit labels enable indexed lookups
- Relationship types enable path-specific queries
- Confidence enables quality filtering

### **5. Maintainability**

- Clear ontology documented
- Standardized terminology prevents duplicates
- Validation ensures data quality

---

## 🔧 Implementation Status

**Phase 2 Complete:** ✅

- [x] 9 strict node labels defined and implemented
- [x] 10 relationship types with confidence scoring
- [x] Entity standardization integrated
- [x] Validation filtering (confidence > 0.5)
- [x] Metadata tracking (source, timestamp)
- [x] Problem-solution mapping
- [x] Brand-component linking

**Ready for Phase 3:** Graph-First Retrieval 🚀
