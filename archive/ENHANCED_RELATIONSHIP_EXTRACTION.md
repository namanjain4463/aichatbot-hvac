# 2. How to Extract More Relationships from Your Code Document

## 🎯 Current Situation

Your graph has:

- ✅ 307 MENTIONS relationships (working great!)
- ✅ 118 PART_OF relationships (all chunks → document)
- ⚠️ Only 5 HAS_SPECIFICATION relationships
- ❌ 0 COMPLIES_WITH relationships (these should exist in a code document!)
- ❌ 0 RESOLVED_BY relationships

**Why?** The relationship extraction patterns might not be matching your PDF's specific language.

---

## 🔍 The Code Already Has Relationship Extractors

Your system DOES have code to extract relationships (line ~1253-1290), but they may not be matching your PDF's specific phrasing.

### **Existing Relationship Patterns:**

#### 1. **COMPLIES_WITH** (Component → Code)

```python
# Line ~2659: Pattern looking for compliance language
(r'(\w+(?:\s+\w+)?)\s+(?:must comply with|complies with|governed by)\s+(Section\s+\d+\.?\d*)', 'COMPLIES_WITH')
```

**What it looks for:**

- "Air handlers must comply with Section 403.2"
- "Furnaces comply with IMC 605"
- "Equipment governed by Section 302.1"

**Your PDF might say instead:**

- "Air handlers **shall meet requirements of** Section 403.2"
- "Section 403.2 **applies to** air handlers"
- "Air handlers **are subject to** Section 403.2"

---

## ✅ Solution 1: Enhanced Relationship Pattern Matching

Let me create an improved version that catches MORE compliance language patterns:

<function_calls>
<invoke name="create_file">
<parameter name="content"># Enhanced Relationship Extraction Patterns

## Add these to your system to extract more relationships from code documents

### Pattern 1: Enhanced COMPLIES_WITH (Component → Code)

```python
# Add to create_knowledge_graph or as a new method
def extract_compliance_relationships_enhanced(self, session, chunks):
    """
    Extract COMPLIES_WITH relationships from code language patterns
    Specifically designed for code/regulation documents
    """
    import re

    # Comprehensive compliance patterns for code documents
    compliance_patterns = [
        # Standard patterns
        (r'(\w+(?:\s+\w+){0,2})\s+(?:must comply with|complies with|shall comply with)\s+((?:Section|IMC|IECC|ASHRAE)\s*\d+(?:\.\d+)*)', 'forward'),

        # Code-first patterns (more common in regulation docs)
        (r'((?:Section|IMC|IECC|ASHRAE)\s*\d+(?:\.\d+)*)\s+(?:applies to|governs|regulates|covers)\s+(\w+(?:\s+\w+){0,2})', 'reverse'),

        # Requirements patterns
        (r'(\w+(?:\s+\w+){0,2})\s+(?:shall meet|must meet|shall satisfy)\s+(?:the )?(?:requirements of|provisions of)\s+((?:Section|IMC|IECC|ASHRAE)\s*\d+(?:\.\d+)*)', 'forward'),

        # Subject-to patterns
        (r'(\w+(?:\s+\w+){0,2})\s+(?:are subject to|is subject to|shall be subject to)\s+((?:Section|IMC|IECC|ASHRAE)\s*\d+(?:\.\d+)*)', 'forward'),

        # Installation/Construction patterns
        (r'(\w+(?:\s+\w+){0,2})\s+(?:shall be installed|shall be constructed|shall be designed)\s+(?:in accordance with|per|according to)\s+((?:Section|IMC|IECC|ASHRAE)\s*\d+(?:\.\d+)*)', 'forward'),

        # Reference patterns
        (r'((?:Section|IMC|IECC|ASHRAE)\s*\d+(?:\.\d+)*)[:\s]+(\w+(?:\s+\w+){0,2})', 'reverse'),

        # Prescriptive patterns
        (r'(\w+(?:\s+\w+){0,2})\s+(?:as required by|as specified in|as prescribed by)\s+((?:Section|IMC|IECC|ASHRAE)\s*\d+(?:\.\d+)*)', 'forward'),
    ]

    relationships_created = 0

    for chunk in chunks:
        chunk_text = chunk['text']

        for pattern, direction in compliance_patterns:
            matches = re.finditer(pattern, chunk_text, re.IGNORECASE)

            for match in matches:
                if direction == 'forward':
                    component_text = match.group(1).strip()
                    code_text = match.group(2).strip()
                else:  # reverse
                    code_text = match.group(1).strip()
                    component_text = match.group(2).strip()

                # Filter out noise (too short, too generic)
                if len(component_text) < 3 or component_text.lower() in ['the', 'a', 'an', 'such', 'all', 'any']:
                    continue

                # Create relationship
                try:
                    session.run("""
                        // Find or create component
                        MERGE (comp:Component {text: $component})
                        ON CREATE SET comp.confidence = 0.7,
                                      comp.source = 'compliance_extraction'

                        // Find or create code
                        MERGE (code:Code {text: $code})
                        ON CREATE SET code.confidence = 0.8,
                                      code.source = 'compliance_extraction'

                        // Create COMPLIES_WITH relationship
                        MERGE (comp)-[r:COMPLIES_WITH]->(code)
                        SET r.confidence = 0.8,
                            r.compliance_type = 'code_requirement',
                            r.source = 'pattern_match',
                            r.pattern_type = $pattern_type,
                            r.chunk_index = $chunk_index
                    """,
                    component=component_text,
                    code=code_text,
                    pattern_type=direction,
                    chunk_index=chunk.get('chunk_index', 0))

                    relationships_created += 1

                except Exception as e:
                    print(f"Error creating compliance relationship: {e}")

    print(f"✓ Created {relationships_created} COMPLIES_WITH relationships from patterns")
    return relationships_created
```

### Pattern 2: Extract HAS_SPECIFICATION (Component → Specification)

```python
def extract_specification_relationships_enhanced(self, session, chunks):
    """
    Extract HAS_SPECIFICATION relationships from technical specs
    Looks for ratings, capacities, dimensions, etc.
    """
    import re

    spec_patterns = [
        # Efficiency ratings
        (r'(\w+(?:\s+\w+){0,2})\s+(?:with|having|rated at|rated)\s+(?:a\s+)?(\d+(?:\.\d+)?\s*SEER)', 'SEER rating'),
        (r'(\w+(?:\s+\w+){0,2})\s+(?:with|having|rated at)\s+(?:a\s+)?(\d+(?:\.\d+)?\s*%?\s*AFUE)', 'AFUE rating'),
        (r'(\w+(?:\s+\w+){0,2})\s+(?:with|having|rated at)\s+(?:a\s+)?(\d+(?:\.\d+)?\s*EER)', 'EER rating'),

        # Capacity ratings
        (r'(\w+(?:\s+\w+){0,2})\s+(?:with|having|rated at)\s+(?:a\s+)?(\d+(?:,\d{3})*\s*BTU/?h?)', 'BTU capacity'),
        (r'(\w+(?:\s+\w+){0,2})\s+(?:with|having|capacity of)\s+(?:a\s+)?(\d+(?:\.\d+)?\s*tons?)', 'cooling capacity'),

        # Airflow
        (r'(\w+(?:\s+\w+){0,2})\s+(?:with|providing|delivering)\s+(?:a\s+)?(\d+(?:,\d{3})*\s*CFM)', 'airflow'),

        # Minimum requirements
        (r'(\w+(?:\s+\w+){0,2})\s+(?:shall have|must have|requires)\s+(?:a\s+)?(?:minimum\s+)?(\d+(?:\.\d+)?\s*(?:SEER|AFUE|EER|BTU|CFM))', 'minimum requirement'),

        # Dimensions
        (r'(\w+(?:\s+\w+){0,2})\s+(?:measuring|sized at|dimensions of)\s+(\d+\s*(?:inch|in|ft|x\s*\d+))', 'dimensions'),
    ]

    relationships_created = 0

    for chunk in chunks:
        chunk_text = chunk['text']

        for pattern, spec_type in spec_patterns:
            matches = re.finditer(pattern, chunk_text, re.IGNORECASE)

            for match in matches:
                component_text = match.group(1).strip()
                spec_value = match.group(2).strip()

                # Filter noise
                if len(component_text) < 3:
                    continue

                try:
                    session.run("""
                        // Find or create component
                        MERGE (comp:Component {text: $component})
                        ON CREATE SET comp.confidence = 0.7

                        // Create specification node
                        MERGE (spec:Specification {text: $spec_value})
                        ON CREATE SET spec.spec_type = $spec_type,
                                      spec.confidence = 0.8

                        // Create HAS_SPECIFICATION relationship
                        MERGE (comp)-[r:HAS_SPECIFICATION]->(spec)
                        SET r.confidence = 0.85,
                            r.spec_type = $spec_type,
                            r.source = 'pattern_match'
                    """,
                    component=component_text,
                    spec_value=spec_value,
                    spec_type=spec_type)

                    relationships_created += 1

                except Exception as e:
                    print(f"Error creating specification relationship: {e}")

    print(f"✓ Created {relationships_created} HAS_SPECIFICATION relationships")
    return relationships_created
```

### Pattern 3: Extract LOCATED_IN (Component → Location)

```python
def extract_location_relationships_enhanced(self, session, chunks):
    """
    Extract INSTALLED_IN / LOCATED_IN relationships
    """
    import re

    location_patterns = [
        (r'(\w+(?:\s+\w+){0,2})\s+(?:located in|installed in|placed in|situated in)\s+(?:the\s+)?(\w+(?:\s+\w+){0,2})', 'specific'),
        (r'(\w+(?:\s+\w+){0,2})\s+(?:for use in|designed for)\s+(\w+(?:\s+\w+){0,2}\s+buildings?)', 'building type'),
        (r'(Atlanta|Georgia)\s+(?:buildings?|facilities?)\s+(?:shall|must)\s+(?:have|install)\s+(\w+(?:\s+\w+){0,2})', 'requirement'),
    ]

    relationships_created = 0

    for chunk in chunks:
        for pattern, location_type in location_patterns:
            matches = re.finditer(pattern, chunk['text'], re.IGNORECASE)

            for match in matches:
                component_text = match.group(2 if location_type == 'requirement' else 1).strip()
                location_text = match.group(1 if location_type == 'requirement' else 2).strip()

                try:
                    session.run("""
                        MERGE (comp:Component {text: $component})
                        MERGE (loc:Location {text: $location})
                        ON CREATE SET loc.location_type = $location_type

                        MERGE (comp)-[r:INSTALLED_IN]->(loc)
                        SET r.confidence = 0.75,
                            r.source = 'pattern_match'
                    """,
                    component=component_text,
                    location=location_text,
                    location_type=location_type)

                    relationships_created += 1
                except:
                    pass

    print(f"✓ Created {relationships_created} INSTALLED_IN relationships")
    return relationships_created
```

---

## 📝 How to Add These to Your System

### Option A: Add to create_knowledge_graph method (RECOMMENDED)

Add these calls after creating entities:

```python
# In create_knowledge_graph method, after Step 5 (Create semantic relationships)
# Add new step:

# Step 5b: Enhanced relationship extraction from text patterns
with self.driver.session() as session:
    print("Step 5b: Extracting enhanced relationships from text patterns...")

    compliance_count = self.extract_compliance_relationships_enhanced(session, chunks)
    spec_count = self.extract_specification_relationships_enhanced(session, chunks)
    location_count = self.extract_location_relationships_enhanced(session, chunks)

    total_enhanced = compliance_count + spec_count + location_count
    print(f"✓ Extracted {total_enhanced} additional relationships from patterns")
```

### Option B: Run as Separate Script (QUICK TEST)

Create a new file `enhance_relationships.py`:

```python
from hvac_code_chatbot import HVACGraphRAG

# Connect to your existing graph
graphrag = HVACGraphRAG(NEO4J_URI, NEO4J_USER, NEO4J_PASS)

# Get chunks from database
with graphrag.driver.session() as session:
    result = session.run("MATCH (c:Chunk) RETURN c ORDER BY c.chunk_index")
    chunks = [dict(record['c']) for record in result]

# Run enhanced extraction
graphrag.extract_compliance_relationships_enhanced(graphrag.driver.session(), chunks)
graphrag.extract_specification_relationships_enhanced(graphrag.driver.session(), chunks)
graphrag.extract_location_relationships_enhanced(graphrag.driver.session(), chunks)

print("✓ Enhanced relationships extracted!")
```

Then run: `python enhance_relationships.py`

---

## 🧪 Test After Adding

After adding enhanced patterns, run these validation queries:

```cypher
// 1. Count new COMPLIES_WITH relationships
MATCH ()-[r:COMPLIES_WITH]->()
RETURN count(r) as compliance_count;
// Expected: 20-50 (was 0)

// 2. Sample compliance relationships
MATCH (comp:Component)-[r:COMPLIES_WITH]->(code:Code)
RETURN comp.text, code.text, r.confidence
ORDER BY r.confidence DESC
LIMIT 10;

// 3. Count HAS_SPECIFICATION
MATCH ()-[r:HAS_SPECIFICATION]->()
RETURN count(r) as spec_count;
// Expected: 15-40 (was 5)

// 4. See what patterns matched
MATCH ()-[r:COMPLIES_WITH]->()
WHERE r.pattern_type IS NOT NULL
RETURN r.pattern_type, count(*) as count
ORDER BY count DESC;
```

---

## 📊 Expected Improvement

**Before Enhanced Extraction:**

- COMPLIES_WITH: 0
- HAS_SPECIFICATION: 5
- INSTALLED_IN: 1

**After Enhanced Extraction:**

- COMPLIES_WITH: 20-50 ✅ (code documents have lots of these!)
- HAS_SPECIFICATION: 15-40 ✅
- INSTALLED_IN: 5-15 ✅

**Total Relationships:**

- Before: 433
- After: 470-530 (10-20% increase)

---

## 💡 Why This Works for Code Documents

Code documents use **specific legal/regulatory language**:

- "shall comply with"
- "are subject to"
- "as specified in"
- "requirements of Section X"

The enhanced patterns catch **MORE variations** of these phrases, extracting relationships that were previously missed!

---

Next: See **"3. How to Query This Graph Structure Effectively"** →
