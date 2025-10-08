# Sample Test Queries for Hybrid Graph + Vector RAG System

## 🎯 Basic Functionality Tests

### Query Set 1: Code Compliance (Intent Pattern 1)

1. "What codes apply to air handlers?"
2. "What are the ASHRAE requirements for ventilation?"
3. "What IMC sections govern furnace installation?"
4. "What are the IECC energy efficiency requirements?"
5. "Show me code compliance for boilers"

**Expected:** COMPLIES_WITH relationships from graph + code sections from chunks

---

### Query Set 2: Problem-Solution (Intent Pattern 2)

1. "How do I fix low airflow issues?"
2. "What causes heating system failures?"
3. "How to troubleshoot air conditioning problems?"
4. "What are common furnace repair issues?"
5. "How to solve ductwork leakage?"

**Expected:** RESOLVED_BY relationships + troubleshooting chunks

---

### Query Set 3: Specifications (Intent Pattern 3)

1. "What is the SEER rating for heat pumps?"
2. "What is the BTU capacity of the furnace?"
3. "What are the efficiency ratings mentioned?"
4. "What is the CFM rating for air handlers?"
5. "What are the performance specifications?"

**Expected:** HAS_SPECIFICATION relationships + spec values from chunks

---

### Query Set 4: Brand/Manufacturer (Intent Pattern 4)

1. "What Carrier equipment is mentioned?"
2. "Who manufactures the air conditioning unit?"
3. "What Trane products are discussed?"
4. "Which manufacturers are referenced?"
5. "What equipment does Lennox make?"

**Expected:** MANUFACTURED_BY relationships + brand mentions

---

### Query Set 5: System Hierarchy (Intent Pattern 5)

1. "What components are part of the HVAC system?"
2. "What is connected to the air handler?"
3. "What parts make up the heating system?"
4. "What equipment is in the ductwork system?"
5. "Show me the system hierarchy"

**Expected:** PART_OF relationships + system structure

---

### Query Set 6: Location (Intent Pattern 6)

1. "What equipment is in Atlanta?"
2. "Where is the furnace installed?"
3. "What are the local code requirements?"
4. "What installations are in Georgia?"
5. "Show me location-specific information"

**Expected:** INSTALLED_IN relationships + location mentions

---

## 🔥 Advanced/Complex Queries

### Multi-Intent Queries

1. "What Carrier air handlers are installed in Atlanta and what codes apply?"

   - **Tests:** Brand + Location + Code intents combined

2. "What are the SEER ratings for heat pumps and what problems might occur?"

   - **Tests:** Specifications + Problem-Solution intents

3. "How do I fix efficiency issues in furnaces that comply with ASHRAE?"

   - **Tests:** Problem-Solution + Code + Specifications

4. "What Trane equipment has BTU specifications above 100,000?"

   - **Tests:** Brand + Specifications with filtering

5. "Show me all components in the ventilation system and their manufacturers"
   - **Tests:** Hierarchy + Brand intents

---

## 🧪 Edge Case Tests

### Empty/Minimal Queries

1. "" (empty string)
2. " " (whitespace only)
3. "?" (single character)
4. "a" (meaningless single letter)
5. "the and or" (only stop words)

**Expected:** Graceful error handling, no crashes

---

### Ambiguous Queries

1. "Tell me about it"
2. "What?"
3. "More information please"
4. "Explain"
5. "Details"

**Expected:** Request for clarification or generic response

---

### Non-HVAC Queries

1. "What is the capital of France?"
2. "How do I bake a cake?"
3. "What is Python programming?"
4. "Who won the World Series?"
5. "What is quantum physics?"

**Expected:** "No HVAC-related information found" or polite redirect

---

### Special Characters

1. "What is the CFM for 12\" ducts?"
2. "Model #1234-ABC/XYZ specifications"
3. "Air handler (Carrier brand) requirements"
4. "Code section 403.2.1.1"
5. "Temperature range: 60°F - 80°F"

**Expected:** Special characters handled correctly, no parsing errors

---

### Very Long Queries

1. "I am looking for extremely detailed and comprehensive information about all possible HVAC system components including but not limited to air handlers, heat exchangers, evaporator coils, condenser units, compressors, expansion valves, thermostats, ductwork, dampers, filters, and ventilation systems that might be installed in commercial buildings in Atlanta, Georgia, and I need to know about all applicable codes from ASHRAE, IMC, IECC, and local Atlanta ordinances, plus any manufacturer specifications from brands like Carrier, Trane, Lennox, York, Rheem, Goodman, and others, along with installation procedures, maintenance schedules, troubleshooting guides, efficiency ratings like SEER and HSPF, capacity measurements in BTU and tons, and airflow ratings in CFM for various operating conditions?"

**Expected:** Query processed (may be truncated), results returned

---

### Contradictory Queries

1. "What are the low efficiency requirements for high SEER systems?"
2. "How do I install a furnace without following codes?"
3. "What are non-compliant code requirements?"

**Expected:** Logical response addressing contradiction

---

## 🎭 Semantic Similarity Tests

### Paraphrase Consistency

Ask the same question in different ways:

**Topic: Code Requirements**

1. "What codes apply to furnaces?"
2. "Which regulations govern furnace installation?"
3. "What are the legal requirements for furnaces?"
4. "Show me furnace code compliance"
5. "What standards must furnaces meet?"

**Expected:** Similar answers across all 5 variations

**Topic: Troubleshooting**

1. "How do I fix low airflow?"
2. "What's the solution for poor air circulation?"
3. "How to resolve insufficient airflow?"
4. "What causes weak airflow and how to repair?"
5. "Troubleshooting steps for low air volume"

**Expected:** Consistent troubleshooting steps

---

## 📊 Performance Benchmark Queries

### Speed Test Set (measure response time)

1. "What codes apply to furnaces?" → \_\_\_ sec
2. "How to fix airflow issues?" → \_\_\_ sec
3. "What is the SEER rating?" → \_\_\_ sec
4. "Which Carrier equipment is mentioned?" → \_\_\_ sec
5. "What components are in the system?" → \_\_\_ sec

**Target:** < 5 seconds average

---

## 🔍 Precision Tests

### Very Specific Queries

1. "What is code section 403.2 about?"
2. "What is the exact SEER rating for model XYZ-1234?"
3. "How many BTUs does the specific furnace model have?"
4. "What is the CFM at 0.1 inches water column?"
5. "Which specific ASHRAE section applies?"

**Expected:** Precise answers if data exists, else "not found"

---

### Broad Queries

1. "Tell me about HVAC systems"
2. "What equipment is discussed?"
3. "Explain the system"
4. "What information is available?"
5. "Overview of the document"

**Expected:** General overview, multiple entities mentioned

---

## 🎪 Stress Tests

### Rapid Fire (same query repeated quickly)

**Query:** "What codes apply to air handlers?"

- Run 1: \_\_\_ sec
- Run 2: \_\_\_ sec (should be faster - caching?)
- Run 3: \_\_\_ sec
- Run 4: \_\_\_ sec
- Run 5: \_\_\_ sec

**Expected:** Consistent results, no degradation

---

### Query Bombing (different queries in quick succession)

1. "What codes apply to furnaces?"
2. "How to fix airflow?"
3. "What is SEER?"
4. "Carrier equipment?"
5. "System components?"
6. "Atlanta installations?"
7. "BTU capacity?"
8. "Troubleshooting?"
9. "Efficiency ratings?"
10. "Code compliance?"

**Expected:** All queries answered correctly, no errors

---

## 🧮 Data Coverage Tests

### Entity Type Coverage

Ask about each entity type to verify extraction:

1. **Components:** "What components are mentioned?"
2. **Codes:** "What code sections are referenced?"
3. **Problems:** "What problems are discussed?"
4. **Solutions:** "What solutions are provided?"
5. **Brands:** "What manufacturers are mentioned?"
6. **Locations:** "What locations are referenced?"
7. **Specifications:** "What specifications are listed?"

**Expected:** At least some results for each type

---

### Relationship Coverage

Test each relationship type:

1. **COMPLIES_WITH:** "What codes apply to equipment?"
2. **RESOLVED_BY:** "What solutions fix problems?"
3. **HAS_SPECIFICATION:** "What are the specifications?"
4. **MANUFACTURED_BY:** "Who makes the equipment?"
5. **PART_OF:** "What components are in systems?"
6. **INSTALLED_IN:** "Where is equipment installed?"
7. **MENTIONS:** (implicit in chunk search)

**Expected:** Graph returns relationships where they exist

---

## 🎨 Answer Quality Tests

### Citation Check

1. "What codes apply to air handlers?"

   - **Verify:** Answer cites specific code sections
   - **Verify:** Sources are mentioned

2. "How to fix low airflow?"
   - **Verify:** Answer references document sections
   - **Verify:** Steps are numbered/organized

---

### Completeness Check

1. "What are ASHRAE ventilation requirements?"
   - **Verify:** Answer is complete (not cut off)
   - **Verify:** Multiple aspects covered
   - **Verify:** Both graph and vector sources used

---

### Accuracy Check (if you know the answer)

1. [Your known HVAC fact]
   - **Verify:** Answer matches document content
   - **Verify:** No hallucinations
   - **Verify:** Numbers/specs are correct

---

## 📝 Test Results Template

```
=== TEST SESSION ===
Date: __________
Time: __________
PDF: __________

BASIC TESTS:
- Code queries: ✅ / ❌
- Problem-Solution: ✅ / ❌
- Specifications: ✅ / ❌
- Brands: ✅ / ❌
- Hierarchy: ✅ / ❌
- Location: ✅ / ❌

EDGE CASES:
- Empty queries: ✅ / ❌
- Special chars: ✅ / ❌
- Non-HVAC: ✅ / ❌
- Very long: ✅ / ❌

PERFORMANCE:
- Avg response time: ___ sec
- Slowest query: ___ sec
- Fastest query: ___ sec

QUALITY:
- Consistent results: ✅ / ❌
- Accurate citations: ✅ / ❌
- No hallucinations: ✅ / ❌
- Graph + Vector fusion: ✅ / ❌

OVERALL: PASS / FAIL
SCORE: ___ / 10
```

---

## 🚀 Quick Start Testing Sequence

**5-Minute Smoke Test:**

1. Upload PDF
2. Create graph
3. Query: "What codes apply to furnaces?"
4. Query: "How to fix low airflow?"
5. Query: "What is the SEER rating?"

If all 5 pass → System is working ✅

**15-Minute Basic Test:**

- All 6 intent patterns (1 query each)
- 3 edge cases
- Check graph in Neo4j Browser
- Verify response times

**60-Minute Full Test:**

- Complete RIGOROUS_TESTING_GUIDE.md
- All intent patterns (multiple queries)
- All edge cases
- Data quality validation
- Performance benchmarking
- Consistency checks

---

## 💡 Pro Tips

1. **Copy-paste queries** from this file into Streamlit for consistent testing
2. **Keep Neo4j Browser open** to validate graph results
3. **Note response times** to identify slow queries
4. **Compare graph vs vector results** to verify hybrid search
5. **Test with multiple PDFs** if you have different documents
6. **Document failures** with exact query and error message
7. **Test both short and long PDFs** for performance comparison

---

Good luck with testing! 🎯
