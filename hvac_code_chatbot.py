import streamlit as st
import os
import glob
import PyPDF2
import pdfplumber
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from typing import List, Tuple, Dict, Any
import tempfile
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# Neo4j imports
from neo4j import GraphDatabase

# spaCy for NLP
import spacy

# OpenAI for embeddings (text-embedding-3-small)
from openai import OpenAI as OpenAIClient
import tiktoken

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("spaCy model not found. Please run: python -m spacy download en_core_web_sm")
    st.stop()

# Configuration from environment variables
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# HVAC-specific configuration
HVAC_DOCUMENTS_PATH = "hvac_documents/"
ATLANTA_CODES_PATH = "atlanta_codes/"

# HVAC Terminology Standardization Dictionary
HVAC_TERMINOLOGY = {
    # Equipment acronyms
    "AHU": "Air Handler",
    "VAV": "Variable Air Volume",
    "CAV": "Constant Air Volume",
    "DX": "Direct Expansion",
    "RTU": "Rooftop Unit",
    "FCU": "Fan Coil Unit",
    "PTAC": "Packaged Terminal Air Conditioner",
    "VRF": "Variable Refrigerant Flow",
    "VRV": "Variable Refrigerant Volume",
    "ERV": "Energy Recovery Ventilator",
    "HRV": "Heat Recovery Ventilator",
    "MAU": "Makeup Air Unit",
    "DOAS": "Dedicated Outdoor Air System",
    
    # Standards and codes
    "ASHRAE": "American Society of Heating Refrigerating and Air-Conditioning Engineers",
    "IMC": "International Mechanical Code",
    "IECC": "International Energy Conservation Code",
    "IFGC": "International Fuel Gas Code",
    "ACCA": "Air Conditioning Contractors of America",
    "SMACNA": "Sheet Metal and Air Conditioning Contractors National Association",
    
    # Measurements (keep as is for recognition)
    "BTU": "BTU",
    "CFM": "CFM",
    "SEER": "SEER",
    "EER": "EER",
    "AFUE": "AFUE",
    "COP": "COP",
    "HSPF": "HSPF"
}

# Known HVAC Manufacturers
KNOWN_MANUFACTURERS = {
    "Carrier", "Trane", "Lennox", "York", "Rheem", "Goodman", "Amana", 
    "Bryant", "Coleman", "Daikin", "Mitsubishi", "Fujitsu", "LG", 
    "American Standard", "Payne", "Heil", "Tempstar", "Comfortmaker",
    "Bosch", "Ruud", "Friedrich", "Nordyne", "Armstrong Air"
}

# Known HVAC Component Types
KNOWN_COMPONENTS = {
    "Air Handler", "Furnace", "Boiler", "Heat Pump", "Air Conditioner",
    "Chiller", "Compressor", "Condenser", "Evaporator", "Thermostat",
    "Ductwork", "Diffuser", "Register", "Grille", "Damper", "Filter",
    "Humidifier", "Dehumidifier", "Fan", "Blower", "Pump", "Valve",
    "Coil", "Plenum", "Expansion Valve", "Refrigerant Line"
}

# Known HVAC System Types
KNOWN_SYSTEM_TYPES = {
    "Heating System", "Cooling System", "Ventilation System", 
    "HVAC System", "Hydronic System", "Steam System", "Geothermal System",
    "Split System", "Packaged System", "Ductless System", "Zoned System"
}

# Common HVAC Problems
KNOWN_PROBLEMS = {
    "Refrigerant Leak", "Low Airflow", "Frozen Coil", "System Not Heating",
    "System Not Cooling", "Thermostat Malfunction", "Compressor Failure",
    "Fan Motor Failure", "Duct Leakage", "Poor Indoor Air Quality",
    "High Humidity", "Low Humidity", "Strange Noises", "Water Leakage",
    "Short Cycling", "Continuous Running"
}

# Initialize session state
if 'hvac_documents' not in st.session_state:
    st.session_state.hvac_documents = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'graphrag' not in st.session_state:
    st.session_state.graphrag = None
if 'retrieval_chain' not in st.session_state:
    st.session_state.retrieval_chain = None

class HVACCodeProcessor:
    """Specialized processor for HVAC code documents"""
    
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,  # Larger chunks for faster processing
            chunk_overlap=100,  # Smaller overlap
            length_function=len,
        )
    
    def standardize_hvac_term(self, term: str) -> str:
        """Standardize HVAC terminology to canonical form"""
        # Check if it's an acronym that needs expansion
        term_upper = term.upper().strip()
        if term_upper in HVAC_TERMINOLOGY:
            return HVAC_TERMINOLOGY[term_upper]
        return term.strip()
    
    def validate_entity(self, entity_text: str, entity_type: str) -> tuple:
        """
        Validate and standardize extracted entities
        Returns: (is_valid, standardized_text, confidence)
        """
        entity_text = entity_text.strip()
        
        # Standardize HVAC terms first
        standardized = self.standardize_hvac_term(entity_text)
        
        # Validate based on type
        if entity_type == "equipment":
            # Check against known components
            if standardized in KNOWN_COMPONENTS:
                return (True, standardized, 1.0)
            # Check for partial matches (fuzzy matching)
            for known in KNOWN_COMPONENTS:
                if known.lower() in standardized.lower() or standardized.lower() in known.lower():
                    return (True, known, 0.8)
            return (False, standardized, 0.5)
        
        elif entity_type == "brand":
            # Check against known manufacturers
            for manufacturer in KNOWN_MANUFACTURERS:
                if manufacturer.lower() in standardized.lower() or standardized.lower() in manufacturer.lower():
                    return (True, manufacturer, 0.9)
            # Allow unknown brands but with lower confidence
            if len(standardized) > 2 and standardized[0].isupper():
                return (True, standardized, 0.6)
            return (False, standardized, 0.3)
        
        elif entity_type == "system":
            if standardized in KNOWN_SYSTEM_TYPES:
                return (True, standardized, 1.0)
            for known in KNOWN_SYSTEM_TYPES:
                if known.lower() in standardized.lower():
                    return (True, known, 0.8)
            return (False, standardized, 0.5)
        
        elif entity_type == "problem":
            if standardized in KNOWN_PROBLEMS:
                return (True, standardized, 1.0)
            return (True, standardized, 0.7)  # Allow new problems
        
        # Default: accept but with medium confidence
        return (True, standardized, 0.7)
    
    def preprocess_pdf_text(self, text: str) -> str:
        """
        Remove noise from PDF text before entity extraction
        Strips headers, footers, page numbers, TOC, legal disclaimers
        """
        import re
        
        # Remove page numbers (various formats)
        text = re.sub(r'\n\s*Page\s+\d+\s*\n', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'\n\s*\d+\s*of\s*\d+\s*\n', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'\n\s*-\s*\d+\s*-\s*\n', '\n', text, flags=re.IGNORECASE)
        
        # Remove common headers/footers
        text = re.sub(r'\n\s*HVAC\s+CODE\s+BOOK.*?\n', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'\n\s*Copyright\s+©.*?\n', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'\n\s*All\s+rights\s+reserved.*?\n', '\n', text, flags=re.IGNORECASE)
        
        # Remove table of contents patterns
        text = re.sub(r'\.{3,}\s*\d+', '', text)  # Remove dot leaders with page numbers
        
        # Remove multiple consecutive blank lines
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Remove very short lines (often artifacts)
        lines = text.split('\n')
        filtered_lines = [line for line in lines if len(line.strip()) > 3 or line.strip() == '']
        
        return '\n'.join(filtered_lines)
        
    def load_hvac_documents(self, documents_path: str) -> List[Document]:
        """Load all HVAC documents from specified path"""
        documents = []
        
        if not os.path.exists(documents_path):
            os.makedirs(documents_path)
            print(f"Created directory {documents_path}. Please add your HVAC PDF files there.")
            return documents
        
        # Find all PDF files
        pdf_files = glob.glob(os.path.join(documents_path, "*.pdf"))
        
        for pdf_file in pdf_files:
            try:
                text = self.extract_text_from_pdf(pdf_file)
                if text.strip():
                    # Create document with metadata
                    doc = Document(
                        page_content=text,
                        metadata={
                            "source": os.path.basename(pdf_file),
                            "type": "hvac_code",
                            "location": "atlanta",
                            "processed_at": datetime.now().isoformat()
                        }
                    )
                    documents.append(doc)
            except Exception as e:
                print(f"Error loading {pdf_file}: {str(e)}")
        
        return documents
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF with HVAC-specific preprocessing"""
        text = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        # Add page context for HVAC codes
                        text += f"\n--- Page {page_num + 1} ---\n"
                        text += page_text + "\n"
            
            # Apply preprocessing to remove noise
            text = self.preprocess_pdf_text(text)
            
        except Exception as e:
            st.error(f"Error extracting text from {file_path}: {str(e)}")
        return text
    
    def chunk_by_sections(self, text: str, source: str) -> List[Document]:
        """
        Create chunks based on code sections for coherent context
        Looks for patterns like 'Section 301.1', 'Chapter 3', etc.
        """
        import re
        
        chunks = []
        
        # Try to split by section markers
        section_pattern = r'(Section\s+\d+(?:\.\d+)*(?:\.\d+)?(?:\s+[A-Z][^\.]+)?)'
        sections = re.split(section_pattern, text, flags=re.IGNORECASE)
        
        current_section = None
        current_text = ""
        
        for i, part in enumerate(sections):
            # Check if this is a section header
            if re.match(section_pattern, part, re.IGNORECASE):
                # Save previous section if exists
                if current_section and current_text.strip():
                    chunks.append(Document(
                        page_content=current_text.strip(),
                        metadata={
                            "source": source,
                            "section": current_section,
                            "chunk_type": "section-based"
                        }
                    ))
                # Start new section
                current_section = part.strip()
                current_text = part + "\n"
            else:
                current_text += part
        
        # Add last section
        if current_section and current_text.strip():
            chunks.append(Document(
                page_content=current_text.strip(),
                metadata={
                    "source": source,
                    "section": current_section,
                    "chunk_type": "section-based"
                }
            ))
        
        # Fallback to regular chunking if no sections found
        if len(chunks) == 0:
            return self.text_splitter.split_documents([Document(
                page_content=text,
                metadata={"source": source, "chunk_type": "regular"}
            )])
        
        return chunks
    
    def create_hvac_chunks(self, documents: List[Document]) -> List[Document]:
        """Create specialized chunks for HVAC code documents"""
        all_chunks = []
        
        for doc in documents:
            # Try section-based chunking first
            chunks = self.chunk_by_sections(doc.page_content, doc.metadata.get("source", "unknown"))
            
            # Enhance chunks with HVAC-specific metadata
            for chunk in chunks:
                chunk.metadata.update({
                    "chunk_type": chunk.metadata.get("chunk_type", "hvac_code"),
                    "atlanta_specific": self.is_atlanta_specific(chunk.page_content),
                    "code_section": chunk.metadata.get("section") or self.extract_code_section(chunk.page_content),
                    "hvac_system_type": self.identify_hvac_system_type(chunk.page_content),
                    "processed_at": doc.metadata.get("processed_at", datetime.now().isoformat())
                })
            
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def is_atlanta_specific(self, text: str) -> bool:
        """Check if text contains Atlanta-specific requirements"""
        atlanta_keywords = [
            "atlanta", "georgia", "ga", "fulton county", "dekalb county",
            "city of atlanta", "atlanta building code", "atlanta mechanical code"
        ]
        return any(keyword.lower() in text.lower() for keyword in atlanta_keywords)
    
    def extract_code_section(self, text: str) -> str:
        """Extract HVAC code section references"""
        import re
        
        # Look for common HVAC code section patterns
        patterns = [
            r"Section\s+(\d+\.?\d*)",
            r"Chapter\s+(\d+)",
            r"Article\s+(\d+)",
            r"(\d+\.\d+\.\d+)",
            r"Code\s+(\d+\.?\d*)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return "Unknown"
    
    def identify_hvac_system_type(self, text: str) -> str:
        """Identify the type of HVAC system mentioned"""
        system_types = {
            "residential": ["residential", "single family", "dwelling", "home"],
            "commercial": ["commercial", "office", "retail", "business"],
            "industrial": ["industrial", "manufacturing", "warehouse", "factory"],
            "institutional": ["school", "hospital", "church", "institutional"]
        }
        
        text_lower = text.lower()
        for system_type, keywords in system_types.items():
            if any(keyword in text_lower for keyword in keywords):
                return system_type
        
        return "general"

class HVACGraphRAG:
    """GraphRAG implementation for HVAC knowledge graph"""
    
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_pass: str):
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_pass = neo4j_pass
        self.driver = None
        self.processor = HVACCodeProcessor()  # Initialize processor for validation
        
        # Initialize OpenAI client for embeddings (text-embedding-3-small)
        self.openai_client = OpenAIClient(api_key=OPENAI_API_KEY)
        self.embedding_model = "text-embedding-3-small"
        self.embedding_dimension = 1536  # text-embedding-3-small default dimension
        
        # Token counter for chunking
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        self._connect_to_neo4j()
    
    def _connect_to_neo4j(self):
        """Connect to Neo4j with error handling"""
        try:
            self.driver = GraphDatabase.driver(
                self.neo4j_uri, 
                auth=(self.neo4j_user, self.neo4j_pass),
                max_connection_lifetime=30 * 60,  # 30 minutes
                max_connection_pool_size=50,
                connection_timeout=30,  # 30 seconds
                max_transaction_retry_time=30  # 30 seconds
            )
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
        except Exception as e:
            print(f"Could not connect to Neo4j: {str(e)}")
            self.driver = None
    
    def chunk_document_with_embeddings(self, document: Document) -> List[Dict]:
        """
        TASK 2: Create document chunks with embeddings for full document coverage
        
        Strategy: Paragraph-based chunking with 500-1000 token limit
        - Preserves narrative flow and semantic boundaries
        - Each chunk gets embedded with text-embedding-3-small
        - Chunks linked to Document via PART_OF relationships
        - Enables full document accessibility via vector similarity search
        """
        chunks = []
        text = document.page_content
        source = document.metadata.get('source', 'unknown')
        
        # Split by paragraphs first (natural semantic boundaries)
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        current_tokens = 0
        chunk_index = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_tokens = len(self.tokenizer.encode(para))
            
            # If single paragraph exceeds max, split it further
            if para_tokens > 1000:
                # Split long paragraph by sentences
                sentences = para.split('. ')
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    
                    sent_tokens = len(self.tokenizer.encode(sentence))
                    
                    if current_tokens + sent_tokens > 1000:
                        # Save current chunk
                        if current_chunk:
                            chunks.append(self._create_chunk_dict(
                                current_chunk, chunk_index, source
                            ))
                            chunk_index += 1
                            current_chunk = sentence
                            current_tokens = sent_tokens
                    else:
                        current_chunk += (". " if current_chunk else "") + sentence
                        current_tokens += sent_tokens
            
            # Normal paragraph processing
            elif current_tokens + para_tokens > 1000:
                # Save current chunk
                if current_chunk:
                    chunks.append(self._create_chunk_dict(
                        current_chunk, chunk_index, source
                    ))
                    chunk_index += 1
                    current_chunk = para
                    current_tokens = para_tokens
            else:
                # Add paragraph to current chunk
                current_chunk += ("\n\n" if current_chunk else "") + para
                current_tokens += para_tokens
        
        # Save last chunk
        if current_chunk:
            chunks.append(self._create_chunk_dict(
                current_chunk, chunk_index, source
            ))
        
        print(f"Created {len(chunks)} chunks for document {source}")
        return chunks
    
    def _create_chunk_dict(self, text: str, index: int, source: str) -> Dict:
        """Helper method to create chunk dictionary with embedding"""
        # Generate embedding using OpenAI text-embedding-3-small
        embedding = self._get_embedding(text)
        
        return {
            'text': text,
            'index': index,
            'source': source,
            'embedding': embedding,
            'token_count': len(self.tokenizer.encode(text))
        }
    
    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenAI text-embedding-3-small"""
        try:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=text,
                encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            # Return zero vector as fallback
            return [0.0] * self.embedding_dimension
    
    def create_hvac_entities(self, documents: List[Document]) -> List[Dict]:
        """
        TASK 1: Two-tier entity extraction
        
        Tier 1 (STRICT - for graph nodes): Confidence >= 0.6, strict filtering
        - Creates high-quality graph nodes with semantic relationships
        - Filters out noise, generic terms, non-HVAC entities
        
        Tier 2 (PERMISSIVE - for text chunks): Confidence >= 0.3, broader extraction
        - Captured in document chunks with embeddings (handled separately)
        - Ensures full document coverage for vector similarity search
        
        This method returns Tier 1 entities only (for graph structure)
        """
        if not documents:
            return []
            
        # Get the single document
        doc = documents[0]
        entities = []
        
        # Process document in larger chunks to reduce overhead
        chunk_size = 20000  # Larger chunks since we only have one document
        overlap = 1000  # Smaller overlap for speed
        
        # Split into chunks with minimal overlap
        for i in range(0, len(doc.page_content), chunk_size - overlap):
            chunk = doc.page_content[i:i + chunk_size]
            # Extract with STRICT filtering (Tier 1 - confidence >= 0.6)
            chunk_entities = self.extract_hvac_entities(chunk, strict=True)
            entities.extend(chunk_entities)
        
        # Advanced deduplication - merge similar entities
        # Groups "Air Handler", "air handler unit", "AHU" into single canonical form
        deduplicated = self.deduplicate_entities(entities)
        
        # Add source metadata
        for entity in deduplicated:
            entity['source'] = doc.metadata.get('source', 'unknown')
        
        print(f"Extracted {len(deduplicated)} high-quality entities (Tier 1 - graph nodes)")
        return deduplicated
    
    def deduplicate_entities(self, entities: List[Dict]) -> List[Dict]:
        """
        Advanced entity deduplication and normalization
        Merges similar entities like 'Air Handler' and 'air handler unit'
        """
        from difflib import SequenceMatcher
        
        unique_entities = []
        seen_texts = {}  # Maps normalized text -> entity
        
        for entity in entities:
            text = entity['text']
            text_lower = text.lower().strip()
            hvac_type = entity['hvac_type']
            
            # Normalize text (remove extra spaces, articles, common suffixes)
            normalized = text_lower.replace('  ', ' ')
            normalized = normalized.replace(' unit', '').replace(' system', '').strip()
            normalized = normalized.replace('the ', '').replace('a ', '').replace('an ', '').strip()
            
            # Check for exact match first
            key = (normalized, hvac_type)
            if key in seen_texts:
                # Already have this entity, skip duplicate
                continue
            
            # Check for fuzzy similarity to existing entities of same type
            is_duplicate = False
            for existing_key, existing_entity in seen_texts.items():
                if existing_key[1] != hvac_type:  # Different type, can't be duplicate
                    continue
                
                # Calculate similarity
                similarity = SequenceMatcher(None, normalized, existing_key[0]).ratio()
                
                # If very similar (>85%), consider it a duplicate
                if similarity > 0.85:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen_texts[key] = entity
                unique_entities.append(entity)
        
        return unique_entities
    
    def is_valid_hvac_entity(self, entity_text: str, hvac_type: str) -> bool:
        """
        Strict filtering to keep only meaningful HVAC entities
        Reduces noise and prevents thousands of low-value nodes
        """
        text_lower = entity_text.lower().strip()
        
        # Minimum length filter (avoid single letters, short abbreviations)
        if len(entity_text) < 3:
            return False
        
        # Blacklist common words that add no value
        blacklist = {
            'the', 'and', 'or', 'but', 'for', 'with', 'from', 'this', 'that',
            'these', 'those', 'are', 'was', 'were', 'been', 'being', 'have',
            'has', 'had', 'can', 'will', 'shall', 'may', 'must', 'should',
            'page', 'section', 'chapter', 'table', 'figure', 'note', 'see',
            'also', 'per', 'via', 'use', 'used', 'using', 'each', 'all',
            'such', 'other', 'any', 'some', 'more', 'most', 'less', 'least',
            'inc', 'llc', 'corp', 'ltd', 'co', 'company', 'copyright',
            'reserved', 'rights', 'property', 'disclaimer'
        }
        
        if text_lower in blacklist:
            return False
        
        # Filter generic numbers and dates (unless they're code references)
        if hvac_type not in ['code_section', 'btu_rating', 'cfm_rating', 'seer_rating']:
            if text_lower.isdigit() or text_lower.replace('.', '').isdigit():
                return False
        
        # Keep only HVAC-relevant entity types
        relevant_types = {
            'equipment', 'system', 'code', 'measurement', 'location',
            'equipment_model', 'code_section', 'standard_reference',
            'btu_rating', 'cfm_rating', 'seer_rating', 'system_type',
            'duct_spec', 'control_point'
        }
        
        if hvac_type not in relevant_types and hvac_type != 'general':
            return False
        
        # For 'general' type, must contain HVAC keywords
        if hvac_type == 'general':
            hvac_keywords = [
                'hvac', 'air', 'heat', 'cool', 'vent', 'duct', 'furnace',
                'boiler', 'pump', 'compressor', 'condenser', 'evaporator',
                'thermostat', 'filter', 'fan', 'blower', 'coil', 'refrigerant',
                'atlanta', 'georgia', 'ashrae', 'imc', 'iecc', 'code'
            ]
            if not any(keyword in text_lower for keyword in hvac_keywords):
                return False
        
        return True
    
    def extract_hvac_entities(self, text: str, strict: bool = True) -> List[Dict]:
        """
        Extract HVAC-specific entities using spaCy and custom patterns
        
        Args:
            text: Text to extract entities from
            strict: If True, use Tier 1 filtering (confidence >= 0.6) for graph nodes
                   If False, use Tier 2 filtering (confidence >= 0.3) for chunk embeddings
        """
        doc = nlp(text)
        entities = []
        
        # Confidence threshold based on tier
        confidence_threshold = 0.6 if strict else 0.3
        
        # Limit spaCy entity types to reduce noise
        relevant_labels = {'ORG', 'PRODUCT', 'GPE', 'FAC', 'CARDINAL'}
        
        # Standard spaCy entities (filtered to relevant types only)
        for ent in doc.ents:
            # Skip irrelevant entity types
            if ent.label_ not in relevant_labels:
                continue
                
            hvac_type = self.classify_hvac_entity(ent.text, ent.label_)
            
            # Apply strict or permissive filtering based on tier
            if strict and not self.is_valid_hvac_entity(ent.text, hvac_type):
                continue
            
            entities.append({
                "text": ent.text.strip(),
                "label": ent.label_,
                "hvac_type": hvac_type,
                "code_reference": self.extract_code_reference(text, ent.text),
                "atlanta_specific": "atlanta" in text.lower() or "georgia" in text.lower()
            })
        
        # HVAC-specific entity patterns (these are already focused)
        hvac_patterns = self.get_hvac_patterns()
        for pattern_name, pattern in hvac_patterns.items():
            matches = pattern.findall(text)
            for match in matches:
                # Apply strict or permissive filtering
                if strict and not self.is_valid_hvac_entity(match, pattern_name):
                    continue
                
                entities.append({
                    "text": match.strip(),
                    "label": "HVAC_" + pattern_name.upper(),
                    "hvac_type": pattern_name,
                    "code_reference": self.extract_code_reference(text, match),
                    "atlanta_specific": "atlanta" in text.lower()
                })
        
        return entities
    
    def classify_hvac_entity(self, text: str, label: str) -> str:
        """Classify entities as HVAC-specific types"""
        text_lower = text.lower()
        
        hvac_keywords = {
            "equipment": ["furnace", "boiler", "heat pump", "air conditioner", "chiller", "compressor"],
            "system": ["ductwork", "ventilation", "air handling", "hvac system", "mechanical system"],
            "code": ["iecc", "ashrae", "imc", "ifgc", "building code", "mechanical code"],
            "measurement": ["btu", "cfm", "ton", "seer", "hspf", "afue"],
            "location": ["attic", "basement", "crawl space", "mechanical room", "equipment room"]
        }
        
        for hvac_type, keywords in hvac_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return hvac_type
        
        return "general"
    
    def get_hvac_patterns(self):
        """Get comprehensive regex patterns for HVAC-specific entities"""
        import re
        
        return {
            # Equipment and Models
            "equipment_model": re.compile(r'\b[A-Z]{2,4}[-]?\d{3,6}[A-Z]?\b', re.IGNORECASE),
            "serial_number": re.compile(r'\b[A-Z]\d{2}-\d{4}[A-Z]\b', re.IGNORECASE),
            
            # Measurements and Ratings
            "btu_rating": re.compile(r'\b\d{1,6}(?:,\d{3})*\s*(?:BTU/?h?|BTUH?)\b', re.IGNORECASE),
            "cfm_rating": re.compile(r'\b\d{1,4}(?:,\d{3})*\s*CFM\b', re.IGNORECASE),
            "seer_rating": re.compile(r'\b\d{1,2}(?:\.\d{1,2})?\s*SEER\b', re.IGNORECASE),
            "efficiency_rating": re.compile(r'\b\d{2,3}%?\s*(?:AFUE|EER|COP)\b', re.IGNORECASE),
            "power_rating": re.compile(r'\b\d+(?:\.\d+)?\s*(?:HP|KW|TON)\b', re.IGNORECASE),
            
            # Code References
            "code_section": re.compile(r'(?:Section|§)\s*\d+(?:\.\d+)*(?:\([a-z]\))?', re.IGNORECASE),
            "standard_reference": re.compile(r'\b(?:ASHRAE|ACCA|SMACNA|IMC|IECC)\s*(?:\d{2,4}(?:-\d{1,2})?)?', re.IGNORECASE),
            
            # Measurements
            "temperature": re.compile(r'\b\d{1,3}(?:\.\d{1,2})?\s*°?[FC]\b', re.IGNORECASE),
            "pressure": re.compile(r'\b\d+(?:\.\d+)?\s*(?:PSI|WC|Pa|kPa)\b', re.IGNORECASE),
            "velocity": re.compile(r'\b\d+(?:\.\d+)?\s*(?:FPM|MPH|m/s)\b', re.IGNORECASE),
            "dimension": re.compile(r'\b\d+(?:\.\d+)?\s*(?:inch|in|ft|mm|cm|m)(?:\s*x\s*\d+(?:\.\d+)?\s*(?:inch|in|ft|mm|cm|m)){0,2}\b', re.IGNORECASE),
            
            # System Components
            "duct_spec": re.compile(r'\b(?:supply|return|exhaust)\s+(?:duct|plenum|register|grille|diffuser)\b', re.IGNORECASE),
            "system_type": re.compile(r'\b(?:VAV|CAV|RTU|AHU|FCU|PTAC|VRF|VRV)\b', re.IGNORECASE),
            "control_point": re.compile(r'\b(?:thermostat|sensor|controller|BMS|setpoint)\b', re.IGNORECASE)
        }
    
    def extract_code_reference(self, text: str, entity: str) -> str:
        """Extract code reference near an entity"""
        import re
        
        # Find code references near the entity
        entity_pos = text.lower().find(entity.lower())
        if entity_pos != -1:
            # Look for code patterns in surrounding text
            start = max(0, entity_pos - 100)
            end = min(len(text), entity_pos + 100)
            context = text[start:end]
            
            code_patterns = [
                r'Section\s+\d+\.?\d*',
                r'Chapter\s+\d+',
                r'Article\s+\d+',
                r'\d+\.\d+\.\d+'
            ]
            
            for pattern in code_patterns:
                match = re.search(pattern, context, re.IGNORECASE)
                if match:
                    return match.group(0)
        
        return ""
    
    def create_knowledge_graph(self, entities: List[Dict], documents: List[Document]):
        """
        Create hybrid Graph + Vector RAG system
        
        Components:
        1. Graph nodes (high-quality entities with semantic relationships)
        2. Chunk nodes (document chunks with embeddings for full coverage)
        3. Vector index (Neo4j native cosine similarity search)
        4. MENTIONS relationships (chunks to entities, confidence >= 0.85)
        
        NOTE: Uses multiple transactions to avoid timeouts
        """
        if not self.driver:
            print("Neo4j not available, skipping knowledge graph creation")
            return
            
        try:
            # STEP 1: Create chunks with embeddings (OUTSIDE transaction - slow API calls)
            print("Creating document chunks with embeddings...")
            doc = documents[0]
            chunks = self.chunk_document_with_embeddings(doc)
            print(f"✓ Created {len(chunks)} chunks with embeddings")
            
            # STEP 2: Clear existing data (separate transaction)
            print("Clearing existing graph data...")
            with self.driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
            print("✓ Graph cleared")
            
            # STEP 3: Create Document node (quick transaction)
            print("Creating Document node...")
            with self.driver.session() as session:
                session.run("""
                    CREATE (d:Document {
                        source: $source,
                        type: $type,
                        location: $location,
                        processed_at: $processed_at,
                        title: $title
                    })
                """, 
                source=doc.metadata.get("source", "unknown"),
                type=doc.metadata.get("type", "hvac_code"),
                location=doc.metadata.get("location", "atlanta"),
                processed_at=datetime.now().isoformat(),
                title=doc.metadata.get("source", "HVAC Document"))
            print("✓ Document node created")
            
            # STEP 4: Create Chunk nodes in batches (separate transactions to avoid timeout)
            print("Creating Chunk nodes in batches...")
            batch_size = 10  # Smaller batches to avoid timeout
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                with self.driver.session() as session:
                    for chunk in batch:
                        session.run("""
                            MATCH (d:Document {source: $source})
                            CREATE (c:Chunk {
                                text: $text,
                                chunk_index: $index,
                                source: $source,
                                token_count: $token_count,
                                embedding: $embedding
                            })
                            CREATE (c)-[:PART_OF]->(d)
                        """,
                        source=chunk['source'],
                        text=chunk['text'],
                        index=chunk['index'],
                        token_count=chunk['token_count'],
                        embedding=chunk['embedding'])
                print(f"  Progress: {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")
            print(f"✓ Created {len(chunks)} Chunk nodes")
            
            # STEP 5: Create Entity nodes (separate transaction)
            print("Creating Entity nodes...")
            with self.driver.session() as session:
                for e in entities:
                    # Determine strict node label based on entity type
                    node_label = self.get_strict_node_label(e)
                    standardized_text = e['text']
                    
                    # Standardize entity text if it's a term
                    if e['hvac_type'] in ['equipment', 'system']:
                        standardized_text = self.processor.standardize_hvac_term(e['text'])
                    
                    # Validate entity
                    is_valid, validated_text, confidence = self.processor.validate_entity(standardized_text, e['hvac_type'])
                    
                    # Stricter confidence threshold to reduce noise (was 0.5, now 0.6)
                    if not is_valid or confidence < 0.6:
                        continue  # Skip invalid or low-confidence entities
                    
                    # Create entity with strict label
                    params = {
                        'text': validated_text,
                        'original_text': e['text'],
                        'label': e['label'],
                        'hvac_type': e['hvac_type'],
                        'code_reference': e.get('code_reference', ''),
                        'atlanta_specific': e.get('atlanta_specific', False),
                        'importance': self.calculate_entity_importance(e),
                        'confidence': confidence,
                        'extracted_at': datetime.now().isoformat()
                    }
                    
                    # Use MERGE to prevent duplicates, with strict node labels
                    query = f"""
                    MERGE (e:{node_label} {{text: $text}})
                    SET e.original_text = $original_text,
                        e.label = $label,
                        e.hvac_type = $hvac_type,
                        e.code_reference = $code_reference,
                        e.atlanta_specific = $atlanta_specific,
                        e.importance = $importance,
                        e.confidence = $confidence,
                        e.extracted_at = $extracted_at
                    """
                    session.run(query, **params)
            print(f"✓ Created Entity nodes")
            
            # STEP 6: Create semantic relationships (separate transaction)
            print("Creating semantic relationships...")
            with self.driver.session() as session:
                self.create_entity_relationships(session, entities)
            print("✓ Semantic relationships created")
            
            # STEP 7: Create HVAC hierarchy (separate transaction)
            print("Creating HVAC hierarchy...")
            with self.driver.session() as session:
                self.create_hvac_hierarchy(session)
            print("✓ HVAC hierarchy created")
            
            # STEP 8: Create MENTIONS relationships in batches (separate transaction)
            print("Creating MENTIONS relationships (chunk-to-entity)...")
            mentions_count = self.create_chunk_entity_mentions_batched(chunks, entities)
            print(f"✓ Created {mentions_count} MENTIONS relationships")
            
            # STEP 9: Create vector index (after all nodes created)
            print("Creating vector index on Chunk nodes...")
            self.create_vector_index()
            print("✓ Vector index created successfully")
            
            print("\n✅ Knowledge graph creation complete!")
                
        except Exception as e:
            print(f"Error creating knowledge graph: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def create_chunk_entity_mentions_batched(self, chunks: List[Dict], entities: List[Dict]) -> int:
        """
        TASK 4: Create MENTIONS relationships from chunks to entities (BATCHED version)
        
        Uses separate transactions in batches to avoid timeout
        """
        mentions_count = 0
        batch_size = 50  # Process 50 chunks at a time
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            with self.driver.session() as session:
                for chunk in batch:
                    chunk_text_lower = chunk['text'].lower()
                    
                    for entity in entities:
                        entity_text = entity['text']
                        entity_text_lower = entity_text.lower()
                        
                        # Check if entity is mentioned in chunk
                        if entity_text_lower in chunk_text_lower:
                            # Calculate confidence based on entity importance and length
                            confidence = 0.85
                            
                            # Boost confidence for longer, more specific entities
                            if len(entity_text) > 10:
                                confidence = 0.90
                            if len(entity_text) > 20:
                                confidence = 0.95
                            
                            # Get entity node label
                            node_label = self.get_strict_node_label(entity)
                            
                            # Create MENTIONS relationship
                            try:
                                session.run(f"""
                                    MATCH (c:Chunk {{chunk_index: $chunk_index}})
                                    MATCH (e:{node_label} {{text: $entity_text}})
                                    MERGE (c)-[r:MENTIONS]->(e)
                                    SET r.confidence = $confidence,
                                        r.created_at = $created_at
                                """,
                                chunk_index=chunk['index'],
                                entity_text=entity_text,
                                confidence=confidence,
                                created_at=datetime.now().isoformat())
                                
                                mentions_count += 1
                            except Exception as e:
                                # Entity might not exist (filtered out), skip silently
                                pass
            
            # Progress update
            if (i + batch_size) % 100 == 0 or (i + batch_size) >= len(chunks):
                print(f"  Progress: {min(i + batch_size, len(chunks))}/{len(chunks)} chunks processed")
        
        return mentions_count
    
    def create_chunk_entity_mentions(self, tx, chunks: List[Dict], entities: List[Dict]):
        """
        LEGACY: Original single-transaction version (kept for reference, not used)
        
        TASK 4: Create MENTIONS relationships from chunks to entities
        
        Only creates relationships when:
        - Entity text appears in chunk text
        - Confidence >= 0.85 (high threshold to avoid noise)
        
        Enables provenance tracking: "Which chunks mention this entity?"
        """
        mentions_count = 0
        
        for chunk in chunks:
            chunk_text_lower = chunk['text'].lower()
            
            for entity in entities:
                entity_text = entity['text']
                entity_text_lower = entity_text.lower()
                
                # Check if entity is mentioned in chunk
                if entity_text_lower in chunk_text_lower:
                    # Calculate confidence based on entity importance and length
                    confidence = 0.85
                    
                    # Boost confidence for longer, more specific entities
                    if len(entity_text) > 10:
                        confidence = 0.90
                    if len(entity_text) > 20:
                        confidence = 0.95
                    
                    # Get entity node label
                    node_label = self.get_strict_node_label(entity)
                    
                    # Create MENTIONS relationship
                    tx.run(f"""
                        MATCH (c:Chunk {{chunk_index: $chunk_index}})
                        MATCH (e:{node_label} {{text: $entity_text}})
                        MERGE (c)-[r:MENTIONS]->(e)
                        SET r.confidence = $confidence,
                            r.created_at = $created_at
                    """,
                    chunk_index=chunk['index'],
                    entity_text=entity_text,
                    confidence=confidence,
                    created_at=datetime.now().isoformat())
                    
                    mentions_count += 1
        
        print(f"Created {mentions_count} MENTIONS relationships (chunk-to-entity)")
    
    def create_vector_index(self):
        """
        TASK 3: Create Neo4j vector index on Chunk.embedding property
        
        Enables fast cosine similarity search for:
        - Finding relevant chunks based on question embedding
        - Hybrid retrieval (graph traversal + vector similarity)
        """
        if not self.driver:
            return
        
        try:
            with self.driver.session() as session:
                # Create vector index with cosine similarity
                session.run("""
                    CREATE VECTOR INDEX chunk_embedding_index IF NOT EXISTS
                    FOR (c:Chunk)
                    ON c.embedding
                    OPTIONS {
                        indexConfig: {
                            `vector.dimensions`: $dimension,
                            `vector.similarity_function`: 'cosine'
                        }
                    }
                """, dimension=self.embedding_dimension)
                
                print(f"Vector index created: {self.embedding_dimension}D cosine similarity")
        except Exception as e:
            print(f"Error creating vector index: {str(e)}")
    
    def get_strict_node_label(self, entity: Dict) -> str:
        """Get strict node label based on entity type and content"""
        text_lower = entity['text'].lower()
        hvac_type = entity['hvac_type']
        
        # Manufacturers/Brands
        if any(brand.lower() in text_lower for brand in KNOWN_MANUFACTURERS):
            return "Brand"
        
        # Code sections
        if hvac_type == 'code' or 'section' in text_lower or 'chapter' in text_lower:
            return "Code"
        
        # Problems/Issues
        if any(problem.lower() in text_lower for problem in KNOWN_PROBLEMS):
            return "Problem"
        
        # Solutions (keywords: replace, repair, fix, install, adjust)
        if any(keyword in text_lower for keyword in ['replace', 'repair', 'fix', 'install', 'adjust', 'clean', 'inspect']):
            return "Solution"
        
        # Locations
        if hvac_type == 'location' or any(loc in text_lower for loc in ['atlanta', 'zone', 'room', 'building', 'floor']):
            return "Location"
        
        # Default: Components (equipment, systems, parts)
        return "Component"
    
    def calculate_entity_importance(self, entity: Dict) -> str:
        """Calculate importance level of an entity"""
        text = entity["text"].lower()
        hvac_type = entity["hvac_type"]
        
        # High importance entities
        if any(keyword in text for keyword in ["section", "chapter", "article", "code"]):
            return "high"
        elif hvac_type in ["equipment", "system"]:
            return "medium"
        else:
            return "low"
    
    def create_entity_relationships(self, session, entities: List[Dict]):
        """Create rich semantic relationships between HVAC entities"""
        # Enhanced entity grouping with subcategories
        entity_groups = {
            "equipment": {
                "heating": [e for e in entities if e["hvac_type"] == "equipment" and any(k in e["text"].lower() for k in ["furnace", "boiler", "heat pump", "radiator"])],
                "cooling": [e for e in entities if e["hvac_type"] == "equipment" and any(k in e["text"].lower() for k in ["air conditioner", "chiller", "cooling coil"])],
                "ventilation": [e for e in entities if e["hvac_type"] == "equipment" and any(k in e["text"].lower() for k in ["fan", "blower", "air handler", "ventilator"])],
                "control": [e for e in entities if e["hvac_type"] == "equipment" and any(k in e["text"].lower() for k in ["thermostat", "sensor", "controller"])]
            },
            "system": [e for e in entities if e["hvac_type"] == "system"],
            "code": [e for e in entities if e["hvac_type"] == "code"],
            "measurement": {
                "temperature": [e for e in entities if e["hvac_type"] == "measurement" and any(k in e["text"].lower() for k in ["°f", "°c", "degree"])],
                "flow": [e for e in entities if e["hvac_type"] == "measurement" and any(k in e["text"].lower() for k in ["cfm", "gpm"])],
                "pressure": [e for e in entities if e["hvac_type"] == "measurement" and any(k in e["text"].lower() for k in ["psi", "pa", "wc"])],
                "efficiency": [e for e in entities if e["hvac_type"] == "measurement" and any(k in e["text"].lower() for k in ["seer", "eer", "cop", "afue"])]
            },
            "location": [e for e in entities if e["hvac_type"] == "location"],
            "requirement": [e for e in entities if any(k in e["text"].lower() for k in ["must", "shall", "required", "minimum", "maximum"])]
        }
        
        # Create hierarchy nodes first
        self.create_hvac_hierarchy(session)
        
        # Process main entity categories
        equipment_entities = [e for e in entities if e["hvac_type"] == "equipment"]
        system_entities = [e for e in entities if e["hvac_type"] == "system"]
        code_entities = [e for e in entities if e["hvac_type"] == "code"]
        measurement_entities = [e for e in entities if e["hvac_type"] == "measurement"]
        location_entities = [e for e in entities if e["hvac_type"] == "location"]
        
        # 1. Equipment-System Relationships
        self.create_equipment_system_relationships(session, equipment_entities, system_entities)
        
        # 2. Code Compliance Relationships
        self.create_code_compliance_relationships(session, entities)
        
        # 3. Equipment Specifications Relationships
        self.create_equipment_spec_relationships(session, equipment_entities, measurement_entities)
        
        # 4. Location-Based Relationships
        self.create_location_relationships(session, entities, location_entities)
        
        # 5. HVAC System Hierarchy
        self.create_hvac_hierarchy_relationships(session, entities)
        
        # 6. Code Section Relationships
        self.create_code_section_relationships(session, code_entities)
        
        # 7. Problem-Solution Relationships
        self.create_problem_solution_relationships(session, entities)
        
        # 8. Brand-Component Relationships
        self.create_brand_component_relationships(session, entities)
    
    def create_equipment_system_relationships(self, session, equipment_entities, system_entities):
        """Create specific equipment-system relationships with strict types"""
        equipment_system_mapping = {
            "furnace": ["heating system", "hvac system", "residential hvac"],
            "boiler": ["heating system", "hydronic system", "steam system"],
            "heat pump": ["hvac system", "heating system", "cooling system"],
            "air conditioner": ["cooling system", "hvac system", "ac system"],
            "chiller": ["cooling system", "commercial hvac", "water cooling"],
            "air handler": ["hvac system", "air handling unit", "ventilation system"],
            "ductwork": ["ventilation system", "air distribution", "hvac system"]
        }
        
        for equipment in equipment_entities:
            equipment_text = equipment["text"].lower()
            standardized_eq = self.processor.standardize_hvac_term(equipment["text"])
            
            for system in system_entities:
                system_text = system["text"].lower()
                standardized_sys = self.processor.standardize_hvac_term(system["text"])
                
                # Check if this equipment belongs to this system
                confidence = 0.0
                for keyword, systems in equipment_system_mapping.items():
                    if keyword in equipment_text and any(sys in system_text for sys in systems):
                        confidence = 0.95  # High confidence for known mappings
                        
                        session.run("""
                            MATCH (e:Component {text: $equipment})
                            MATCH (s:Component {text: $system})
                            MERGE (e)-[r:PART_OF]->(s)
                            SET r.confidence = $confidence,
                                r.source = 'entity_extraction',
                                r.relationship_type = 'component_hierarchy',
                                r.created_at = $created_at
                        """, 
                        equipment=standardized_eq, 
                        system=standardized_sys,
                        confidence=confidence,
                        created_at=datetime.now().isoformat())
    
    def create_code_compliance_relationships(self, session, entities):
        """Create code compliance relationships with strict types"""
        for entity in entities:
            if entity["code_reference"]:
                standardized_entity = self.processor.standardize_hvac_term(entity["text"])
                
                # Entity is governed by this code section
                session.run("""
                    MATCH (e) WHERE e.text = $entity AND (e:Component OR e:Brand OR e:Problem OR e:Solution)
                    MATCH (c:Code {text: $code_ref})
                    MERGE (e)-[r:COMPLIES_WITH]->(c)
                    SET r.compliance_type = 'code_requirement',
                        r.confidence = 0.9,
                        r.source = 'code_reference',
                        r.created_at = $created_at
                """, 
                entity=standardized_entity, 
                code_ref=entity["code_reference"],
                created_at=datetime.now().isoformat())
            
            # Create relationships based on HVAC standards
            text_lower = entity["text"].lower()
            if any(std in text_lower for std in ["ashrae", "iecc", "imc", "ifgc"]):
                session.run("""
                    MATCH (e:Code {text: $entity})
                    MERGE (s:Standard {name: $standard, type: 'hvac_standard'})
                    MERGE (e)-[r:FOLLOWS]->(s)
                    SET r.confidence = 0.95,
                        r.source = 'standard_reference',
                        r.created_at = $created_at
                """, 
                entity=entity["text"], 
                standard=entity["text"],
                created_at=datetime.now().isoformat())
    
    def create_equipment_spec_relationships(self, session, equipment_entities, measurement_entities):
        """Create equipment specification relationships with confidence scoring"""
        for equipment in equipment_entities:
            standardized_eq = self.processor.standardize_hvac_term(equipment["text"])
            
            for measurement in measurement_entities:
                if self.are_equipment_spec_related(equipment["text"], measurement["text"]):
                    # Calculate confidence based on proximity and context
                    confidence = 0.85  # Default confidence for spec relationships
                    
                    session.run("""
                        MATCH (e:Component {text: $equipment})
                        MATCH (m) WHERE m.text = $measurement AND (m:Component OR m:Measurement)
                        MERGE (e)-[r:HAS_SPECIFICATION]->(m)
                        SET r.spec_type = 'performance_rating',
                            r.confidence = $confidence,
                            r.source = 'entity_extraction',
                            r.created_at = $created_at
                    """, 
                    equipment=standardized_eq, 
                    measurement=measurement["text"],
                    confidence=confidence,
                    created_at=datetime.now().isoformat())
    
    def create_location_relationships(self, session, entities, location_entities):
        """Create location-based relationships with strict types"""
        for entity in entities:
            standardized_entity = self.processor.standardize_hvac_term(entity["text"])
            
            for location in location_entities:
                if self.are_location_related(entity["text"], location["text"]):
                    confidence = 0.8  # Default confidence for location relationships
                    
                    session.run("""
                        MATCH (e) WHERE e.text = $entity AND (e:Component OR e:Brand OR e:Problem)
                        MATCH (l:Location {text: $location})
                        MERGE (e)-[r:INSTALLED_IN]->(l)
                        SET r.installation_type = 'physical_location',
                            r.confidence = $confidence,
                            r.source = 'entity_extraction',
                            r.created_at = $created_at
                    """, 
                    entity=standardized_entity, 
                    location=location["text"],
                    confidence=confidence,
                    created_at=datetime.now().isoformat())
    
    def create_hvac_hierarchy_relationships(self, session, entities):
        """Create HVAC system hierarchy relationships with strict ontology"""
        # Create main HVAC categories with strict labels
        categories = [
            {"name": "Heating Systems", "subsystems": ["furnace", "boiler", "heat pump"]},
            {"name": "Cooling Systems", "subsystems": ["air conditioner", "chiller", "heat pump"]},
            {"name": "Ventilation Systems", "subsystems": ["air handler", "ductwork", "exhaust fan"]},
            {"name": "Control Systems", "subsystems": ["thermostat", "control panel", "sensor"]}
        ]
        
        for category in categories:
            session.run("""
                MERGE (c:Category {name: $name, type: 'hvac_category'})
            """, name=category["name"])
            
            for entity in entities:
                entity_text_lower = entity["text"].lower()
                standardized_entity = self.processor.standardize_hvac_term(entity["text"])
                
                if any(subsystem in entity_text_lower for subsystem in category["subsystems"]):
                    session.run("""
                        MATCH (e:Component {text: $entity})
                        MATCH (c:Category {name: $category})
                        MERGE (e)-[r:BELONGS_TO]->(c)
                        SET r.category_type = 'hvac_system',
                            r.confidence = 0.9,
                            r.source = 'category_mapping',
                            r.created_at = $created_at
                    """, 
                    entity=standardized_entity, 
                    category=category["name"],
                    created_at=datetime.now().isoformat())
    
    def create_code_section_relationships(self, session, code_entities):
        """Create relationships between code sections with confidence scoring"""
        for code1 in code_entities:
            for code2 in code_entities:
                if code1["text"] != code2["text"] and self.are_code_sections_related(code1["text"], code2["text"]):
                    confidence = 0.75  # Default confidence for code relationships
                    
                    session.run("""
                        MATCH (c1:Code {text: $code1})
                        MATCH (c2:Code {text: $code2})
                        MERGE (c1)-[r:RELATED_TO]->(c2)
                        SET r.relationship_type = 'code_reference',
                            r.confidence = $confidence,
                            r.source = 'section_analysis',
                            r.created_at = $created_at
                    """, 
                    code1=code1["text"], 
                    code2=code2["text"],
                    confidence=confidence,
                    created_at=datetime.now().isoformat())
    
    def create_problem_solution_relationships(self, session, entities):
        """Create Problem -> Solution relationships with confidence scoring"""
        # Common HVAC problem-solution mappings
        problem_solution_mapping = {
            "refrigerant leak": ["repair leak", "recharge refrigerant", "replace coil"],
            "low airflow": ["clean filter", "clear ductwork", "adjust dampers", "replace blower"],
            "frozen coil": ["check airflow", "clean filter", "check refrigerant", "adjust thermostat"],
            "high energy bills": ["tune up system", "seal ducts", "upgrade thermostat", "improve insulation"],
            "noisy operation": ["lubricate bearings", "tighten components", "replace worn parts"],
            "short cycling": ["adjust thermostat", "check refrigerant", "clean coils", "replace capacitor"],
            "uneven heating": ["balance system", "seal ducts", "upgrade zones", "adjust dampers"],
            "uneven cooling": ["balance system", "seal ducts", "upgrade zones", "adjust dampers"]
        }
        
        problem_entities = [e for e in entities if any(problem.lower() in e["text"].lower() for problem in KNOWN_PROBLEMS)]
        solution_entities = [e for e in entities if any(keyword in e["text"].lower() for keyword in ['replace', 'repair', 'fix', 'install', 'adjust', 'clean', 'inspect'])]
        
        for problem in problem_entities:
            problem_text_lower = problem["text"].lower()
            
            for solution in solution_entities:
                solution_text_lower = solution["text"].lower()
                
                # Check if problem and solution are related
                confidence = 0.0
                for prob_key, solutions in problem_solution_mapping.items():
                    if prob_key in problem_text_lower:
                        for sol in solutions:
                            if sol in solution_text_lower:
                                confidence = 0.9  # High confidence for known mappings
                                break
                
                if confidence > 0:
                    session.run("""
                        MATCH (p:Problem {text: $problem})
                        MATCH (s:Solution {text: $solution})
                        MERGE (p)-[r:RESOLVED_BY]->(s)
                        SET r.confidence = $confidence,
                            r.source = 'problem_solution_mapping',
                            r.created_at = $created_at
                    """, 
                    problem=problem["text"], 
                    solution=solution["text"],
                    confidence=confidence,
                    created_at=datetime.now().isoformat())
    
    def create_brand_component_relationships(self, session, entities):
        """Create Brand -> Component (MANUFACTURED_BY) relationships"""
        brand_entities = [e for e in entities if any(brand.lower() in e["text"].lower() for brand in KNOWN_MANUFACTURERS)]
        component_entities = [e for e in entities if any(comp.lower() in e["text"].lower() for comp in KNOWN_COMPONENTS)]
        
        for component in component_entities:
            comp_text = component["text"]
            
            for brand in brand_entities:
                brand_text = brand["text"]
                
                # Check if brand and component appear in proximity (basic heuristic)
                # In practice, this would use document context or co-occurrence
                if self.are_brand_component_related(brand_text, comp_text):
                    confidence = 0.85  # High confidence for manufacturer relationships
                    
                    session.run("""
                        MATCH (c:Component {text: $component})
                        MATCH (b:Brand {text: $brand})
                        MERGE (c)-[r:MANUFACTURED_BY]->(b)
                        SET r.confidence = $confidence,
                            r.source = 'entity_extraction',
                            r.created_at = $created_at
                    """, 
                    component=comp_text, 
                    brand=brand_text,
                    confidence=confidence,
                    created_at=datetime.now().isoformat())
    
    def are_brand_component_related(self, brand_text: str, component_text: str) -> bool:
        """Check if brand and component are related (basic heuristic)"""
        # In production, this would check document proximity or co-occurrence
        # For now, return True if both are valid entities
        return True  # Simplified - in production, check context proximity
    
    def are_equipment_spec_related(self, equipment_text: str, measurement_text: str) -> bool:
        """Check if equipment and measurement are related"""
        equipment_lower = equipment_text.lower()
        measurement_lower = measurement_text.lower()
        
        # BTU ratings for heating equipment
        if any(equip in equipment_lower for equip in ["furnace", "boiler", "heat pump"]) and "btu" in measurement_lower:
            return True
        
        # SEER ratings for cooling equipment
        if any(equip in equipment_lower for equip in ["air conditioner", "heat pump"]) and "seer" in measurement_lower:
            return True
        
        # CFM ratings for air handling equipment
        if any(equip in equipment_lower for equip in ["air handler", "fan", "blower"]) and "cfm" in measurement_lower:
            return True
        
        return False
    
    def are_location_related(self, entity_text: str, location_text: str) -> bool:
        """Check if entity and location are related"""
        entity_lower = entity_text.lower()
        location_lower = location_text.lower()
        
        # Equipment typically installed in specific locations
        location_equipment_map = {
            "attic": ["air handler", "ductwork", "insulation"],
            "basement": ["furnace", "boiler", "water heater"],
            "mechanical room": ["chiller", "boiler", "air handler"],
            "crawl space": ["ductwork", "insulation", "vapor barrier"]
        }
        
        for location, equipment_list in location_equipment_map.items():
            if location in location_lower:
                return any(equip in entity_lower for equip in equipment_list)
        
        return False
    
    def are_code_sections_related(self, code1: str, code2: str) -> bool:
        """Check if two code sections are related"""
        # Extract section numbers
        import re
        section1 = re.search(r'(\d+\.\d+)', code1)
        section2 = re.search(r'(\d+\.\d+)', code2)
        
        if section1 and section2:
            # Same chapter (first number)
            return section1.group(1).split('.')[0] == section2.group(1).split('.')[0]
        
        return False
    
    def are_related(self, text1: str, text2: str) -> bool:
        """Check if two entities are related"""
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        
        # Simple relationship detection
        hvac_terms = ["hvac", "heating", "ventilation", "air conditioning", "duct", "furnace", "boiler"]
        return any(term in text1_lower and term in text2_lower for term in hvac_terms)
    
    def create_hvac_hierarchy(self, session):
        """Create HVAC system hierarchy"""
        # Create main HVAC categories
        categories = [
            {"name": "Heating Systems", "type": "category"},
            {"name": "Ventilation Systems", "type": "category"},
            {"name": "Air Conditioning", "type": "category"},
            {"name": "Ductwork", "type": "category"},
            {"name": "Controls", "type": "category"}
        ]
        
        for category in categories:
            session.run("""
                MERGE (c:HVACCategory {name: $name})
                SET c.type = $type
            """, name=category["name"], type=category["type"])

class HVACChatbot:
    """Specialized HVAC code chatbot with GraphRAG and LangChain"""
    
    def __init__(self):
        self.processor = HVACCodeProcessor()
        self.graphrag = HVACGraphRAG(NEO4J_URI, NEO4J_USER, NEO4J_PASS)
        self.llm = None
        self.vector_store = None
        self.retrieval_chain = None
        self.initialized = False
    
    def initialize_system(self):
        """Initialize the entire system with documents and knowledge graph"""
        if self.initialized:
            return True
            
        try:
            # Initialize LLM
            if OPENAI_API_KEY:
                self.llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.1)
            else:
                st.error("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
                return False
            
            # Load and process documents
            documents = self.processor.load_hvac_documents(HVAC_DOCUMENTS_PATH)
            
            if not documents:
                st.warning("No HVAC documents found. Please add PDF files to the hvac_documents/ folder.")
                return False
            
            # Create smaller chunks for faster processing
            chunks = self.processor.create_hvac_chunks(documents)
            self.vector_store = FAISS.from_documents(chunks, self.processor.embeddings)
            
            # Create retrieval chain first (most important for chat)
            self.retrieval_chain = self.create_retrieval_chain(self.vector_store)
            self.initialized = True
            
            # Create optimized knowledge graph
            try:
                print("Creating knowledge graph from HVAC document...")
                # Extract entities first
                entities = self.graphrag.create_hvac_entities(documents)
                
                # Create the knowledge graph with batched operations
                self.graphrag.create_knowledge_graph(entities, documents)
                print("Knowledge graph creation completed")
            except Exception as neo4j_error:
                print(f"Neo4j knowledge graph creation failed: {str(neo4j_error)}")
            
            return True
                
        except Exception as e:
            st.error(f"Error initializing system: {str(e)}")
            return False
    
    def initialize_llm(self, api_key: str):
        """Initialize OpenAI LLM"""
        self.llm = OpenAI(openai_api_key=api_key, temperature=0, request_timeout=60)
    
    def create_retrieval_chain(self, vector_store):
        """Create LangChain retrieval chain for HVAC queries"""
        if not self.llm:
            st.error("Please provide OpenAI API key to initialize the LLM")
            return None
        
        # HVAC-specific prompt template
        prompt_template = """You are a specialized HVAC code assistant for Atlanta, Georgia. 
Answer the question concisely and accurately using the provided context.

Context: {context}

Question: {question}

Instructions:
- Keep responses concise and to the point (2-3 paragraphs maximum)
- Use complete, grammatically correct sentences
- Focus on the most relevant information only
- Reference specific code sections when available
- If information is not Atlanta-specific, mention this
- Always cite the source document and code section
- Ensure your response is complete and properly formatted
- Do not repeat information unnecessarily

Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        # Create retrieval chain with better formatting
        retrieval_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),  # Fewer chunks for better formatting
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        return retrieval_chain
    
    def query_graph_with_natural_language(self, question: str) -> Dict[str, Any]:
        """
        Convert natural language question to targeted Cypher queries
        Returns structured knowledge graph data directly via Cypher
        
        Examples:
        - "What codes apply to Carrier air handlers?" 
          -> Query: MATCH (c:Component)-[r:COMPLIES_WITH]->(code:Code) WHERE...
        - "What problems are resolved by replacing the filter?"
          -> Query: MATCH (p:Problem)-[r:RESOLVED_BY]->(s:Solution) WHERE...
        """
        if not self.graphrag.driver:
            return {"error": "Neo4j not connected", "results": []}
        
        # Extract key entities and intent from question
        doc = nlp(question.lower())
        entities = [ent.text for ent in doc.ents]
        question_lower = question.lower()
        
        # Determine query intent and build appropriate Cypher
        results = {
            "question": question,
            "entities_found": entities,
            "graph_results": [],
            "cypher_queries": []
        }
        
        with self.graphrag.driver.session() as session:
            # Intent 1: Code compliance queries
            if any(word in question_lower for word in ['code', 'comply', 'complies', 'requirement', 'regulation', 'section']):
                cypher = """
                MATCH (comp)-[r:COMPLIES_WITH]->(code:Code)
                WHERE any(entity IN $entities WHERE toLower(comp.text) CONTAINS toLower(entity))
                   OR any(word IN ['furnace', 'boiler', 'air handler', 'hvac'] WHERE toLower(comp.text) CONTAINS word)
                RETURN comp.text as component,
                       labels(comp)[0] as component_type,
                       code.text as code_section,
                       r.confidence as confidence,
                       r.compliance_type as compliance_type
                ORDER BY r.confidence DESC
                LIMIT 10
                """
                try:
                    result = session.run(cypher, entities=entities + [question_lower])
                    records = [dict(record) for record in result]
                    if records:
                        results["graph_results"].extend(records)
                        results["cypher_queries"].append("Code Compliance Query")
                except Exception as e:
                    pass  # Graph may not have this data yet
            
            # Intent 2: Problem-Solution queries
            if any(word in question_lower for word in ['problem', 'issue', 'fix', 'solve', 'repair', 'troubleshoot', 'wrong']):
                cypher = """
                MATCH (p:Problem)-[r:RESOLVED_BY]->(s:Solution)
                WHERE any(entity IN $entities WHERE toLower(p.text) CONTAINS toLower(entity) 
                                                 OR toLower(s.text) CONTAINS toLower(entity))
                   OR toLower($question) CONTAINS toLower(p.text)
                RETURN p.text as problem,
                       s.text as solution,
                       r.confidence as confidence
                ORDER BY r.confidence DESC
                LIMIT 10
                """
                try:
                    result = session.run(cypher, entities=entities, question=question_lower)
                    records = [dict(record) for record in result]
                    if records:
                        results["graph_results"].extend(records)
                        results["cypher_queries"].append("Problem-Solution Query")
                except Exception as e:
                    pass
            
            # Intent 3: Manufacturer/Brand queries
            if any(word in question_lower for word in ['carrier', 'trane', 'lennox', 'manufacturer', 'brand', 'made by', 'makes']):
                cypher = """
                MATCH (comp:Component)-[r:MANUFACTURED_BY]->(brand:Brand)
                WHERE any(entity IN $entities WHERE toLower(comp.text) CONTAINS toLower(entity)
                                                 OR toLower(brand.text) CONTAINS toLower(entity))
                   OR any(brand_name IN ['carrier', 'trane', 'lennox', 'york', 'rheem'] 
                          WHERE toLower(brand.text) CONTAINS brand_name AND toLower($question) CONTAINS brand_name)
                RETURN comp.text as component,
                       brand.text as manufacturer,
                       r.confidence as confidence
                ORDER BY r.confidence DESC
                LIMIT 10
                """
                try:
                    result = session.run(cypher, entities=entities, question=question_lower)
                    records = [dict(record) for record in result]
                    if records:
                        results["graph_results"].extend(records)
                        results["cypher_queries"].append("Manufacturer Query")
                except Exception as e:
                    pass
            
            # Intent 4: Specification queries
            if any(word in question_lower for word in ['spec', 'rating', 'btu', 'cfm', 'seer', 'efficiency', 'capacity']):
                cypher = """
                MATCH (comp:Component)-[r:HAS_SPECIFICATION]->(spec)
                WHERE any(entity IN $entities WHERE toLower(comp.text) CONTAINS toLower(entity))
                   OR any(word IN ['furnace', 'air handler', 'hvac', 'unit'] WHERE toLower(comp.text) CONTAINS word)
                RETURN comp.text as component,
                       spec.text as specification,
                       r.spec_type as spec_type,
                       r.confidence as confidence
                ORDER BY r.confidence DESC
                LIMIT 10
                """
                try:
                    result = session.run(cypher, entities=entities)
                    records = [dict(record) for record in result]
                    if records:
                        results["graph_results"].extend(records)
                        results["cypher_queries"].append("Specification Query")
                except Exception as e:
                    pass
            
            # Intent 5: System hierarchy queries
            if any(word in question_lower for word in ['part of', 'component', 'system', 'includes', 'contains', 'consists']):
                cypher = """
                MATCH (comp:Component)-[r:PART_OF]->(system:Component)
                WHERE any(entity IN $entities WHERE toLower(comp.text) CONTAINS toLower(entity)
                                                 OR toLower(system.text) CONTAINS toLower(entity))
                RETURN comp.text as component,
                       system.text as system,
                       r.confidence as confidence
                ORDER BY r.confidence DESC
                LIMIT 10
                """
                try:
                    result = session.run(cypher, entities=entities)
                    records = [dict(record) for record in result]
                    if records:
                        results["graph_results"].extend(records)
                        results["cypher_queries"].append("System Hierarchy Query")
                except Exception as e:
                    pass
            
            # Intent 6: Location queries (Atlanta-specific)
            if any(word in question_lower for word in ['atlanta', 'georgia', 'location', 'where', 'installed']):
                cypher = """
                MATCH (comp)-[r:INSTALLED_IN]->(loc:Location)
                WHERE any(entity IN $entities WHERE toLower(comp.text) CONTAINS toLower(entity))
                   OR toLower(loc.text) CONTAINS 'atlanta'
                   OR toLower(loc.text) CONTAINS 'georgia'
                RETURN comp.text as component,
                       labels(comp)[0] as component_type,
                       loc.text as location,
                       r.confidence as confidence
                ORDER BY r.confidence DESC
                LIMIT 10
                """
                try:
                    result = session.run(cypher, entities=entities)
                    records = [dict(record) for record in result]
                    if records:
                        results["graph_results"].extend(records)
                        results["cypher_queries"].append("Location Query")
                except Exception as e:
                    pass
        
        # If no specific intent matched, do a general graph search
        if not results["graph_results"]:
            with self.graphrag.driver.session() as session:
                cypher = """
                MATCH (n)
                WHERE any(entity IN $entities WHERE toLower(n.text) CONTAINS toLower(entity))
                RETURN n.text as entity,
                       labels(n)[0] as type,
                       n.hvac_type as hvac_type,
                       n.confidence as confidence
                ORDER BY n.confidence DESC
                LIMIT 5
                """
                try:
                    result = session.run(cypher, entities=entities)
                    records = [dict(record) for record in result]
                    results["graph_results"] = records
                    results["cypher_queries"].append("General Entity Search")
                except Exception as e:
                    pass
        
        return results
    
    def query_vector_similarity(self, question: str, top_k: int = 5) -> List[Dict]:
        """
        TASK 5 (Part 2): Vector similarity search using Neo4j vector index
        
        Performs cosine similarity search on Chunk.embedding to find relevant document chunks
        Complements graph queries with full document coverage
        """
        if not self.graphrag.driver:
            return []
        
        # Generate embedding for the question
        question_embedding = self.graphrag._get_embedding(question)
        
        results = []
        
        try:
            with self.graphrag.driver.session() as session:
                # Use Neo4j vector index for cosine similarity search
                cypher = """
                CALL db.index.vector.queryNodes('chunk_embedding_index', $top_k, $question_embedding)
                YIELD node, score
                RETURN node.text as text,
                       node.chunk_index as chunk_index,
                       node.source as source,
                       node.token_count as token_count,
                       score as similarity_score
                ORDER BY similarity_score DESC
                """
                
                result = session.run(cypher, top_k=top_k, question_embedding=question_embedding)
                records = [dict(record) for record in result]
                results = records
                
        except Exception as e:
            print(f"Error in vector similarity search: {str(e)}")
        
        return results
    
    def hybrid_search(self, question: str, top_k_vector: int = 5) -> Dict[str, Any]:
        """
        TASK 5: Hybrid parallel search combining graph queries + vector similarity
        
        Executes in parallel:
        1. Graph queries (6 Cypher intent patterns) - structured facts
        2. Vector similarity (cosine on embeddings) - semantic content
        
        Returns combined results with weighted ranking:
        - Graph results (higher weight) - verified relationships
        - Vector results (supporting role) - relevant document chunks
        """
        # Execute graph and vector searches in parallel (conceptually)
        # In Python, we'll do them sequentially but treat as parallel operations
        
        # 1. Graph query (structured knowledge)
        graph_results = self.query_graph_with_natural_language(question)
        
        # 2. Vector similarity search (semantic content)
        vector_results = self.query_vector_similarity(question, top_k=top_k_vector)
        
        # 3. Combine and rank results
        combined = {
            "question": question,
            "graph_results": graph_results["graph_results"],
            "graph_queries_used": graph_results["cypher_queries"],
            "vector_results": vector_results,
            "num_graph_results": len(graph_results["graph_results"]),
            "num_vector_results": len(vector_results),
            "search_method": "hybrid_parallel"
        }
        
        # 4. Add metadata for result fusion
        combined["fusion_strategy"] = "graph_prioritized"  # Graph facts ranked higher
        
        return combined
    
    def format_hybrid_results_as_context(self, hybrid_results: Dict) -> str:
        """
        Format hybrid search results (graph + vector) into unified context
        
        Structure:
        1. Graph facts (verified relationships) - prioritized
        2. Relevant chunks (vector similarity) - supporting context
        """
        context_parts = []
        
        # Part 1: Graph Facts (if any)
        if hybrid_results["num_graph_results"] > 0:
            context_parts.append("**KNOWLEDGE GRAPH FACTS (Verified Relationships):**\n")
            
            for i, result in enumerate(hybrid_results["graph_results"], 1):
                # Format based on result type (same as existing format_graph_results_as_context)
                if "problem" in result and "solution" in result:
                    context_parts.append(
                        f"{i}. Problem: {result['problem']} → Solution: {result['solution']} "
                        f"(confidence: {result.get('confidence', 'N/A')})"
                    )
                elif "component" in result and "code_section" in result:
                    context_parts.append(
                        f"{i}. {result.get('component_type', 'Component')}: {result['component']} → "
                        f"Code: {result['code_section']} (confidence: {result.get('confidence', 'N/A')})"
                    )
                elif "component" in result and "manufacturer" in result:
                    context_parts.append(
                        f"{i}. {result['component']} manufactured by {result['manufacturer']} "
                        f"(confidence: {result.get('confidence', 'N/A')})"
                    )
                elif "component" in result and "specification" in result:
                    context_parts.append(
                        f"{i}. {result['component']} → Spec: {result['specification']} "
                        f"({result.get('spec_type', 'general')}, confidence: {result.get('confidence', 'N/A')})"
                    )
                elif "component" in result and "system" in result:
                    context_parts.append(
                        f"{i}. {result['component']} is part of {result['system']} "
                        f"(confidence: {result.get('confidence', 'N/A')})"
                    )
                elif "component" in result and "location" in result:
                    context_parts.append(
                        f"{i}. {result.get('component_type', 'Component')}: {result['component']} → "
                        f"Location: {result['location']} (confidence: {result.get('confidence', 'N/A')})"
                    )
                else:
                    # Generic format for other result types
                    formatted = ", ".join([f"{k}: {v}" for k, v in result.items() if v is not None])
                    context_parts.append(f"{i}. {formatted}")
            
            context_parts.append("\n")
        
        # Part 2: Relevant Document Chunks (vector similarity)
        if hybrid_results["num_vector_results"] > 0:
            context_parts.append("**RELEVANT DOCUMENT SECTIONS (Cosine Similarity):**\n")
            
            for i, chunk in enumerate(hybrid_results["vector_results"], 1):
                similarity_pct = chunk.get('similarity_score', 0) * 100
                context_parts.append(
                    f"{i}. [Similarity: {similarity_pct:.1f}%] {chunk['text'][:500]}{'...' if len(chunk['text']) > 500 else ''}\n"
                )
        
        # Part 3: Metadata footer
        context_parts.append(f"\n**Search Metadata:**")
        context_parts.append(f"- Graph queries executed: {', '.join(hybrid_results['graph_queries_used']) if hybrid_results['graph_queries_used'] else 'None'}")
        context_parts.append(f"- Graph results: {hybrid_results['num_graph_results']}")
        context_parts.append(f"- Vector results: {hybrid_results['num_vector_results']}")
        context_parts.append(f"- Fusion strategy: {hybrid_results['fusion_strategy']}")
        
        return "\n".join(context_parts)
    
    def query_hvac_system(self, question: str, use_hybrid: bool = True) -> str:
        """
        TASK 6: Main query method with hybrid Graph + Vector RAG
        
        Args:
            question: User's question in natural language
            use_hybrid: If True (default), use hybrid parallel search (graph + vector)
                       If False, use legacy graph-first approach
        
        Flow:
        1. Execute hybrid search (graph Cypher + vector cosine similarity in parallel)
        2. Format combined results as context
        3. Generate response with LLM
        4. Return formatted answer
        """
        if not self.initialized or not self.retrieval_chain:
            return "System not initialized. Please wait for initialization to complete."
        
        try:
            if use_hybrid:
                # NEW: Hybrid parallel search (Task 5 + 6)
                
                # Step 1: Execute hybrid search (graph + vector in parallel)
                hybrid_results = self.hybrid_search(question, top_k_vector=3)
                
                # Step 2: Format hybrid results into unified context
                combined_context = self.format_hybrid_results_as_context(hybrid_results)
                
                # Step 3: Generate response using combined context
                answer = self.generate_response_with_context(question, combined_context)
                
                # Post-process to improve formatting
                answer = self.clean_response_formatting(str(answer))
                
                return answer
            else:
                # LEGACY: Graph-first approach (backward compatibility)
                
                # Step 1: Query knowledge graph with natural language -> Cypher
                graph_results = self.graphrag.query_graph_with_natural_language(question)
                
                # Step 2: Get vector search results (fewer needed now)
                vector_context = self.get_vector_context(question, k=2)  # Reduced from 3
                
                # Step 3: Format graph results into context
                kg_context = self.format_graph_results_as_context(graph_results)
                
                # Step 4: Combine contexts (prioritize graph facts)
                combined_context = self.combine_contexts_v2(kg_context, vector_context, graph_results)
                
                # Step 5: Generate response using combined context
                answer = self.generate_response_with_context(question, combined_context)
                
                # Post-process to improve formatting
                answer = self.clean_response_formatting(str(answer))
                return answer
            
        except Exception as e:
            return f"Error processing query: {str(e)}"
    
    def format_graph_results_as_context(self, graph_results: Dict) -> str:
        """Format Cypher query results into readable context"""
        if not graph_results.get("graph_results"):
            return ""
        
        context_parts = ["**KNOWLEDGE GRAPH FACTS (Verified Relationships):**\n"]
        
        for i, result in enumerate(graph_results["graph_results"], 1):
            # Format based on result type
            if "problem" in result and "solution" in result:
                context_parts.append(
                    f"{i}. Problem: {result['problem']} → Solution: {result['solution']} "
                    f"(confidence: {result.get('confidence', 'N/A')})"
                )
            elif "component" in result and "code_section" in result:
                context_parts.append(
                    f"{i}. {result.get('component_type', 'Component')}: {result['component']} → "
                    f"Code: {result['code_section']} (confidence: {result.get('confidence', 'N/A')})"
                )
            elif "component" in result and "manufacturer" in result:
                context_parts.append(
                    f"{i}. {result['component']} manufactured by {result['manufacturer']} "
                    f"(confidence: {result.get('confidence', 'N/A')})"
                )
            elif "component" in result and "specification" in result:
                context_parts.append(
                    f"{i}. {result['component']} → Spec: {result['specification']} "
                    f"({result.get('spec_type', 'general')}, confidence: {result.get('confidence', 'N/A')})"
                )
            elif "component" in result and "system" in result:
                context_parts.append(
                    f"{i}. {result['component']} is part of {result['system']} "
                    f"(confidence: {result.get('confidence', 'N/A')})"
                )
            elif "component" in result and "location" in result:
                context_parts.append(
                    f"{i}. {result.get('component_type', 'Component')}: {result['component']} → "
                    f"Location: {result['location']} (confidence: {result.get('confidence', 'N/A')})"
                )
            else:
                # Generic format for other result types
                formatted = ", ".join([f"{k}: {v}" for k, v in result.items() if v is not None])
                context_parts.append(f"{i}. {formatted}")
        
        return "\n".join(context_parts)
    
    def combine_contexts_v2(self, kg_context: str, vector_context: str, graph_results: Dict) -> str:
        """
        Combine knowledge graph and vector contexts
        Prioritizes verified graph facts over document excerpts
        """
        combined = []
        
        # Prioritize graph facts if available
        if kg_context:
            combined.append(kg_context)
            combined.append("\n**SOURCE DOCUMENTS (Supporting Context):**\n")
        
        if vector_context:
            combined.append(vector_context)
        
        # Add query metadata
        if graph_results.get("cypher_queries"):
            combined.append(f"\n(Graph queries executed: {', '.join(graph_results['cypher_queries'])})")
        
        return "\n".join(combined)
    
    def get_vector_context(self, question: str, k: int = 3) -> str:
        """Get context from vector search"""
        try:
            docs = self.vector_store.similarity_search(question, k=k)
            context_parts = []
            for i, doc in enumerate(docs, 1):
                context_parts.append(f"Source {i} ({doc.metadata.get('source', 'Unknown')}):\n{doc.page_content}")
            return "\n\n".join(context_parts)
        except Exception as e:
            print(f"Vector search error: {str(e)}")
            return ""
    
    def get_knowledge_graph_context(self, question: str) -> str:
        """Get context from knowledge graph using meaningful relationships (DEPRECATED - use query_graph_with_natural_language)"""
        if not self.graphrag.driver:
            return ""
        
        try:
            with self.graphrag.driver.session() as session:
                # Extract key terms from question
                key_terms = self.extract_key_terms(question)
                
                kg_context_parts = []
                
                # 1. Find direct entities and their relationships
                for term in key_terms:
                    # Get entities and their connected components
                    result = session.run("""
                        MATCH (e:HVACEntity)-[r]-(related)
                        WHERE toLower(e.text) CONTAINS toLower($term)
                        RETURN e.text as entity, 
                               type(r) as relationship_type, 
                               related.text as related_entity,
                               related.hvac_type as related_type,
                               e.code_reference as code_ref
                        LIMIT 10
                    """, term=term)
                    
                    relationships = []
                    for record in result:
                        rel_type = record["relationship_type"]
                        related_entity = record["related_entity"]
                        related_type = record["related_type"]
                        code_ref = record["code_ref"]
                        
                        if rel_type == "COMPONENT_OF":
                            relationships.append(f"{record['entity']} is a component of {related_entity}")
                        elif rel_type == "GOVERNED_BY":
                            relationships.append(f"{record['entity']} is governed by {related_entity}")
                        elif rel_type == "HAS_SPECIFICATION":
                            relationships.append(f"{record['entity']} has specification {related_entity}")
                        elif rel_type == "INSTALLED_IN":
                            relationships.append(f"{record['entity']} is installed in {related_entity}")
                        elif rel_type == "BELONGS_TO":
                            relationships.append(f"{record['entity']} belongs to {related_entity} category")
                    
                    if relationships:
                        kg_context_parts.append(f"Knowledge about '{term}': {'; '.join(relationships[:5])}")
                
                # 2. Find code compliance information
                code_result = session.run("""
                    MATCH (e:HVACEntity)-[:GOVERNED_BY]->(code:HVACEntity)
                    WHERE toLower(e.text) CONTAINS toLower($term)
                    RETURN e.text as entity, code.text as code_section
                    LIMIT 5
                """, term=key_terms[0] if key_terms else "")
                
                code_relationships = []
                for record in code_result:
                    code_relationships.append(f"{record['entity']} governed by {record['code_section']}")
                
                if code_relationships:
                    kg_context_parts.append(f"Code compliance: {'; '.join(code_relationships)}")
                
                # 3. Find equipment specifications
                spec_result = session.run("""
                    MATCH (equipment:HVACEntity)-[:HAS_SPECIFICATION]->(spec:HVACEntity)
                    WHERE toLower(equipment.text) CONTAINS toLower($term)
                    RETURN equipment.text as equipment, spec.text as specification
                    LIMIT 5
                """, term=key_terms[0] if key_terms else "")
                
                spec_relationships = []
                for record in spec_result:
                    spec_relationships.append(f"{record['equipment']} has {record['specification']}")
                
                if spec_relationships:
                    kg_context_parts.append(f"Specifications: {'; '.join(spec_relationships)}")
                
                return "\n\n".join(kg_context_parts)
                
        except Exception as e:
            print(f"Knowledge graph query error: {str(e)}")
            return ""
    
    def extract_key_terms(self, question: str) -> list:
        """Extract key HVAC terms from the question"""
        hvac_terms = [
            "duct", "ventilation", "hvac", "heating", "cooling", "air conditioning",
            "furnace", "boiler", "heat pump", "seer", "btu", "cfm", "ashrae",
            "iecc", "imc", "ifgc", "atlanta", "georgia", "code", "section"
        ]
        
        question_lower = question.lower()
        found_terms = [term for term in hvac_terms if term in question_lower]
        
        # Also extract any words that might be HVAC-related
        words = question_lower.split()
        hvac_words = [word for word in words if len(word) > 3 and any(hvac_term in word for hvac_term in hvac_terms)]
        
        return list(set(found_terms + hvac_words))[:5]  # Limit to 5 terms
    
    def query_graph_for_facts(self, question: str) -> dict:
        """
        Phase 3: Graph-first retrieval - Query knowledge graph for factual relationships
        Returns structured facts from the graph with high confidence
        """
        if not self.graphrag.driver:
            return {"facts": [], "entities": [], "relationships": []}
        
        try:
            # Extract entities from question
            entities = self.extract_key_terms(question)
            
            with self.graphrag.driver.session() as session:
                graph_facts = {
                    "facts": [],
                    "entities": [],
                    "relationships": [],
                    "codes": [],
                    "specifications": [],
                    "problems_solutions": []
                }
                
                # 1. Query for code compliance relationships (highest priority)
                code_result = session.run("""
                    MATCH (c)-[r:COMPLIES_WITH]->(code:Code)
                    WHERE ANY(entity IN $entities WHERE toLower(c.text) CONTAINS toLower(entity))
                      AND r.confidence > 0.85
                      AND (c:Component OR c:Brand)
                    RETURN c.text as component,
                           code.text as code_section,
                           r.confidence as confidence,
                           r.compliance_type as compliance_type
                    ORDER BY r.confidence DESC
                    LIMIT 10
                """, entities=entities)
                
                for record in code_result:
                    fact = f"{record['component']} must comply with {record['code_section']}"
                    graph_facts["facts"].append(fact)
                    graph_facts["codes"].append({
                        "component": record["component"],
                        "code": record["code_section"],
                        "confidence": record["confidence"],
                        "type": record.get("compliance_type", "code_requirement")
                    })
                
                # 2. Query for problem-solution relationships
                problem_result = session.run("""
                    MATCH (p:Problem)-[r:RESOLVED_BY]->(s:Solution)
                    WHERE ANY(entity IN $entities WHERE 
                              toLower(p.text) CONTAINS toLower(entity) OR
                              toLower($question) CONTAINS toLower(p.text))
                      AND r.confidence > 0.8
                    RETURN p.text as problem,
                           s.text as solution,
                           r.confidence as confidence
                    ORDER BY r.confidence DESC
                    LIMIT 5
                """, entities=entities, question=question)
                
                for record in problem_result:
                    fact = f"Problem: {record['problem']} → Solution: {record['solution']}"
                    graph_facts["facts"].append(fact)
                    graph_facts["problems_solutions"].append({
                        "problem": record["problem"],
                        "solution": record["solution"],
                        "confidence": record["confidence"]
                    })
                
                # 3. Query for component specifications
                spec_result = session.run("""
                    MATCH (c:Component)-[r:HAS_SPECIFICATION]->(spec)
                    WHERE ANY(entity IN $entities WHERE toLower(c.text) CONTAINS toLower(entity))
                      AND r.confidence > 0.8
                    RETURN c.text as component,
                           spec.text as specification,
                           r.confidence as confidence,
                           r.spec_type as spec_type
                    ORDER BY r.confidence DESC
                    LIMIT 5
                """, entities=entities)
                
                for record in spec_result:
                    fact = f"{record['component']} has specification: {record['specification']}"
                    graph_facts["facts"].append(fact)
                    graph_facts["specifications"].append({
                        "component": record["component"],
                        "specification": record["specification"],
                        "confidence": record["confidence"],
                        "type": record.get("spec_type", "performance_rating")
                    })
                
                # 4. Query for brand-component relationships
                brand_result = session.run("""
                    MATCH (c:Component)-[r:MANUFACTURED_BY]->(b:Brand)
                    WHERE ANY(entity IN $entities WHERE 
                              toLower(c.text) CONTAINS toLower(entity) OR 
                              toLower(b.text) CONTAINS toLower(entity))
                      AND r.confidence > 0.8
                    RETURN c.text as component,
                           b.text as brand,
                           r.confidence as confidence
                    ORDER BY r.confidence DESC
                    LIMIT 5
                """, entities=entities)
                
                for record in brand_result:
                    fact = f"{record['component']} is manufactured by {record['brand']}"
                    graph_facts["facts"].append(fact)
                    graph_facts["relationships"].append({
                        "type": "MANUFACTURED_BY",
                        "from": record["component"],
                        "to": record["brand"],
                        "confidence": record["confidence"]
                    })
                
                # 5. Query for system hierarchy
                hierarchy_result = session.run("""
                    MATCH (c:Component)-[r:PART_OF]->(system:Component)
                    WHERE ANY(entity IN $entities WHERE toLower(c.text) CONTAINS toLower(entity))
                      AND r.confidence > 0.9
                    RETURN c.text as component,
                           system.text as system,
                           r.confidence as confidence
                    ORDER BY r.confidence DESC
                    LIMIT 5
                """, entities=entities)
                
                for record in hierarchy_result:
                    fact = f"{record['component']} is part of {record['system']}"
                    graph_facts["facts"].append(fact)
                    graph_facts["relationships"].append({
                        "type": "PART_OF",
                        "from": record["component"],
                        "to": record["system"],
                        "confidence": record["confidence"]
                    })
                
                # 6. Query for location information
                location_result = session.run("""
                    MATCH (c)-[r:INSTALLED_IN]->(l:Location)
                    WHERE ANY(entity IN $entities WHERE toLower(c.text) CONTAINS toLower(entity))
                      AND r.confidence > 0.75
                      AND (c:Component OR c:Brand)
                    RETURN c.text as component,
                           l.text as location,
                           r.confidence as confidence
                    ORDER BY r.confidence DESC
                    LIMIT 3
                """, entities=entities)
                
                for record in location_result:
                    fact = f"{record['component']} is installed in {record['location']}"
                    graph_facts["facts"].append(fact)
                
                return graph_facts
                
        except Exception as e:
            print(f"Graph query error: {str(e)}")
            return {"facts": [], "entities": [], "relationships": []}
    
    def graph_first_hybrid_search(self, question: str) -> dict:
        """
        Phase 3: Implement graph-first hybrid search
        1. Query graph for factual relationships (high confidence)
        2. Extract terms from graph results
        3. Use graph terms to seed vector search
        4. Combine with priority to graph facts
        """
        # Step 1: Get graph facts
        graph_facts = self.query_graph_for_facts(question)
        
        # Step 2: Extract terms from graph results to enhance vector search
        graph_terms = []
        for fact in graph_facts.get("facts", []):
            # Extract key terms from facts
            words = fact.split()
            graph_terms.extend([w for w in words if len(w) > 4])
        
        # Step 3: Enhanced vector search using graph terms
        enhanced_query = question
        if graph_terms:
            # Add top graph terms to query for better vector retrieval
            unique_terms = list(set(graph_terms[:10]))
            enhanced_query = f"{question} {' '.join(unique_terms)}"
        
        # Get vector context with enhanced query
        try:
            docs = self.vector_store.similarity_search(enhanced_query, k=3)
            vector_context = []
            for doc in docs:
                vector_context.append({
                    "content": doc.page_content,
                    "source": doc.metadata.get('source', 'Unknown'),
                    "type": "document_context"
                })
        except Exception as e:
            print(f"Vector search error: {str(e)}")
            vector_context = []
        
        # Step 4: Return combined results with priority structure
        return {
            "graph_facts": graph_facts,  # High confidence, factual
            "vector_context": vector_context,  # Supporting details
            "enhanced_query": enhanced_query,
            "original_query": question
        }
    
    def combine_contexts(self, vector_context: str, kg_context: str) -> str:
        """Combine vector and knowledge graph contexts"""
        contexts = []
        
        if vector_context:
            contexts.append(f"Document Context:\n{vector_context}")
        
        if kg_context:
            contexts.append(f"Knowledge Graph Context:\n{kg_context}")
        
        return "\n\n".join(contexts) if contexts else "No relevant context found."
    
    def generate_enhanced_response(self, question: str, hybrid_results: dict) -> str:
        """
        Phase 3: Generate response with enhanced prompt template
        Separates graph facts (high confidence) from document context (supporting details)
        """
        try:
            graph_facts = hybrid_results.get("graph_facts", {})
            vector_context = hybrid_results.get("vector_context", [])
            
            # Format graph facts section
            graph_facts_text = ""
            if graph_facts.get("facts"):
                facts_list = graph_facts["facts"]
                graph_facts_text = "**KNOWLEDGE GRAPH FACTS** (High Confidence - Verified Relationships):\n"
                for i, fact in enumerate(facts_list[:10], 1):
                    graph_facts_text += f"  {i}. {fact}\n"
                
                # Add code compliance details
                if graph_facts.get("codes"):
                    graph_facts_text += "\n**Code Compliance Requirements:**\n"
                    for code_info in graph_facts["codes"][:5]:
                        graph_facts_text += f"  • {code_info['component']} → {code_info['code']} (confidence: {code_info['confidence']:.2f})\n"
                
                # Add problem-solution mappings
                if graph_facts.get("problems_solutions"):
                    graph_facts_text += "\n**Problem-Solution Mappings:**\n"
                    for ps in graph_facts["problems_solutions"][:3]:
                        graph_facts_text += f"  • {ps['problem']} → {ps['solution']} (confidence: {ps['confidence']:.2f})\n"
            
            # Format vector context section
            vector_context_text = ""
            if vector_context:
                vector_context_text = "\n**SOURCE DOCUMENT CONTEXT** (Supporting Details):\n"
                for i, doc in enumerate(vector_context[:3], 1):
                    source = doc.get('source', 'Unknown')
                    content = doc.get('content', '')[:500]  # Limit length
                    vector_context_text += f"\nSource {i} ({source}):\n{content}\n"
            
            # Enhanced prompt template with clear separation
            prompt = f"""You are an expert HVAC code assistant for Atlanta, Georgia with access to a verified knowledge graph.

Answer the user's question using ONLY the information provided below. Prioritize KNOWLEDGE GRAPH FACTS as they are verified and high-confidence.

{graph_facts_text}

{vector_context_text}

**USER QUESTION:** {question}

**INSTRUCTIONS:**
1. **Prioritize graph facts** - These are verified relationships with high confidence
2. **Cite sources** - Reference specific code sections and documents
3. **Be specific** - Include complete code section numbers (e.g., Section 302.5.1)
4. **Indicate confidence** - Note if information comes from high-confidence graph facts vs. general context
5. **Be concise** - 2-3 well-structured paragraphs maximum
6. **Complete sentences** - Ensure your response ends properly
7. **Atlanta focus** - Mention if information is Atlanta-specific or general

**YOUR ANSWER:**"""

            # Generate response
            response = self.llm.invoke(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            
            # Check if response seems incomplete and retry if needed
            if self.is_response_incomplete(answer):
                print("Response appears incomplete, retrying...")
                response = self.llm.invoke(prompt)
                answer = response.content if hasattr(response, 'content') else str(response)
            
            return answer
            
        except Exception as e:
            print(f"Error generating enhanced response: {str(e)}")
            # Fallback to simple generation
            return self.generate_response_with_context(question, str(hybrid_results))
    
    def generate_response_with_context(self, question: str, context: str) -> str:
        """Generate response using the combined context"""
        try:
            # Use the LLM directly with our custom prompt
            prompt = f"""You are a specialized HVAC code assistant for Atlanta, Georgia. 
Answer the question concisely and accurately using the provided context.

Context: {context}

Question: {question}

Instructions:
- Provide a complete, well-formed response
- Use complete, grammatically correct sentences
- Include all relevant code section references (e.g., 302.5.1, not just 302.)
- Focus on the most relevant information only
- If information is not Atlanta-specific, mention this
- Always cite the source document and complete code section
- Ensure your response ends with a complete sentence
- Do not cut off mid-sentence or mid-reference

Answer:"""

            response = self.llm.invoke(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            
            # Check if response seems incomplete and retry if needed
            if self.is_response_incomplete(answer):
                print("Response appears incomplete, retrying...")
                response = self.llm.invoke(prompt)
                answer = response.content if hasattr(response, 'content') else str(response)
            
            return answer
            
        except Exception as e:
            print(f"LLM generation error: {str(e)}")
            return "Unable to generate response at this time."
    
    def is_response_incomplete(self, text: str) -> bool:
        """Check if response appears to be cut off"""
        import re
        
        if not text or len(text.strip()) < 20:
            return True
        
        # Check for patterns that suggest truncation
        text = text.strip()
        
        # Only check for truly incomplete patterns
        # 1. Ends with ellipsis or dash (clearly incomplete)
        if text.endswith(('...', '--', '-', ',')):
            return True
        
        # 2. Ends mid-sentence with common conjunctions/prepositions
        incomplete_endings = ['and', 'or', 'but', 'with', 'from', 'to', 'in', 'on', 'at', 'by', 'for']
        last_word = text.split()[-1].lower().strip('.,!?;:')
        if last_word in incomplete_endings:
            return True
        
        # 3. Ends with "Section" or "Code" without a number (clearly incomplete reference)
        if re.search(r'\b(Section|Code|Article|Chapter)\s*$', text, re.IGNORECASE):
            return True
        
        # 4. Very short response that doesn't end with proper punctuation (< 5 words)
        if len(text.split()) < 5 and not text.endswith(('.', '!', '?')):
            return True
        
        # Otherwise, consider it complete
        # NOTE: We're being less strict now - if response ends with period, it's probably complete
        # Even if it ends with "Section 302." that's a valid complete sentence
        return False
    
    def clean_response_formatting(self, text: str) -> str:
        """Clean up response formatting for better display"""
        import re
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Don't truncate responses - let them be complete
        # Only fix obvious formatting issues
        
        # Fix incomplete sentences by looking for patterns like "302." at the end
        if text.endswith(('.', '!', '?')) and len(text.split()) > 1:
            # Check if the last "sentence" is actually incomplete
            last_sentence = text.split('.')[-2] if '.' in text else text
            if len(last_sentence.strip()) < 20 and not last_sentence.strip().endswith(('!', '?')):
                # This might be an incomplete sentence, try to find a better break point
                sentences = re.split(r'[.!?]+', text)
                if len(sentences) > 2:
                    # Take all but the last incomplete sentence
                    text = '. '.join(sentences[:-1])
                    if not text.endswith(('.', '!', '?')):
                        text += '.'
        
        # Ensure proper paragraph breaks (but not too many)
        text = re.sub(r'\. ([A-Z])', r'.\n\n\1', text)
        text = re.sub(r'\? ([A-Z])', r'?\n\n\1', text)
        text = re.sub(r'! ([A-Z])', r'!\n\n\1', text)
        
        # Clean up multiple newlines
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Remove repetitive phrases
        text = re.sub(r'\b(Additionally|Furthermore|Moreover|In addition)\b.*?\.', '', text, flags=re.IGNORECASE)
        
        # Ensure the response ends properly
        if not text.endswith(('.', '!', '?')):
            text = text.rstrip() + '.'
        
        return text.strip()
    
    def validate_response_against_graph(self, response: str, question: str) -> dict:
        """
        Phase 3: Post-generation validation
        Validate LLM response against knowledge graph facts
        Returns validation results with confidence scores
        """
        if not self.graphrag.driver:
            return {
                "validated": True,
                "confidence": 0.5,
                "warnings": ["Graph validation unavailable"],
                "verified_claims": [],
                "unverified_claims": []
            }
        
        try:
            import re
            
            validation_results = {
                "validated": True,
                "confidence": 1.0,
                "warnings": [],
                "verified_claims": [],
                "unverified_claims": [],
                "code_references": []
            }
            
            # Extract claims from response
            # 1. Extract code section references (e.g., "Section 301.1", "IMC 302.5")
            code_pattern = r'(?:Section|§|IMC|IECC|ASHRAE)\s*(\d+\.?\d*\.?\d*)'
            code_refs = re.findall(code_pattern, response, re.IGNORECASE)
            
            # 2. Extract relationship claims (e.g., "X is part of Y", "X complies with Y")
            relationship_patterns = [
                (r'(\w+(?:\s+\w+)?)\s+(?:is part of|belongs to|is a component of)\s+(\w+(?:\s+\w+)?)', 'PART_OF'),
                (r'(\w+(?:\s+\w+)?)\s+(?:must comply with|complies with|governed by)\s+(Section\s+\d+\.?\d*)', 'COMPLIES_WITH'),
                (r'(\w+(?:\s+\w+)?)\s+(?:manufactured by|made by|produced by)\s+(\w+)', 'MANUFACTURED_BY'),
                (r'(\w+(?:\s+\w+)?)\s+(?:can be resolved by|fixed by|solved by)\s+(\w+(?:\s+\w+)?)', 'RESOLVED_BY')
            ]
            
            with self.graphrag.driver.session() as session:
                # Validate code references
                for code_ref in code_refs[:10]:  # Limit validation
                    result = session.run("""
                        MATCH (c:Code)
                        WHERE c.text CONTAINS $code_ref
                        RETURN c.text as code, c.confidence as confidence
                        LIMIT 1
                    """, code_ref=code_ref)
                    
                    record = result.single()
                    if record:
                        validation_results["verified_claims"].append(f"Code reference {record['code']} verified")
                        validation_results["code_references"].append({
                            "reference": code_ref,
                            "verified": True,
                            "confidence": record["confidence"]
                        })
                    else:
                        validation_results["warnings"].append(f"Code reference '{code_ref}' not found in graph")
                        validation_results["unverified_claims"].append(f"Code {code_ref}")
                        validation_results["confidence"] *= 0.9  # Reduce confidence
                
                # Validate relationship claims
                for pattern, rel_type in relationship_patterns:
                    matches = re.findall(pattern, response, re.IGNORECASE)
                    
                    for match in matches[:5]:  # Limit per pattern
                        source, target = match
                        
                        # Query graph to verify this relationship exists
                        if rel_type == 'PART_OF':
                            result = session.run("""
                                MATCH (s:Component)-[r:PART_OF]->(t:Component)
                                WHERE toLower(s.text) CONTAINS toLower($source)
                                  AND toLower(t.text) CONTAINS toLower($target)
                                  AND r.confidence > 0.8
                                RETURN r.confidence as confidence
                                LIMIT 1
                            """, source=source.strip(), target=target.strip())
                        
                        elif rel_type == 'COMPLIES_WITH':
                            result = session.run("""
                                MATCH (c)-[r:COMPLIES_WITH]->(code:Code)
                                WHERE toLower(c.text) CONTAINS toLower($source)
                                  AND toLower(code.text) CONTAINS toLower($target)
                                  AND r.confidence > 0.8
                                RETURN r.confidence as confidence
                                LIMIT 1
                            """, source=source.strip(), target=target.strip())
                        
                        elif rel_type == 'MANUFACTURED_BY':
                            result = session.run("""
                                MATCH (c:Component)-[r:MANUFACTURED_BY]->(b:Brand)
                                WHERE toLower(c.text) CONTAINS toLower($source)
                                  AND toLower(b.text) CONTAINS toLower($target)
                                  AND r.confidence > 0.8
                                RETURN r.confidence as confidence
                                LIMIT 1
                            """, source=source.strip(), target=target.strip())
                        
                        elif rel_type == 'RESOLVED_BY':
                            result = session.run("""
                                MATCH (p:Problem)-[r:RESOLVED_BY]->(s:Solution)
                                WHERE toLower(p.text) CONTAINS toLower($source)
                                  AND toLower(s.text) CONTAINS toLower($target)
                                  AND r.confidence > 0.8
                                RETURN r.confidence as confidence
                                LIMIT 1
                            """, source=source.strip(), target=target.strip())
                        else:
                            result = None
                        
                        if result:
                            record = result.single()
                            if record:
                                validation_results["verified_claims"].append(
                                    f"{rel_type}: {source} → {target} (confidence: {record['confidence']:.2f})"
                                )
                            else:
                                validation_results["warnings"].append(
                                    f"Relationship '{source} {rel_type} {target}' not verified in graph"
                                )
                                validation_results["unverified_claims"].append(f"{source} → {target}")
                                validation_results["confidence"] *= 0.95
            
            # Final validation decision
            if len(validation_results["unverified_claims"]) > 3:
                validation_results["validated"] = False
                validation_results["warnings"].append("Multiple unverified claims detected")
            
            if validation_results["confidence"] < 0.7:
                validation_results["validated"] = False
            
            return validation_results
            
        except Exception as e:
            print(f"Validation error: {str(e)}")
            return {
                "validated": True,
                "confidence": 0.5,
                "warnings": [f"Validation error: {str(e)}"],
                "verified_claims": [],
                "unverified_claims": []
            }
    
    def query_hvac_system_with_validation(self, question: str) -> tuple:
        """
        Phase 3: Complete query system with graph-first retrieval and validation
        Returns (answer, validation_results)
        """
        if not self.initialized or not self.retrieval_chain:
            return "System not initialized. Please wait for initialization to complete.", {}
        
        try:
            # Use graph-first hybrid search
            hybrid_results = self.graph_first_hybrid_search(question)
            
            # Generate response with enhanced prompt
            answer = self.generate_enhanced_response(question, hybrid_results)
            
            # Post-generation validation
            validation = self.validate_response_against_graph(answer, question)
            
            # Clean up formatting
            answer = self.clean_response_formatting(str(answer))
            
            # Append validation warnings if needed
            if not validation["validated"] or validation["confidence"] < 0.8:
                warning_text = "\n\n⚠️ **Validation Notice:** Some claims in this response could not be fully verified against the knowledge graph."
                if validation["warnings"]:
                    warning_text += f" ({len(validation['warnings'])} warnings)"
                answer += warning_text
            
            return answer, validation
            
        except Exception as e:
            return f"Error processing query: {str(e)}", {}

def main():
    try:
        st.set_page_config(
            page_title="HVAC Code Assistant - Atlanta",
            page_icon="🏗️",
            layout="wide"
        )
        
        st.title("🏗️ HVAC Code Assistant - Atlanta")
        st.markdown("Specialized chatbot for HVAC codes and regulations in Atlanta, Georgia")
        
        # Initialize chatbot with error handling
        if 'chatbot' not in st.session_state:
            with st.spinner("Setting up chatbot..."):
                st.session_state.chatbot = HVACChatbot()
                st.success("Chatbot initialized successfully!")
    except Exception as e:
        st.error(f"Error during initialization: {str(e)}")
        print(f"Initialization error: {str(e)}")  # Console logging
    
    # Sidebar - Clean professional design
    with st.sidebar:
        # Branding
        st.image("https://img.icons8.com/fluency/96/000000/air-conditioner.png", width=80)
        st.markdown("### HVAC Code Assistant")
        st.markdown("*Atlanta, Georgia*")
        st.divider()
        
        # System Status - Simplified
        st.markdown("#### 📊 System Status")
        
        # Auto-initialize system
        if not st.session_state.chatbot.initialized:
            with st.spinner("Initializing system..."):
                if st.session_state.chatbot.initialize_system():
                    st.success("✅ Ready")
                else:
                    st.error("❌ System Error")
                    st.stop()
        else:
            st.success("✅ System Ready")
        
        # Knowledge base status
        if st.session_state.chatbot.vector_store:
            st.info(f"📚 Knowledge Base: **Loaded**")
        
        st.divider()
        
        # About section
        st.markdown("#### ℹ️ About")
        st.markdown("""
        This AI assistant helps with:
        - HVAC code compliance
        - Atlanta-specific regulations
        - Equipment specifications
        - Installation requirements
        - Troubleshooting guidance
        """)
        
        st.divider()
        
        st.caption("Powered by OpenAI & Neo4j")
        st.caption("© 2025 HVAC Code Assistant")
    
    # Main chat interface
    st.header("💬 Chat with HVAC Assistant")
    
    # Chat messages
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add welcome greeting on first load
        welcome_message = "👋 Hello! I'm your personal HVAC Assistant for Atlanta, Georgia. Ask me anything about HVAC codes, regulations, equipment specifications, or compliance requirements."
        st.session_state.messages.append({"role": "assistant", "content": welcome_message})
    
    # Display messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about HVAC codes, regulations, or requirements..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching HVAC knowledge base..."):
                response = st.session_state.chatbot.query_hvac_system(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {str(e)}")  # Console logging
        import traceback
        traceback.print_exc()
        # Try to show error in browser
        try:
            st.error("Application failed to start. Please check the console for errors.")
        except:
            pass