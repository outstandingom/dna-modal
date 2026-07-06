"""
DYNAMIC KNOWLEDGE LOADER
Loads knowledge from multiple sources dynamically into the DNA Memory System.
"""

import os
import json
import csv
import yaml
import time
import logging
import requests
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import random
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ConceptData:
    """Data structure for a concept to be loaded"""
    name: str
    features: List[str]
    domain: str = "General"
    importance: float = 1.0
    relationships: List[Tuple[str, str, float]] = field(default_factory=list)  # (target, color, weight)
    numeric_value: Optional[float] = None

class DynamicKnowledgeLoader:
    """
    Dynamically loads knowledge from multiple sources into the DNA memory system.
    Can be extended with any data source.
    """
    
    def __init__(self):
        self.concepts: List[ConceptData] = []
        self.sources_loaded = []
        self.total_concepts_loaded = 0
        
        # Built-in knowledge bases (can be extended)
        self.builtin_sources = {
            "science": self._load_science_knowledge,
            "history": self._load_history_knowledge,
            "art": self._load_art_knowledge,
            "philosophy": self._load_philosophy_knowledge,
            "technology": self._load_technology_knowledge,
            "geography": self._load_geography_knowledge,
            "biology": self._load_biology_knowledge,
            "mathematics": self._load_mathematics_knowledge,
        }
    
    def load_from_config(self, config_path: str) -> int:
        """
        Load knowledge from a configuration file.
        Config can be JSON, YAML, or TOML.
        """
        with open(config_path) as f:
            if config_path.endswith('.json'):
                config = json.load(f)
            elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
                config = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path}")
        
        total_loaded = 0
        
        # Load from each source in config
        for source in config.get('sources', []):
            source_type = source.get('type')
            if source_type == 'builtin':
                total_loaded += self._load_builtin(source.get('name'))
            elif source_type == 'file':
                total_loaded += self.load_from_file(source.get('path'), source.get('format', 'auto'))
            elif source_type == 'api':
                total_loaded += self.load_from_api(
                    source.get('url'),
                    source.get('api_key'),
                    source.get('params', {})
                )
            elif source_type == 'database':
                total_loaded += self.load_from_database(source.get('connection'))
            elif source_type == 'web':
                total_loaded += self.load_from_web(source.get('url'), source.get('selector', ''))
            elif source_type == 'llm':
                total_loaded += self.load_from_llm(source.get('provider'), source.get('prompt'))
            elif source_type == 'folder':
                total_loaded += self.load_from_folder(source.get('path'))
        
        self.total_concepts_loaded += total_loaded
        return total_loaded
    
    def load_builtin_knowledge(self, categories: List[str] = None) -> int:
        """
        Load built-in knowledge for specified categories.
        If no categories specified, loads all.
        """
        if categories is None:
            categories = list(self.builtin_sources.keys())
        
        total_loaded = 0
        for category in categories:
            if category in self.builtin_sources:
                total_loaded += self._load_builtin(category)
            else:
                logger.warning(f"Unknown builtin category: {category}")
        
        self.total_concepts_loaded += total_loaded
        return total_loaded
    
    def _load_builtin(self, category: str) -> int:
        """Load a built-in knowledge category"""
        logger.info(f"Loading builtin knowledge: {category}")
        loader = self.builtin_sources.get(category)
        if loader:
            return loader()
        return 0
    
    # ============================================================
    # BUILT-IN KNOWLEDGE BASES
    # ============================================================
    
    def _load_science_knowledge(self) -> int:
        """Load scientific concepts"""
        science_data = [
            ConceptData(
                name="Quantum Mechanics",
                features=["physics", "subatomic", "particles", "wave_function", "observation"],
                domain="Science",
                importance=2.0,
                relationships=[("Physics", "IS_A", 0.9)]
            ),
            ConceptData(
                name="Evolution",
                features=["biology", "natural_selection", "species", "adaptation", "genetics"],
                domain="Science",
                importance=2.0,
                relationships=[("Biology", "IS_A", 0.9)]
            ),
            ConceptData(
                name="General Relativity",
                features=["physics", "spacetime", "gravity", "Einstein", "curvature"],
                domain="Science",
                importance=2.0,
                relationships=[("Physics", "IS_A", 0.9)]
            ),
            ConceptData(
                name="Photosynthesis",
                features=["biology", "plants", "sunlight", "chlorophyll", "energy"],
                domain="Science",
                importance=1.8,
                relationships=[("Biology", "IS_A", 0.9)]
            ),
            ConceptData(
                name="Entropy",
                features=["thermodynamics", "disorder", "energy", "second_law", "statistical"],
                domain="Science",
                importance=1.8,
                relationships=[("Physics", "IS_A", 0.9)]
            ),
            ConceptData(
                name="DNA",
                features=["genetics", "heredity", "molecules", "cells", "coding"],
                domain="Science",
                importance=2.0,
                relationships=[("Biology", "IS_A", 0.9)]
            ),
            ConceptData(
                name="E=mc²",
                features=["physics", "energy", "mass", "speed_of_light", "relativity"],
                domain="Science",
                importance=2.0,
                numeric_value=1.0,
                relationships=[("General Relativity", "RELATED_TO", 0.8)]
            ),
            ConceptData(
                name="Dark Matter",
                features=["astrophysics", "cosmology", "universe", "invisible", "gravity"],
                domain="Science",
                importance=1.5,
                relationships=[("Cosmology", "IS_A", 0.8)]
            ),
            ConceptData(
                name="String Theory",
                features=["physics", "strings", "dimensions", "quantum_gravity", "multiverse"],
                domain="Science",
                importance=1.5,
                relationships=[("Quantum Mechanics", "RELATED_TO", 0.7)]
            ),
            ConceptData(
                name="CRISPR",
                features=["biology", "gene_editing", "DNA", "technology", "medicine"],
                domain="Science",
                importance=1.8,
                relationships=[("Genetics", "IS_A", 0.9)]
            ),
        ]
        self._add_concepts(science_data)
        return len(science_data)
    
    def _load_history_knowledge(self) -> int:
        """Load historical concepts"""
        history_data = [
            ConceptData(
                name="Ancient Rome",
                features=["empire", "Roman", "gladiators", "Senate", "aqueducts"],
                domain="History",
                importance=2.0,
                relationships=[("Ancient Civilizations", "IS_A", 0.9)]
            ),
            ConceptData(
                name="The Renaissance",
                features=["art", "culture", "Europe", "Michelangelo", "DaVinci"],
                domain="History",
                importance=2.0,
                relationships=[("Art History", "IS_A", 0.9)]
            ),
            ConceptData(
                name="World War 2",
                features=["war", "Holocaust", "Allies", "Axis", "tanks"],
                domain="History",
                importance=2.0,
                relationships=[("Modern History", "IS_A", 0.9)]
            ),
            ConceptData(
                name="Industrial Revolution",
                features=["machines", "steam", "factory", "urbanization", "technology"],
                domain="History",
                importance=2.0,
                relationships=[("History", "HAS", 0.8)]
            ),
            ConceptData(
                name="Ancient Egypt",
                features=["pyramids", "pharaohs", "Nile", "hieroglyphics", "mummies"],
                domain="History",
                importance=1.8,
                relationships=[("Ancient Civilizations", "IS_A", 0.9)]
            ),
            ConceptData(
                name="The French Revolution",
                features=["democracy", "liberty", "equality", "fraternity", "Napoleon"],
                domain="History",
                importance=1.8,
                relationships=[("Revolution", "IS_A", 0.8)]
            ),
            ConceptData(
                name="The Cold War",
                features=["USA", "USSR", "nuclear", "space_race", "proxy_wars"],
                domain="History",
                importance=1.5,
                relationships=[("Modern History", "IS_A", 0.9)]
            ),
            ConceptData(
                name="Ancient Greece",
                features=["philosophy", "democracy", "mythology", "Olympics", "Socrates"],
                domain="History",
                importance=1.8,
                relationships=[("Ancient Civilizations", "IS_A", 0.9)]
            ),
            ConceptData(
                name="The Roman Empire",
                features=["emperors", "legions", "roads", "law", "Latin"],
                domain="History",
                importance=1.8,
                relationships=[("Ancient Rome", "IS_A", 0.9)]
            ),
            ConceptData(
                name="The Crusades",
                features=["religion", "knights", "Holy_Land", "Islam", "Christianity"],
                domain="History",
                importance=1.5,
                relationships=[("Medieval History", "IS_A", 0.8)]
            ),
        ]
        self._add_concepts(history_data)
        return len(history_data)
    
    def _load_art_knowledge(self) -> int:
        """Load art concepts"""
        art_data = [
            ConceptData(
                name="Impressionism",
                features=["painting", "light", "color", "Monet", "Renoir"],
                domain="Art",
                importance=2.0,
                relationships=[("Art Movement", "IS_A", 0.9)]
            ),
            ConceptData(
                name="Classical Music",
                features=["orchestra", "Mozart", "Beethoven", "symphony", "sonata"],
                domain="Art",
                importance=2.0,
                relationships=[("Music", "IS_A", 0.9)]
            ),
            ConceptData(
                name="Cubism",
                features=["abstract", "Picasso", "geometric", "fragmented", "perspective"],
                domain="Art",
                importance=1.5,
                relationships=[("Art Movement", "IS_A", 0.9)]
            ),
            ConceptData(
                name="Sculpture",
                features=["statue", "marble", "Michelangelo", "bronze", "art"],
                domain="Art",
                importance=1.5,
                relationships=[("Art", "IS_A", 0.9)]
            ),
            ConceptData(
                name="Jazz",
                features=["music", "improvisation", "saxophone", "piano", "blues"],
                domain="Art",
                importance=1.8,
                relationships=[("Music", "IS_A", 0.9)]
            ),
            ConceptData(
                name="Renaissance Art",
                features=["painting", "sculpture", "architecture", "DaVinci", "Michelangelo"],
                domain="Art",
                importance=1.8,
                relationships=[("Art History", "IS_A", 0.9)]
            ),
            ConceptData(
                name="Surrealism",
                features=["dreams", "surreal", "Dali", "fantasy", "subconscious"],
                domain="Art",
                importance=1.5,
                relationships=[("Art Movement", "IS_A", 0.9)]
            ),
            ConceptData(
                name="Opera",
                features=["singing", "orchestra", "drama", "music", "theater"],
                domain="Art",
                importance=1.5,
                relationships=[("Music", "IS_A", 0.8)]
            ),
            ConceptData(
                name="Poetry",
                features=["writing", "rhythm", "metaphor", "emotion", "verse"],
                domain="Art",
                importance=1.8,
                relationships=[("Literature", "IS_A", 0.9)]
            ),
            ConceptData(
                name="Architecture",
                features=["building", "design", "engineering", "aesthetics", "structure"],
                domain="Art",
                importance=1.8,
                relationships=[("Art", "HAS", 0.8)]
            ),
        ]
        self._add_concepts(art_data)
        return len(art_data)
    
    def _load_philosophy_knowledge(self) -> int:
        """Load philosophy concepts"""
        philosophy_data = [
            ConceptData(
                name="Ethics",
                features=["morality", "good", "evil", "virtue", "consequences"],
                domain="Philosophy",
                importance=2.0,
                relationships=[("Philosophy", "IS_A", 0.9)]
            ),
            ConceptData(
                name="Existentialism",
                features=["freedom", "meaning", "choice", "individuality", "anxiety"],
                domain="Philosophy",
                importance=1.8,
                relationships=[("Philosophy", "IS_A", 0.9)]
            ),
            ConceptData(
                name="Logic",
                features=["reasoning", "arguments", "truth", "validity", "deduction"],
                domain="Philosophy",
                importance=2.0,
                relationships=[("Philosophy", "IS_A", 0.9)]
            ),
            ConceptData(
                name="Nihilism",
                features=["nothingness", "meaninglessness", "absurdity", "rejection", "values"],
                domain="Philosophy",
                importance=1.5,
                relationships=[("Existentialism", "RELATED_TO", 0.8)]
            ),
            ConceptData(
                name="Stoicism",
                features=["virtue", "wisdom", "courage", "justice", "temperance"],
                domain="Philosophy",
                importance=1.8,
                relationships=[("Philosophy", "IS_A", 0.9)]
            ),
            ConceptData(
                name="Utilitarianism",
                features=["utility", "happiness", "consequences", "greatest_good", "Bentham"],
                domain="Philosophy",
                importance=1.5,
                relationships=[("Ethics", "IS_A", 0.9)]
            ),
            ConceptData(
                name="Rationalism",
                features=["reason", "intuition", "deduction", "innate_ideas", "Descartes"],
                domain="Philosophy",
                importance=1.8,
                relationships=[("Philosophy", "IS_A", 0.9)]
            ),
            ConceptData(
                name="The Socratic Method",
                features=["questioning", "dialogue", "critical_thinking", "wisdom", "Socrates"],
                domain="Philosophy",
                importance=1.8,
                relationships=[("Philosophy", "HAS", 0.8)]
            ),
            ConceptData(
                name="Mind-Body Dualism",
                features=["consciousness", "body", "Descartes", "mind", "physical"],
                domain="Philosophy",
                importance=1.5,
                relationships=[("Philosophy", "RELATED_TO", 0.8)]
            ),
            ConceptData(
                name="Free Will",
                features=["choice", "determinism", "freedom", "responsibility", "consciousness"],
                domain="Philosophy",
                importance=1.8,
                relationships=[("Philosophy", "IS_A", 0.9)]
            ),
        ]
        self._add_concepts(philosophy_data)
        return len(philosophy_data)
    
    def _load_technology_knowledge(self) -> int:
        """Load technology concepts"""
        technology_data = [
            ConceptData(
                name="Artificial Intelligence",
                features=["machine_learning", "neural_networks", "deep_learning", "NLP", "robotics"],
                domain="Technology",
                importance=2.0,
                relationships=[("Computer Science", "IS_A", 0.9)]
            ),
            ConceptData(
                name="Quantum Computing",
                features=["qubits", "superposition", "entanglement", "quantum_gates", "computation"],
                domain="Technology",
                importance=1.8,
                relationships=[("Computer Science", "IS_A", 0.9)]
            ),
            ConceptData(
                name="Blockchain",
                features=["distributed", "cryptography", "consensus", "decentralized", "ledger"],
                domain="Technology",
                importance=1.8,
                relationships=[("Technology", "IS_A", 0.9)]
            ),
            ConceptData(
                name="Internet of Things",
                features=["devices", "sensors", "connectivity", "smart", "automation"],
                domain="Technology",
                importance=1.5,
                relationships=[("Technology", "IS_A", 0.9)]
            ),
            ConceptData(
                name="Cybersecurity",
                features=["security", "encryption", "firewall", "malware", "protection"],
                domain="Technology",
                importance=1.8,
                relationships=[("Computer Science", "IS_A", 0.9)]
            ),
            ConceptData(
                name="Cloud Computing",
                features=["servers", "storage", "SaaS", "IaaS", "PaaS"],
                domain="Technology",
                importance=1.8,
                relationships=[("Technology", "IS_A", 0.9)]
            ),
            ConceptData(
                name="Virtual Reality",
                features=["immersive", "3D", "gaming", "simulation", "headset"],
                domain="Technology",
                importance=1.5,
                relationships=[("Technology", "IS_A", 0.9)]
            ),
            ConceptData(
                name="Natural Language Processing",
                features=["language", "text", "speech", "sentiment_analysis", "translation"],
                domain="Technology",
                importance=1.8,
                relationships=[("Artificial Intelligence", "IS_A", 0.9)]
            ),
            ConceptData(
                name="Robotics",
                features=["automation", "mechanics", "electronics", "AI", "physical"],
                domain="Technology",
                importance=1.8,
                relationships=[("Artificial Intelligence", "RELATED_TO", 0.8)]
            ),
            ConceptData(
                name="5G Technology",
                features=["mobile", "network", "data", "speed", "connectivity"],
                domain="Technology",
                importance=1.5,
                relationships=[("Technology", "IS_A", 0.9)]
            ),
        ]
        self._add_concepts(technology_data)
        return len(technology_data)
    
    def _load_geography_knowledge(self) -> int:
        """Load geography concepts"""
        geography_data = [
            ConceptData(
                name="Himalayas",
                features=["mountains", "Asia", "Everest", "mountain", "tectonic"],
                domain="Geography",
                importance=1.5,
                relationships=[("Mountains", "IS_A", 0.9)]
            ),
            ConceptData(
                name="Amazon River",
                features=["river", "South America", "rainforest", "water", "ecosystem"],
                domain="Geography",
                importance=1.5,
                relationships=[("Rivers", "IS_A", 0.9)]
            ),
            ConceptData(
                name="Sahara Desert",
                features=["desert", "Africa", "sand", "arid", "heat"],
                domain="Geography",
                importance=1.5,
                relationships=[("Deserts", "IS_A", 0.9)]
            ),
            ConceptData(
                name="Pacific Ocean",
                features=["ocean", "largest", "water", "Mariana_Trench", "tectonic"],
                domain="Geography",
                importance=1.5,
                relationships=[("Oceans", "IS_A", 0.9)]
            ),
            ConceptData(
                name="Mount Everest",
                features=["mountain", "tallest", "Nepal", "Tibet", "climbing"],
                domain="Geography",
                importance=1.8,
                relationships=[("Himalayas", "PART_OF", 0.9)]
            ),
        ]
        self._add_concepts(geography_data)
        return len(geography_data)
    
    def _load_biology_knowledge(self) -> int:
        """Load biology concepts"""
        biology_data = [
            ConceptData(
                name="Cell Biology",
                features=["cells", "organelles", "mitosis", "membrane", "nucleus"],
                domain="Biology",
                importance=1.8,
                relationships=[("Biology", "IS_A", 0.9)]
            ),
            ConceptData(
                name="Ecosystem",
                features=["environment", "species", "habitat", "food_web", "biodiversity"],
                domain="Biology",
                importance=1.8,
                relationships=[("Biology", "IS_A", 0.9)]
            ),
            ConceptData(
                name="Neuroscience",
                features=["brain", "neurons", "nervous_system", "consciousness", "neurotransmitters"],
                domain="Biology",
                importance=1.8,
                relationships=[("Biology", "IS_A", 0.9)]
            ),
            ConceptData(
                name="Evolutionary Biology",
                features=["evolution", "natural_selection", "adaptation", "speciation", "fitness"],
                domain="Biology",
                importance=1.8,
                relationships=[("Biology", "IS_A", 0.9)]
            ),
            ConceptData(
                name="Genetics",
                features=["genes", "inheritance", "traits", "DNA", "chromosomes"],
                domain="Biology",
                importance=1.8,
                relationships=[("Biology", "IS_A", 0.9)]
            ),
        ]
        self._add_concepts(biology_data)
        return len(biology_data)
    
    def _load_mathematics_knowledge(self) -> int:
        """Load mathematics concepts"""
        mathematics_data = [
            ConceptData(
                name="Calculus",
                features=["derivatives", "integrals", "limits", "functions", "continuity"],
                domain="Mathematics",
                importance=1.8,
                relationships=[("Mathematics", "IS_A", 0.9)]
            ),
            ConceptData(
                name="Linear Algebra",
                features=["vectors", "matrices", "determinants", "spaces", "transformations"],
                domain="Mathematics",
                importance=1.8,
                relationships=[("Mathematics", "IS_A", 0.9)]
            ),
            ConceptData(
                name="Probability",
                features=["randomness", "statistics", "events", "distributions", "expectation"],
                domain="Mathematics",
                importance=1.8,
                relationships=[("Mathematics", "IS_A", 0.9)]
            ),
            ConceptData(
                name="Number Theory",
                features=["primes", "integers", "modular_arithmetic", "divisibility", "cryptography"],
                domain="Mathematics",
                importance=1.5,
                relationships=[("Mathematics", "IS_A", 0.9)]
            ),
            ConceptData(
                name="Topology",
                features=["spaces", "continuity", "compactness", "connectedness", "metrization"],
                domain="Mathematics",
                importance=1.5,
                relationships=[("Mathematics", "IS_A", 0.9)]
            ),
        ]
        self._add_concepts(mathematics_data)
        return len(mathematics_data)
    
    # ============================================================
    # CORE LOADING METHODS
    # ============================================================
    
    def _add_concepts(self, concepts: List[ConceptData]):
        """Add concepts to the loader's collection"""
        self.concepts.extend(concepts)
    
    def load_from_file(self, filepath: str, format: str = 'auto') -> int:
        """Load concepts from a file"""
        logger.info(f"Loading from file: {filepath}")
        
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            return 0
        
        with open(filepath) as f:
            if format == 'auto':
                if filepath.endswith('.json'):
                    data = json.load(f)
                elif filepath.endswith('.csv'):
                    reader = csv.DictReader(f)
                    data = [row for row in reader]
                elif filepath.endswith('.yaml') or filepath.endswith('.yml'):
                    data = yaml.safe_load(f)
                else:
                    logger.error(f"Unknown file format: {filepath}")
                    return 0
            elif format == 'json':
                data = json.load(f)
            elif format == 'csv':
                reader = csv.DictReader(f)
                data = [row for row in reader]
            elif format == 'yaml':
                data = yaml.safe_load(f)
            else:
                logger.error(f"Unsupported format: {format}")
                return 0
        
        return self._parse_data(data)
    
    def load_from_folder(self, folderpath: str, recursive: bool = True) -> int:
        """Load all knowledge files from a folder"""
        logger.info(f"Loading from folder: {folderpath}")
        
        if not os.path.exists(folderpath):
            logger.error(f"Folder not found: {folderpath}")
            return 0
        
        total = 0
        for root, dirs, files in os.walk(folderpath):
            for file in files:
                if file.endswith(('.json', '.csv', '.yaml', '.yml')):
                    filepath = os.path.join(root, file)
                    total += self.load_from_file(filepath)
            if not recursive:
                break
        
        return total
    
    def load_from_api(self, url: str, api_key: Optional[str] = None, params: Dict = None) -> int:
        """Load concepts from an API"""
        logger.info(f"Loading from API: {url}")
        
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        response = requests.get(url, headers=headers, params=params or {})
        
        if response.status_code != 200:
            logger.error(f"API error: {response.status_code}")
            return 0
        
        data = response.json()
        
        # Auto-detect concept structure in response
        # Try common formats
        items = data.get('results', data.get('items', data.get('data', [])))
        if not items:
            # Maybe the data is directly the list
            if isinstance(data, list):
                items = data
            else:
                # Try to find any list in the data
                for key, value in data.items():
                    if isinstance(value, list) and len(value) > 0:
                        items = value
                        break
        
        return self._parse_data(items)
    
    def load_from_database(self, connection_string: str, table: str = 'concepts') -> int:
        """Load concepts from a database"""
        logger.info(f"Loading from database: {connection_string}")
        
        import sqlite3
        conn = sqlite3.connect(connection_string)
        cursor = conn.cursor()
        
        try:
            cursor.execute(f"SELECT * FROM {table} LIMIT 10000")
        except:
            logger.error(f"Table not found: {table}")
            return 0
        
        columns = [desc[0] for desc in cursor.description]
        data = [dict(zip(columns, row)) for row in cursor.fetchall()]
        conn.close()
        
        return self._parse_data(data)
    
    def load_from_web(self, url: str, selector: str = '') -> int:
        """Load concepts from a webpage by scraping"""
        logger.info(f"Loading from web: {url}")
        
        try:
            from bs4 import BeautifulSoup
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            if selector:
                elements = soup.select(selector)
            else:
                # Auto-detect content
                elements = soup.find_all(['h1', 'h2', 'h3', 'p', 'li'])
            
            concepts = []
            for el in elements:
                text = el.get_text().strip()
                if text and len(text) > 3:
                    # Extract concepts from text
                    words = text.split()[:5]  # Take first 5 words as features
                    concepts.append(ConceptData(
                        name=text[:50],  # Truncate long text
                        features=words,
                        domain="General",
                        importance=0.5
                    ))
            
            self._add_concepts(concepts)
            return len(concepts)
            
        except ImportError:
            logger.error("BeautifulSoup not installed. Install with: pip install beautifulsoup4")
            return 0
        except Exception as e:
            logger.error(f"Web scraping error: {e}")
            return 0
    
    def load_from_llm(self, provider: str, prompt: str, api_key: Optional[str] = None) -> int:
        """Load knowledge from an LLM using a prompt"""
        logger.info(f"Loading from LLM: {provider}")
        
        try:
            import openai
            
            # Get API key
            if api_key is None:
                if provider == "groq":
                    api_key = os.getenv("GROQ_API_KEY")
                elif provider == "openai":
                    api_key = os.getenv("OPENAI_API_KEY")
                elif provider == "huggingface":
                    api_key = os.getenv("HF_TOKEN")
            
            if not api_key:
                logger.error(f"No API key found for {provider}")
                return 0
            
            # Build client
            provider_configs = {
                "groq": {"base_url": "https://api.groq.com/openai/v1", "model": "llama-3.3-70b-versatile"},
                "openai": {"base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini"},
                "huggingface": {"base_url": "https://router.huggingface.co/v1", "model": "Qwen/Qwen2.5-72B-Instruct"},
            }
            
            config = provider_configs.get(provider)
            if not config:
                logger.error(f"Unknown provider: {provider}")
                return 0
            
            client = openai.OpenAI(
                base_url=config["base_url"],
                api_key=api_key
            )
            
            # Make request
            response = client.chat.completions.create(
                model=config["model"],
                messages=[
                    {"role": "system", "content": "You are a knowledge base. Return concepts as a JSON array with name, features (list), domain, and importance (1-3)."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            
            # Parse JSON response
            try:
                # Find JSON in response
                import re
                json_match = re.search(r'\[.*\]', content, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    data = json.loads(content)
                
                return self._parse_data(data)
                
            except json.JSONDecodeError:
                logger.error("Failed to parse LLM response as JSON")
                return 0
                
        except ImportError:
            logger.error("OpenAI not installed. Install with: pip install openai")
            return 0
        except Exception as e:
            logger.error(f"LLM loading error: {e}")
            return 0
    
    def _parse_data(self, data: Any) -> int:
        """Parse loaded data into ConceptData objects"""
        concepts = []
        
        if isinstance(data, list):
            for item in data:
                concept = self._parse_item(item)
                if concept:
                    concepts.append(concept)
        elif isinstance(data, dict):
            # Try to find list in dict
            for key, value in data.items():
                if isinstance(value, list):
                    for item in value:
                        concept = self._parse_item(item)
                        if concept:
                            concepts.append(concept)
                    break
        
        self._add_concepts(concepts)
        return len(concepts)
    
    def _parse_item(self, item: Dict) -> Optional[ConceptData]:
        """Parse a single item into ConceptData"""
        try:
            # Try to find concept name
            name = item.get('name', item.get('concept', item.get('title', item.get('label'))))
            if not name:
                return None
            
            # Try to find features
            features = item.get('features', item.get('keywords', item.get('tags')))
            if isinstance(features, str):
                features = features.split(',') if ',' in features else features.split()
            elif not features:
                # Use name and description
                desc = item.get('description', '')
                features = desc.split()[:5]
                if not features:
                    features = [name]
            
            # Ensure features is list
            if isinstance(features, str):
                features = [features]
            
            domain = item.get('domain', item.get('category', 'General'))
            importance = float(item.get('importance', item.get('score', 1.0)))
            numeric_value = item.get('numeric_value')
            
            return ConceptData(
                name=str(name).lower(),
                features=[str(f).lower() for f in features if f],
                domain=str(domain),
                importance=importance,
                numeric_value=numeric_value
            )
            
        except Exception as e:
            logger.error(f"Error parsing item: {e}")
            return None
    
    def get_concepts(self) -> List[ConceptData]:
        """Get all loaded concepts"""
        return self.concepts
    
    def get_concepts_count(self) -> int:
        """Get number of loaded concepts"""
        return len(self.concepts)
    
    def clear(self):
        """Clear all loaded concepts"""
        self.concepts = []
        self.total_concepts_loaded = 0
    
    def save_to_file(self, filepath: str):
        """Save loaded concepts to a file"""
        data = []
        for concept in self.concepts:
            data.append({
                "name": concept.name,
                "features": concept.features,
                "domain": concept.domain,
                "importance": concept.importance,
                "numeric_value": concept.numeric_value
            })
        
        with open(filepath, 'w') as f:
            if filepath.endswith('.json'):
                json.dump(data, f, indent=2)
            elif filepath.endswith('.yaml') or filepath.endswith('.yml'):
                yaml.dump(data, f)
            else:
                # Default to JSON
                json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(data)} concepts to {filepath}")
