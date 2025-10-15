"""
LoRA Factory - Production Ready (Optimized with DRY Principles)
Modern, Robust, Professional Implementation
"""

import asyncio
import hashlib
import ipaddress
import json
import logging
import os
import re
import socket
import subprocess
import time
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import modal
from cerebras.cloud.sdk import Cerebras

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}
ALLOWED_MIME_TYPES = {'image/jpeg', 'image/png', 'image/webp'}
MAX_FILE_SIZE = 20 * 1024 * 1024
MIN_IMAGE_SIZE = 512
THUMBNAIL_SIZE = 300
MIN_IMAGE_COUNT = 10
MAX_IMAGE_COUNT = 1000

# Blocked IP ranges for security
BLOCKED_IP_RANGES = [
    ipaddress.ip_network('10.0.0.0/8'),
    ipaddress.ip_network('172.16.0.0/12'),
    ipaddress.ip_network('192.168.0.0/16'),
    ipaddress.ip_network('127.0.0.0/8'),
    ipaddress.ip_network('169.254.0.0/16'),
    ipaddress.ip_network('::1/128'),
    ipaddress.ip_network('fc00::/7'),
]

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class ImageCandidate:
    url: str
    thumbnail_path: str = ""
    approved: bool = False
    rejected: bool = False
    width: int = 0
    height: int = 0
    file_size: int = 0
    hash: str = ""

@dataclass
class ConceptConfig:
    name: str
    description: str
    image_count: int
    collected: int = 0
    candidates: List[ImageCandidate] = field(default_factory=list)

@dataclass
class LoRAInfo:
    task_id: str
    name: str
    concepts: List[Dict[str, str]]
    file_path: str
    file_size: int
    created_at: float
    description: str = ""
    test_images: List[str] = field(default_factory=list)
    training_images_count: int = 0
    status: str = "completed"
    checkpoint_path: str = ""
    can_continue: bool = True

@dataclass
class TrainingTask:
    task_id: str
    status: str
    progress: int
    stage: str
    message: str
    lora_name: str
    concepts: List[Dict]
    created_at: float
    updated_at: float
    parent_lora_id: str = ""

# ============================================================================
# PATH MANAGER
# ============================================================================

class PathManager:
    """Centralized path management"""
    
    def __init__(self):
        self.base_paths = {
            'raw_images': Path("/raw_images"),
            'dataset': Path("/datasets"),
            'loras': Path("/loras"),
            'models': Path("/models"),
            'state': Path("/state"),
            'test': Path("/test_outputs"),
            'thumbnails': Path("/thumbnails"),
            'kohya': Path("/root/kohya_ss"),
        }
    
    def get_path(self, name: str) -> Path:
        return self.base_paths.get(name, Path(f"/{name}"))
    
    def ensure_dir(self, name: str) -> Path:
        path = self.get_path(name)
        path.mkdir(parents=True, exist_ok=True)
        return path

paths = PathManager()

# ============================================================================
# CONFIGURATION MANAGER
# ============================================================================

class ConfigManager:
    """Centralized configuration management"""
    
    # Model configurations
    LLM_MODEL_NAME = "gpt-oss-120b"
    VISION_MODEL = "Salesforce/blip2-opt-2.7b"
    
    # Model URLs
    MODEL_URLS = {
        'base': "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors",
        'test': "https://civitai.com/api/download/models/456194"
    }
    
    MODEL_NAMES = {
        'base': "sd_xl_base_1.0.safetensors",
        'test': "juggernaut_xl_v9.safetensors"
    }
    
    # Training parameters
    TRAINING_PARAMS = {
        'resolution': 1024,
        'batch_size': 1,
        'epochs': 10,
        'learning_rate': 1e-4,
        'network_dim': 128,
        'network_alpha': 64,
    }
    
    # API Keys
    CEREBRAS_API_KEY = "csk-j439vyke89px4we44r29wcvetwcfm6mjmp5xwmxx4m2mpmcn"

config = ConfigManager()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal"""
    filename = os.path.basename(filename)
    filename = re.sub(r'[^\w\s.-]', '', filename)
    filename = re.sub(r'[-\s]+', '_', filename)
    return filename[:255]

def sanitize_text(text: str) -> str:
    """Sanitize text input"""
    dangerous_chars = ['<', '>', '{', '}', '`', '$', '|', ';', '&']
    for char in dangerous_chars:
        text = text.replace(char, '')
    return text.strip()[:2000]

def safe_execute(func: Callable, *args, default=None, **kwargs):
    """Safely execute a function with error handling"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Error in {func.__name__}: {e}")
        return default

def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator for retrying functions on failure"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(delay * (2 ** attempt))
            return None
        return wrapper
    return decorator

# ============================================================================
# SECURITY VALIDATOR
# ============================================================================

class SecurityValidator:
    """Centralized security validation"""
    
    @staticmethod
    def is_safe_url(url: str) -> bool:
        """Check if URL is safe to fetch"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            
            if parsed.scheme not in ['http', 'https']:
                return False
            
            if not parsed.hostname:
                return False
            
            ip = socket.gethostbyname(parsed.hostname)
            ip_obj = ipaddress.ip_address(ip)
            
            return not any(ip_obj in range for range in BLOCKED_IP_RANGES)
        except Exception as e:
            logger.error(f"URL validation error: {e}")
            return False
    
    @staticmethod
    def validate_image_file(file_path: Path) -> bool:
        """Validate image file"""
        try:
            import magic
            from PIL import Image
            
            if file_path.stat().st_size > MAX_FILE_SIZE:
                return False
            
            if file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
                return False
            
            mime = magic.from_file(str(file_path), mime=True)
            if mime not in ALLOWED_MIME_TYPES:
                return False
            
            img = Image.open(file_path)
            img.verify()
            img = Image.open(file_path)
            
            if min(img.size) < MIN_IMAGE_SIZE:
                return False
            
            return True
        except Exception as e:
            logger.error(f"Image validation error: {e}")
            return False

# ============================================================================
# DATA MANAGER - Generic persistence manager
# ============================================================================

class DataManager:
    """Generic data manager with persistence"""
    
    def __init__(self, filename: str, data_class: type, id_field: str = "task_id"):
        self.file_path = paths.get_path('state') / filename
        self.data_class = data_class
        self.id_field = id_field
        self.data = []
        self._load()
    
    def _load(self):
        """Load data from file"""
        safe_execute(self._load_internal)
    
    def _load_internal(self):
        """Internal loading logic"""
        if self.file_path.exists():
            with open(self.file_path) as f:
                data = json.load(f)
                self.data = [self.data_class(**item) for item in data]
    
    def _save(self):
        """Save data to file"""
        safe_execute(self._save_internal)
    
    def _save_internal(self):
        """Internal saving logic"""
        paths.ensure_dir('state')
        with open(self.file_path, 'w') as f:
            json.dump([asdict(item) for item in self.data], f, indent=2)
    
    def add(self, item):
        """Add new item"""
        self.data.append(item)
        self._save()
    
    def get_all(self):
        """Get all items"""
        return self.data
    
    def get_by_id(self, id_value: str):
        """Get item by ID"""
        return next((item for item in self.data if getattr(item, self.id_field) == id_value), None)
    
    def update(self, id_value: str, updates: Dict):
        """Update item by ID"""
        for item in self.data:
            if getattr(item, self.id_field) == id_value:
                for key, value in updates.items():
                    if hasattr(item, key):
                        setattr(item, key, value)
                self._save()
                return True
        return False

# ============================================================================
# SPECIALIZED DATA MANAGERS
# ============================================================================

class LoRALibrary(DataManager):
    """LoRA library manager"""
    
    def __init__(self):
        super().__init__("lora_library.json", LoRAInfo)
    
    def add_lora(self, lora: LoRAInfo):
        self.add(lora)
    
    def get_by_id(self, task_id: str) -> Optional[LoRAInfo]:
        return super().get_by_id(task_id)
    
    def update_lora(self, task_id: str, updates: Dict):
        return super().update(task_id, updates)

class TaskManager(DataManager):
    """Task manager with special handling for tasks"""
    
    def __init__(self):
        super().__init__("tasks.json", TrainingTask)
        self.tasks = {task.task_id: task for task in self.data}
    
    def _save_internal(self):
        """Override save for tasks dictionary format"""
        paths.ensure_dir('state')
        with open(self.file_path, 'w') as f:
            json.dump({k: asdict(v) for k, v in self.tasks.items()}, f, indent=2)
    
    def create_task(self, task_id: str, lora_name: str, concepts: List[Dict], parent_lora_id: str = "") -> TrainingTask:
        """Create new task"""
        task = TrainingTask(
            task_id=task_id,
            status="scraping",
            progress=0,
            stage="Initializing",
            message="Starting image collection...",
            lora_name=lora_name,
            concepts=concepts,
            created_at=time.time(),
            updated_at=time.time(),
            parent_lora_id=parent_lora_id
        )
        self.tasks[task_id] = task
        self._save()
        return task
    
    def update_task(self, task_id: str, **kwargs):
        """Update task"""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            for key, value in kwargs.items():
                if hasattr(task, key):
                    setattr(task, key, value)
            task.updated_at = time.time()
            self._save()
    
    def get_task(self, task_id: str) -> Optional[TrainingTask]:
        """Get task by ID"""
        return self.tasks.get(task_id)
    
    def get_all_tasks(self) -> List[TrainingTask]:
        """Get all tasks"""
        return list(self.tasks.values())

# ============================================================================
# MODEL STATUS MANAGER
# ============================================================================

class ModelStatusManager:
    """Centralized model status management"""
    
    @staticmethod
    def check_model(model_type: str, model_path: Path = None, model_name: str = None, purpose: str = "") -> Dict[str, Any]:
        """Generic model status checker"""
        if model_type == "llm":
            return {
                "ready": True,
                "files": 1,
                "size_mb": 0,
                "name": "Cerebras GPT-OSS-120B",
                "purpose": "Smart query generation & caption enhancement"
            }
        
        if not model_path or not model_name:
            return {"ready": False, "size_mb": 0, "name": model_name or "Unknown"}
        
        try:
            if model_path.exists():
                size = model_path.stat().st_size
                return {
                    "ready": True,
                    "size_mb": round(size / 1024 / 1024, 2),
                    "name": model_name,
                    "purpose": purpose
                }
            return {
                "ready": False,
                "size_mb": 0,
                "name": model_name,
                "purpose": purpose
            }
        except Exception as e:
            logger.error(f"Model check error for {model_name}: {e}")
            return {"ready": False, "size_mb": 0, "name": model_name}
    
    @classmethod
    def get_all_status(cls) -> Dict[str, Any]:
        """Get all model statuses"""
        models_path = paths.get_path('models')
        return {
            "llm": cls.check_model("llm"),
            "base_model": cls.check_model(
                "base",
                models_path / config.MODEL_NAMES['base'],
                "SDXL Base 1.0",
                "Base model for LoRA training"
            ),
            "test_model": cls.check_model(
                "test",
                models_path / config.MODEL_NAMES['test'],
                "Juggernaut XL v9",
                "Test image generation"
            )
        }

# ============================================================================
# HTTP CLIENT MANAGER
# ============================================================================

class HTTPClientManager:
    """Centralized HTTP client management"""
    
    def __init__(self):
        self.session = None
    
    def get_session(self):
        """Get or create HTTP session"""
        if not self.session:
            import requests
            self.session = requests.Session()
            self.session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
        return self.session
    
    @retry_on_failure(max_retries=3)
    def get(self, url: str, **kwargs):
        """Make GET request with retry"""
        session = self.get_session()
        return session.get(url, timeout=kwargs.get('timeout', 15), **kwargs)
    
    @retry_on_failure(max_retries=3)
    def download_stream(self, url: str, output_path: Path, **kwargs):
        """Download file with streaming"""
        session = self.get_session()
        response = session.get(url, stream=True, timeout=kwargs.get('timeout', 600), **kwargs)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(8192):
                f.write(chunk)
        
        return response

http_client = HTTPClientManager()

# ============================================================================
# IMAGE SCRAPER
# ============================================================================

class ImageScraper:
    """Multi-source image scraper with generic scraping methods"""
    
    def __init__(self):
        self.scrapers = {
            'duckduckgo': self._scrape_duckduckgo,
            'pixabay': self._scrape_pixabay,
            'pexels': self._scrape_pexels,
            'unsplash': self._scrape_unsplash,
            'flickr': self._scrape_flickr,
            'wikimedia': self._scrape_wikimedia
        }
    
    def _scrape_with_regex(self, url: str, pattern: str, limit: int = 100) -> List[str]:
        """Generic scraping with regex"""
        try:
            response = http_client.get(url)
            if response.status_code == 200:
                matches = re.findall(pattern, response.text)
                return list(set(matches))[:limit]
        except Exception as e:
            logger.debug(f"Regex scraping error: {e}")
        return []
    
    def _scrape_with_bs4(self, url: str, extractor: Callable, limit: int = 100) -> List[str]:
        """Generic scraping with BeautifulSoup"""
        try:
            from bs4 import BeautifulSoup
            response = http_client.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                urls = extractor(soup)
                return list(set(urls))[:limit]
        except Exception as e:
            logger.debug(f"BS4 scraping error: {e}")
        return []
    
    def _scrape_duckduckgo(self, query: str) -> List[str]:
        """Scrape DuckDuckGo images"""
        url = f"https://duckduckgo.com/?q={query}&iax=images&ia=images"
        pattern = r'https?://[^\s<"]+?\.(?:jpg|jpeg|png|webp)'
        return self._scrape_with_regex(url, pattern)
    
    def _scrape_pixabay(self, query: str) -> List[str]:
        """Scrape Pixabay"""
        url = f"https://pixabay.com/images/search/{query}/"
        
        def extractor(soup):
            urls = []
            for img in soup.find_all('img', {'srcset': True}):
                srcset = img.get('srcset', '')
                parts = srcset.split(',')
                for part in parts:
                    url_match = re.search(r'(https?://[^\s]+)', part)
                    if url_match:
                        urls.append(url_match.group(1).split()[0])
            return urls
        
        return self._scrape_with_bs4(url, extractor)
    
    def _scrape_pexels(self, query: str) -> List[str]:
        """Scrape Pexels"""
        url = f"https://www.pexels.com/search/{query}/"
        
        def extractor(soup):
            urls = []
            for img in soup.find_all('img', {'src': True}):
                src = img.get('src', '')
                if 'images.pexels.com' in src and src.startswith('http'):
                    urls.append(src)
            return urls
        
        return self._scrape_with_bs4(url, extractor)
    
    def _scrape_unsplash(self, query: str) -> List[str]:
        """Scrape Unsplash"""
        url = f"https://unsplash.com/s/photos/{query}"
        pattern = r'https://images\.unsplash\.com/[^\s<"]+'
        return self._scrape_with_regex(url, pattern)
    
    def _scrape_flickr(self, query: str) -> List[str]:
        """Scrape Flickr"""
        url = f"https://www.flickr.com/search/?text={query}"
        pattern = r'https://[^\s<"]+flickr\.com/[^\s<"]+\.jpg'
        return self._scrape_with_regex(url, pattern)
    
    def _scrape_wikimedia(self, query: str) -> List[str]:
        """Scrape Wikimedia Commons"""
        url = f"https://commons.wikimedia.org/w/api.php?action=query&generator=search&gsrsearch={query}&gsrnamespace=6&gsrlimit=50&prop=imageinfo&iiprop=url&format=json"
        
        try:
            response = http_client.get(url)
            if response.status_code == 200:
                data = response.json()
                urls = []
                if 'query' in data and 'pages' in data['query']:
                    for page in data['query']['pages'].values():
                        if 'imageinfo' in page:
                            for img in page['imageinfo']:
                                if 'url' in img:
                                    urls.append(img['url'])
                return urls[:100]
        except Exception as e:
            logger.debug(f"Wikimedia scraping error: {e}")
        return []
    
    def scrape_all_sources(self, queries: List[str]) -> List[str]:
        """Scrape all sources in parallel"""
        all_urls = []
        
        with ThreadPoolExecutor(max_workers=25) as executor:
            futures = []
            
            for query in queries:
                query_encoded = query.replace(' ', '+')
                for scraper_func in self.scrapers.values():
                    futures.append(executor.submit(scraper_func, query_encoded))
            
            for future in as_completed(futures):
                try:
                    urls = future.result()
                    all_urls.extend(urls)
                except Exception as e:
                    logger.debug(f"Scraper error: {e}")
        
        unique_urls = list(set(all_urls))
        logger.info(f"Total unique URLs found: {len(unique_urls)}")
        return unique_urls

# ============================================================================
# CEREBRAS API MANAGER
# ============================================================================

class CerebrasAPIManager:
    """Centralized Cerebras API management"""
    
    def __init__(self):
        self.client = None
    
    def get_client(self):
        """Get or create Cerebras client"""
        if not self.client:
            api_key = os.environ.get("CEREBRAS_API_KEY")
            if not api_key:
                raise Exception("CEREBRAS_API_KEY not found in environment")
            self.client = Cerebras(api_key=api_key)
        return self.client
    
    @retry_on_failure(max_retries=3)
    def generate_text(self, prompt: str, max_tokens: int = 800, temperature: float = 0.7) -> str:
        """Generate text using Cerebras API"""
        client = self.get_client()
        
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant specialized in image search and LoRA training."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model=config.LLM_MODEL_NAME,
            max_completion_tokens=max_tokens,
            temperature=temperature,
            top_p=1,
            reasoning_effort="medium"
        )
        
        return response.choices[0].message.content

cerebras_api = CerebrasAPIManager()

# ============================================================================
# SMART AGENT
# ============================================================================

class SmartAgent:
    """AI agent powered by Cerebras"""
    
    def __init__(self):
        self.scraper = ImageScraper()
        self.fallback_queries = [
            lambda name, desc: [f"{name} professional photo", f"{name} high quality portrait", f"{desc}"],
            lambda name, desc: [f"{name} studio photo", f"{name} high resolution", f"professional {name}"]
        ]
    
    def generate_search_queries(self, concept_name: str, concept_description: str, target_count: int) -> List[str]:
        """Generate diverse search queries"""
        try:
            prompt = f"""Generate 8-10 diverse search queries to find high-quality images.

Subject: {concept_name}
Description: {concept_description}
Target: {target_count} images

Create specific, varied queries that will find professional, high-quality images from different angles and contexts.
Include variations like: "professional photo", "portrait", "full body", "high resolution", "studio photography", etc.

Return ONLY the queries, one per line, numbered 1-10.

Queries:"""

            response = cerebras_api.generate_text(prompt, max_tokens=500)
            
            queries = []
            for line in response.split('\n'):
                line = line.strip()
                line = re.sub(r'^[\d\.\)\-\s]+', '', line)
                if line and 5 < len(line) < 150:
                    queries.append(line)
            
            if len(queries) >= 3:
                logger.info(f"✅ Generated {len(queries)} queries using Cerebras")
                return queries[:10]
        except Exception as e:
            logger.error(f"Query generation error: {e}")
        
        # Fallback queries
        logger.warning("Using fallback queries")
        return [
            f"{concept_name} professional photo",
            f"{concept_name} high quality portrait",
            f"{concept_name} full body professional photography",
            f"{concept_description}",
            f"{concept_name} studio photo",
            f"{concept_name} high resolution image",
            f"professional {concept_name} photograph",
            f"{concept_name} detailed photo",
            f"{concept_name} 4k photo",
            f"{concept_name} hd image"
        ]
    
    def scrape_with_monitoring(self, concept_name: str, concept_description: str, target_count: int, task_id: str) -> List[str]:
        """Scrape images with AI monitoring"""
        task_mgr = TaskManager()
        task_mgr.update_task(task_id, stage="AI generating search queries", progress=5)
        
        queries = self.generate_search_queries(concept_name, concept_description, target_count)
        
        task_mgr.update_task(task_id, stage="Scraping 6 sources simultaneously", progress=10)
        
        all_urls = self.scraper.scrape_all_sources(queries)
        
        task_mgr.update_task(task_id, stage=f"Found {len(all_urls)} URLs, validating", progress=30)
        
        if len(all_urls) >= target_count:
            logger.info(f"Target reached: {len(all_urls)} >= {target_count}")
            return all_urls[:target_count * 3]
        else:
            logger.info(f"Collected {len(all_urls)} URLs (target: {target_count})")
            return all_urls
    
    def parse_user_request(self, user_input: str, available_loras: List[LoRAInfo]) -> Dict:
        """Parse user request using AI"""
        user_input = sanitize_text(user_input)
        
        lora_context = ""
        if available_loras:
            lora_context = "\n\nAvailable LoRAs:\n"
            for lora in available_loras[:15]:
                concepts_str = ', '.join([c['name'] for c in lora.concepts])
                lora_context += f"- [{lora.task_id}] {lora.name}: {concepts_str}\n"
        
        prompt = f"""Parse this LoRA training request and extract the information.

{lora_context}

User Request: "{user_input}"

Extract:
1. Action: "train_new" / "continue_training" / "list" / "help"
2. If continue_training: parent_lora_id (task_id of existing LoRA)
3. Concepts: name and detailed description
4. Image count per concept (default: 300, max: 1000)
5. LoRA name

Return ONLY valid JSON in this exact format:
{{
  "action": "train_new",
  "parent_lora_id": "",
  "concepts": [{{"name": "Concept Name", "description": "detailed visual description", "image_count": 300}}],
  "lora_name": "descriptive_name_v1",
  "message": "User-friendly confirmation message"
}}

JSON:"""

        try:
            response = cerebras_api.generate_text(prompt, max_tokens=1000)
            
            json_match = re.search(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                
                # Sanitize extracted data
                if 'concepts' in data:
                    for concept in data['concepts']:
                        concept['name'] = sanitize_text(concept.get('name', ''))
                        concept['description'] = sanitize_text(concept.get('description', ''))
                        concept['image_count'] = max(
                            MIN_IMAGE_COUNT,
                            min(concept.get('image_count', 300), MAX_IMAGE_COUNT)
                        )
                
                if 'lora_name' in data:
                    data['lora_name'] = sanitize_filename(data['lora_name'])
                
                return data
        except Exception as e:
            logger.error(f"Parsing error: {e}")
        
        return {
            'action': 'help',
            'concepts': [],
            'lora_name': '',
            'parent_lora_id': '',
            'message': 'Please describe what you want to train. Example: "Train Taylor Swift with 500 images"'
        }
    
    def enhance_caption(self, basic_caption: str, concept_description: str) -> str:
        """Enhance image caption with AI"""
        try:
            prompt = f"""Improve this image caption for LoRA training.

Basic caption: "{basic_caption}"
Subject: "{concept_description}"

Create a natural, detailed caption (max 75 words) that describes the image accurately.
Focus on visual details, pose, lighting, and composition.

Enhanced caption:"""

            enhanced = cerebras_api.generate_text(prompt, max_tokens=150, temperature=0.7)
            result = sanitize_text(enhanced)[:200]
            
            if len(result) > 10:
                return result
        except Exception as e:
            logger.warning(f"Caption enhancement failed: {e}")
        
        return f"{concept_description}, {basic_caption}"

# ============================================================================
# IMAGE PROCESSOR
# ============================================================================

class ImageProcessor:
    """Centralized image processing"""
    
    @staticmethod
    @retry_on_failure(max_retries=2)
    def download_and_validate(url: str, output_path: Path) -> Optional[ImageCandidate]:
        """Download and validate single image"""
        if not SecurityValidator.is_safe_url(url):
            return None
        
        try:
            response = http_client.get(url, timeout=20, stream=True)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '')
            if 'image' not in content_type:
                return None
            
            temp_path = output_path.with_suffix('.tmp')
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(8192):
                    f.write(chunk)
            
            if not SecurityValidator.validate_image_file(temp_path):
                temp_path.unlink()
                return None
            
            return ImageProcessor._process_image(temp_path, output_path, url)
            
        except Exception as e:
            logger.debug(f"Download error for {url}: {e}")
            return None
    
    @staticmethod
    def _process_image(temp_path: Path, output_path: Path, url: str) -> ImageCandidate:
        """Process downloaded image"""
        from PIL import Image
        
        img = Image.open(temp_path)
        width, height = img.size
        file_size = temp_path.stat().st_size
        
        with open(temp_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        
        img = img.convert('RGB')
        img.save(output_path, 'JPEG', quality=95)
        temp_path.unlink()
        
        # Create thumbnail
        thumbnail_path = output_path.parent / "thumbnails" / output_path.name
        thumbnail_path.parent.mkdir(exist_ok=True)
        img.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE), Image.Resampling.LANCZOS)
        img.save(thumbnail_path, 'JPEG', quality=85)
        
        return ImageCandidate(
            url=url,
            thumbnail_path=str(thumbnail_path),
            width=width,
            height=height,
            file_size=file_size,
            hash=file_hash,
            approved=False,
            rejected=False
        )
    
    @staticmethod
    def download_batch_parallel(urls: List[str], output_dir: Path, max_workers: int = 15) -> List[ImageCandidate]:
        """Download multiple images in parallel"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        candidates = []
        seen_hashes = set()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for idx, url in enumerate(urls):
                output_path = output_dir / f"img_{idx:04d}.jpg"
                future = executor.submit(ImageProcessor.download_and_validate, url, output_path)
                futures[future] = url
            
            for future in as_completed(futures):
                try:
                    candidate = future.result()
                    if candidate and candidate.hash not in seen_hashes:
                        candidates.append(candidate)
                        seen_hashes.add(candidate.hash)
                except Exception as e:
                    logger.debug(f"Download error: {e}")
        
        return candidates

# ============================================================================
# MODEL DOWNLOADER
# ============================================================================

class ModelDownloader:
    """Centralized model downloading"""
    
    @staticmethod
    def download_model(model_type: str) -> bool:
        """Download a specific model"""
        models_path = paths.ensure_dir('models')
        model_path = models_path / config.MODEL_NAMES[model_type]
        
        if model_path.exists():
            logger.info(f"{model_type.title()} model already exists")
            return True
        
        logger.info(f"Downloading {model_type} model...")
        try:
            from tqdm import tqdm
            
            response = http_client.get(
                config.MODEL_URLS[model_type],
                stream=True,
                headers={'User-Agent': 'Mozilla/5.0'} if model_type == 'test' else {}
            )
            response.raise_for_status()
            
            total = int(response.headers.get('content-length', 0))
            with open(model_path, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True) as pbar:
                for chunk in response.iter_content(8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
            
            logger.info(f"{model_type.title()} model downloaded")
            return True
        except Exception as e:
            logger.error(f"Failed to download {model_type} model: {e}")
            return False

# ============================================================================
# MODAL SETUP
# ============================================================================

app = modal.App("lora-factory-production")

# Modal Image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # Core
        "fastapi[all]==0.104.1",
        "pydantic==2.5.0",
        "uvicorn[standard]==0.24.0",
        "requests==2.31.0",
        "aiohttp==3.9.1",
        "aiofiles==23.2.1",
        
        # ML & Vision
        "torch==2.1.0",
        "torchvision==0.16.0",
        "transformers==4.36.0",
        "accelerate==0.25.0",
        "sentencepiece==0.1.99",
        "protobuf==3.20.3",
        
        # Image Processing
        "pillow==10.1.0",
        "opencv-python-headless==4.8.1.78",
        
        # Diffusion
        "diffusers==0.25.0",
        "safetensors==0.4.1",
        "compel==2.0.2",
        
        # Utils
        "beautifulsoup4==4.12.2",
        "python-magic==0.4.27",
        "tenacity==8.2.3",
        "tqdm==4.66.1",
        "huggingface-hub==0.20.0",
        "cerebras-cloud-sdk==1.0.0",
    )
    .apt_install(
        "git", "wget", "curl", 
        "libgl1-mesa-glx", "libglib2.0-0", 
        "libmagic1", "file"
    )
    .run_commands(
        "git clone --depth 1 https://github.com/bmaltais/kohya_ss.git /root/kohya_ss",
    )
)

# Modal Volumes
volumes = {
    str(paths.get_path('raw_images')): modal.Volume.from_name("lora-raw-v2", create_if_missing=True),
    str(paths.get_path('dataset')): modal.Volume.from_name("lora-dataset-v2", create_if_missing=True),
    str(paths.get_path('loras')): modal.Volume.from_name("lora-outputs-v2", create_if_missing=True),
    str(paths.get_path('models')): modal.Volume.from_name("lora-models-v2", create_if_missing=True),
    str(paths.get_path('state')): modal.Volume.from_name("lora-state-v2", create_if_missing=True),
    str(paths.get_path('test')): modal.Volume.from_name("lora-test-v2", create_if_missing=True),
    str(paths.get_path('thumbnails')): modal.Volume.from_name("lora-thumbnails-v2", create_if_missing=True),
}

# ============================================================================
# MODAL FUNCTIONS
# ============================================================================

@app.function(
    image=image,
    volumes={str(paths.get_path('models')): volumes[str(paths.get_path('models'))]},
    timeout=2400,
)
def download_base_models():
    """Download base models"""
    results = {}
    for model_type in ['base', 'test']:
        results[model_type] = ModelDownloader.download_model(model_type)
        if results[model_type]:
            volumes[str(paths.get_path('models'))].commit()
    return results

@app.function(
    image=image,
    secrets=[config.CEREBRAS_API_KEY],
    timeout=600,
    volumes={str(paths.get_path('state')): volumes[str(paths.get_path('state'))]}
)
def ai_agent_process(user_input: str) -> Dict:
    """Process user input with AI agent"""
    try:
        volumes[str(paths.get_path('state'))].reload()
        lib = LoRALibrary()
        agent = SmartAgent()
        result = agent.parse_user_request(user_input, lib.get_all())
        return result
    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
        return {
            'action': 'error',
            'concepts': [],
            'lora_name': '',
            'parent_lora_id': '',
            'message': f'Processing error: {str(e)}'
        }

@app.function(
    image=image,
    secrets=[config.CEREBRAS_API_KEY],
    timeout=300,
    volumes={str(paths.get_path('state')): volumes[str(paths.get_path('state'))]}
)
def orchestrate(user_input: str) -> Dict:
    """Orchestrate training workflow"""
    task_id = str(uuid.uuid4())[:8]
    
    try:
        volumes[str(paths.get_path('state'))].reload()
        
        decision_result = ai_agent_process.remote(user_input)
        action = decision_result.get('action', 'help')
        
        if action in ['train_new', 'continue_training']:
            concepts = [
                ConceptConfig(
                    name=c['name'],
                    description=c.get('description', c['name']),
                    image_count=c.get('image_count', 300),
                )
                for c in decision_result.get('concepts', [])
            ]
            
            if not concepts:
                return {'error': 'No concepts found', 'message': decision_result.get('message', '')}
            
            parent_lora_id = decision_result.get('parent_lora_id', '')
            lora_name = decision_result.get('lora_name', f"lora_{task_id}")
            
            if action == 'continue_training' and parent_lora_id:
                lora_name = f"{lora_name}_continued"
            
            task_mgr = TaskManager()
            task_mgr.create_task(task_id, lora_name, [asdict(c) for c in concepts], parent_lora_id)
            
            scrape_images.spawn(task_id, [asdict(c) for c in concepts], lora_name)
            
            return {
                'task_id': task_id,
                'action': action,
                'lora_name': lora_name,
                'parent_lora_id': parent_lora_id,
                'concepts': [{'name': c.name, 'description': c.description, 'count': c.image_count} for c in concepts],
                'message': decision_result.get('message', 'Training started!')
            }
        
        elif action == 'list':
            lib = LoRALibrary()
            return {
                'action': action,
                'loras': [asdict(l) for l in lib.get_all()],
                'message': decision_result.get('message', '')
            }
        
        else:
            return {
                'action': action,
                'message': decision_result.get('message', 'How can I help?')
            }
    
    except Exception as e:
        logger.error(f"Orchestration error: {e}", exc_info=True)
        return {'error': str(e), 'message': f'Error: {str(e)}', 'action': 'error'}

@app.function(
    image=image,
    volumes={
        str(paths.get_path('raw_images')): volumes[str(paths.get_path('raw_images'))],
        str(paths.get_path('thumbnails')): volumes[str(paths.get_path('thumbnails'))],
        str(paths.get_path('state')): volumes[str(paths.get_path('state'))]
    },
    secrets=[config.CEREBRAS_API_KEY],
    timeout=3600,
)
def scrape_images(task_id: str, concepts_dict: List[Dict], lora_name: str):
    """Scrape images for all concepts"""
    concepts = [ConceptConfig(**c) for c in concepts_dict]
    
    task_mgr = TaskManager()
    task_mgr.update_task(task_id, status="scraping", stage="Initializing AI Agent", progress=0)
    
    # Initialize AI agent
    agent = safe_execute(SmartAgent)
    use_ai = agent is not None
    
    if not use_ai:
        task_mgr.update_task(task_id, stage="Using fallback scraper (AI unavailable)", progress=0)
    
    all_concepts_data = []
    
    for idx, concept in enumerate(concepts):
        logger.info(f"Processing concept {idx+1}/{len(concepts)}: {concept.name}")
        
        task_mgr.update_task(
            task_id,
            stage=f"{'AI' if use_ai else 'Fallback'} processing: {concept.name}",
            progress=int((idx / len(concepts)) * 40)
        )
        
        # Get URLs
        if use_ai and agent:
            urls = safe_execute(
                agent.scrape_with_monitoring,
                concept.name, concept.description, concept.image_count, task_id,
                default=[]
            )
            if not urls:
                # Fallback if AI fails
                scraper = ImageScraper()
                queries = [f"{concept.name} professional photo", f"{concept.description}"]
                urls = scraper.scrape_all_sources(queries)
        else:
            scraper = ImageScraper()
            queries = [
                f"{concept.name} professional photo",
                f"{concept.name} high quality portrait",
                f"{concept.description}",
                f"{concept.name} studio photo"
            ]
            urls = scraper.scrape_all_sources(queries)
        
        task_mgr.update_task(
            task_id,
            stage=f"Downloading & validating: {concept.name}",
            progress=int((idx / len(concepts)) * 40) + 20
        )
        
        # Download images
        folder_name = sanitize_filename(concept.name.lower().replace(' ', '_'))
        out_dir = paths.ensure_dir('raw_images') / task_id / folder_name
        
        candidates = ImageProcessor.download_batch_parallel(urls, out_dir, max_workers=20)
        
        concept.candidates = candidates
        concept.collected = len(candidates)
        all_concepts_data.append(asdict(concept))
        
        logger.info(f"Collected {len(candidates)} valid images for {concept.name}")
        
        # Commit volumes periodically
        if idx % 2 == 0:
            volumes[str(paths.get_path('raw_images'))].commit()
            volumes[str(paths.get_path('thumbnails'))].commit()
    
    # Final commit
    volumes[str(paths.get_path('raw_images'))].commit()
    volumes[str(paths.get_path('thumbnails'))].commit()
    
    # Save candidates data
    candidates_file = paths.get_path('state') / f"{task_id}_candidates.json"
    with open(candidates_file, 'w') as f:
        json.dump(all_concepts_data, f, indent=2)
    volumes[str(paths.get_path('state'))].commit()
    
    total_collected = sum(c.collected for c in concepts)
    task_mgr.update_task(
        task_id,
        status="review",
        stage="Ready for manual review",
        progress=50,
        message=f"{'AI' if use_ai else 'Fallback scraper'} collected {total_collected} images. Please review and approve."
    )
    
    logger.info(f"Scraping complete for task {task_id}. Awaiting manual review.")

# ============================================================================
# WEB SERVER (Optional - for API endpoints)
# ============================================================================

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

web_app = FastAPI(title="LoRA Factory API")

@web_app.get("/status")
def get_status():
    """Get system status"""
    return {
        "status": "running",
        "models": ModelStatusManager.get_all_status()
    }

@web_app.post("/process")
def process_request(user_input: str):
    """Process user request"""
    try:
        result = orchestrate.remote(user_input)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@web_app.get("/task/{task_id}")
def get_task_status(task_id: str):
    """Get task status"""
    task_mgr = TaskManager()
    task = task_mgr.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return asdict(task)

@web_app.get("/loras")
def list_loras():
    """List all LoRAs"""
    lib = LoRALibrary()
    return [asdict(l) for l in lib.get_all()]

# Mount web app to Modal
@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    return web_app
