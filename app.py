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
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import modal
from cerebras.cloud.sdk import Cerebras

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = modal.App("lora-factory-production-cerebras")

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

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi[all]",
        "pydantic",
        "uvicorn[standard]",
        "requests",
        "aiohttp",
        "aiofiles",
        "torch",
        "torchvision",
        "transformers",
        "accelerate",
        "protobuf",
        "pillow",
        "opencv-python-headless",
        "diffusers",
        "safetensors",
        "compel",
        "beautifulsoup4",
        "python-magic",
        "tenacity",
        "tqdm",
        "huggingface-hub",
        "cerebras-sdk", 
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

raw_images_vol = modal.Volume.from_name("lora-raw-v2", create_if_missing=True)
dataset_vol = modal.Volume.from_name("lora-dataset-v2", create_if_missing=True)
loras_vol = modal.Volume.from_name("lora-outputs-v2", create_if_missing=True)
models_vol = modal.Volume.from_name("lora-models-v2", create_if_missing=True)
state_vol = modal.Volume.from_name("lora-state-v2", create_if_missing=True)
test_vol = modal.Volume.from_name("lora-test-v2", create_if_missing=True)
thumbnails_vol = modal.Volume.from_name("lora-thumbnails-v2", create_if_missing=True)

class Config:
    RAW_IMAGES_PATH = Path("/raw_images")
    DATASET_PATH = Path("/datasets")
    LORAS_PATH = Path("/loras")
    MODELS_PATH = Path("/models")
    STATE_PATH = Path("/state")
    TEST_PATH = Path("/test_outputs")
    THUMBNAILS_PATH = Path("/thumbnails")
    KOHYA_PATH = Path("/root/kohya_ss")

    VISION_MODEL = "Salesforce/blip2-opt-2.7b"
    BASE_MODEL_URL = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors"
    BASE_MODEL_NAME = "sd_xl_base_1.0.safetensors"
    TEST_MODEL_URL = "https://civitai.com/api/download/models/456194"
    TEST_MODEL_NAME = "juggernaut_xl_v9.safetensors"

    HF_TOKEN = modal.Secret.from_name("huggingface-token")
    CEREBRAS_SECRET = modal.Secret.from_dict({"CEREBRAS_API_KEY": "csk-j439vyke89px4we44r29wcvetwcfm6mjmp5xwmxx4m2mpmcn"})

    RESOLUTION = 1024
    BATCH_SIZE = 1
    EPOCHS = 10
    LEARNING_RATE = 1e-4
    NETWORK_DIM = 128
    NETWORK_ALPHA = 64

    MAX_IMAGE_COUNT = 1000
    MIN_IMAGE_COUNT = 10
    MIN_IMAGE_SIZE = 512
    THUMBNAIL_SIZE = 300

    BLOCKED_IP_RANGES = [
        ipaddress.ip_network('10.0.0.0/8'),
        ipaddress.ip_network('172.16.0.0/12'),
        ipaddress.ip_network('192.168.0.0/16'),
        ipaddress.ip_network('127.0.0.0/8'),
        ipaddress.ip_network('169.254.0.0/16'),
        ipaddress.ip_network('::1/128'),
        ipaddress.ip_network('fc00::/7'),
    ]

class SecurityValidator:
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}
    ALLOWED_MIME_TYPES = {'image/jpeg', 'image/png', 'image/webp'}
    MAX_FILE_SIZE = 20 * 1024 * 1024

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        filename = os.path.basename(filename)
        filename = re.sub(r'[^\w\s.-]', '', filename)
        filename = re.sub(r'[-\s]+', '_', filename)
        return filename[:255]

    @staticmethod
    def sanitize_text(text: str) -> str:
        dangerous_chars = ['<', '>', '{', '}', '`', '$', '|', ';', '&']
        for char in dangerous_chars:
            text = text.replace(char, '')
        return text.strip()[:2000]

    @staticmethod
    def is_safe_url(url: str) -> bool:
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            if parsed.scheme not in ['http', 'https']:
                return False
            if not parsed.hostname:
                return False
            ip = socket.gethostbyname(parsed.hostname)
            ip_obj = ipaddress.ip_address(ip)
            for blocked_range in Config.BLOCKED_IP_RANGES:
                if ip_obj in blocked_range:
                    return False
            return True
        except Exception as e:
            logger.error(f"URL validation error: {e}")
            return False

    @staticmethod
    def validate_image_file(file_path: Path) -> bool:
        try:
            import magic
            from PIL import Image
            if file_path.stat().st_size > SecurityValidator.MAX_FILE_SIZE:
                return False
            if file_path.suffix.lower() not in SecurityValidator.ALLOWED_EXTENSIONS:
                return False
            mime = magic.from_file(str(file_path), mime=True)
            if mime not in SecurityValidator.ALLOWED_MIME_TYPES:
                return False
            img = Image.open(file_path)
            img.verify()
            img = Image.open(file_path)
            if min(img.size) < Config.MIN_IMAGE_SIZE:
                return False
            return True
        except Exception as e:
            logger.error(f"Image validation error: {e}")
            return False

class ModelStatus:
    @staticmethod
    def check_llm() -> Dict[str, Any]:
        return {
            "ready": True,
            "files": 0,
            "size_mb": "API",
            "name": "Cerebras gpt-oss-120b",
            "purpose": "Smart query generation & caption enhancement"
        }

    @staticmethod
    def check_base_model() -> Dict[str, Any]:
        try:
            base_model = Config.MODELS_PATH / Config.BASE_MODEL_NAME
            if base_model.exists():
                size = base_model.stat().st_size
                return {"ready": True, "size_mb": round(size / 1024 / 1024, 2), "name": "SDXL Base 1.0", "purpose": "Base model for LoRA training"}
            return {"ready": False, "size_mb": 0, "name": "SDXL Base 1.0", "purpose": "Base model for LoRA training"}
        except Exception as e:
            logger.error(f"Base model check error: {e}")
            return {"ready": False, "size_mb": 0, "name": "SDXL Base 1.0"}

    @staticmethod
    def check_test_model() -> Dict[str, Any]:
        try:
            test_model = Config.MODELS_PATH / Config.TEST_MODEL_NAME
            if test_model.exists():
                size = test_model.stat().st_size
                return {"ready": True, "size_mb": round(size / 1024 / 1024, 2), "name": "Juggernaut XL v9", "purpose": "Test image generation"}
            return {"ready": False, "size_mb": 0, "name": "Juggernaut XL v9", "purpose": "Test image generation"}
        except Exception as e:
            logger.error(f"Test model check error: {e}")
            return {"ready": False, "size_mb": 0}

    @staticmethod
    def get_all_status() -> Dict[str, Any]:
        return {
            "llm": ModelStatus.check_llm(),
            "base_model": ModelStatus.check_base_model(),
            "test_model": ModelStatus.check_test_model()
        }

class LoRALibrary:
    def __init__(self):
        self.library_file = Config.STATE_PATH / "lora_library.json"
        self.loras: List[LoRAInfo] = []
        self._load()

    def _load(self):
        try:
            if self.library_file.exists():
                with open(self.library_file) as f:
                    data = json.load(f)
                    self.loras = [LoRAInfo(**item) for item in data]
        except Exception as e:
            logger.warning(f"Failed to load library: {e}")

    def _save(self):
        try:
            Config.STATE_PATH.mkdir(parents=True, exist_ok=True)
            with open(self.library_file, 'w') as f:
                json.dump([asdict(l) for l in self.loras], f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save library: {e}")

    def add_lora(self, lora: LoRAInfo):
        self.loras.append(lora)
        self._save()

    def get_all(self) -> List[LoRAInfo]:
        return self.loras

    def get_by_id(self, task_id: str) -> Optional[LoRAInfo]:
        return next((l for l in self.loras if l.task_id == task_id), None)

    def update_lora(self, task_id: str, updates: Dict):
        for lora in self.loras:
            if lora.task_id == task_id:
                for key, value in updates.items():
                    if hasattr(lora, key):
                        setattr(lora, key, value)
                self._save()
                return True
        return False

class TaskManager:
    def __init__(self):
        self.tasks_file = Config.STATE_PATH / "tasks.json"
        self.tasks: Dict[str, TrainingTask] = {}
        self._load()

    def _load(self):
        try:
            if self.tasks_file.exists():
                with open(self.tasks_file) as f:
                    data = json.load(f)
                    self.tasks = {k: TrainingTask(**v) for k, v in data.items()}
        except Exception as e:
            logger.warning(f"Failed to load tasks: {e}")

    def _save(self):
        try:
            Config.STATE_PATH.mkdir(parents=True, exist_ok=True)
            with open(self.tasks_file, 'w') as f:
                json.dump({k: asdict(v) for k, v in self.tasks.items()}, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save tasks: {e}")

    def create_task(self, task_id: str, lora_name: str, concepts: List[Dict], parent_lora_id: str = "") -> TrainingTask:
        task = TrainingTask(
            task_id=task_id, status="scraping", progress=0, stage="Initializing",
            message="Starting image collection...", lora_name=lora_name, concepts=concepts,
            created_at=time.time(), updated_at=time.time(), parent_lora_id=parent_lora_id
        )
        self.tasks[task_id] = task
        self._save()
        return task

    def update_task(self, task_id: str, **kwargs):
        if task_id in self.tasks:
            task = self.tasks[task_id]
            for key, value in kwargs.items():
                if hasattr(task, key):
                    setattr(task, key, value)
            task.updated_at = time.time()
            self._save()

    def get_task(self, task_id: str) -> Optional[TrainingTask]:
        return self.tasks.get(task_id)

    def get_all_tasks(self) -> List[TrainingTask]:
        return list(self.tasks.values())

class MultiSourceImageScraper:
    def __init__(self):
        self.session = None

    def _get_session(self):
        if not self.session:
            import requests
            self.session = requests.Session()
            self.session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
        return self.session

    def scrape_duckduckgo(self, query: str) -> List[str]:
        try:
            session = self._get_session()
            search_url = f"https://duckduckgo.com/?q={query}&iax=images&ia=images"
            response = session.get(search_url, timeout=15)
            if response.status_code == 200:
                img_urls = re.findall(r'https?://[^\s<>"]+?\.(?:jpg|jpeg|png|webp)', response.text)
                return list(set(img_urls))[:100]
        except Exception as e:
            logger.debug(f"DuckDuckGo error: {e}")
        return []

    def scrape_pixabay(self, query: str) -> List[str]:
        try:
            from bs4 import BeautifulSoup
            session = self._get_session()
            url = f"https://pixabay.com/images/search/{query}/"
            response = session.get(url, timeout=15)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                urls = []
                for img in soup.find_all('img', {'srcset': True}):
                    srcset = img.get('srcset', '')
                    parts = srcset.split(',')
                    for part in parts:
                        url_match = re.search(r'(https?://[^\s]+)', part)
                        if url_match:
                            urls.append(url_match.group(1).split()[0])
                return list(set(urls))[:100]
        except Exception as e:
            logger.debug(f"Pixabay error: {e}")
        return []

    def scrape_pexels(self, query: str) -> List[str]:
        try:
            from bs4 import BeautifulSoup
            session = self._get_session()
            url = f"https://www.pexels.com/search/{query}/"
            response = session.get(url, timeout=15)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                urls = [img.get('src', '') for img in soup.find_all('img', {'src': True}) if 'images.pexels.com' in img.get('src', '') and img.get('src', '').startswith('http')]
                return list(set(urls))[:100]
        except Exception as e:
            logger.debug(f"Pexels error: {e}")
        return []

    def scrape_unsplash(self, query: str) -> List[str]:
        try:
            session = self._get_session()
            url = f"https://unsplash.com/s/photos/{query}"
            response = session.get(url, timeout=15)
            if response.status_code == 200:
                img_urls = re.findall(r'https://images\.unsplash\.com/[^\s<>"]+', response.text)
                return list(set(img_urls))[:100]
        except Exception as e:
            logger.debug(f"Unsplash error: {e}")
        return []

    def scrape_flickr(self, query: str) -> List[str]:
        try:
            session = self._get_session()
            url = f"https://www.flickr.com/search/?text={query}"
            response = session.get(url, timeout=15)
            if response.status_code == 200:
                img_urls = re.findall(r'https://[^\s<>"]+flickr\.com/[^\s<>"]+\.jpg', response.text)
                return list(set(img_urls))[:100]
        except Exception as e:
            logger.debug(f"Flickr error: {e}")
        return []

    def scrape_wikimedia(self, query: str) -> List[str]:
        try:
            session = self._get_session()
            url = f"https://commons.wikimedia.org/w/api.php?action=query&generator=search&gsrsearch={query}&gsrnamespace=6&gsrlimit=50&prop=imageinfo&iiprop=url&format=json"
            response = session.get(url, timeout=15)
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
            logger.debug(f"Wikimedia error: {e}")
        return []

    def scrape_all_sources(self, queries: List[str]) -> List[str]:
        all_urls = []
        with ThreadPoolExecutor(max_workers=25) as executor:
            futures = []
            for query in queries:
                query_encoded = query.replace(' ', '+')
                futures.extend([
                    executor.submit(self.scrape_duckduckgo, query_encoded),
                    executor.submit(self.scrape_pixabay, query_encoded),
                    executor.submit(self.scrape_pexels, query_encoded),
                    executor.submit(self.scrape_unsplash, query_encoded),
                    executor.submit(self.scrape_flickr, query_encoded),
                    executor.submit(self.scrape_wikimedia, query_encoded)
                ])
            for future in as_completed(futures):
                try:
                    all_urls.extend(future.result())
                except Exception as e:
                    logger.debug(f"Scraper error: {e}")
        unique_urls = list(set(all_urls))
        logger.info(f"Total unique URLs found: {len(unique_urls)}")
        return unique_urls

class SmartAgent:
    def __init__(self):
        self.scraper = MultiSourceImageScraper()
        self.client = Cerebras(api_key=os.environ.get("CEREBRAS_API_KEY"))

    def _generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 800, temperature: float = 0.7) -> str:
        try:
            stream = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model="gpt-oss-120b", stream=True, max_completion_tokens=max_tokens,
                temperature=temperature, top_p=0.9
            )
            return "".join(chunk.choices[0].delta.content or "" for chunk in stream).strip()
        except Exception as e:
            logger.error(f"Cerebras API generation error: {e}")
            raise

    def generate_search_queries(self, concept_name: str, concept_description: str, target_count: int) -> List[str]:
        fallback_queries = [f"{concept_name} professional photo", f"{concept_name} high quality portrait", f"{concept_name} full body professional photography", f"{concept_description}", f"{concept_name} studio photo", f"{concept_name} high resolution image", f"professional {concept_name} photograph", f"{concept_name} detailed photo", f"{concept_name} 4k photo", f"{concept_name} hd image"]
        try:
            system_prompt = "You are an AI assistant specialized in creating diverse, effective search queries for finding high-quality images for machine learning datasets. You only output a numbered list of queries and nothing else."
            user_prompt = f"""Generate 8-10 diverse search queries to find high-quality images.

Subject: {concept_name}
Description: {concept_description}
Target: {target_count} images

Create specific, varied queries that will find professional, high-quality images from different angles and contexts. Include variations like: "professional photo", "portrait", "full body", "high resolution", "studio photography", etc.
Return ONLY the queries, one per line, numbered 1-10.
Queries:"""
            response = self._generate(system_prompt, user_prompt, max_tokens=500)
            queries = [re.sub(r'^[\d\.\)\-\s]+', '', line.strip()) for line in response.split('\n') if line.strip() and 5 < len(line.strip()) < 150]
            if len(queries) >= 3:
                logger.info(f"Generated {len(queries)} queries using AI")
                return queries[:10]
            else:
                logger.warning("AI generated insufficient queries, using fallback")
                return fallback_queries
        except Exception as e:
            logger.error(f"Query generation error: {e}, using fallback")
            return fallback_queries

    def scrape_with_monitoring(self, concept_name: str, concept_description: str, target_count: int, task_id: str) -> List[str]:
        task_mgr = TaskManager()
        task_mgr.update_task(task_id, stage="AI generating search queries", progress=5)
        queries = self.generate_search_queries(concept_name, concept_description, target_count)
        task_mgr.update_task(task_id, stage="Scraping 6 sources simultaneously", progress=10)
        all_urls = self.scraper.scrape_all_sources(queries)
        task_mgr.update_task(task_id, stage=f"Found {len(all_urls)} URLs, validating", progress=30)
        return all_urls[:target_count * 3] if len(all_urls) >= target_count else all_urls

    def parse_user_request(self, user_input: str, available_loras: List[LoRAInfo]) -> Dict:
        user_input = SecurityValidator.sanitize_text(user_input)
        lora_context = ""
        if available_loras:
            lora_context = "\n\nAvailable LoRAs:\n"
            for lora in available_loras[:15]:
                concepts_str = ', '.join([c['name'] for c in lora.concepts])
                lora_context += f"- [{lora.task_id}] {lora.name}: {concepts_str}\n"

        system_prompt = "You are an AI assistant that parses user requests for training LoRA models. You must extract specific information and return it ONLY in a valid JSON format. Do not add any extra text or explanations outside the JSON structure."
        user_prompt = f"""Parse this LoRA training request and extract the information.
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
  "action": "train_new", "parent_lora_id": "",
  "concepts": [{{"name": "Concept Name", "description": "detailed visual description", "image_count": 300}}],
  "lora_name": "descriptive_name_v1", "message": "User-friendly confirmation message"
}}
JSON:"""
        try:
            response = self._generate(system_prompt, user_prompt, max_tokens=1000)
            json_match = re.search(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                if 'concepts' in data:
                    for concept in data['concepts']:
                        concept['name'] = SecurityValidator.sanitize_text(concept.get('name', ''))
                        concept['description'] = SecurityValidator.sanitize_text(concept.get('description', ''))
                        concept['image_count'] = max(Config.MIN_IMAGE_COUNT, min(concept.get('image_count', 300), Config.MAX_IMAGE_COUNT))
                if 'lora_name' in data:
                    data['lora_name'] = SecurityValidator.sanitize_filename(data['lora_name'])
                return data
        except Exception as e:
            logger.error(f"Parsing error: {e}")
        return {'action': 'help', 'concepts': [], 'lora_name': '', 'parent_lora_id': '', 'message': 'Please describe what you want to train. Example: "Train Taylor Swift with 500 images"'}

    def enhance_caption(self, basic_caption: str, concept_description: str) -> str:
        try:
            system_prompt = "You are an expert image analyst. Your task is to improve a basic image caption to be more descriptive and natural for training a LoRA model. You only output the enhanced caption."
            user_prompt = f"""Improve this image caption for LoRA training.
Basic caption: "{basic_caption}"
Subject: "{concept_description}"
Create a natural, detailed caption (max 75 words) that describes the image accurately. Focus on visual details, pose, lighting, and composition.
Enhanced caption:"""
            enhanced = self._generate(system_prompt, user_prompt, max_tokens=150)
            result = SecurityValidator.sanitize_text(enhanced)[:200]
            return result if result and len(result) > 10 else f"{concept_description}, {basic_caption}"
        except Exception as e:
            logger.warning(f"Caption enhancement failed: {e}, using basic caption")
            return f"{concept_description}, {basic_caption}"

class ImageDownloader:
    @staticmethod
    def download_and_validate(url: str, output_path: Path, max_retries: int = 2) -> Optional[ImageCandidate]:
        import requests
        from PIL import Image
        if not SecurityValidator.is_safe_url(url): return None
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=20, stream=True)
                response.raise_for_status()
                if 'image' not in response.headers.get('content-type', ''): return None
                temp_path = output_path.with_suffix('.tmp')
                with open(temp_path, 'wb') as f:
                    for chunk in response.iter_content(8192): f.write(chunk)
                if not SecurityValidator.validate_image_file(temp_path):
                    temp_path.unlink(); return None
                img = Image.open(temp_path)
                width, height = img.size
                file_size = temp_path.stat().st_size
                with open(temp_path, 'rb') as f: file_hash = hashlib.md5(f.read()).hexdigest()
                img = img.convert('RGB')
                img.save(output_path, 'JPEG', quality=95)
                temp_path.unlink()
                thumbnail_path = output_path.parent / "thumbnails" / output_path.name
                thumbnail_path.parent.mkdir(exist_ok=True)
                img.thumbnail((Config.THUMBNAIL_SIZE, Config.THUMBNAIL_SIZE), Image.Resampling.LANCZOS)
                img.save(thumbnail_path, 'JPEG', quality=85)
                return ImageCandidate(url=url, thumbnail_path=str(thumbnail_path), width=width, height=height, file_size=file_size, hash=file_hash)
            except Exception:
                if attempt == max_retries - 1: return None
                time.sleep(0.5)
        return None

    @staticmethod
    def download_batch_parallel(urls: List[str], output_dir: Path, max_workers: int = 15) -> List[ImageCandidate]:
        output_dir.mkdir(parents=True, exist_ok=True)
        candidates, seen_hashes = [], set()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(ImageDownloader.download_and_validate, url, output_dir / f"img_{idx:04d}.jpg"): url for idx, url in enumerate(urls)}
            for future in as_completed(futures):
                try:
                    candidate = future.result()
                    if candidate and candidate.hash not in seen_hashes:
                        candidates.append(candidate)
                        seen_hashes.add(candidate.hash)
                except Exception as e:
                    logger.debug(f"Download error: {e}")
        return candidates

def _download_file_with_progress(url: str, path: Path, description: str, headers: Dict = None):
    import requests
    from tqdm import tqdm
    logger.info(f"Downloading {description}...")
    try:
        final_headers = {'User-Agent': 'Mozilla/5.0'}
        if headers: final_headers.update(headers)
        r = requests.get(url, stream=True, headers=final_headers, timeout=600)
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        with open(path, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True, desc=description) as pbar:
            for chunk in r.iter_content(8192):
                f.write(chunk)
                pbar.update(len(chunk))
        logger.info(f"{description} downloaded")
        return True
    except Exception as e:
        logger.error(f"Failed to download {description}: {e}")
        return False

@app.function(image=image, volumes={str(Config.MODELS_PATH): models_vol}, timeout=2400)
def download_base_models():
    results = {"base": False, "test": False}
    Config.MODELS_PATH.mkdir(parents=True, exist_ok=True)
    base_model_path = Config.MODELS_PATH / Config.BASE_MODEL_NAME
    if not base_model_path.exists():
        if _download_file_with_progress(Config.BASE_MODEL_URL, base_model_path, "SDXL base model"):
            results["base"] = True
            models_vol.commit()
    else:
        results["base"] = True; logger.info("SDXL base model already exists")

    test_model_path = Config.MODELS_PATH / Config.TEST_MODEL_NAME
    if not test_model_path.exists():
        if _download_file_with_progress(Config.TEST_MODEL_URL, test_model_path, "Juggernaut XL test model"):
            results["test"] = True
            models_vol.commit()
    else:
        results["test"] = True; logger.info("Test model already exists")
    return {"success": results["base"] and results["test"], "message": "Model download process finished."}

@app.function(image=image, volumes={str(Config.STATE_PATH): state_vol}, secrets=[Config.CEREBRAS_SECRET], timeout=600)
def ai_agent_process(user_input: str) -> Dict:
    try:
        state_vol.reload()
        lib = LoRALibrary()
        agent = SmartAgent()
        return agent.parse_user_request(user_input, lib.get_all())
    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
        return {'action': 'error', 'concepts': [], 'lora_name': '', 'parent_lora_id': '', 'message': f'Processing error: {str(e)}'}

@app.function(image=image, secrets=[Config.CEREBRAS_SECRET], timeout=300, volumes={str(Config.STATE_PATH): state_vol})
def orchestrate(user_input: str) -> Dict:
    task_id = str(uuid.uuid4())[:8]
    try:
        state_vol.reload()
        decision_result = ai_agent_process.remote(user_input)
        action = decision_result.get('action', 'help')
        if action in ['train_new', 'continue_training']:
            concepts = [ConceptConfig(name=c['name'], description=c.get('description', c['name']), image_count=c.get('image_count', 300)) for c in decision_result.get('concepts', [])]
            if not concepts: return {'error': 'No concepts found', 'message': decision_result.get('message', '')}
            parent_lora_id = decision_result.get('parent_lora_id', '')
            lora_name = decision_result.get('lora_name', f"lora_{task_id}")
            if action == 'continue_training' and parent_lora_id: lora_name = f"{lora_name}_continued"
            task_mgr = TaskManager()
            task_mgr.create_task(task_id, lora_name, [asdict(c) for c in concepts], parent_lora_id)
            scrape_images.spawn(task_id, [asdict(c) for c in concepts], lora_name)
            return {'task_id': task_id, 'action': action, 'lora_name': lora_name, 'parent_lora_id': parent_lora_id, 'concepts': [{'name': c.name, 'description': c.description, 'count': c.image_count} for c in concepts], 'message': decision_result.get('message', 'Training started!')}
        elif action == 'list':
            lib = LoRALibrary()
            return {'action': action, 'loras': [asdict(l) for l in lib.get_all()], 'message': decision_result.get('message', '')}
        else:
            return {'action': action, 'message': decision_result.get('message', 'How can I help?')}
    except Exception as e:
        logger.error(f"Orchestration error: {e}", exc_info=True)
        return {'error': str(e), 'message': f'Error: {str(e)}', 'action': 'error'}

@app.function(image=image, gpu="A10G", volumes={str(Config.RAW_IMAGES_PATH): raw_images_vol, str(Config.THUMBNAILS_PATH): thumbnails_vol, str(Config.STATE_PATH): state_vol}, secrets=[Config.CEREBRAS_SECRET], timeout=3600)
def scrape_images(task_id: str, concepts_dict: List[Dict], lora_name: str):
    concepts = [ConceptConfig(**c) for c in concepts_dict]
    task_mgr = TaskManager()
    task_mgr.update_task(task_id, status="scraping", stage="Initializing AI Agent", progress=0)
    try:
        agent = SmartAgent(); use_ai = True
    except Exception as e:
        logger.warning(f"AI unavailable: {e}, using fallback mode"); agent = None; use_ai = False
        task_mgr.update_task(task_id, stage="Using fallback scraper (AI unavailable)", progress=0)
    all_concepts_data = []
    for idx, concept in enumerate(concepts):
        logger.info(f"Processing concept {idx+1}/{len(concepts)}: {concept.name}")
        task_mgr.update_task(task_id, stage=f"{'AI' if use_ai else 'Fallback'} processing: {concept.name}", progress=int((idx / len(concepts)) * 40))
        if use_ai and agent:
            try: urls = agent.scrape_with_monitoring(concept.name, concept.description, concept.image_count, task_id)
            except Exception as e:
                logger.error(f"AI scraping failed: {e}, using fallback"); scraper = MultiSourceImageScraper()
                urls = scraper.scrape_all_sources([f"{concept.name} professional photo", f"{concept.name} high quality", f"{concept.description}", f"{concept.name} portrait"])
        else:
            scraper = MultiSourceImageScraper()
            urls = scraper.scrape_all_sources([f"{concept.name} professional photo", f"{concept.name} high quality portrait", f"{concept.description}", f"{concept.name} studio photo"])
        task_mgr.update_task(task_id, stage=f"Downloading & validating: {concept.name}", progress=int((idx / len(concepts)) * 40) + 20)
        folder_name = SecurityValidator.sanitize_filename(concept.name.lower().replace(' ', '_'))
        out_dir = Config.RAW_IMAGES_PATH / task_id / folder_name
        out_dir.mkdir(parents=True, exist_ok=True)
        candidates = ImageDownloader.download_batch_parallel(urls, out_dir, max_workers=20)
        concept.candidates, concept.collected = candidates, len(candidates)
        all_concepts_data.append(asdict(concept))
        logger.info(f"Collected {len(candidates)} valid images for {concept.name}")
        if idx % 2 == 0: raw_images_vol.commit(); thumbnails_vol.commit()
    raw_images_vol.commit(); thumbnails_vol.commit()
    candidates_file = Config.STATE_PATH / f"{task_id}_candidates.json"
    with open(candidates_file, 'w') as f: json.dump(all_concepts_data, f, indent=2)
    state_vol.commit()
    total_collected = sum(c.collected for c in concepts)
    task_mgr.update_task(task_id, status="review", stage="Ready for manual review", progress=50, message=f"{'AI' if use_ai else 'Fallback scraper'} collected {total_collected} images. Please review and approve.")
    logger.info(f"Scraping complete for task {task_id}. Awaiting manual review.")

@app.function(image=image, gpu="A10G", volumes={str(Config.RAW_IMAGES_PATH): raw_images_vol, str(Config.DATASET_PATH): dataset_vol, str(Config.STATE_PATH): state_vol, str(Config.LORAS_PATH): loras_vol}, secrets=[Config.HF_TOKEN, Config.CEREBRAS_SECRET], timeout=3600)
def preprocess_approved_images(task_id: str, approved_images: Dict[str, List[str]], lora_name: str):
    from PIL import Image
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    import torch, shutil
    task_mgr = TaskManager(); task = task_mgr.get_task(task_id)
    parent_lora_id = task.parent_lora_id if task else ""
    task_mgr.update_task(task_id, status="preprocessing", stage="Loading models", progress=55)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Blip2Processor.from_pretrained(Config.VISION_MODEL)
    vision_model = Blip2ForConditionalGeneration.from_pretrained(Config.VISION_MODEL, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device)
    try: agent = SmartAgent(); use_agent = True
    except Exception as e: logger.warning(f"AI Agent unavailable for captions: {e}"); use_agent = False
    dataset_dir = Config.DATASET_PATH / task_id
    if parent_lora_id:
        parent_dataset = Config.DATASET_PATH / parent_lora_id
        if parent_dataset.exists():
            for item in parent_dataset.iterdir():
                if item.is_dir(): shutil.copytree(item, dataset_dir / item.name, dirs_exist_ok=True)
            logger.info(f"Copied existing dataset from {parent_lora_id}")
    total_processed = 0
    total_images_to_process = sum(len(v) for v in approved_images.values())
    for concept_name, image_paths in approved_images.items():
        folder_name = SecurityValidator.sanitize_filename(concept_name.lower().replace(' ', '_'))
        concept_dir = dataset_dir / f"{Config.EPOCHS}_{folder_name}"
        concept_dir.mkdir(parents=True, exist_ok=True)
        task_mgr.update_task(task_id, stage=f"Processing: {concept_name}", progress=55 + int((total_processed / total_images_to_process) * 25))
        existing_count = len(list(concept_dir.glob("img_*.jpg"))); processed = existing_count
        for img_path_str in image_paths:
            try:
                img_path = Path(img_path_str)
                if not img_path.exists(): continue
                img = Image.open(img_path).convert('RGB')
                img.thumbnail((Config.RESOLUTION, Config.RESOLUTION), Image.Resampling.LANCZOS)
                canvas = Image.new('RGB', (Config.RESOLUTION, Config.RESOLUTION), (255, 255, 255))
                canvas.paste(img, ((Config.RESOLUTION - img.width) // 2, (Config.RESOLUTION - img.height) // 2))
                try:
                    inputs = processor(images=canvas, return_tensors="pt").to(device)
                    with torch.no_grad(): gen_ids = vision_model.generate(**inputs, max_length=50)
                    basic_caption = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
                    full_caption = agent.enhance_caption(basic_caption, concept_name) if use_agent else f"{concept_name}, {basic_caption}"
                except: full_caption = f"{concept_name}, high quality professional photo"
                out_name = f"img_{processed:04d}"
                canvas.save(concept_dir / f"{out_name}.jpg", "JPEG", quality=95)
                (concept_dir / f"{out_name}.txt").write_text(full_caption)
                processed += 1; total_processed += 1
                if processed % 10 == 0: dataset_vol.commit()
            except Exception as e: logger.error(f"Preprocessing error: {e}"); continue
        logger.info(f"Processed {processed - existing_count} new images for {concept_name} (total: {processed})")
    dataset_vol.commit()
    task_mgr.update_task(task_id, stage="Starting training", progress=80)
    train_lora.spawn(dataset_dir, lora_name, task_id, total_processed, parent_lora_id)

@app.function(image=image, gpu="A100", volumes={str(Config.DATASET_PATH): dataset_vol, str(Config.LORAS_PATH): loras_vol, str(Config.MODELS_PATH): models_vol, str(Config.STATE_PATH): state_vol}, timeout=7200)
def train_lora(dataset_dir: Path, lora_name: str, task_id: str, image_count: int, parent_lora_id: str = ""):
    task_mgr = TaskManager()
    task_mgr.update_task(task_id, status="training", stage="Preparing training", progress=85)
    base_model = Config.MODELS_PATH / Config.BASE_MODEL_NAME
    if not base_model.exists(): task_mgr.update_task(task_id, status="failed", message="Base model not found."); return
    out_dir = Config.LORAS_PATH / task_id
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_lora_name = SecurityValidator.sanitize_filename(lora_name)
    if not safe_lora_name or len(safe_lora_name) < 3: safe_lora_name = f"lora_{task_id}"
    task_mgr.update_task(task_id, stage="Training LoRA", progress=90)
    cmd = ["python", str(Config.KOHYA_PATH / "sdxl_train_network.py"), "--pretrained_model_name_or_path", str(base_model), "--train_data_dir", str(dataset_dir), "--output_dir", str(out_dir), "--output_name", safe_lora_name, "--network_module", "networks.lora", "--network_dim", str(Config.NETWORK_DIM), "--network_alpha", str(Config.NETWORK_ALPHA), "--resolution", f"{Config.RESOLUTION},{Config.RESOLUTION}", "--train_batch_size", str(Config.BATCH_SIZE), "--max_train_epochs", str(Config.EPOCHS), "--learning_rate", str(Config.LEARNING_RATE), "--optimizer_type", "AdamW8bit", "--xformers", "--mixed_precision", "fp16", "--cache_latents", "--cache_latents_to_disk", "--save_model_as", "safetensors", "--max_data_loader_n_workers", "4", "--gradient_checkpointing", "--save_every_n_epochs", "2"]
    if parent_lora_id:
        lib = LoRALibrary(); parent_lora = lib.get_by_id(parent_lora_id)
        if parent_lora and parent_lora.file_path:
            parent_path = Config.LORAS_PATH / parent_lora.file_path
            if parent_path.exists(): cmd.extend(["--network_weights", str(parent_path)]); logger.info(f"Continuing from parent LoRA: {parent_lora_id}")
    try:
        subprocess.run(cmd, cwd=str(Config.KOHYA_PATH), check=True, timeout=7000)
        loras_vol.commit()
        lora_file = out_dir / f"{safe_lora_name}.safetensors"
        if lora_file.exists():
            candidates_file = Config.STATE_PATH / f"{task_id}_candidates.json"; concepts_info = []
            if candidates_file.exists():
                with open(candidates_file) as f: concepts_data = json.load(f); concepts_info = [{"name": c['name'], "description": c['description']} for c in concepts_data]
            if parent_lora_id:
                lib = LoRALibrary(); parent_lora = lib.get_by_id(parent_lora_id)
                if parent_lora:
                    existing_concepts = {c['name']: c for c in parent_lora.concepts}
                    for new_c in concepts_info:
                        if new_c['name'] not in existing_concepts: existing_concepts[new_c['name']] = new_c
                    concepts_info = list(existing_concepts.values())
            lib = LoRALibrary()
            lora_info = LoRAInfo(task_id=task_id, name=safe_lora_name, concepts=concepts_info, file_path=str(lora_file.relative_to(Config.LORAS_PATH)), file_size=lora_file.stat().st_size, created_at=time.time(), description=f"Trained on {image_count} images" + (f" (continued from {parent_lora_id})" if parent_lora_id else ""), training_images_count=image_count, checkpoint_path=str(out_dir.relative_to(Config.LORAS_PATH)))
            lib.add_lora(lora_info); state_vol.commit()
            task_mgr.update_task(task_id, status="completed", stage="Training complete", progress=100, message="LoRA training successful!")
            test_lora_with_juggernaut.spawn(task_id, safe_lora_name)
        else: task_mgr.update_task(task_id, status="failed", message="Training failed - no output file")
    except Exception as e: logger.error(f"Training error: {e}"); task_mgr.update_task(task_id, status="failed", message=f"Training error: {str(e)}")

@app.function(image=image, gpu="A100", volumes={str(Config.LORAS_PATH): loras_vol, str(Config.MODELS_PATH): models_vol, str(Config.TEST_PATH): test_vol, str(Config.STATE_PATH): state_vol}, timeout=1800)
def test_lora_with_juggernaut(task_id: str, lora_name: str):
    import torch; from diffusers import DiffusionPipeline
    test_model = Config.MODELS_PATH / Config.TEST_MODEL_NAME
    if not test_model.exists(): logger.warning("Test model not found, skipping tests"); return
    try:
        pipe = DiffusionPipeline.from_single_file(str(test_model), torch_dtype=torch.float16, use_safetensors=True).to("cuda")
        lora_path = Config.LORAS_PATH / task_id / f"{lora_name}.safetensors"
        pipe.load_lora_weights(str(lora_path.parent), weight_name=lora_path.name)
        lib = LoRALibrary(); lora_info = lib.get_by_id(task_id);
        if not lora_info: return
        test_dir = Config.TEST_PATH / task_id; test_dir.mkdir(parents=True, exist_ok=True)
        test_prompts = []
        for concept in lora_info.concepts[:3]: test_prompts.extend([f"{concept.get('description', concept['name'])}, portrait, professional photography, high quality, detailed", f"{concept.get('description', concept['name'])}, full body shot, studio lighting, 8k, masterpiece"])
        if len(lora_info.concepts) > 1: test_prompts.append(f"{' and '.join([c['name'] for c in lora_info.concepts[:2]])} together, professional photo, high quality")
        generated_paths = []
        for idx, prompt in enumerate(test_prompts[:6]):
            try:
                image = pipe(prompt=prompt, num_inference_steps=30, guidance_scale=7.5, width=1024, height=1024).images[0]
                test_path = test_dir / f"test_{idx:02d}.jpg"
                image.save(test_path, "JPEG", quality=95)
                generated_paths.append(str(test_path.relative_to(Config.TEST_PATH)))
            except Exception as e: logger.error(f"Test generation error: {e}")
        test_vol.commit(); lib.update_lora(task_id, {'test_images': generated_paths}); state_vol.commit()
        logger.info(f"Generated {len(generated_paths)} test images")
    except Exception as e: logger.error(f"Testing error: {e}")

@app.function(image=image, volumes={str(Config.STATE_PATH): state_vol, str(Config.MODELS_PATH): models_vol}, timeout=60)
def check_models_status() -> Dict:
    try:
        models_vol.reload(); state_vol.reload()
        return ModelStatus.get_all_status()
    except Exception as e:
        logger.error(f"Model status check error: {e}", exc_info=True)
        return {"llm": {"ready": False}, "base_model": {"ready": False}, "test_model": {"ready": False}, "error": str(e)}

HTML = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LoRA Factory - Production</title><script src="https://cdn.tailwindcss.com"></script><script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js"></script><style>@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');*{font-family:'Inter',sans-serif}.gradient-bg{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%)}.glass{background:rgba(255,255,255,0.1);backdrop-filter:blur(10px);border:1px solid rgba(255,255,255,0.2)}@keyframes spin{to{transform:rotate(360deg)}}.animate-spin{animation:spin 1s linear infinite}@keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}.animate-pulse{animation:pulse 2s cubic-bezier(.4,0,.6,1) infinite}.hover-scale{transition:transform .2s}.hover-scale:hover{transform:scale(1.05)}</style></head><body class="bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 min-h-screen text-white"><div x-data="loraFactory()" x-init="init()" class="container mx-auto px-4 py-8 max-w-7xl"><header class="gradient-bg rounded-2xl shadow-2xl p-8 mb-8"><h1 class="text-4xl font-bold mb-2">🎨 LoRA Factory</h1><p class="text-lg opacity-90 mb-4">Production-Ready AI-Powered LoRA Training Platform</p><div class="flex flex-wrap gap-2"><span class="glass px-3 py-1 rounded-full text-sm">✅ Cerebras API</span><span class="glass px-3 py-1 rounded-full text-sm">🤖 AI Agent</span><span class="glass px-3 py-1 rounded-full text-sm">🌐 6 Sources</span><span class="glass px-3 py-1 rounded-full text-sm">🖼️ Manual Review</span><span class="glass px-3 py-1 rounded-full text-sm">⚡ GPU Accelerated</span><span class="glass px-3 py-1 rounded-full text-sm">🔄 Continue Training</span></div></header><div x-show="modelAlert.show" x-transition class="mb-6"><div :class="modelAlert.type === 'success' ? 'bg-green-500/20 border-green-500' : modelAlert.type === 'warning' ? 'bg-yellow-500/20 border-yellow-500' : 'bg-red-500/20 border-red-500'" class="border-l-4 p-4 rounded-lg"><p class="font-medium" x-html="modelAlert.message"></p></div></div><div class="bg-slate-800/50 backdrop-blur rounded-2xl shadow-xl p-6 mb-8"><h2 class="text-2xl font-bold mb-4">📦 Model Status</h2><div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4"><template x-for="(model, key) in models" :key="key"><div :class="model.ready ? 'border-green-500' : 'border-red-500'" class="bg-slate-700/50 border-2 rounded-xl p-4 relative"><div class="flex items-center gap-2 mb-2"><div :class="model.ready ? 'bg-green-500' : 'bg-red-500'" class="w-3 h-3 rounded-full"></div><span class="font-semibold" x-text="model.name"></span></div><p class="text-sm text-gray-300" x-text="model.ready ? `${model.size_mb || model.files}` : 'Not downloaded'"></p><p class="text-xs text-gray-400 mt-1" x-text="model.purpose"></p></div></template></div><button x-show="!allModelsReady" @click="downloadModels()" :disabled="downloading" class="w-full bg-yellow-500 hover:bg-yellow-600 disabled:opacity-50 text-black font-semibold py-3 px-6 rounded-lg transition flex items-center justify-center gap-2"><svg x-show="downloading" class="animate-spin h-5 w-5" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg><span x-text="downloading ? 'Downloading...' : '⬇️ Download Missing Models'"></span></button></div><div class="grid grid-cols-1 lg:grid-cols-2 gap-8"><div class="bg-slate-800/50 backdrop-blur rounded-2xl shadow-xl p-6"><h2 class="text-2xl font-bold mb-4">💬 Create LoRA</h2><textarea x-model="prompt" placeholder="Examples:&#10;• Train Taylor Swift with 500 images&#10;• Create LoRA for anime character Naruto&#10;• Continue training [task_id] with new concept&#10;• Add red sneakers to existing LoRA" class="w-full bg-slate-700/50 border border-slate-600 rounded-lg p-4 text-white placeholder-gray-400 focus:ring-2 focus:ring-purple-500 focus:border-transparent outline-none min-h-[150px] resize-none mb-4"></textarea><button @click="startTraining()" :disabled="training || !prompt.trim()" class="w-full bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 disabled:opacity-50 text-white font-semibold py-3 px-6 rounded-lg transition flex items-center justify-center gap-2"><svg x-show="training" class="animate-spin h-5 w-5" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg><span x-text="training ? 'Processing...' : '🚀 Start Training'"></span></button><div x-show="response.show" x-transition class="mt-4"><div :class="response.type === 'success' ? 'bg-green-500/20 border-green-500' : 'bg-red-500/20 border-red-500'" class="border-l-4 p-4 rounded-lg"><p class="font-medium" x-html="response.message"></p></div></div></div><div class="bg-slate-800/50 backdrop-blur rounded-2xl shadow-xl p-6"><div class="flex justify-between items-center mb-4"><h2 class="text-2xl font-bold">📚 LoRA Library</h2><button @click="loadLibrary()" class="bg-slate-700 hover:bg-slate-600 px-4 py-2 rounded-lg transition">🔄 Refresh</button></div><div class="space-y-4 max-h-[600px] overflow-y-auto"><template x-for="lora in loras" :key="lora.task_id"><div class="bg-slate-700/50 rounded-lg p-4 border border-slate-600 hover:border-purple-500 transition"><h3 class="font-bold text-lg mb-2" x-text="lora.name"></h3><p class="text-sm text-gray-300 mb-2" x-text="`${lora.task_id} • ${(lora.file_size/1024/1024).toFixed(2)} MB • ${lora.training_images_count} images`"></p><div class="flex flex-wrap gap-2 mb-3"><template x-for="concept in lora.concepts" :key="concept.name"><span class="bg-slate-600 px-2 py-1 rounded-full text-xs" x-text="concept.name"></span></template></div><div class="flex flex-wrap gap-2"><button @click="downloadLora(lora.file_path, lora.name)" class="bg-green-500 hover:bg-green-600 text-white text-sm px-3 py-1 rounded transition">⬇️ Download</button><button @click="testLora(lora.task_id, lora.name)" class="bg-blue-500 hover:bg-blue-600 text-white text-sm px-3 py-1 rounded transition" x-text="lora.test_images.length > 0 ? '🎨 Retest' : '🎨 Test'"></button><button x-show="lora.can_continue" @click="continueLora(lora.task_id, lora.name)" class="bg-yellow-500 hover:bg-yellow-600 text-black text-sm px-3 py-1 rounded transition">🔄 Continue</button><button x-show="lora.test_images.length > 0" @click="toggleGallery(lora.task_id)" class="bg-purple-500 hover:bg-purple-600 text-white text-sm px-3 py-1 rounded transition" x-text="`🖼️ Tests (${lora.test_images.length})`"></button></div><div x-show="galleryOpen === lora.task_id" x-transition class="grid grid-cols-3 gap-2 mt-4"><template x-for="img in lora.test_images" :key="img"><img :src="`/api/test-image/${img}`" @click="window.open(`/api/test-image/${img}`)" class="rounded cursor-pointer hover-scale w-full h-24 object-cover"></template></div></div></template><div x-show="loras.length === 0" class="text-center py-12 text-gray-400"><p class="text-lg">📦 No LoRAs yet</p><p class="text-sm">Start training to create your first LoRA!</p></div></div></div></div><div x-show="tasks.length > 0" class="bg-slate-800/50 backdrop-blur rounded-2xl shadow-xl p-6 mt-8"><h2 class="text-2xl font-bold mb-4">⚡ Active Tasks</h2><div class="space-y-4"><template x-for="task in tasks" :key="task.task_id"><div class="bg-slate-700/50 rounded-lg p-4 border border-slate-600"><div class="flex justify-between items-center mb-2"><span class="font-semibold" x-text="task.lora_name"></span><span :class="task.status === 'completed' ? 'text-green-400' : task.status === 'failed' ? 'text-red-400' : 'text-yellow-400'" class="text-sm font-medium" x-text="task.status"></span></div><p class="text-sm text-gray-300 mb-2" x-text="task.stage"></p><div class="w-full bg-slate-600 rounded-full h-2 mb-2"><div :style="`width: ${task.progress}%`" class="bg-gradient-to-r from-purple-500 to-pink-500 h-2 rounded-full transition-all duration-300"></div></div><p class="text-xs text-gray-400" x-text="task.message"></p></div></template></div></div><div x-show="reviewModal.show" x-transition class="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4" @click.self="closeReview()"><div class="bg-slate-800 rounded-2xl shadow-2xl max-w-6xl w-full max-h-[90vh] overflow-auto"><div class="sticky top-0 bg-slate-800 border-b border-slate-700 p-6 flex justify-between items-center"><h2 class="text-2xl font-bold">🖼️ Review Images</h2><button @click="closeReview()" class="bg-red-500 hover:bg-red-600 w-10 h-10 rounded-full flex items-center justify-center text-xl">×</button></div><div class="p-6"><div class="grid grid-cols-3 gap-4 mb-6"><div class="text-center"><div class="text-3xl font-bold" x-text="reviewModal.total"></div><div class="text-sm text-gray-400">Total</div></div><div class="text-center"><div class="text-3xl font-bold text-green-400" x-text="reviewModal.approved"></div><div class="text-sm text-gray-400">Approved</div></div><div class="text-center"><div class="text-3xl font-bold text-red-400" x-text="reviewModal.rejected"></div><div class="text-sm text-gray-400">Rejected</div></div></div><div class="flex gap-2 mb-6"><button @click="approveAll()" class="flex-1 bg-green-500 hover:bg-green-600 py-2 rounded-lg font-semibold">✓ Approve All</button><button @click="rejectAll()" class="flex-1 bg-red-500 hover:bg-red-600 py-2 rounded-lg font-semibold">✗ Reject All</button><button @click="submitReview()" :disabled="submitting" class="flex-1 bg-purple-500 hover:bg-purple-600 disabled:opacity-50 py-2 rounded-lg font-semibold"><span x-text="submitting ? 'Starting...' : '🚀 Train Selected'"></span></button></div><div x-html="reviewModal.content"></div></div></div></div></div><script>function loraFactory(){return{models:{},allModelsReady:!1,downloading:!1,modelAlert:{show:!1,type:"",message:""},prompt:"",training:!1,response:{show:!1,type:"",message:""},loras:[],tasks:[],galleryOpen:null,reviewModal:{show:!1,total:0,approved:0,rejected:0,content:"",taskId:""},approvals:{},submitting:!1,pollers:{},async init(){await this.checkModels(),await this.loadLibrary(),this.startPolling()},async checkModels(){try{const t=await fetch("/api/models/status"),e=await t.json();this.models=e,this.allModelsReady=e.base_model?.ready&&e.test_model?.ready,this.allModelsReady?this.modelAlert={show:!0,type:"success",message:"✅ All models ready! AI-powered features enabled."}:(this.modelAlert={show:!0,type:"warning",message:"⚠️ Some models are missing: "+["Base Model","Test Model"].filter((t,s)=>[!e.base_model?.ready,!e.test_model?.ready][s]).join(", ")+"."})}catch(t){this.modelAlert={show:!0,type:"error",message:"❌ Cannot check model status"}}},async downloadModels(){this.downloading=!0,this.modelAlert={show:!0,type:"info",message:"⏳ Downloading models... This may take 10-20 minutes. Please wait."};try{const t=await fetch("/api/models/download",{method:"POST"}),e=await t.json();e.success?(this.modelAlert={show:!0,type:"success",message:"✅ Models downloaded! Rechecking..."},setTimeout(()=>this.checkModels(),3e3)):this.modelAlert={show:!0,type:"error",message:"❌ Download failed: "+e.message}}catch(t){this.modelAlert={show:!0,type:"error",message:"❌ Error: "+t.message}}finally{this.downloading=!1}},async startTraining(){if(!this.prompt.trim())return;this.training=!0,this.response={show:!1,type:"",message:""};try{const t=await fetch("/api/train",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({user_input:this.prompt})}),e=await t.json();e.error?this.response={show:!0,type:"error",message:"❌ "+e.error}:(this.response={show:!0,type:"success",message:"✅ "+e.message+(e.parent_lora_id?"<br><small>🔄 Continuing from: "+e.parent_lora_id+"</small>":"")},e.task_id&&(this.prompt="",this.pollTask(e.task_id)))}catch(t){this.response={show:!0,type:"error",message:"❌ "+t.message}}finally{this.training=!1}},async loadLibrary(){try{const t=await fetch("/api/loras"),e=await t.json();this.loras=e.loras||[]}catch(t){console.error("Load library error:",t)}},toggleGallery(t){this.galleryOpen=this.galleryOpen===t?null:t},async testLora(t,e){if(!confirm(`Generate test images for "${e}"?`))return;try{await fetch("/api/test-lora",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({task_id:t,lora_name:e})}),alert("Test generation started!"),setTimeout(()=>this.loadLibrary(),2e3)}catch(t){alert("Error: "+t.message)}},continueLora(t,e){this.prompt=`Continue training ${t} (${e}) with new concept`,alert("Now describe the new concept you want to add to this LoRA")},async downloadLora(t,e){try{const s=await fetch(`/api/download/${encodeURIComponent(t)}`),o=await s.blob(),a=URL.createObjectURL(o),i=document.createElement("a");i.href=a,i.download=e+".safetensors",i.click(),URL.revokeObjectURL(a)}catch(t){alert("Download failed")}},pollTask(t){this.pollers[t]||(this.pollers[t]=setInterval(async()=>{try{const e=await fetch(`/api/task/${t}`),s=await e.json();s.error?(clearInterval(this.pollers[t]),delete this.pollers[t]):(this.updateTask(s),"review"===s.status?(clearInterval(this.pollers[t]),delete this.pollers[t],await this.showReview(t)):"completed"!==s.status&&"failed"!==s.status||(clearInterval(this.pollers[t]),delete this.pollers[t],await this.loadLibrary()))}catch(t){console.error("Poll error:",t)}},1e4))},updateTask(t){const e=this.tasks.findIndex(e=>e.task_id===t.task_id);e>=0?this.tasks[e]=t:this.tasks.push(t),"completed"!==t.status&&"failed"!==t.status||setTimeout(()=>{this.tasks=this.tasks.filter(e=>e.task_id!==t.task_id)},1e4)},async showReview(t){try{const e=await fetch(`/api/candidates/${t}`),s=await e.json();if(!s.concepts)return;this.approvals={};let o=0,a="";s.concepts.forEach(t=>{const e=t.name;this.approvals[e]={},a+=`<div><h3 class="text-xl font-bold my-4">${t.name} (${t.candidates.length} images)</h3><div class="grid grid-cols-4 md:grid-cols-6 gap-2">`,t.candidates.forEach((t,s)=>{const i=`${e}-${s}`;this.approvals[e][i]=!1,o++,a+=`<div id="item-${i}" onclick="app.toggleImage('${e}','${i}')" class="relative cursor-pointer group rounded-lg overflow-hidden border-2 border-slate-600 hover:border-purple-500 transition aspect-square">\n<img src="/api/thumbnail/${this.reviewModal.taskId||t.task_id}/${encodeURIComponent(t.thumbnail_path)}" class="w-full h-full object-cover">\n<div class="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition flex items-center justify-center gap-2">\n<button onclick="event.stopPropagation();app.approve('${e}','${i}')" class="bg-green-500 hover:bg-green-600 w-8 h-8 rounded-full">✓</button>\n<button onclick="event.stopPropagation();app.reject('${e}','${i}')" class="bg-red-500 hover:bg-red-600 w-8 h-8 rounded-full">✗</button>\n</div></div>`}),a+="</div></div>"}),this.reviewModal={show:!0,total:o,approved:0,rejected:0,content:a,taskId:t},window.app=this}catch(t){alert("Error: "+t.message)}},toggleImage(t,e){const s=this.approvals[t][e];!1===s?this.approve(t,e):!0===s?this.reject(t,e):(this.approvals[t][e]=!1,this.updateDisplay(e,""),this.updateStats())},approve(t,e){this.approvals[t][e]=!0,this.updateDisplay(e,"approved"),this.updateStats()},reject(t,e){this.approvals[t][e]=!1,this.updateDisplay(e,"rejected"),this.updateStats()},updateDisplay(t,e){const s=document.getElementById(`item-${t}`);s&&(s.className="relative cursor-pointer group rounded-lg overflow-hidden border-2 transition aspect-square","approved"===e?s.classList.add("border-green-500","ring-2","ring-green-500"):"rejected"===e?s.classList.add("opacity-30","grayscale","border-red-500"):s.classList.add("border-slate-600","hover:border-purple-500"))},updateStats(){let t=0,e=0;Object.values(this.approvals).forEach(s=>{Object.values(s).forEach(s=>{!0===s&&t++,!1===s&&e++})}),this.reviewModal.approved=t,this.reviewModal.rejected=e},approveAll(){Object.keys(this.approvals).forEach(t=>{Object.keys(this.approvals[t]).forEach(e=>{this.approve(t,e)})})},rejectAll(){Object.keys(this.approvals).forEach(t=>{Object.keys(this.approvals[t]).forEach(e=>{this.reject(t,e)})})},async submitReview(){const t=this.reviewModal.taskId;let e={};let s=0;if(Object.keys(this.approvals).forEach(t=>{e[t]=[],Object.entries(this.approvals[t]).forEach(([o,a])=>{!0===a&&(e[t].push(o),s++)})}),s<10)return void alert("Please approve at least 10 images");this.submitting=!0;try{const s=await fetch("/api/approve-images",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({task_id:t,approved_images:e})}),o=await s.json();o.success?(this.closeReview(),alert("Training started with approved images!"),this.pollTask(t)):alert("Error: "+o.error)}catch(t){alert("Error: "+t.message)}finally{this.submitting=!1}},closeReview(){this.reviewModal.show=!1,this.approvals={},window.app=null},startPolling(){setInterval(()=>this.loadLibrary(),6e4),setInterval(()=>this.checkModels(),3e4),setInterval(async()=>{try{const t=await fetch("/api/tasks/active"),e=await t.json();e.tasks&&e.tasks.forEach(t=>{this.pollers[t.task_id]||"completed"===t.status||"failed"===t.status||this.pollTask(t.task_id),this.updateTask(t)})}catch(t){console.error("Task sync error:",t)}},1e4)}}}<"""+"""/script></body></html>"""

@app.function(image=image, volumes={str(Config.LORAS_PATH): loras_vol, str(Config.STATE_PATH): state_vol, str(Config.TEST_PATH): test_vol, str(Config.RAW_IMAGES_PATH): raw_images_vol, str(Config.THUMBNAILS_PATH): thumbnails_vol, str(Config.MODELS_PATH): models_vol})
@modal.asgi_app()
def web():
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, FileResponse
    from pydantic import BaseModel, Field
    web_app = FastAPI(title="LoRA Factory Production")
    class TrainReq(BaseModel): user_input: str = Field(..., max_length=2000, min_length=3)
    class ApproveReq(BaseModel): task_id: str; approved_images: Dict[str, List[str]]
    class TestReq(BaseModel): task_id: str; lora_name: str

    @web_app.get("/", response_class=HTMLResponse)
    async def root(): return HTML
    @web_app.get("/api/models/status")
    async def models_status(): return check_models_status.remote()
    @web_app.post("/api/models/download")
    async def download_models(): return download_base_models.remote()
    @web_app.post("/api/train")
    async def train(req: TrainReq): return orchestrate.remote(req.user_input)
    @web_app.get("/api/task/{task_id}")
    async def get_task(task_id: str):
        state_vol.reload(); task_mgr = TaskManager(); task = task_mgr.get_task(task_id)
        return asdict(task) if task else {"error": "Task not found"}
    @web_app.get("/api/tasks/active")
    async def get_active_tasks():
        state_vol.reload(); task_mgr = TaskManager()
        active = [t for t in task_mgr.get_all_tasks() if t.status not in ['completed', 'failed']]
        return {"tasks": [asdict(t) for t in active[-10:]]}
    @web_app.get("/api/candidates/{task_id}")
    async def get_candidates(task_id: str):
        state_vol.reload(); candidates_file = Config.STATE_PATH / f"{task_id}_candidates.json"
        if not candidates_file.exists(): raise HTTPException(404, "Candidates not found")
        with open(candidates_file) as f: return {"concepts": json.load(f)}
    @web_app.get("/api/thumbnail/{task_id}/{path:path}")
    async def get_thumbnail(task_id: str, path: str):
        path = SecurityValidator.sanitize_filename(path)
        full_path = (Config.RAW_IMAGES_PATH / task_id).resolve()
        thumbnail_path = (full_path / "thumbnails" / path).resolve()
        if not str(thumbnail_path).startswith(str(full_path)): raise HTTPException(403, "Access denied")
        if not thumbnail_path.exists(): raise HTTPException(404, "Thumbnail not found")
        return FileResponse(str(thumbnail_path), media_type="image/jpeg")
    @web_app.post("/api/approve-images")
    async def approve_images(req: ApproveReq):
        state_vol.reload(); candidates_file = Config.STATE_PATH / f"{req.task_id}_candidates.json"
        if not candidates_file.exists(): return {"success": False, "error": "Candidates not found"}
        with open(candidates_file) as f: concepts_data = json.load(f)
        approved_paths = {}
        for concept in concepts_data:
            concept_name = concept['name']
            if concept_name not in req.approved_images: continue
            folder_name = SecurityValidator.sanitize_filename(concept_name.lower().replace(' ', '_'))
            base_path = Config.RAW_IMAGES_PATH / req.task_id / folder_name
            approved_paths[concept_name] = []
            for img_id in req.approved_images[concept_name]:
                try:
                    idx = int(img_id.split('-')[-1])
                    img_path = base_path / f"img_{idx:04d}.jpg"
                    if img_path.exists(): approved_paths[concept_name].append(str(img_path))
                except: continue
        task_mgr = TaskManager(); task = task_mgr.get_task(req.task_id)
        if not task: return {"success": False, "error": "Task not found"}
        preprocess_approved_images.spawn(req.task_id, approved_paths, task.lora_name)
        task_mgr.update_task(req.task_id, status="preprocessing", stage="Processing approved images", progress=55)
        return {"success": True, "message": "Training started with approved images"}
    @web_app.get("/api/loras")
    async def get_loras(): state_vol.reload(); lib = LoRALibrary(); return {"loras": [asdict(l) for l in lib.get_all()]}
    @web_app.post("/api/test-lora")
    async def test_lora_endpoint(req: TestReq): test_lora_with_juggernaut.spawn(req.task_id, req.lora_name); return {"success": True}
    @web_app.get("/api/download/{path:path}")
    async def download_lora(path: str):
        full = (Config.LORAS_PATH / path).resolve()
        if not str(full).startswith(str(Config.LORAS_PATH.resolve())) or not full.is_file(): raise HTTPException(404, "File not found")
        return FileResponse(str(full), filename=full.name)
    @web_app.get("/api/test-image/{path:path}")
    async def test_image(path: str):
        full = (Config.TEST_PATH / path).resolve()
        if not str(full).startswith(str(Config.TEST_PATH.resolve())) or not full.is_file(): raise HTTPException(404, "File not found")
        return FileResponse(str(full), media_type="image/jpeg")
    return web_app
