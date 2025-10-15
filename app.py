import asyncio
import hashlib
import json
import logging
import os
import re
import subprocess
import time
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import modal
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from time import sleep

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = modal.App("lora-factory-pinterest-crawler-v2")

class EndPageException(Exception):
    def __init__(self, message="End of page"):
        self.message = message
    def __str__(self):
        return self.message

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
        "fastapi[all]", "pydantic", "uvicorn[standard]", "requests",
        "aiohttp", "aiofiles", "torch", "torchvision", "transformers",
        "accelerate", "protobuf", "pillow", "opencv-python-headless",
        "diffusers", "safetensors", "compel", "python-magic",
        "tenacity", "tqdm", "huggingface-hub", "beautifulsoup4",
        "selenium", "pyyaml"
    )
    .apt_install(
        "git", "wget", "curl", "libgl1-mesa-glx", "libglib2.0-0",
        "libmagic1", "file", "chromium", "chromium-driver"
    )
    .run_commands(
        "git clone --depth 1 https://github.com/bmaltais/kohya_ss.git /root/kohya_ss"
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
    RESOLUTION = 1024
    BATCH_SIZE = 1
    EPOCHS = 10
    LEARNING_RATE = 1e-4
    NETWORK_DIM = 128
    NETWORK_ALPHA = 64
    MIN_IMAGE_COUNT = 10
    MIN_IMAGE_SIZE = 512
    THUMBNAIL_SIZE = 300

class SecurityValidator:
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        filename = os.path.basename(filename)
        return re.sub(r'[^\w\s.-]', '', filename)[:255]

class ModelStatus:
    @staticmethod
    def check_base_model():
        path = Config.MODELS_PATH / Config.BASE_MODEL_NAME
        return {"ready": path.exists(), "name": "SDXL Base 1.0"}
    @staticmethod
    def check_test_model():
        path = Config.MODELS_PATH / Config.TEST_MODEL_NAME
        return {"ready": path.exists(), "name": "Juggernaut XL v9"}
    @staticmethod
    def get_all_status():
        return { "base_model": ModelStatus.check_base_model(), "test_model": ModelStatus.check_test_model() }

class LoRALibrary:
    def __init__(self):
        self.library_file = Config.STATE_PATH / "lora_library.json"
        self._load()
    def _load(self):
        if self.library_file.exists():
            try:
                with open(self.library_file) as f:
                    self.loras = [LoRAInfo(**item) for item in json.load(f)]
            except Exception as e:
                logger.warning(f"Failed to load library: {e}")
    def _save(self):
        Config.STATE_PATH.mkdir(parents=True, exist_ok=True)
        with open(self.library_file, 'w') as f:
            json.dump([asdict(l) for l in self.loras], f, indent=2)
    def add_lora(self, lora: LoRAInfo):
        self.loras.append(lora)
        self._save()
    def get_all(self) -> List[LoRAInfo]:
        return self.loras
    def get_by_id(self, task_id: str) -> Optional[LoRAInfo]:
        return next((l for l in self.loras if l.task_id == task_id), None)
    def update_lora(self, task_id: str, updates: Dict):
        lora = self.get_by_id(task_id)
        if lora:
            for key, value in updates.items():
                setattr(lora, key, value)
            self._save()

class TaskManager:
    def __init__(self):
        self.tasks_file = Config.STATE_PATH / "tasks.json"
        self._load()
    def _load(self):
        if self.tasks_file.exists():
            try:
                with open(self.tasks_file) as f:
                    self.tasks = {k: TrainingTask(**v) for k, v in json.load(f).items()}
            except Exception as e:
                logger.warning(f"Failed to load tasks: {e}")
    def _save(self):
        Config.STATE_PATH.mkdir(parents=True, exist_ok=True)
        with open(self.tasks_file, 'w') as f:
            json.dump({k: asdict(v) for k, v in self.tasks.items()}, f, indent=2)
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
                setattr(task, key, value)
            task.updated_at = time.time()
            self._save()
    def get_task(self, task_id: str) -> Optional[TrainingTask]:
        return self.tasks.get(task_id)
    def get_all_tasks(self) -> List[TrainingTask]:
        return list(self.tasks.values())

async def _download_image(src, dir, i):
    filename = src.split('/')[-1]
    savedir = Path(dir) / filename
    src = src.replace("/236x/", "/originals/").replace("/474x/", "/originals/").replace("/736x/", "/originals/")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(src, timeout=30) as response:
                if response.status != 200:
                    logger.warning(f"Download {src} fail! Status: {response.status}")
                    return False
                content = await response.read()
                with open(savedir, 'wb') as file:
                    file.write(content)
                return True
    except Exception as e:
        logger.error(f"{src} : Download fail! Error: {e}")
        return False

async def _download_image_host(plist, dir):
    tasks = [_download_image(plist[i], dir, i) for i in range(len(plist))]
    results = await asyncio.gather(*tasks)
    successful_downloads = sum(1 for r in results if r)
    logger.info(f"Downloaded {successful_downloads}/{len(plist)} images successfully.")

class PinterestCrawler:
    def __init__(self, email, password):
        self.email = email
        self.password = password
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")
        self.driver = webdriver.Chrome(options=chrome_options)
        self.login()

    def login(self):
        self.driver.get("https://www.pinterest.com/login/")
        sleep(3)
        self.driver.find_element_by_name("id").send_keys(self.email)
        self.driver.find_element_by_name("password").send_keys(self.password)
        self.driver.find_element_by_xpath('//button[@type="submit"]').click()
        sleep(5)
        logger.info("Successfully logged into Pinterest.")

    def scrape_image_urls(self, link, pages_to_scroll):
        self.driver.get(link)
        logger.info(f"Navigated to {link}. Starting scroll...")
        sleep(5)
        
        image_urls = set()
        for _ in range(pages_to_scroll):
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            sleep(3)
        
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        for img in soup.find_all('img'):
            if img.get('src') and 'i.pinimg.com' in img.get('src'):
                image_urls.add(img['src'])
        
        logger.info(f"Found {len(image_urls)} unique image URLs.")
        return list(image_urls)

    def close(self):
        self.driver.quit()

@app.function(
    image=image,
    gpu="A10G",
    volumes={
        str(Config.RAW_IMAGES_PATH): raw_images_vol,
        str(Config.THUMBNAILS_PATH): thumbnails_vol,
        str(Config.STATE_PATH): state_vol
    },
    timeout=3600
)
def scrape_pinterest_images(task_id: str, lora_name: str, email: str, password: str, link: str, pages: int):
    from PIL import Image
    task_mgr = TaskManager()
    crawler = None
    try:
        task_mgr.update_task(task_id, stage="Initializing Pinterest Crawler...", progress=5)
        crawler = PinterestCrawler(email, password)
        
        task_mgr.update_task(task_id, stage=f"Scrolling {pages} pages...", progress=10)
        image_urls = crawler.scrape_image_urls(link, pages)
        
        if not image_urls:
            raise Exception("No image URLs found. Please check the link or credentials.")

        output_dir = Config.RAW_IMAGES_PATH / task_id / "pinterest_images"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        task_mgr.update_task(task_id, stage=f"Downloading {len(image_urls)} images...", progress=25)
        asyncio.run(_download_image_host(image_urls, str(output_dir)))
        raw_images_vol.commit()
        
        task_mgr.update_task(task_id, stage="Processing & creating thumbnails...", progress=40)
        candidates = []
        downloaded_files = list(output_dir.glob("*"))

        thumbnails_dir = Config.THUMBNAILS_PATH / task_id
        thumbnails_dir.mkdir(parents=True, exist_ok=True)

        for img_path in downloaded_files:
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
                    file_size = img_path.stat().st_size
                    thumb_path = thumbnails_dir / img_path.name
                    img.thumbnail((Config.THUMBNAIL_SIZE, Config.THUMBNAIL_SIZE))
                    img.save(thumb_path, "JPEG")
                    candidates.append(ImageCandidate(
                        url=str(img_path),
                        thumbnail_path=str(thumb_path.relative_to(Config.THUMBNAILS_PATH)),
                        width=width,
                        height=height,
                        file_size=file_size
                    ))
            except Exception as e:
                logger.warning(f"Could not process image {img_path}: {e}")
        
        thumbnails_vol.commit()

        concept_data = {
            "name": SecurityValidator.sanitize_filename(lora_name),
            "description": f"Images from Pinterest: {link}",
            "candidates": [asdict(c) for c in candidates]
        }
        candidates_file = Config.STATE_PATH / f"{task_id}_candidates.json"
        with open(candidates_file, 'w') as f:
            json.dump([concept_data], f, indent=2)
        state_vol.commit()

        task_mgr.update_task(
            task_id,
            status="review",
            stage="Ready for manual review",
            progress=50,
            message=f"Collected {len(candidates)} images. Please review."
        )

    except Exception as e:
        logger.error(f"Scraping failed for task {task_id}: {e}", exc_info=True)
        task_mgr.update_task(task_id, status="failed", message=str(e))
    finally:
        if crawler:
            crawler.close()
            logger.info("Crawler closed.")

def _download_file_with_progress(url: str, path: Path, description: str):
    import requests
    from tqdm import tqdm
    try:
        r = requests.get(url, stream=True, timeout=600)
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        with open(path, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True, desc=description) as pbar:
            for chunk in r.iter_content(8192):
                f.write(chunk)
                pbar.update(len(chunk))
        return True
    except Exception as e:
        logger.error(f"Failed to download {description}: {e}")
        return False

@app.function(image=image, volumes={str(Config.MODELS_PATH): models_vol}, timeout=3600)
def download_base_models():
    Config.MODELS_PATH.mkdir(parents=True, exist_ok=True)
    base_ok = _download_file_with_progress(Config.BASE_MODEL_URL, Config.MODELS_PATH / Config.BASE_MODEL_NAME, "SDXL Base")
    test_ok = _download_file_with_progress(Config.TEST_MODEL_URL, Config.MODELS_PATH / Config.TEST_MODEL_NAME, "Juggernaut XL")
    if base_ok and test_ok:
        models_vol.commit()
    return {"success": base_ok and test_ok}

@app.function(image=image, gpu="A10G", volumes={...}, secrets=[...], timeout=3600)
def preprocess_approved_images(task_id: str, approved_images: Dict[str, List[str]], lora_name: str):
    pass

@app.function(image=image, gpu="A100", volumes={...}, timeout=7200)
def train_lora(dataset_dir: Path, lora_name: str, task_id: str, image_count: int, parent_lora_id: str = ""):
    pass

@app.function(image=image, gpu="A100", volumes={...}, timeout=1800)
def test_lora_with_juggernaut(task_id: str, lora_name: str):
    pass

@app.function(image=image, volumes={...}, timeout=60)
def check_models_status():
    return ModelStatus.get_all_status()

HTML = """
<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LoRA Factory - Pinterest</title><script src="https://cdn.tailwindcss.com"></script><script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js"></script><style>@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');*{font-family:'Inter',sans-serif}.gradient-bg{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%)}</style></head><body class="bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 min-h-screen text-white"><div x-data="loraFactory()" x-init="init()" class="container mx-auto px-4 py-8 max-w-7xl"><header class="gradient-bg rounded-2xl shadow-2xl p-8 mb-8"><h1 class="text-4xl font-bold mb-2">🎨 LoRA Factory</h1><p class="text-lg opacity-90">Image Training Platform with Pinterest Crawler</p></header><div x-show="modelAlert.show" x-transition class="mb-6"><div :class="modelAlert.type === 'success' ? 'bg-green-500/20' : 'bg-yellow-500/20'" class="border-l-4 p-4 rounded-lg" :class="modelAlert.type === 'success' ? 'border-green-500' : 'border-yellow-500'"><p class="font-medium" x-html="modelAlert.message"></p></div></div><div class="bg-slate-800/50 backdrop-blur rounded-2xl shadow-xl p-6 mb-8"><h2 class="text-2xl font-bold mb-4">📦 Model Status</h2><div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4"><template x-for="(model, key) in models" :key="key"><div :class="model.ready ? 'border-green-500' : 'border-red-500'" class="bg-slate-700/50 border-2 rounded-xl p-4"><div class="flex items-center gap-2 mb-2"><div :class="model.ready ? 'bg-green-500' : 'bg-red-500'" class="w-3 h-3 rounded-full"></div><span class="font-semibold" x-text="model.name"></span></div></div></template></div><button x-show="!allModelsReady" @click="downloadModels()" :disabled="downloading" class="w-full bg-yellow-500 hover:bg-yellow-600 disabled:opacity-50 text-black font-semibold py-3 px-6 rounded-lg transition" x-text="downloading ? 'Downloading...' : '⬇️ Download Missing Models'"></button></div><div class="grid grid-cols-1 lg:grid-cols-2 gap-8"><div class="bg-slate-800/50 backdrop-blur rounded-2xl shadow-xl p-6"><h2 class="text-2xl font-bold mb-4">📌 Start with Pinterest</h2><div class="space-y-4"><input type="text" x-model="pinterest.lora_name" placeholder="Enter LoRA Name (e.g., character_style_v1)" class="w-full bg-slate-700/50 p-3 rounded-lg border border-slate-600 focus:ring-purple-500"><input type="email" x-model="pinterest.email" placeholder="Your Pinterest Email" class="w-full bg-slate-700/50 p-3 rounded-lg border border-slate-600 focus:ring-purple-500"><input type="password" x-model="pinterest.password" placeholder="Your Pinterest Password" class="w-full bg-slate-700/50 p-3 rounded-lg border border-slate-600 focus:ring-purple-500"><input type="url" x-model="pinterest.link" placeholder="Pinterest URL (Board, Pin, or Search)" class="w-full bg-slate-700/50 p-3 rounded-lg border border-slate-600 focus:ring-purple-500"><input type="number" x-model.number="pinterest.pages" placeholder="Number of Pages to Scroll (e.g., 5)" class="w-full bg-slate-700/50 p-3 rounded-lg border border-slate-600 focus:ring-purple-500"></div><button @click="startPinterestScrape()" :disabled="scraping" class="w-full mt-4 bg-gradient-to-r from-red-500 to-pink-500 hover:from-red-600 hover:to-pink-600 disabled:opacity-50 text-white font-semibold py-3 px-6 rounded-lg transition" x-text="scraping ? 'Scraping...' : '🚀 Start Scraping'"></button><div x-show="response.show" x-transition class="mt-4"><div :class="response.type === 'success' ? 'bg-green-500/20 border-green-500' : 'bg-red-500/20 border-red-500'" class="border-l-4 p-4 rounded-lg"><p class="font-medium" x-html="response.message"></p></div></div></div><div class="bg-slate-800/50 backdrop-blur rounded-2xl shadow-xl p-6"></div></div></div>
<script>
function loraFactory() {
    return {
        models: {}, allModelsReady: false, downloading: false,
        modelAlert: { show: false, type: '', message: '' },
        scraping: false, response: { show: false, type: '', message: '' },
        pinterest: { lora_name: '', email: '', password: '', link: '', pages: 5 },
        tasks: [], loras: [],
        async init() {},
        async checkModels() {},
        async downloadModels() {},
        async startPinterestScrape() {
            if (!this.pinterest.lora_name || !this.pinterest.email || !this.pinterest.password || !this.pinterest.link || !this.pinterest.pages) {
                this.response = { show: true, type: 'error', message: 'All fields are required.' }; return;
            }
            this.scraping = true; this.response = { show: false };
            try {
                const r = await fetch('/api/train_pinterest', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(this.pinterest)
                });
                const data = await r.json();
                if (data.error) { this.response = { show: true, type: 'error', message: `❌ ${data.error}` }; } 
                else {
                    this.response = { show: true, type: 'success', message: `✅ Task started! ID: ${data.task_id}` };
                    this.pollTask(data.task_id);
                    this.pinterest = { lora_name: '', email: '', password: '', link: '', pages: 5 };
                }
            } catch (e) { this.response = { show: true, type: 'error', message: `❌ ${e.message}` }; } 
            finally { this.scraping = false; }
        },
    }
}
</script></body></html>
"""

@app.function(image=image, volumes={...})
@modal.asgi_app()
def web():
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel

    web_app = FastAPI(title="LoRA Factory - Pinterest Edition")

    class PinterestTrainReq(BaseModel):
        lora_name: str
        email: str
        password: str
        link: str
        pages: int

    @web_app.get("/", response_class=HTMLResponse)
    async def root():
        return HTML

    @web_app.post("/api/train_pinterest")
    async def train_pinterest(req: PinterestTrainReq):
        if not all([req.lora_name, req.email, req.password, req.link, req.pages]):
            raise HTTPException(status_code=400, detail="All fields are required.")

        task_id = str(uuid.uuid4())[:8]
        task_mgr = TaskManager()
        task_mgr.create_task(
            task_id,
            lora_name=req.lora_name,
            concepts=[{"name": req.lora_name, "description": f"Pinterest: {req.link}"}]
        )
        
        scrape_pinterest_images.spawn(
            task_id, req.lora_name, req.email, req.password, req.link, req.pages
        )
        
        return {"success": True, "task_id": task_id}

    return web_app