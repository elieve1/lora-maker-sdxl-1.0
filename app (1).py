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

import modal
import requests
import aiohttp
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from time import sleep

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = modal.App("lora-factory-pinterest-crawler-final")

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
        self.loras = []
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
        self.tasks = {}
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

async def _download_image(src, directory):
    filename = src.split('/')[-1].split('?')[0]
    if not filename or len(filename) < 5:
        filename = f"{uuid.uuid4().hex[:8]}.jpg"
    savedir = Path(directory) / filename
    src = src.replace("/236x/", "/originals/").replace("/474x/", "/originals/").replace("/736x/", "/originals/")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(src, timeout=30) as response:
                if response.status != 200:
                    logger.warning(f"Download {src} failed! Status: {response.status}")
                    return False
                content = await response.read()
                with open(savedir, 'wb') as file:
                    file.write(content)
                return True
    except Exception as e:
        logger.error(f"Download {src} failed! Error: {e}")
        return False

async def _download_image_host(plist, directory):
    tasks = [_download_image(url, directory) for url in plist]
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
        self.driver.find_element(By.NAME, "id").send_keys(self.email)
        self.driver.find_element(By.NAME, "password").send_keys(self.password)
        self.driver.find_element(By.XPATH, '//button[@type="submit"]').click()
        sleep(5)
        logger.info("Successfully logged into Pinterest.")

    def scrape_image_urls(self, link, pages_to_scroll):
        self.driver.get(link)
        logger.info(f"Navigated to {link}. Starting scroll...")
        sleep(5)
        
        image_urls = set()
        for i in range(pages_to_scroll):
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            logger.info(f"Scrolling page {i+1}/{pages_to_scroll}")
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
                    if width < 200 or height < 200:
                        continue
                    file_size = img_path.stat().st_size
                    thumb_filename = f"{img_path.stem}_thumb.jpg"
                    thumb_path = thumbnails_dir / thumb_filename
                    img_copy = img.copy()
                    img_copy.thumbnail((Config.THUMBNAIL_SIZE, Config.THUMBNAIL_SIZE), Image.Resampling.LANCZOS)
                    img_copy.save(thumb_path, "JPEG", quality=85)
                    
                    candidates.append(ImageCandidate(
                        url=str(img_path),
                        thumbnail_path=str(Path(task_id) / thumb_filename),
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
            message=f"Collected {len(candidates)} images. Please review and start training."
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

@app.function(
    image=image,
    gpu="A10G",
    volumes={
        str(Config.RAW_IMAGES_PATH): raw_images_vol,
        str(Config.DATASET_PATH): dataset_vol,
        str(Config.STATE_PATH): state_vol,
        str(Config.LORAS_PATH): loras_vol,
    },
    secrets=[Config.HF_TOKEN],
    timeout=3600
)
def preprocess_approved_images(task_id: str, approved_images: Dict[str, List[str]], lora_name: str, parent_lora_id: str):
    from PIL import Image
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    import torch
    import shutil

    task_mgr = TaskManager()
    task_mgr.update_task(task_id, status="preprocessing", stage="Loading vision model", progress=55, parent_lora_id=parent_lora_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Blip2Processor.from_pretrained(Config.VISION_MODEL)
    vision_model = Blip2ForConditionalGeneration.from_pretrained(Config.VISION_MODEL, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device)
    
    dataset_dir = Config.DATASET_PATH / task_id
    if parent_lora_id:
        parent_dataset = Config.DATASET_PATH / parent_lora_id
        if parent_dataset.exists():
            for item in parent_dataset.iterdir():
                if item.is_dir():
                    shutil.copytree(item, dataset_dir / item.name, dirs_exist_ok=True)
            logger.info(f"Copied existing dataset from {parent_lora_id}")

    total_processed = 0
    total_images_to_process = sum(len(v) for v in approved_images.values())

    for concept_name, image_paths in approved_images.items():
        folder_name = SecurityValidator.sanitize_filename(concept_name.lower().replace(' ', '_'))
        concept_dir = dataset_dir / f"{Config.EPOCHS}_{folder_name}"
        concept_dir.mkdir(parents=True, exist_ok=True)
        task_mgr.update_task(task_id, stage=f"Processing concept: {concept_name}", progress=55 + int((total_processed / total_images_to_process) * 25))
        
        existing_count = len(list(concept_dir.glob("img_*.jpg")))
        processed = existing_count

        for img_path_str in image_paths:
            try:
                img_path = Path(img_path_str)
                if not img_path.exists():
                    logger.warning(f"Image not found: {img_path}")
                    continue
                
                img = Image.open(img_path).convert('RGB')
                img.thumbnail((Config.RESOLUTION, Config.RESOLUTION), Image.Resampling.LANCZOS)
                canvas = Image.new('RGB', (Config.RESOLUTION, Config.RESOLUTION), (255, 255, 255))
                canvas.paste(img, ((Config.RESOLUTION - img.width) // 2, (Config.RESOLUTION - img.height) // 2))
                
                try:
                    inputs = processor(images=canvas, return_tensors="pt").to(device)
                    with torch.no_grad():
                        gen_ids = vision_model.generate(**inputs, max_length=50)
                    basic_caption = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
                    full_caption = f"{concept_name}, {basic_caption}"
                except Exception as e:
                    logger.warning(f"Caption generation failed: {e}")
                    full_caption = f"{concept_name}, high quality professional photo"
                
                out_name = f"img_{processed:04d}"
                canvas.save(concept_dir / f"{out_name}.jpg", "JPEG", quality=95)
                (concept_dir / f"{out_name}.txt").write_text(full_caption)
                processed += 1
                total_processed += 1
                
                if processed % 10 == 0:
                    dataset_vol.commit()
            except Exception as e:
                logger.error(f"Preprocessing error for {img_path_str}: {e}")
                continue
        logger.info(f"Processed {processed - existing_count} new images for {concept_name} (total: {processed})")
    
    dataset_vol.commit()
    task_mgr.update_task(task_id, stage="Starting LoRA training", progress=80)
    train_lora.spawn(dataset_dir, lora_name, task_id, total_processed, parent_lora_id)

@app.function(
    image=image,
    gpu="A100",
    volumes={
        str(Config.DATASET_PATH): dataset_vol,
        str(Config.LORAS_PATH): loras_vol,
        str(Config.MODELS_PATH): models_vol,
        str(Config.STATE_PATH): state_vol,
    },
    timeout=7200
)
def train_lora(dataset_dir: Path, lora_name: str, task_id: str, image_count: int, parent_lora_id: str = ""):
    task_mgr = TaskManager()
    task_mgr.update_task(task_id, status="training", stage="Preparing training environment", progress=85)
    
    base_model = Config.MODELS_PATH / Config.BASE_MODEL_NAME
    if not base_model.exists():
        task_mgr.update_task(task_id, status="failed", message="Base model not found.")
        return
    
    out_dir = Config.LORAS_PATH / task_id
    out_dir.mkdir(parents=True, exist_ok=True)
    
    safe_lora_name = SecurityValidator.sanitize_filename(lora_name)
    if not safe_lora_name or len(safe_lora_name) < 3:
        safe_lora_name = f"lora_{task_id}"
    
    task_mgr.update_task(task_id, stage="Training LoRA network", progress=90)
    
    cmd = [
        "python", str(Config.KOHYA_PATH / "sdxl_train_network.py"),
        "--pretrained_model_name_or_path", str(base_model),
        "--train_data_dir", str(dataset_dir),
        "--output_dir", str(out_dir),
        "--output_name", safe_lora_name,
        "--network_module", "networks.lora",
        "--network_dim", str(Config.NETWORK_DIM),
        "--network_alpha", str(Config.NETWORK_ALPHA),
        "--resolution", f"{Config.RESOLUTION},{Config.RESOLUTION}",
        "--train_batch_size", str(Config.BATCH_SIZE),
        "--max_train_epochs", str(Config.EPOCHS),
        "--learning_rate", str(Config.LEARNING_RATE),
        "--optimizer_type", "AdamW8bit",
        "--xformers",
        "--mixed_precision", "fp16",
        "--cache_latents",
        "--cache_latents_to_disk",
        "--save_model_as", "safetensors",
        "--max_data_loader_n_workers", "4",
        "--gradient_checkpointing",
        "--save_every_n_epochs", "2",
    ]
    
    if parent_lora_id:
        lib = LoRALibrary()
        parent_lora = lib.get_by_id(parent_lora_id)
        if parent_lora and parent_lora.file_path:
            parent_path = Config.LORAS_PATH / parent_lora.file_path
            if parent_path.exists():
                cmd.extend(["--network_weights", str(parent_path)])
                logger.info(f"Continuing from parent LoRA: {parent_lora_id}")
    
    try:
        subprocess.run(cmd, cwd=str(Config.KOHYA_PATH), check=True, timeout=7000)
        loras_vol.commit()
        
        lora_file = out_dir / f"{safe_lora_name}.safetensors"
        if lora_file.exists():
            candidates_file = Config.STATE_PATH / f"{task_id}_candidates.json"
            concepts_info = []
            if candidates_file.exists():
                with open(candidates_file) as f:
                    concepts_data = json.load(f)
                    concepts_info = [{"name": c['name'], "description": c['description']} for c in concepts_data]
            
            lib = LoRALibrary()
            lora_info = LoRAInfo(
                task_id=task_id,
                name=safe_lora_name,
                concepts=concepts_info,
                file_path=str(lora_file.relative_to(Config.LORAS_PATH)),
                file_size=lora_file.stat().st_size,
                created_at=time.time(),
                description=f"Multi-concept LoRA trained on {image_count} images",
                training_images_count=image_count,
                checkpoint_path=str(out_dir.relative_to(Config.LORAS_PATH))
            )
            lib.add_lora(lora_info)
            state_vol.commit()
            
            task_mgr.update_task(task_id, status="completed", stage="Training complete", progress=100, message="LoRA training successful!")
            test_lora_with_juggernaut.spawn(task_id, safe_lora_name)
        else:
            task_mgr.update_task(task_id, status="failed", message="Training failed - no output file")
    except Exception as e:
        logger.error(f"Training error: {e}")
        task_mgr.update_task(task_id, status="failed", message=f"Training error: {str(e)}")

@app.function(
    image=image,
    gpu="A100",
    volumes={
        str(Config.LORAS_PATH): loras_vol,
        str(Config.MODELS_PATH): models_vol,
        str(Config.TEST_PATH): test_vol,
        str(Config.STATE_PATH): state_vol,
    },
    timeout=1800
)
def test_lora_with_juggernaut(task_id: str, lora_name: str):
    import torch
    from diffusers import DiffusionPipeline
    
    test_model = Config.MODELS_PATH / Config.TEST_MODEL_NAME
    if not test_model.exists():
        logger.warning("Test model not found, skipping tests")
        return
    
    try:
        pipe = DiffusionPipeline.from_single_file(
            str(test_model),
            torch_dtype=torch.float16,
            use_safetensors=True
        ).to("cuda")
        
        lora_path = Config.LORAS_PATH / task_id / f"{lora_name}.safetensors"
        pipe.load_lora_weights(str(lora_path.parent), weight_name=lora_path.name)
        
        lib = LoRALibrary()
        lora_info = lib.get_by_id(task_id)
        if not lora_info:
            return
        
        test_dir = Config.TEST_PATH / task_id
        test_dir.mkdir(parents=True, exist_ok=True)
        
        test_prompts = []
        for concept in lora_info.concepts[:3]:
            desc = concept.get('description', concept['name'])
            test_prompts.extend([
                f"{desc}, portrait, professional photography, high quality, detailed",
                f"{desc}, full body shot, studio lighting, 8k, masterpiece",
            ])
        
        generated_paths = []
        for idx, prompt in enumerate(test_prompts[:6]):
            try:
                image = pipe(
                    prompt=prompt,
                    num_inference_steps=30,
                    guidance_scale=7.5,
                    width=1024,
                    height=1024
                ).images[0]
                
                test_path = test_dir / f"test_{idx:02d}.jpg"
                image.save(test_path, "JPEG", quality=95)
                generated_paths.append(str(test_path.relative_to(Config.TEST_PATH)))
            except Exception as e:
                logger.error(f"Test generation error: {e}")
        
        test_vol.commit()
        lib.update_lora(task_id, {'test_images': generated_paths})
        state_vol.commit()
        
        logger.info(f"Generated {len(generated_paths)} test images")
    except Exception as e:
        logger.error(f"Testing error: {e}")

@app.function(
    image=image,
    volumes={
        str(Config.STATE_PATH): state_vol,
        str(Config.MODELS_PATH): models_vol,
    },
    timeout=60
)
def check_models_status():
    return ModelStatus.get_all_status()

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>LoRA Factory - Multi-Concept Training</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        * { font-family: 'Inter', sans-serif; }
        .gradient-bg { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
        .image-grid-item { position: relative; aspect-ratio: 1; overflow: hidden; border-radius: 0.5rem; }
        .image-grid-item img { width: 100%; height: 100%; object-fit: cover; }
        .delete-btn { position: absolute; top: 0.25rem; right: 0.25rem; opacity: 0; transition: opacity 0.2s; }
        .image-grid-item:hover .delete-btn { opacity: 1; }
    </style>
</head>
<body class="bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 min-h-screen text-white">
    <div x-data="loraFactory()" x-init="init()" class="container mx-auto px-4 py-8 max-w-7xl">
        <header class="gradient-bg rounded-2xl shadow-2xl p-8 mb-8">
            <h1 class="text-4xl font-bold mb-2">🎨 LoRA Factory</h1>
            <p class="text-lg opacity-90">Multi-Concept Training Platform with Pinterest Crawler</p>
        </header>

        <div x-show="modelAlert.show" x-transition class="mb-6">
            <div :class="modelAlert.type === 'success' ? 'bg-green-500/20 border-green-500' : 'bg-yellow-500/20 border-yellow-500'" class="border-l-4 p-4 rounded-lg">
                <p class="font-medium" x-html="modelAlert.message"></p>
            </div>
        </div>

        <div class="bg-slate-800/50 backdrop-blur rounded-2xl shadow-xl p-6 mb-8">
            <h2 class="text-2xl font-bold mb-4">📦 Model Status</h2>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                <template x-for="(model, key) in models" :key="key">
                    <div :class="model.ready ? 'border-green-500' : 'border-red-500'" class="bg-slate-700/50 border-2 rounded-xl p-4">
                        <div class="flex items-center gap-2 mb-2">
                            <div :class="model.ready ? 'bg-green-500' : 'bg-red-500'" class="w-3 h-3 rounded-full"></div>
                            <span class="font-semibold" x-text="model.name"></span>
                        </div>
                    </div>
                </template>
            </div>
            <button x-show="!allModelsReady" @click="downloadModels()" :disabled="downloading" class="w-full bg-yellow-500 hover:bg-yellow-600 disabled:opacity-50 text-black font-semibold py-3 px-6 rounded-lg transition" x-text="downloading ? 'Downloading...' : '⬇️ Download Missing Models'"></button>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div class="bg-slate-800/50 backdrop-blur rounded-2xl shadow-xl p-6">
                <h2 class="text-2xl font-bold mb-4">📌 Start Pinterest Scraping</h2>
                <div class="space-y-4">
                    <input type="email" x-model="pinterest.email" placeholder="Pinterest Email" class="w-full bg-slate-700/50 p-3 rounded-lg border border-slate-600 focus:ring-purple-500 text-white">
                    <input type="password" x-model="pinterest.password" placeholder="Pinterest Password" class="w-full bg-slate-700/50 p-3 rounded-lg border border-slate-600 focus:ring-purple-500 text-white">
                    <input type="url" x-model="pinterest.link" placeholder="Pinterest URL (Board, Pin, Search)" class="w-full bg-slate-700/50 p-3 rounded-lg border border-slate-600 focus:ring-purple-500 text-white">
                    <input type="number" x-model.number="pinterest.pages" placeholder="Pages to Scroll (e.g., 10)" class="w-full bg-slate-700/50 p-3 rounded-lg border border-slate-600 focus:ring-purple-500 text-white">
                </div>
                <div class="flex gap-2 mt-4 mb-4">
                    <button @click="saveInfo()" class="flex-1 text-sm bg-blue-500 hover:bg-blue-600 rounded-lg py-2 font-medium">💾 Save Info</button>
                    <button @click="clearInfo()" class="flex-1 text-sm bg-gray-600 hover:bg-gray-700 rounded-lg py-2 font-medium">🗑️ Clear Info</button>
                </div>
                <button @click="startPinterestScrape()" :disabled="scraping" class="w-full bg-gradient-to-r from-red-500 to-pink-500 hover:from-red-600 hover:to-pink-600 disabled:opacity-50 text-white font-semibold py-3 px-6 rounded-lg transition" x-text="scraping ? 'Scraping...' : '🚀 Start Scraping'"></button>
                <div x-show="response.show" x-transition class="mt-4">
                    <div :class="response.type === 'success' ? 'bg-green-500/20 border-green-500' : 'bg-red-500/20 border-red-500'" class="border-l-4 p-4 rounded-lg">
                        <p class="font-medium" x-html="response.message"></p>
                    </div>
                </div>
            </div>

            <div class="bg-slate-800/50 backdrop-blur rounded-2xl shadow-xl p-6">
                <div class="flex justify-between items-center mb-4">
                    <h2 class="text-2xl font-bold">📚 LoRA Library</h2>
                    <button @click="loadLoras()" class="text-sm bg-blue-500 hover:bg-blue-600 px-3 py-1 rounded">🔄 Refresh</button>
                </div>
                <div x-show="loras.length === 0" class="text-center py-12 text-gray-400">
                    <p>No trained LoRAs yet.</p>
                </div>
                <div class="space-y-3 max-h-96 overflow-y-auto">
                    <template x-for="lora in loras" :key="lora.task_id">
                        <div class="bg-slate-700/50 p-4 rounded-lg border border-slate-600">
                            <div class="flex justify-between items-start mb-2">
                                <div>
                                    <p class="font-semibold text-lg" x-text="lora.name"></p>
                                    <p class="text-xs text-gray-400" x-text="`${lora.training_images_count} images • ${new Date(lora.created_at * 1000).toLocaleDateString()}`"></p>
                                </div>
                                <span class="text-xs bg-green-500/20 text-green-400 px-2 py-1 rounded">✓ Ready</span>
                            </div>
                            <p class="text-sm text-gray-300 mb-2" x-text="lora.description"></p>
                            <div class="bg-slate-800 p-2 rounded text-xs">
                                <p class="text-gray-400 mb-1">Download command:</p>
                                <code class="text-green-400">modal volume get lora-outputs-v2 <span x-text="lora.file_path"></span> ./<span x-text="lora.name + '.safetensors'"></span></code>
                            </div>
                        </div>
                    </template>
                </div>
            </div>
        </div>
        
        <div x-show="tasks.length > 0" class="mt-8 bg-slate-800/50 backdrop-blur rounded-2xl shadow-xl p-6">
            <h2 class="text-2xl font-bold mb-4">⚡ Active Tasks</h2>
            <div class="space-y-4">
                <template x-for="task in tasks" :key="task.task_id">
                    <div class="bg-slate-700/50 rounded-lg p-4 border border-slate-600">
                        <div class="flex justify-between items-center mb-2">
                            <span class="font-semibold" x-text="task.lora_name"></span>
                            <span :class="{
                                'text-green-400': task.status === 'completed',
                                'text-red-400': task.status === 'failed',
                                'text-blue-400': task.status === 'review',
                                'text-yellow-400': !['completed', 'failed', 'review'].includes(task.status)
                            }" class="text-sm font-medium capitalize" x-text="task.status"></span>
                        </div>
                        <p class="text-sm text-gray-300 mb-2" x-text="task.stage"></p>
                        <div class="w-full bg-slate-600 rounded-full h-2 mb-2">
                            <div :style="`width: ${task.progress}%`" class="bg-gradient-to-r from-purple-500 to-pink-500 h-2 rounded-full transition-all duration-300"></div>
                        </div>
                        <p class="text-xs text-gray-400" x-text="task.message"></p>
                        <button x-show="task.status === 'review'" @click="showReview(task.task_id)" class="mt-2 w-full bg-blue-500 hover:bg-blue-600 text-white py-2 rounded-lg font-medium">👁️ Review & Train</button>
                    </div>
                </template>
            </div>
        </div>

        <div x-show="review.images.length > 0" x-transition class="mt-8 bg-slate-800/50 backdrop-blur rounded-2xl shadow-xl p-6">
            <h2 class="text-2xl font-bold mb-4" x-text="`🖼️ Review Images (${review.images.length} collected)`"></h2>
            
            <div class="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-8 gap-2 mb-6 max-h-96 overflow-y-auto p-2">
                <template x-for="(image, index) in review.images" :key="index">
                    <div class="image-grid-item">
                        <img :src="image.thumbnail_url" :alt="`Image ${index + 1}`" class="rounded-lg">
                        <button @click="deleteImage(index)" class="delete-btn bg-red-600 hover:bg-red-700 text-white rounded-full w-7 h-7 flex items-center justify-center font-bold shadow-lg">×</button>
                    </div>
                </template>
            </div>

            <div class="bg-slate-700/50 p-6 rounded-lg space-y-4">
                <div>
                    <label class="block text-sm font-medium mb-2">Training Mode:</label>
                    <select x-model="review.selectedLora" class="w-full bg-slate-800 p-3 rounded-lg border border-slate-600 text-white">
                        <option value="new">🆕 Create New LoRA</option>
                        <option disabled>────────────────────</option>
                        <template x-for="lora in loras" :key="lora.task_id">
                            <option :value="lora.task_id" x-text="`➕ Continue: ${lora.name}`"></option>
                        </template>
                    </select>
                </div>

                <div x-show="review.selectedLora === 'new'">
                    <label class="block text-sm font-medium mb-2">New LoRA Name:</label>
                    <input type="text" x-model="review.newLoraName" placeholder="e.g., my_custom_style" class="w-full bg-slate-800 p-3 rounded-lg border border-slate-600 text-white">
                </div>

                <div x-show="review.selectedLora !== 'new'" class="bg-blue-500/20 border border-blue-500 rounded p-3 text-sm">
                    <p>💡 <strong>Continuing Training:</strong> New images will be added to the existing LoRA to improve it with additional concepts.</p>
                </div>

                <button @click="startTraining()" :disabled="training" class="w-full bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-600 hover:to-emerald-700 disabled:opacity-50 text-white font-bold py-4 px-6 rounded-lg transition text-lg" x-text="training ? '⏳ Starting Training...' : '🚀 Start Training'"></button>
            </div>
        </div>

    </div>

<script>
function loraFactory() {
    return {
        models: {}, 
        allModelsReady: false, 
        downloading: false,
        modelAlert: { show: false, type: '', message: '' },
        scraping: false, 
        training: false,
        response: { show: false, type: '', message: '' },
        pinterest: { email: '', password: '', link: '', pages: 10 },
        review: { images: [], selectedLora: 'new', newLoraName: '', taskId: '' },
        tasks: [], 
        loras: [], 
        pollers: {},

        init() {
            this.checkModels();
            this.loadInfo();
            this.loadLoras();
            this.startPollingActiveTasks();
        },

        loadInfo() {
            const savedInfo = sessionStorage.getItem('pinterestCrawlerInfo');
            if (savedInfo) {
                try {
                    this.pinterest = { ...this.pinterest, ...JSON.parse(savedInfo) };
                } catch (e) {
                    console.error('Failed to load saved info:', e);
                }
            }
        },

        saveInfo() {
            sessionStorage.setItem('pinterestCrawlerInfo', JSON.stringify(this.pinterest));
            this.showTempResponse('success', '✅ Info saved to this session.');
        },

        clearInfo() {
            sessionStorage.removeItem('pinterestCrawlerInfo');
            this.pinterest = { email: '', password: '', link: '', pages: 10 };
            this.showTempResponse('info', 'ℹ️ Saved info has been cleared.');
        },

        showTempResponse(type, message) {
            this.response = { show: true, type: type, message: message };
            setTimeout(() => this.response.show = false, 4000);
        },

        async checkModels() {
            try {
                const r = await fetch('/api/models/status');
                this.models = await r.json();
                this.allModelsReady = this.models.base_model?.ready && this.models.test_model?.ready;
            } catch (e) { 
                console.error('Failed to check models:', e); 
            }
        },

        async downloadModels() {
            this.downloading = true;
            try { 
                await fetch('/api/models/download', { method: 'POST' }); 
                this.showTempResponse('success', '✅ Models downloaded successfully!');
            } catch (e) { 
                console.error('Download failed:', e); 
                this.showTempResponse('error', '❌ Failed to download models.');
            } finally { 
                this.downloading = false; 
                setTimeout(() => this.checkModels(), 2000);
            }
        },

        async startPinterestScrape() {
            if (!this.pinterest.email || !this.pinterest.password || !this.pinterest.link || !this.pinterest.pages) {
                this.response = { show: true, type: 'error', message: '❌ All fields are required.' }; 
                return;
            }
            
            this.scraping = true; 
            this.response = { show: false };
            
            try {
                const r = await fetch('/api/train_pinterest', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(this.pinterest)
                });
                const data = await r.json();
                
                if (data.error) {
                    this.response = { show: true, type: 'error', message: `❌ ${data.error}` };
                } else {
                    this.response = { show: true, type: 'success', message: `✅ Task ${data.task_id} started! Scraping images...` };
                    this.pollTask(data.task_id);
                }
            } catch (e) {
                this.response = { show: true, type: 'error', message: `❌ ${e.message}` };
            } finally {
                this.scraping = false;
            }
        },

        pollTask(id) {
            if (this.pollers[id]) return;
            
            this.pollers[id] = setInterval(async () => {
                try {
                    const r = await fetch(`/api/task/${id}`);
                    const task = await r.json();
                    
                    this.updateTask(task);

                    if (task.error) { 
                        throw new Error(task.error); 
                    }

                    if (task.status === 'review') {
                        clearInterval(this.pollers[id]);
                        delete this.pollers[id];
                        await this.showReview(id);
                        this.showTempResponse('success', '✅ Images ready for review!');
                    } else if (task.status === 'completed') {
                        clearInterval(this.pollers[id]);
                        delete this.pollers[id];
                        await this.loadLoras();
                        this.tasks = this.tasks.filter(t => t.task_id !== id);
                        this.showTempResponse('success', '🎉 Training completed successfully!');
                    } else if (task.status === 'failed') {
                        clearInterval(this.pollers[id]);
                        delete this.pollers[id];
                        this.tasks = this.tasks.filter(t => t.task_id !== id);
                        this.showTempResponse('error', `❌ Task failed: ${task.message}`);
                    }
                } catch (e) {
                    clearInterval(this.pollers[id]);
                    delete this.pollers[id];
                    console.error('Poll error:', e);
                }
            }, 3000);
        },
        
        updateTask(task) {
            const idx = this.tasks.findIndex(t => t.task_id === task.task_id);
            if (idx >= 0) {
                this.tasks[idx] = task;
            } else {
                this.tasks.push(task);
            }
        },

        startPollingActiveTasks() {
            setInterval(async () => {
                try {
                    const r = await fetch('/api/tasks/active');
                    const data = await r.json();
                    if (data.tasks) {
                        data.tasks.forEach(t => {
                            this.updateTask(t);
                            if (!this.pollers[t.task_id] && t.status !== 'review') { 
                                this.pollTask(t.task_id); 
                            }
                        });
                    }
                } catch (e) { 
                    console.error('Active task sync error:', e); 
                }
            }, 10000);
        },

        async showReview(id) {
            try {
                const r = await fetch(`/api/candidates/${id}`);
                const data = await r.json();
                
                if (!data.concepts || data.concepts.length === 0) {
                    console.error('No candidates found');
                    return;
                }

                const concept = data.concepts[0];
                this.review.images = concept.candidates.map(c => ({
                    ...c,
                    thumbnail_url: `/api/thumbnail/${encodeURIComponent(c.thumbnail_path)}`
                }));
                this.review.taskId = id;
                this.review.newLoraName = concept.name || `lora_${id}`;
                this.review.selectedLora = 'new';
                
                console.log(`Loaded ${this.review.images.length} images for review`);
            } catch (e) {
                console.error("Failed to show review:", e);
                this.showTempResponse('error', '❌ Failed to load images for review.');
            }
        },
        
        deleteImage(index) {
            this.review.images.splice(index, 1);
        },
        
        async loadLoras() {
            try {
                const r = await fetch('/api/loras');
                const data = await r.json();
                this.loras = data.loras || [];
            } catch (e) { 
                console.error('Failed to load LoRAs:', e); 
            }
        },
        
        async startTraining() {
            if (this.review.images.length === 0) {
                alert('⚠️ No images selected for training!');
                return;
            }

            const isNew = this.review.selectedLora === 'new';
            let loraName = isNew ? this.review.newLoraName.trim() : this.loras.find(l => l.task_id === this.review.selectedLora)?.name;
            
            if (isNew && !loraName) {
                alert('⚠️ Please provide a name for the new LoRA.');
                return;
            }
            
            if (!isNew && !loraName) {
                alert('⚠️ Selected LoRA not found.');
                return;
            }
            
            this.training = true;
            
            try {
                const approvedImages = { 
                    [loraName]: this.review.images.map(img => img.url) 
                };
                
                const response = await fetch('/api/start_training', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        task_id: this.review.taskId,
                        lora_name: loraName,
                        parent_lora_id: isNew ? '' : this.review.selectedLora,
                        approved_images: approvedImages
                    })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    this.showTempResponse('success', `✅ Training started for "${loraName}" with ${this.review.images.length} images!`);
                    this.review.images = [];
                    this.review.taskId = '';
                    this.pollTask(this.review.taskId);
                } else {
                    throw new Error(result.error || 'Training failed to start');
                }
            } catch (e) {
                console.error('Training start error:', e);
                this.showTempResponse('error', `❌ Failed to start training: ${e.message}`);
            } finally {
                this.training = false;
            }
        }
    }
}
</script>
</body>
</html>
"""

@app.function(
    image=image,
    volumes={
        str(Config.RAW_IMAGES_PATH): raw_images_vol,
        str(Config.THUMBNAILS_PATH): thumbnails_vol,
        str(Config.LORAS_PATH): loras_vol,
        str(Config.STATE_PATH): state_vol,
        str(Config.TEST_PATH): test_vol,
        str(Config.MODELS_PATH): models_vol
    }
)
@modal.asgi_app()
def web():
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, FileResponse
    from pydantic import BaseModel
    import urllib.parse

    web_app = FastAPI(title="LoRA Factory - Multi-Concept Edition")

    class PinterestTrainReq(BaseModel):
        email: str
        password: str
        link: str
        pages: int

    class StartTrainingReq(BaseModel):
        task_id: str
        lora_name: str
        parent_lora_id: str
        approved_images: Dict[str, List[str]]

    @web_app.get("/", response_class=HTMLResponse)
    async def root():
        return HTML

    @web_app.post("/api/train_pinterest")
    async def train_pinterest(req: PinterestTrainReq):
        if not all([req.email, req.password, req.link, req.pages]):
            raise HTTPException(status_code=400, detail="All fields are required.")

        task_id = str(uuid.uuid4())[:8]
        lora_name = f"lora_{task_id}"
        task_mgr = TaskManager()
        task_mgr.create_task(
            task_id,
            lora_name=lora_name,
            concepts=[{"name": lora_name, "description": f"Pinterest: {req.link}"}]
        )
        
        scrape_pinterest_images.spawn(
            task_id, lora_name, req.email, req.password, req.link, req.pages
        )
        
        return {"success": True, "task_id": task_id}

    @web_app.get("/api/models/status")
    async def models_status():
        return check_models_status.remote()

    @web_app.post("/api/models/download")
    async def download_models_endpoint():
        result = download_base_models.remote()
        return result
    
    @web_app.get("/api/task/{task_id}")
    async def get_task_status(task_id: str):
        state_vol.reload()
        task_mgr = TaskManager()
        task = task_mgr.get_task(task_id)
        if not task:
            raise HTTPException(404, "Task not found")
        return asdict(task)
        
    @web_app.get("/api/tasks/active")
    async def get_active_tasks():
        state_vol.reload()
        task_mgr = TaskManager()
        active_tasks = [t for t in task_mgr.get_all_tasks() if t.status not in ['completed', 'failed']]
        return {"tasks": [asdict(t) for t in active_tasks]}

    @web_app.get("/api/loras")
    async def get_loras():
        state_vol.reload()
        lib = LoRALibrary()
        return {"loras": sorted([asdict(l) for l in lib.get_all()], key=lambda x: x['created_at'], reverse=True)}

    @web_app.get("/api/candidates/{task_id}")
    async def get_candidates(task_id: str):
        state_vol.reload()
        thumbnails_vol.reload()
        candidates_file = Config.STATE_PATH / f"{task_id}_candidates.json"
        if not candidates_file.exists():
            raise HTTPException(404, "Candidates not found")
        with open(candidates_file) as f:
            return {"concepts": json.load(f)}

    @web_app.get("/api/thumbnail/{path:path}")
    async def get_thumbnail(path: str):
        thumbnails_vol.reload()
        decoded_path = urllib.parse.unquote(path)
        thumb_path = Config.THUMBNAILS_PATH / decoded_path
        
        if not thumb_path.exists():
            logger.error(f"Thumbnail not found: {thumb_path}")
            raise HTTPException(404, f"Thumbnail not found: {decoded_path}")
        
        return FileResponse(str(thumb_path), media_type="image/jpeg")

    @web_app.post("/api/start_training")
    async def start_training(req: StartTrainingReq):
        task_mgr = TaskManager()
        task = task_mgr.get_task(req.task_id)
        
        if not task:
            raise HTTPException(404, "Task not found")
        
        task_mgr.update_task(req.task_id, lora_name=req.lora_name, status="preprocessing")
        preprocess_approved_images.spawn(req.task_id, req.approved_images, req.lora_name, req.parent_lora_id)
        
        return {"success": True, "message": "Training started"}

    return web_app