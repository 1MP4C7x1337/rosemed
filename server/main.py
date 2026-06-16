"""RoseMed FastAPI inference server."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Tuple

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from config import get_config
from server.limiter import limiter
from server.rag import KnowledgeRetriever
from server.routes import router

cfg = get_config()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("rosemed")


class ModelEngine:
    """Unified inference engine supporting GPU (transformers) and CPU (GGUF) backends."""

    def __init__(self) -> None:
        self.backend: str = "mock"
        self.model: Any = None
        self.tokenizer: Any = None
        self.llm: Any = None

    def load(self) -> str:
        """Load model from disk, auto-detecting best backend."""
        merged_path = cfg.export.merged_model_path
        gguf_path = cfg.export.gguf_path

        if torch.cuda.is_available() and merged_path.exists():
            return self._load_transformers(merged_path)
        if gguf_path.exists():
            return self._load_gguf(gguf_path)
        if merged_path.exists():
            return self._load_transformers(merged_path)
        logger.warning("No model found — running in mock mode for testing")
        self.backend = "mock"
        return "mock"

    def _load_transformers(self, model_path: Path) -> str:
        """Load merged model via transformers on GPU."""
        logger.info("Loading RoseMed model (GPU/transformers)...")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            self.model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                device_map="auto",
            )
            self.model.eval()
            self.backend = "gpu"
            logger.info("Model loaded on GPU")
            return "gpu"
        except Exception as exc:
            logger.error("GPU load failed: %s", exc)
            gguf_path = cfg.export.gguf_path
            if gguf_path.exists():
                return self._load_gguf(gguf_path)
            raise

    def _load_gguf(self, gguf_path: Path) -> str:
        """Load GGUF model for CPU inference."""
        logger.info("Loading RoseMed model (CPU/GGUF)...")
        try:
            from llama_cpp import Llama

            self.llm = Llama(
                model_path=str(gguf_path),
                n_ctx=4096,
                n_threads=max(4, (os.cpu_count() or 4) // 2),
                verbose=False,
            )
            self.backend = "cpu"
            logger.info("Model loaded on CPU (GGUF)")
            return "cpu"
        except ImportError:
            logger.warning("llama-cpp-python not installed — mock mode")
            self.backend = "mock"
            return "mock"
        except Exception as exc:
            logger.error("GGUF load failed: %s", exc)
            self.backend = "mock"
            return "mock"

    async def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Tuple[str, int]:
        """Generate text from prompt, returning (text, token_count)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._generate_sync,
            prompt,
            max_new_tokens,
            temperature,
            top_p,
        )

    def _generate_sync(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> Tuple[str, int]:
        """Synchronous generation across backends."""
        if self.backend == "mock":
            return self._mock_generate(prompt)

        if self.backend == "gpu" and self.model is not None:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            input_len = inputs["input_ids"].shape[1]
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            generated = outputs[0][input_len:]
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            return text.strip(), len(generated)

        if self.backend == "cpu" and self.llm is not None:
            result = self.llm(
                prompt,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            text = result["choices"][0]["text"]
            tokens = result.get("usage", {}).get("completion_tokens", len(text.split()))
            return text.strip(), tokens

        return self._mock_generate(prompt)

    def _mock_generate(self, prompt: str) -> Tuple[str, int]:
        """Generate a mock response for testing without a loaded model."""
        cfg = get_config()
        marker = cfg.user_turn_start
        if marker in prompt:
            user_part = prompt.rsplit(marker, 1)[-1].split(cfg.turn_end)[0].strip()
        else:
            user_part = prompt[-200:]
        reply = (
            f"RoseMed отговор на: {user_part[:100]}. "
            "Това е тестов отговор. Моля, заредете обучения модел за пълна функционалност. "
            "Информацията е съобразена с българските медицински протоколи и НЗОК."
        )
        return reply, len(reply.split())


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    logger.info("Starting RoseMed-27B-BG inference server...")
    engine = ModelEngine()
    backend = engine.load()
    app.state.model_engine = engine
    app.state.backend = backend
    retriever = KnowledgeRetriever()
    chunk_count = retriever.load()
    app.state.retriever = retriever
    logger.info("Server ready (backend: %s, knowledge chunks: %d)", backend, chunk_count)
    yield
    logger.info("Shutting down RoseMed server...")
    app.state.model_engine = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(
    title="RoseMed-27B-BG API",
    description="Bulgarian Medical AI Assistant",
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unhandled exceptions including OOM."""
    if "out of memory" in str(exc).lower() or isinstance(exc, MemoryError):
        logger.error("OOM error: %s", exc)
        return JSONResponse(
            status_code=503,
            content={"detail": "GPU out of memory. Please try again later."},
        )
    logger.error("Unhandled error: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


def main() -> None:
    """Run the RoseMed API server."""
    uvicorn.run(
        "server.main:app",
        host=cfg.inference.host,
        port=cfg.inference.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
