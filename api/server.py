"""
VisionGuard-AD — FastAPI Server
=================================
Minimal REST API for anomaly detection inference.

Endpoints:
    POST /predict     — Upload an image, get anomaly score + decision
    GET  /health      — Liveness check
    GET  /model-info  — Model metadata

Usage:
    python api/server.py \\
        --model_path ./outputs/benchmark/carpet/patchcore_memory_bank.pt \\
        --backbone resnet18 --category carpet

    # Then in another terminal:
    curl -X POST http://localhost:8000/predict \\
        -F "file=@./data/mvtec/carpet/test/scratch/000.png"
"""

import argparse
import io
import logging
import os
import sys
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

logger = logging.getLogger("visionguard.api.server")

# Global model handle — loaded once at startup
_model_handle = None


def get_model_handle():
    """Return the loaded model handle (raises if not loaded)."""
    if _model_handle is None:
        raise RuntimeError("Model not loaded. Server must be started with model arguments.")
    return _model_handle


def create_app(
    method: str = "patchcore",
    backbone: str = "resnet18",
    category: str = "carpet",
    model_path: str = "",
    device: str = "auto",
    threshold: float = 0.5,
    img_size: int = 224,
):
    """
    Create and configure the FastAPI application.

    Loads the model at app creation time so it's ready for requests.
    """
    try:
        from fastapi import FastAPI, File, UploadFile, HTTPException
        from fastapi.responses import JSONResponse
    except ImportError:
        raise ImportError(
            "FastAPI is required for the API server. "
            "Install with: pip install fastapi uvicorn python-multipart"
        )

    from api.inference_api import load_model, run_inference

    # Load model
    global _model_handle
    _model_handle = load_model(
        method=method,
        backbone=backbone,
        category=category,
        model_path=model_path,
        device=device,
        threshold=threshold,
        img_size=img_size,
    )
    logger.info(f"Model loaded for API: {method}/{backbone} — {category}")

    app = FastAPI(
        title="VisionGuard-AD API",
        description="Industrial anomaly detection inference API",
        version="1.0.0",
    )

    @app.get("/health")
    async def health():
        """Liveness check."""
        return {"status": "ok", "model_loaded": _model_handle is not None}

    @app.get("/model-info")
    async def model_info():
        """Return metadata about the loaded model."""
        handle = get_model_handle()
        info = {
            "method": handle["method"],
            "backbone": handle["backbone"],
            "category": handle["category"],
            "device": str(handle["device"]),
            "img_size": handle["img_size"],
            "threshold": handle["threshold"],
        }
        if handle["method"] == "patchcore" and handle["model"].memory_bank is not None:
            mb = handle["model"].memory_bank
            info["memory_bank_shape"] = list(mb.shape)
            info["memory_bank_mb"] = round(mb.nelement() * mb.element_size() / (1024 * 1024), 2)
        return info

    @app.post("/predict")
    async def predict(
        file: UploadFile = File(...),
        threshold: Optional[float] = None,
    ):
        """
        Run anomaly detection on an uploaded image.

        Returns JSON with:
            - image_score: float anomaly score
            - is_anomalous: bool decision
            - threshold: float threshold used
            - meta: dict with inference metadata
        """
        handle = get_model_handle()

        # Read and decode image
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        nparr = np.frombuffer(contents, np.uint8)
        import cv2
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image file")

        # Run inference
        result = run_inference(
            handle, image,
            return_anomaly_map=False,  # Don't send large arrays over HTTP
            threshold=threshold,
        )

        return JSONResponse(content={
            "image_score": result["image_score"],
            "is_anomalous": result["is_anomalous"],
            "threshold": result["threshold"],
            "filename": file.filename,
            "meta": result["meta"],
        })

    return app


def main():
    parser = argparse.ArgumentParser(
        description="VisionGuard-AD REST API Server\n"
                    "Starts a FastAPI server for anomaly detection inference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model_path", required=True, help="Path to trained model (.pt)")
    parser.add_argument("--method", default="patchcore", choices=["patchcore", "fastflow"],
                        help="Detection method (default: patchcore)")
    parser.add_argument("--backbone", default="resnet18", help="Backbone architecture (default: resnet18)")
    parser.add_argument("--category", default="carpet", help="MVTec category (default: carpet)")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, mps, cuda (default: auto)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold (default: 0.5)")
    parser.add_argument("--img_size", type=int, default=224, help="Input image size (default: 224)")
    parser.add_argument("--host", default="0.0.0.0", help="Server host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    args = parser.parse_args()

    app = create_app(
        method=args.method,
        backbone=args.backbone,
        category=args.category,
        model_path=args.model_path,
        device=args.device,
        threshold=args.threshold,
        img_size=args.img_size,
    )

    try:
        import uvicorn
    except ImportError:
        raise ImportError(
            "uvicorn is required to run the API server. "
            "Install with: pip install uvicorn"
        )

    print(f"\n  Starting VisionGuard-AD API server")
    print(f"  Docs: http://{args.host}:{args.port}/docs")
    print(f"  Health: http://{args.host}:{args.port}/health\n")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
