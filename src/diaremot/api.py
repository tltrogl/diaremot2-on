"""
FastAPI wrapper for diaremot2-on Cloud Run deployment.
Accepts audio file uploads via HTTP and returns processed results.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import tempfile
import os
import shutil
from pathlib import Path
import logging

# Import diaremot
try:
    from diaremot.pipeline.audio_pipeline_core import AudioAnalysisPipelineV2
except ImportError:
    AudioAnalysisPipelineV2 = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="DiaRemot Speech Intelligence API", version="2.2.0")


@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "service": "diaremot2-on",
        "version": "2.2.0",
        "status": "healthy",
        "endpoints": {
            "process": "POST /process - Upload audio file for processing",
            "health": "GET /health - Service health check",
        }
    }


@app.get("/health")
def health():
    """Detailed health check"""
    if AudioAnalysisPipelineV2 is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": "diaremot not installed"}
        )
    
    model_dir = os.getenv("DIAREMOT_MODEL_DIR", "/srv/models")
    models_exist = Path(model_dir).exists()
    
    return {
        "status": "healthy",
        "models_available": models_exist,
        "model_dir": model_dir,
    }


@app.post("/process")
async def process_audio(
    audio: UploadFile = File(...),
    output_format: str = "json"
):
    """
    Process audio file through diaremot pipeline.
    
    Args:
        audio: Audio file (WAV, MP3, M4A, FLAC)
        output_format: Response format (json, csv, all)
    
    Returns:
        Processed transcript with diarization and affect analysis
    """
    if AudioAnalysisPipelineV2 is None:
        raise HTTPException(status_code=503, detail="DiaRemot not installed")
    
    # Create temp directories
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_path = temp_path / "input_audio"
        output_dir = temp_path / "outputs"
        output_dir.mkdir()
        
        # Save uploaded file
        with open(input_path, "wb") as f:
            shutil.copyfileobj(audio.file, f)
        
        logger.info(f"Processing {audio.filename} ({input_path.stat().st_size} bytes)")
        
        try:
            # Run pipeline
            config = {
                "asr_backend": "faster",
                "compute_type": "int8",  # Fast mode
                "disable_sed": False,
                "disable_affect": False,
            }
            
            pipeline = AudioAnalysisPipelineV2(config)
            result = pipeline.process_audio_file(str(input_path), str(output_dir))
            
            logger.info(f"Processing complete: {result.get('num_segments')} segments")
            
            # Return requested format
            if output_format == "csv":
                csv_path = output_dir / "diarized_transcript_with_emotion.csv"
                if csv_path.exists():
                    return FileResponse(
                        csv_path,
                        media_type="text/csv",
                        filename=f"{audio.filename}_transcript.csv"
                    )
            
            elif output_format == "all":
                # Create zip of all outputs
                import zipfile
                zip_path = temp_path / "outputs.zip"
                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    for file in output_dir.glob("*"):
                        zipf.write(file, file.name)
                
                return FileResponse(
                    zip_path,
                    media_type="application/zip",
                    filename=f"{audio.filename}_outputs.zip"
                )
            
            else:  # json (default)
                # Return summary JSON
                import json
                summary_path = output_dir / "qc_report.json"
                if summary_path.exists():
                    with open(summary_path) as f:
                        summary = json.load(f)
                    return {
                        "status": "success",
                        "filename": audio.filename,
                        "result": result,
                        "summary": summary
                    }
                else:
                    return {
                        "status": "success",
                        "filename": audio.filename,
                        "result": result
                    }
        
        except Exception as e:
            logger.error(f"Processing failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
