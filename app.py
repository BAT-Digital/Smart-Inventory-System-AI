from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from forecast_model import forecast_top_products
from fastapi.responses import JSONResponse
import tempfile
import shutil
import json
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

@app.post("/forecast")
async def upload_csv(file: UploadFile = File(...)):
    try:
        logger.info(f"Received file: {file.filename}")
        
        # Save the uploaded CSV temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
            logger.info(f"Saved CSV to: {tmp_path}")
        
        # Reset file position after reading
        await file.seek(0)
        
        # Get forecast results
        logger.info("Calling forecast_top_products...")
        json_result = forecast_top_products(tmp_path)  # This returns a JSON string
        
        # Parse the JSON string back to a dictionary to ensure it's valid
        try:
            result_dict = json.loads(json_result)
            logger.info(f"Successfully parsed result: {len(result_dict.get('top_products', []))} products found")
            
            # Return a proper JSON response
            return result_dict
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON result: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": "Failed to parse forecast results", "details": str(e)}
            )
            
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to process the request", "details": str(e)}
        )