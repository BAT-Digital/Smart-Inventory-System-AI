from fastapi import FastAPI, File, UploadFile
from forecast_model import forecast_top_products  # Renamed from analyze_file
import tempfile
import shutil

app = FastAPI()

@app.post("/forecast")
async def upload_csv(file: UploadFile = File(...)):
    # Save the uploaded CSV temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    # Get forecast results
    result = forecast_top_products(tmp_path)  # Now using forecast_top_products
    return result