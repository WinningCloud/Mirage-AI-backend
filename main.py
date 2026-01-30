from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import shutil
# Import your logic
from predict_logic import predict_deepfake, load_all_models_once

app = FastAPI()

# 1. ALLOW REACT TO CONNECT
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"], # React's default ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. LOAD MODELS ON STARTUP (Fast)
MODELS = load_all_models_once()

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    # Save the uploaded file temporarily
    file_id = str(uuid.uuid4())
    os.makedirs("uploads", exist_ok=True)
    file_path = f"uploads/{file_id}_{file.filename}"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Run the AI prediction
        results = predict_deepfake(file_path, MODELS)
        
        # Format the data for React
        data_list = []
        real_votes = 0
        for name, info in results.items():
            label = info['label']
            conf = info['confidence']
            final_conf = (100 - conf * 100) if label.lower() == 'fake' else (conf * 100)
            real_votes += 1 if label.lower() == 'real' else 0
            data_list.append({
                "model": name, 
                "label": label, 
                "confidence": round(final_conf, 2)
            })

        return {
            "verdict": "Real" if real_votes >= len(data_list)/2 else "Fake",
            "avg_confidence": round(sum(d['confidence'] for d in data_list) / len(data_list), 2),
            "detailed_results": data_list
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))