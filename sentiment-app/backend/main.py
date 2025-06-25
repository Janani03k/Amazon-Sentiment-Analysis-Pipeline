from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from backend.auth import authenticate_user
from backend.utils import trigger_ecs_pipeline, generate_dashboard
from dotenv import load_dotenv
import os

# ✅ Load .env
env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path)

print("✅ FastAPI loaded with:")
print("   - COGNITO_CLIENT_ID =", os.getenv("COGNITO_CLIENT_ID"))
print("   - ECS_TASK_DEFINITION =", os.getenv("ECS_TASK_DEFINITION_ARN"))

app = FastAPI()

# 🔐 Login Request
class LoginRequest(BaseModel):
    username: str
    password: str

@app.post("/login")
def login_user(req: LoginRequest):
    try:
        tokens = authenticate_user(req.username, req.password)
        return {"message": "Login successful", "tokens": tokens}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 🚀 Trigger ECS Task
class PreprocessRequest(BaseModel):
    filename: str
    user: str

@app.post("/trigger_pipeline")
def trigger_pipeline(req: PreprocessRequest):
    try:
        print(f"📩 Received pipeline trigger for user={req.user}, file={req.filename}")
        task_arn = trigger_ecs_pipeline(req.filename, req.user)
        print(f"📦 ECS Task Triggered: {task_arn}")
        return {"status": "pipeline_triggered", "task_arn": task_arn}
    except Exception as e:
        print(f"❌ Error in /trigger_pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 📊 Generate Dashboard After ECS Completion
@app.post("/generate_dashboard")
def generate_dashboard_api(req: PreprocessRequest):
    try:
        print(f"📊 Attempting dashboard generation for {req.user}/{req.filename}")
        generate_dashboard(req.user, req.filename)  # your actual logic
        return {"status": "dashboard_generated"}

    except FileNotFoundError as e:
        error_msg = f"Required files missing: {str(e)}"
        print(f"❌ Dashboard generation failed (missing file): {error_msg}")
        raise HTTPException(status_code=404, detail=error_msg)

    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(f"❌ Error in /generate_dashboard: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)
