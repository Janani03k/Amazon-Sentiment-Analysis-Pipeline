import streamlit as st
import boto3
import requests
import os
import json
import time
from datetime import datetime
from io import BytesIO

API_URL = "http://localhost:8000"
BUCKET_NAME = "mlops-sentiment-app"

# ---------------------------
# 🚀 Upload File to S3
# ---------------------------
def upload_file_to_s3(uploaded_file, username):
    s3 = boto3.client("s3")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    s3_key = f"uploads/raw/{username}/{timestamp}_{uploaded_file.name}"
    try:
        s3.upload_fileobj(BytesIO(uploaded_file.read()), BUCKET_NAME, s3_key)
        st.session_state["uploaded_filename"] = f"{timestamp}_{uploaded_file.name}"
        return s3_key
    except Exception as e:
        st.error(f"❌ Upload failed: {e}")
        return None

# ---------------------------
# 🔄 Reset pipeline status
# ---------------------------
def reset_pipeline_status(user):
    s3 = boto3.client("s3")
    key = f"metadata/{user}/pipeline_status.json"
    try:
        s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=json.dumps({"status": "pending"}), ContentType="application/json")
    except Exception as e:
        st.error(f"❌ Failed to reset status: {e}")

# ---------------------------
# 🛰️ Poll until status or fail
# ---------------------------
def check_status(user, expected_status, timeout=600):
    s3 = boto3.client("s3")
    key = f"metadata/{user}/pipeline_status.json"
    start = time.time()
    interval = 5
    for attempt in range(timeout // interval):
        try:
            obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
            status = json.loads(obj["Body"].read().decode("utf-8")).get("status", "")
            print(f"📡 [{attempt+1}] Status: {status}")
            if status == expected_status:
                return True
            elif status == "failed":
                return False
        except:
            pass
        time.sleep(interval)
    return False

# ---------------------------
# 🚀 Trigger ECS Pipeline
# ---------------------------
def trigger_pipeline(filename, user):
    try:
        payload = {"filename": filename, "user": user}
        response = requests.post(f"{API_URL}/trigger_pipeline", json=payload)
        return response.status_code == 200
    except Exception as e:
        st.error(f"❌ Failed to trigger pipeline: {e}")
        return False

# ---------------------------
# 📊 Trigger Dashboard Generation
# ---------------------------
def generate_dashboard(filename, user):
    try:
        payload = {"filename": filename, "user": user}
        response = requests.post(f"{API_URL}/generate_dashboard", json=payload)
        return response.status_code == 200
    except Exception as e:
        st.error(f"❌ Failed to generate dashboard: {e}")
        return False

# ---------------------------
# 🖥️ Streamlit UI
# ---------------------------
st.title("📤 Upload & Analyze Sentiment")
st.write(f"👤 Logged in as: `{st.session_state.get('user', 'unknown')}`")
st.warning("⚠️ This pipeline may take 5–10 minutes. Please do not refresh or close this page.")

# 📁 Upload
uploaded_file = st.file_uploader("Upload your CSV (max 200MB)", type=["csv"])

# ---------------------------
# 💡 UI Session Control Flags
# ---------------------------
if "dashboard_ready" not in st.session_state:
    st.session_state["dashboard_ready"] = False
if "filename" not in st.session_state:
    st.session_state["filename"] = ""

# 🚀 Start Sentiment Pipeline
if uploaded_file and st.button("🚀 Get Sentiment"):
    user = st.session_state.get("user", "")
    s3_key = upload_file_to_s3(uploaded_file, user)

    if s3_key:
        filename = os.path.basename(s3_key)
        st.session_state["filename"] = filename
        reset_pipeline_status(user)

        st.info("⚙️ Triggering ECS pipeline...")
        if trigger_pipeline(filename, user):
            st.success("✅ Pipeline triggered.")

            # Preprocessing Stage
            with st.spinner("⏳ Preprocessing your data..."):
                if not check_status(user, "preprocessing_complete", timeout=600):
                    st.error("❌ Preprocessing failed or timed out.")
                    st.stop()

            st.success("✅ Preprocessing complete.")

            # Inference Stage
            with st.spinner("🧠 Running model inference..."):
                if not check_status(user, "inference_complete", timeout=600):
                    st.error("❌ Inference failed or timed out.")
                    st.stop()

            st.success("🎉 Inference complete!")

            elapsed = int(time.time() - time.mktime(datetime.now().timetuple()))
            st.info(f"⏱️ Pipeline completed in {elapsed // 60} min {elapsed % 60} sec")

# ---------------------------
# 📊 Create Dashboard
# ---------------------------
if st.session_state.get("filename") and not st.session_state.get("dashboard_ready"):
    if st.button("📊 Create Dashboard"):
        with st.spinner("📊 Creating dashboard..."):
            user = st.session_state.get("user", "")
            success = generate_dashboard(st.session_state["filename"], user)
            if success:
                st.session_state["dashboard_ready"] = True
                st.success("✅ Dashboard created successfully!")
            else:
                st.error("❌ Dashboard generation failed.")

# ---------------------------
# 🎯 Go to Dashboard
# ---------------------------
if st.session_state.get("dashboard_ready"):
    st.markdown("### ✅ Your sentiment dashboard is ready!")
    if st.button("➡️ Go to Dashboard"):
        st.switch_page("pages/dashboard.py")
