import boto3
import os
from fastapi import HTTPException
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

COGNITO_REGION = os.getenv("COGNITO_REGION", "us-east-1")
CLIENT_ID = os.getenv("COGNITO_CLIENT_ID")

if not CLIENT_ID:
    raise RuntimeError("❌ COGNITO_CLIENT_ID not set")

client = boto3.client("cognito-idp", region_name=COGNITO_REGION)

def authenticate_user(username: str, password: str) -> dict:
    try:
        response = client.initiate_auth(
            AuthFlow='USER_PASSWORD_AUTH',
            AuthParameters={"USERNAME": username, "PASSWORD": password},
            ClientId=CLIENT_ID
        )
        return {
            "access_token": response["AuthenticationResult"]["AccessToken"],
            "id_token": response["AuthenticationResult"]["IdToken"],
            "refresh_token": response["AuthenticationResult"]["RefreshToken"]
        }
    except client.exceptions.NotAuthorizedException:
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    except client.exceptions.UserNotFoundException:
        raise HTTPException(status_code=404, detail="User not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
