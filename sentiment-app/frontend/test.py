import boto3

client = boto3.client("cognito-idp", region_name="us-east-1")

response = client.initiate_auth(
    AuthFlow='USER_PASSWORD_AUTH',
    AuthParameters={
        'USERNAME': 'ani',
        'PASSWORD': 'AniPass123!'
    },
    ClientId='23t7o5in28lbnisg81it1vqk57'
)

print("✅ Login success")
print(response["AuthenticationResult"])
