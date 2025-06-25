import boto3

INSTANCE_ID = 'i-0d9483271f7cacc06'
REGION = 'us-east-1'

def stop_instance():
    ec2 = boto3.client('ec2', region_name=REGION)
    print("🛑 Stopping EC2 instance...")
    ec2.stop_instances(InstanceIds=[INSTANCE_ID])
    print("✅ Instance stopped.")

if __name__ == "__main__":
    stop_instance()
