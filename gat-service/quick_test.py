import requests
import json

# Test GAT service health check
try:
    print("Testing GAT Service Health...")
    response = requests.get("http://localhost:8001/health", timeout=5)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    if response.status_code == 200:
        print("\n✅ GAT Service is healthy!")
    else:
        print(f"\n❌ GAT Service health check failed with status {response.status_code}")
        
except Exception as e:
    print(f"\n❌ Failed to connect to GAT service: {e}")
    print("Make sure the service is running on http://localhost:8001")