import requests

url = "https://modal.com/apps/teamleaderleo/main/deployed/bizarre-pose-api/predict"
with open(r"_samples/megumin.png", "rb") as f:
    files = {"file": ("megumin.png", f, "image/png")}
    resp = requests.post(url, files=files)

print("Status code:", resp.status_code)
print("Raw response:")
print(resp.text)
