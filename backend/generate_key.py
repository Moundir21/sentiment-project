
import json
import secrets  # لتوليد كلمات سر قوية
import os

API_KEYS_FILE = "api_keys.json"

def load_keys():
    """تحميل المفاتيح الموجودة"""
    if os.path.exists(API_KEYS_FILE):
        with open(API_KEYS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_keys(keys):
    """حفظ المفاتيح في ملف JSON"""
    with open(API_KEYS_FILE, "w") as f:
        json.dump(keys, f, indent=4)

def generate_key(user_name):
    """توليد API Key جديدة"""
    keys = load_keys()
    # توليد مفتاح قوي 32 بايت ثم تحويله لـ hex
    api_key = secrets.token_hex(32)
    keys[user_name] = api_key
    save_keys(keys)
    return api_key

if __name__ == "__main__":
    user_name = input("Enter user name: ")
    key = generate_key(user_name)
    print(f"Generated API Key for {user_name}: {key}")