import os
import shutil

def init_model():
    source = "/app/saved_models/20250221_132302/ResNet-50_acc78.61_20250221_132302.pth"
    dest_dir = "/app/saved_models/20250221_132302/"
    dest = os.path.join(dest_dir, "ResNet-50_acc78.61_20250221_132302.pth")
    
    os.makedirs(dest_dir, exist_ok=True)
    if not os.path.exists(dest):
        shutil.copy2(source, dest)

if __name__ == "__main__":
    init_model() 