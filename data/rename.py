import os

def rename_images_in_folders(base_path):
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        
        if os.path.isdir(folder_path):
            print(f"Renaming images in folder: {folder_name}")
            count = 1
            
            for file_name in sorted(os.listdir(folder_path)):
                file_path = os.path.join(folder_path, file_name)
                
                if os.path.isfile(file_path) and file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    new_name = f"{folder_name}_{count:02d}.jpg"
                    new_path = os.path.join(folder_path, new_name)
                    
                    os.rename(file_path, new_path)
                    print(f"Renamed {file_name} to {new_name}")
                    count += 1

if __name__ == "__main__":
    base_path = "data/parts_presplit"
    rename_images_in_folders(base_path)