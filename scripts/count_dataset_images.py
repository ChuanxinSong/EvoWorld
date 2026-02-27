import os

def count_images():
    root_dir = "/data2/songcx/dataset/evoworld/unity_curve"
    output_file = "episode_image_counts.txt"
    splits = ["train", "test", "val"]

    with open(output_file, "w") as f:
        for split in splits:
            split_path = os.path.join(root_dir, split)
            if not os.path.exists(split_path):
                print(f"Warning: split path {split_path} does not exist.")
                continue
            
            f.write(f"--- Split: {split} ---\n")
            # Only list directories that start with 'episode_'
            episodes = [d for d in os.listdir(split_path) 
                       if os.path.isdir(os.path.join(split_path, d)) and d.startswith("episode_")]
            episodes.sort()
            
            for episode in episodes:
                panorama_path = os.path.join(split_path, episode, "panorama")
                if os.path.exists(panorama_path):
                    # Count all files in the panorama directory
                    try:
                        files = os.listdir(panorama_path)
                        image_count = len([img for img in files if os.path.isfile(os.path.join(panorama_path, img))])
                        f.write(f"{split}/{episode}: {image_count}\n")
                    except Exception as e:
                        f.write(f"{split}/{episode}: Error reading folder - {str(e)}\n")
                else:
                    f.write(f"{split}/{episode}: panorama folder not found\n")
            f.write("\n")

    print(f"Successfully counted images and saved to {output_file}")

if __name__ == "__main__":
    count_images()
