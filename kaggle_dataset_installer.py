import kagglehub

# Download latest version
path = kagglehub.dataset_download("sriramr/apples-bananas-oranges")

print("Path to dataset files:", path)