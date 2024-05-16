import cv2
import os

# Path to the directory containing colored images
input_dir = "E:\\8th Sem BTech\\0) Final Yr. Project - 8th Sem\\Satellite Data\\3) FINAL (CORRECTLY LABELLED)\\# Final - Merged\\sat\\"

# Path to the directory where grayscale images will be saved
output_dir = "E:\\8th Sem BTech\\0) Final Yr. Project - 8th Sem\\Satellite Data\\3) FINAL (CORRECTLY LABELLED)\\# Final - Merged\\sat_pan\\"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Get a list of all files in the input directory
files = os.listdir(input_dir)

# Iterate through each file in the input directory
for file in files:
    # Read the colored image
    colored_image = cv2.imread(os.path.join(input_dir, file))
    
    # Convert the colored image to grayscale
    grayscale_image = cv2.cvtColor(colored_image, cv2.COLOR_BGR2GRAY)
    
    # Write the grayscale image to the output directory
    cv2.imwrite(os.path.join(output_dir, file), grayscale_image)

print("Conversion complete.")