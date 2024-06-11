function printImage() {
  var imagePath = document.getElementById("img_path").value;
  var imageContainer = document.getElementById("input_img_container");

  // Clear previous image
  imageContainer.innerHTML = "";

  // Create new image element
  var img = document.createElement("img");
  img.src = imagePath;
  console.log("HAAALLLOO2", imagePath);
  img.alt = "Input Image";

  // Append image to container
  imageContainer.appendChild(img);
}

function handlerror(img) {
  img.style.display = 'none';
}
