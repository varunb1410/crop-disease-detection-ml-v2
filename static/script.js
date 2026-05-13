const fileInput = document.getElementById("fileInput");

const preview = document.getElementById("preview");

const uploadForm = document.getElementById("uploadForm");

const loading = document.getElementById("loading");


// Image Preview
fileInput.addEventListener("change", function () {

    const file = this.files[0];

    if (file) {

        preview.src = URL.createObjectURL(file);

        preview.style.display = "block";
    }
});


// Loading Animation
uploadForm.addEventListener("submit", function () {

    loading.style.display = "block";
});