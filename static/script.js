// Preview BEFORE analyze
function previewImage(){
    const file = document.getElementById("imageInput").files[0];
    if(!file) return;

    const reader = new FileReader();
    reader.onload = function(e){
        document.getElementById("previewImage").src = e.target.result;
    }
    reader.readAsDataURL(file);
}

// Analyze
async function analyze(){

    const fileInput = document.getElementById("imageInput");

    if (!fileInput.files.length){
        alert("Upload image first!");
        return;
    }

    const formData = new FormData();
    formData.append("image", fileInput.files[0]);

    document.getElementById("label").innerText = "Processing...";
    document.getElementById("confidence").innerText = "";

    const res = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData
    });

    const data = await res.json();

    if (data.error){
        alert(data.error);
        return;
    }

    // ===== COLORED PREDICTION =====
    let color = "white";

    if(data.prediction === "Fresh") color = "lime";
    else if(data.prediction === "Half Spoiled") color = "orange";
    else color = "red";

    document.getElementById("label").innerHTML =
        `<span style="color:${color}">Prediction: ${data.prediction}</span>`;

    document.getElementById("confidence").innerText =
        "Confidence: " + data.confidence + "%";

    // ===== SPLIT COMBINED IMAGE =====
    const img = new Image();
    img.src = "data:image/jpeg;base64," + data.result_image;

    img.onload = function(){
        const w = img.width / 4;
        const h = img.height;

        const canvas = document.createElement("canvas");
        const ctx = canvas.getContext("2d");

        function crop(x){
            canvas.width = w;
            canvas.height = h;
            ctx.drawImage(img, -x*w, 0);
            return canvas.toDataURL();
        }

        document.getElementById("img_original").src = crop(0);
        document.getElementById("img_heatmap").src = crop(1);
        document.getElementById("img_overlay").src = crop(2);
        document.getElementById("img_local").src = crop(3);
    }
}