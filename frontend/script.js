async function analyze() {

    const fileInput = document.getElementById("imageInput");

    if (!fileInput.files.length) {
        alert("Upload an image first!");
        return;
    }

    const formData = new FormData();
    formData.append("image", fileInput.files[0]);

    document.getElementById("label").innerText = "Processing...";
    document.getElementById("confidence").innerText = "";

    try {
        const response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (data.error) {
            alert(data.error);
            return;
        }

        document.getElementById("label").innerText =
            "Prediction: " + data.prediction;

        document.getElementById("confidence").innerText =
            "Confidence: " + data.confidence + "%";

        document.getElementById("outputImage").src =
            "data:image/jpeg;base64," + data.result_image;

    } catch (error) {
        alert("Server error!");
        console.log(error);
    }
}