from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import base64

app = Flask(__name__)
CORS(app)

# ===== HOME PAGE =====
@app.route("/")
def home():
    return render_template("index.html")


MODEL_PATH = "mobilenet_v2_meat_freshness_new.h5"

# ===== FIX MODEL LOAD =====
from tensorflow.keras.layers import Dense as OriginalDense

class FixedDense(OriginalDense):
    def __init__(self, *args, **kwargs):
        kwargs.pop("quantization_config", None)
        super().__init__(*args, **kwargs)

print("Loading model...")
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
    custom_objects={"Dense": FixedDense}
)
print("✅ Model loaded")

# ⚠️ Change this if class order differs
classes = ["Fresh", "Half Spoiled", "Spoiled"]

# ===== FIND LAST CONV LAYER =====
last_conv_layer = None
for layer in reversed(model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
        last_conv_layer = layer.name
        break

if last_conv_layer is None:
    raise Exception("❌ No Conv2D layer found")

print("🔥 Using layer:", last_conv_layer)


# ===== PREPROCESS =====
def preprocess(image_bytes):
    npimg = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, (224, 224))

    norm = resized.astype("float32") / 255.0
    input_img = np.expand_dims(norm, axis=0)

    return img_rgb, input_img


# ===== GRADCAM =====
def gradcam(img_array):

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[
            model.get_layer(last_conv_layer).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array)
        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy(), preds[0]


# ===== ENCODE IMAGE =====
def encode(img):
    _, buffer = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode()


# ===== API =====
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]
        img_rgb, img_input = preprocess(file.read())

        # Prediction
        preds = model.predict(img_input)
        idx = int(np.argmax(preds[0]))
        confidence = float(np.max(preds[0])) * 100
        label = classes[idx]

        # GradCAM
        heatmap, _ = gradcam(img_input)

        h, w = img_rgb.shape[:2]
        heatmap = cv2.resize(heatmap, (w, h))

        heatmap_color = cv2.applyColorMap(
            np.uint8(255 * heatmap),
            cv2.COLORMAP_JET
        )

        overlay = cv2.addWeighted(img_rgb, 0.6, heatmap_color, 0.4, 0)

        # Localization
        binary = np.uint8(heatmap > 0.6) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        output = img_rgb.copy()

        if label == "Fresh":
            color = (0,255,0)
        elif label == "Half Spoiled":
            color = (255,165,0)
        else:
            color = (255,0,0)

        for c in contours:
            x,y,wc,hc = cv2.boundingRect(c)
            if wc * hc > 500:
                cv2.rectangle(output, (x,y), (x+wc,y+hc), color, 2)

        combined = np.hstack([img_rgb, heatmap_color, overlay, output])

        return jsonify({
            "prediction": label,
            "confidence": round(confidence,2),
            "result_image": encode(combined)
        })

    except Exception as e:
        print("❌ ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(port=8000, debug=True)