from pathlib import Path
from fastapi import FastAPI, File, UploadFile
import onnxruntime as ort
import numpy as np
import cv2
from loguru import logger

ROOT_DIR = Path(__file__).parent
MODEL_PATH = ROOT_DIR / "resources" / "resnet50-v1-7.onnx"
if not MODEL_PATH.exists():
    logger.error(f"Model file not found at {MODEL_PATH.resolve()}")
    exit(1)
logger.info(f"Using model path: {MODEL_PATH.resolve()}")
logger.info(f"ONNX Runtime version: {ort.__version__}")

app = FastAPI()


def get_onnx_providers():
    try:
        providers = ort.get_available_providers()
        return {"available_providers": providers}
    except Exception as e:
        return {"error": str(e)}


session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


@app.get("/")
def read_root():
    return {"Hello": "ONNX World"}


@app.get("/gpu-test")
def gpu_test():
    try:
        providers = ort.get_available_providers()

        if "CUDAExecutionProvider" not in providers:
            return {
                "gpu_test": "failed",
                "reason": "CUDAExecutionProvider not available",
                "providers": providers,
            }

        # Creamos una sesión dummy para probar GPU
        # Creamos un modelo ONNX mínimo en memoria
        # Pero como ejemplo simple, hacemos una operación numpy
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        c = a + b

        return {
            "gpu_test": "success",
            "result": c.tolist(),
            "providers": providers,
        }

    except Exception as e:
        return {"gpu_test": "failed", "reason": str(e)}


@app.get("/gpu-info")
def gpu_info():
    try:
        providers = ort.get_available_providers()

        return {
            "cuda_available": "CUDAExecutionProvider" in providers,
            "available_providers": providers,
        }

    except Exception as e:
        return {"error": str(e)}


@app.get("/load-dummy-model")
def load_dummy_model():
    try:
        session = ort.InferenceSession(
            MODEL_PATH,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

        return {
            "model_loaded": True,
            "providers_used": session.get_providers(),
        }

    except Exception as e:
        return {"model_loaded": False, "error": str(e)}


@app.post("/predict-gpu")
async def predict_gpu(file: UploadFile = File(...)):
    try:
        # Leer imagen
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Preprocesado típico clasificación (ejemplo)
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC → CHW
        img = np.expand_dims(img, axis=0)  # batch dimension

        logger.info(f"Input image shape after preprocessing: {img.shape}")

        # Inferencia
        logger.info(f"Running inference with providers: {session.get_providers()}")
        outputs = session.run(
            [output_name],
            {input_name: img},
        )

        prediction = outputs[0]
        logger.info(f"Prediction shape: {prediction.shape}")

        return {
            "success": True,
            "prediction_shape": prediction.shape,
            "providers_used": session.get_providers(),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/predict-cpu")
async def predict_cpu(file: UploadFile = File(...)):
    try:
        # Set session to CPU provider only
        session = ort.InferenceSession(
            MODEL_PATH,
            providers=["CPUExecutionProvider"],
        )

        # Leer imagen
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Preprocesado 
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC → CHW
        img = np.expand_dims(img, axis=0)  # batch dimension

        logger.info(f"Input image shape after preprocessing: {img.shape}")

        # Inferencia
        logger.info(f"Running inference with providers: {session.get_providers()}")
        outputs = session.run(
            [output_name],
            {input_name: img},
        )

        prediction = outputs[0]
        logger.info(f"Prediction shape: {prediction.shape}")

        return {
            "success": True,
            "prediction_shape": prediction.shape,
            "providers_used": session.get_providers(),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}
