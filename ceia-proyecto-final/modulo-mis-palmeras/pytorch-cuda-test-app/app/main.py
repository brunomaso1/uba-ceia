from fastapi import FastAPI
import torch

app = FastAPI()


def get_gpu_info():
    try:
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpus = []
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpus.append({"id": i, "name": gpu_name})
            return {"gpu_available": True, "gpus": gpus}
        else:
            return {"gpu_available": False, "gpus": []}
    except Exception as e:
        return {"error": str(e)}


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/gpu-test")
def gpu_test():
    try:
        if torch.cuda.is_available():
            # Realizar una operación simple en la GPU para confirmar su funcionamiento
            a = torch.tensor([1.0, 2.0, 3.0]).cuda()
            b = torch.tensor([4.0, 5.0, 6.0]).cuda()
            c = a + b
            return {"gpu_test": "success", "result": c.cpu().tolist()}
        else:
            return {"gpu_test": "failed", "reason": "No GPU available"}
    except Exception as e:
        return {"gpu_test": "failed", "reason": str(e)}


@app.get("/gpu-architecture-list")
def gpu_architecture_list():
    try:
        if torch.cuda.is_available():
            arch_list = torch.cuda.get_arch_list()
            return {"supported_gpu_architectures": arch_list}
        else:
            return {"supported_gpu_architectures": []}
    except Exception as e:
        return {"error": str(e)}


@app.get("/gpu-architecture")
def gpu_architecture():
    try:
        if not torch.cuda.is_available():
            return {"gpu_available": False}

        device = torch.cuda.current_device()
        name = torch.cuda.get_device_name(device)
        major, minor = torch.cuda.get_device_capability(device)

        return {
            "gpu_available": True,
            "gpu_name": name,
            "compute_capability": f"{major}.{minor}",
            "sm": f"sm_{major}{minor}",
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/gpu-compatibility")
def gpu_compatibility():
    try:
        if not torch.cuda.is_available():
            return {"gpu_available": False}

        major, minor = torch.cuda.get_device_capability()
        sm = f"sm_{major}{minor}"
        supported = torch.cuda.get_arch_list()

        return {"gpu_sm": sm, "pytorch_supported_architectures": supported, "is_supported": sm in supported}
    except Exception as e:
        return {"error": str(e)}


@app.get("/gpu-info")
def gpu_info():
    return get_gpu_info()
