from fastapi import FastAPI
import pynvml

app = FastAPI()


def get_gpu_info():
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        gpus = []
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpus.append(
                {
                    "index": i,
                    "name": name,
                    "memory_total_mb": mem_info.total // 1024 // 1024,
                    "memory_used_mb": mem_info.used // 1024 // 1024,
                    "memory_free_mb": mem_info.free // 1024 // 1024,
                }
            )
        return {"gpu_available": True, "gpus": gpus}
    except pynvml.NVMLError as e:
        return {"gpu_available": False, "error": str(e)}


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/gpu")
def gpu_info():
    return get_gpu_info()
