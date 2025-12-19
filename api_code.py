import uvicorn

if __name__ == "__main__":
    # This will run your FastAPI app defined in main.py
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

# 127.0.0.1
