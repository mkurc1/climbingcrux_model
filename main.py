from fastapi import FastAPI, UploadFile

app = FastAPI()


@app.post("/boulder/generate")
async def generate_boulder(file: UploadFile):
    contents = await file.read()

    return {"Hello": "World"}
