from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from vertexai_chuck_norris.chain import chain as vertexai_chuck_norris_chain

app = FastAPI()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


# Edit this to add the chain you want to add
#add_routes(app, NotImplemented)
add_routes(app, vertexai_chuck_norris_chain, path="/vertexai-chuck-norris")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
