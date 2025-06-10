import asyncio
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, HTMLResponse

from src.bot import bot
from src.elastic.elastic_controller import elastic_router
from src.idu_llm.idu_llm_controller import idu_llm_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(bot.infinity_polling(), name="bot-task")
    yield


app = FastAPI(
    lifespan=lifespan,
    root_path="/api/v1"
)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(elastic_router, prefix="")
app.include_router(idu_llm_router, prefix="")

@app.get("/", include_in_schema=False)
async def docs_redirect():
    return RedirectResponse(url="/docs")
