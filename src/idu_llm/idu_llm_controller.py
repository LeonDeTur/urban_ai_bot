from typing import NoReturn

from .dto.base_request_dto import BaseLlmRequest

from fastapi import APIRouter, WebSocket

from src.dependencies import idu_llm_client


idu_llm_router = APIRouter()


@idu_llm_router.post("/generate")
async def generate(
    message_info: BaseLlmRequest,
):
    """
    Main function to generate response through bot api

    Args:

        message_info (BaseLlmRequest): Message to send

    Returns:

        response (dict): Generated response
    """

    response = await idu_llm_client.generate_response(message_info)
    return response

@idu_llm_router.websocket("/ws/generate")
async def websocket_llm_endpoint(websocket: WebSocket) -> NoReturn:
    """
    WebSocket endpoint to generate response through bot api

    Args:

        websocket (WebSocket): WebSocket connection

    Returns:

        None
    """

    await websocket.accept()
    try:
        request = await websocket.receive_json()
        message_info = BaseLlmRequest(**request)
        async for text in idu_llm_client.generate_stream_response(message_info):
            if text:
                await websocket.send_text(text)
            else:
                await websocket.close(1000, "Stream ended")
    except Exception as e:
        await websocket.close(code=1011, reason=e.__str__())
