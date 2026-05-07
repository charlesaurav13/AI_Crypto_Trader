from fastapi import APIRouter, Depends
from sse_starlette.sse import EventSourceResponse
from cryptoswarm.api.deps import get_bus

router = APIRouter()


@router.get("/events")
async def sse_stream(bus=Depends(get_bus)):
    async def generator():
        async for topic, data in bus.psubscribe("*"):
            yield {"event": topic, "data": data}
    return EventSourceResponse(generator())
