import json
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from cryptoswarm.api.deps import get_bus
from cryptoswarm.bus.messages import Signal

router = APIRouter(prefix="/test")


class SignalRequest(BaseModel):
    symbol: str
    side: str           # "LONG" | "SHORT"
    size_usd: float
    sl: float
    tp: float
    leverage: int = 5
    entry: float = 0.0  # optional explicit fill price


@router.post("/signal")
async def post_signal(req: SignalRequest, bus=Depends(get_bus)):
    reasoning = json.dumps({"entry": req.entry}) if req.entry else ""
    sig = Signal(
        symbol=req.symbol, side=req.side,
        size_usd=req.size_usd, sl=req.sl, tp=req.tp,
        leverage=req.leverage, reasoning=reasoning,
    )
    await bus.publish("signal.execute", sig)
    return {"correlation_id": sig.correlation_id, "published": True}
