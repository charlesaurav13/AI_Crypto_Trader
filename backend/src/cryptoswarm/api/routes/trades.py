from fastapi import APIRouter, Depends
from cryptoswarm.api.deps import get_pg

router = APIRouter(prefix="/trades")


@router.get("/")
async def list_trades(limit: int = 50, pg=Depends(get_pg)):
    rows = await pg._pool.fetch(
        "SELECT * FROM trades ORDER BY opened_ts DESC LIMIT $1", limit
    )
    return [dict(r) for r in rows]
