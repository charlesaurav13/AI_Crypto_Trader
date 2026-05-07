from fastapi import APIRouter, Depends
from cryptoswarm.api.deps import get_engine

router = APIRouter(prefix="/positions")


@router.get("/")
async def list_positions(engine=Depends(get_engine)):
    acc = engine._account
    return {
        "balance": acc.balance,
        "equity": acc.equity,
        "positions": [
            {
                "symbol": p.symbol,
                "side": p.side,
                "qty": p.qty,
                "entry_price": p.entry_price,
                "mark_price": p.mark_price,
                "unrealized_pnl": p.unrealized_pnl,
                "liq_price": p.liq_price,
            }
            for p in acc.open_positions.values()
        ],
    }
