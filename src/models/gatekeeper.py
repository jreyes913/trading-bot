import math
import logging
from typing import Optional, List
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger("gatekeeper")

class FundamentalSnapshot(BaseModel):
    """Validated snapshot of a company's financial health."""
    ticker: str
    revenue: float = Field(gt=0)
    ebit: float
    total_debt: float = Field(ge=0)
    cash: float = Field(ge=0)
    shares_outstanding: float = Field(gt=0)
    market_cap: float = Field(gt=0)
    ebitda: Optional[float] = None

    @field_validator('revenue', 'shares_outstanding', 'market_cap')
    @classmethod
    def must_be_positive(cls, v: float) -> float:
        if math.isnan(v) or math.isinf(v):
            raise ValueError("Value cannot be NaN or Inf")
        return v

    @model_validator(mode='after')
    def validate_enterprise_value(self) -> 'FundamentalSnapshot':
        """Calculates EV and ensures it is not negative (accounting impossibility)."""
        ev = self.market_cap + self.total_debt - self.cash
        if ev < 0:
            raise ValueError(f"Negative Enterprise Value (EV={ev:.2f}) detected. Data likely corrupt.")
        return self

class FundamentalGatekeeper:
    """Processes raw financial records and filters out corrupt data."""
    def __init__(self):
        self.rejected: List[dict] = []
        self.accepted: List[FundamentalSnapshot] = []

    def process(self, raw_records: List[dict]) -> List[FundamentalSnapshot]:
        self.accepted = []
        for rec in raw_records:
            try:
                snap = FundamentalSnapshot(**rec)
                self.accepted.append(snap)
            except Exception as e:
                ticker = rec.get("ticker", "UNKNOWN")
                logger.error(f"event=TICKER_REJECTED ticker={ticker} reason='{str(e)}'")
                self.rejected.append({"ticker": ticker, "reason": str(e)})
        
        return self.accepted
