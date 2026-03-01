import logging
import re

logger = logging.getLogger("utils.symbols")

# List of known invalid or placeholder symbols to reject
INVALID_PLACEHOLDERS = {"ALL", "TBD", "NONE", "NULL"}

def is_valid_symbol(symbol: str) -> bool:
    """
    Checks if a symbol is a valid ticker format.
    Alpaca typically uses A-Z, maybe with dots or dashes.
    """
    if not isinstance(symbol, str):
        return False
    
    symbol = symbol.strip().upper()
    
    if not symbol:
        return False
    
    if symbol in INVALID_PLACEHOLDERS:
        return False
    
    # Standard ticker regex: 1-5 letters, optional extension like .A or -B
    # Adjust if your specific exchange/broker has different needs
    if not re.match(r"^[A-Z]{1,5}([.\-][A-Z]{1,2})?$", symbol):
        return False
        
    return True

def sanitize_symbols(symbols: list) -> list:
    """
    Filters a list of symbols, returning only valid ones.
    Logs rejected symbols.
    """
    sanitized = []
    rejected = []
    
    for s in symbols:
        if is_valid_symbol(s):
            sanitized.append(s.strip().upper())
        else:
            rejected.append(s)
            
    if rejected:
        logger.warning(f"event=SYMBOLS_REJECTED count={len(rejected)} symbols={rejected}")
        
    return sorted(list(set(sanitized)))
