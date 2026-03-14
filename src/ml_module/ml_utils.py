import logging

def log_section(title):
    """Log text separator"""
    logging.info(f"{'='*10} {title.upper()} {'='*10}")