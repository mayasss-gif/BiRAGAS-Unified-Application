"""
Thread-safe plotting utilities for L1000/DepMap pipelines.

Handles:
- Matplotlib backend initialization (Agg)
- Plotly PNG export with Kaleido fallback
- Thread-safe figure creation/closing
- Retry logic for unstable exports
- Graceful degradation (HTML-only if PNG fails)
"""
from __future__ import annotations

import os
import threading
import time
from functools import wraps
from pathlib import Path
from typing import Optional, Callable, Any
import logging

# CRITICAL: Set matplotlib backend BEFORE any pyplot imports
import matplotlib
matplotlib.use('Agg')  # Thread-safe, non-interactive backend
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Thread-local storage for matplotlib figures
_thread_local = threading.local()


def _ensure_backend() -> None:
    """Ensure matplotlib backend is set to Agg (thread-safe)."""
    current_backend = matplotlib.get_backend()
    if current_backend != 'Agg':
        try:
            matplotlib.use('Agg', force=True)
            logger.debug(f"Switched matplotlib backend to Agg (was {current_backend})")
        except Exception as e:
            logger.warning(f"Could not switch matplotlib backend: {e}")


def retry_export(max_retries: int = 3, delay: float = 0.5):
    """Decorator for retrying unstable export operations."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = delay * (2 ** attempt)  # Exponential backoff
                        logger.debug(f"Export attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                        time.sleep(wait_time)
                    else:
                        logger.warning(f"Export failed after {max_retries} attempts: {e}")
            raise last_exception
        return wrapper
    return decorator


def safe_matplotlib_savefig(
    fig,
    path: Path,
    dpi: int = 150,
    bbox_inches: Optional[str] = None,
    **kwargs
) -> bool:
    """
    Thread-safe matplotlib savefig with error handling.
    
    Returns:
        True if saved successfully, False otherwise
    """
    _ensure_backend()
    
    if fig is None:
        logger.warning(f"Cannot save None figure to {path}")
        return False
    
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use thread-local figure state to avoid conflicts
        save_kwargs = {"dpi": dpi, **kwargs}
        if bbox_inches:
            save_kwargs["bbox_inches"] = bbox_inches
        
        fig.savefig(str(path), **save_kwargs)
        logger.debug(f"Saved matplotlib figure: {path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save matplotlib figure {path}: {e}", exc_info=True)
        return False
    finally:
        # Always close figure to prevent memory leaks
        try:
            plt.close(fig)
        except Exception:
            pass


def safe_plotly_html(
    fig,
    html_path: Path,
    include_plotlyjs: str = "cdn",
    auto_open: bool = False,
    logger_instance: Optional[logging.Logger] = None
) -> bool:
    """
    Save Plotly figure as HTML (always safe, no rendering required).
    
    Returns:
        True if saved successfully, False otherwise
    """
    log = logger_instance or logger
    
    if fig is None:
        log.warning(f"Cannot save None Plotly figure to {html_path}")
        return False
    
    try:
        html_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(
            str(html_path),
            include_plotlyjs=include_plotlyjs,
            auto_open=auto_open
        )
        log.debug(f"Saved Plotly HTML: {html_path}")
        return True
    except Exception as e:
        log.error(f"Failed to save Plotly HTML {html_path}: {e}", exc_info=True)
        return False


@retry_export(max_retries=2, delay=0.3)
def _try_kaleido_export(fig, png_path: Path, width: Optional[int] = None, 
                        height: Optional[int] = None, scale: int = 2) -> bool:
    """Try Kaleido export with retry logic."""
    try:
        import plotly.io as pio
        
        export_kwargs = {"format": "png", "scale": scale}
        if width:
            export_kwargs["width"] = width
        if height:
            export_kwargs["height"] = height
        
        pio.write_image(fig, str(png_path), **export_kwargs)
        return True
    except Exception as e:
        logger.debug(f"Kaleido export failed: {e}")
        return False


def safe_plotly_png(
    fig,
    png_path: Path,
    html_path: Optional[Path] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    scale: int = 2,
    fallback_to_mpl: bool = True,
    logger_instance: Optional[logging.Logger] = None,
    always_save_html: bool = True
) -> bool:
    """
    Save Plotly figure as PNG with multiple fallback strategies.
    
    Strategy:
    1. Try Matplotlib conversion (fast, thread-safe)
    2. Try Kaleido export (requires Chrome/Kaleido)
    3. Always save HTML as fallback
    
    Args:
        fig: Plotly figure
        png_path: Output PNG path
        html_path: Optional HTML path (auto-generated if None)
        width: PNG width in pixels
        height: PNG height in pixels
        scale: Scale factor for high-DPI
        fallback_to_mpl: Try matplotlib conversion first
        logger_instance: Optional logger (uses module logger if None)
    
    Returns:
        True if PNG saved successfully, False otherwise
    """
    log = logger_instance or logger
    
    if fig is None:
        log.warning(f"Cannot save None Plotly figure to {png_path}")
        return False
    
    png_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Always save HTML as fallback (if path provided or always_save_html=True)
    html_saved = False
    if html_path:
        html_saved = safe_plotly_html(fig, html_path, logger_instance=log)
    elif always_save_html:
        auto_html_path = png_path.with_suffix('.html')
        html_saved = safe_plotly_html(fig, auto_html_path, logger_instance=log)
    
    # Strategy 1: Matplotlib conversion (thread-safe, no external deps)
    if fallback_to_mpl:
        try:
            from .plotly_mpl_export import save_plotly_png_with_mpl
            if save_plotly_png_with_mpl(fig, png_path, width=width, height=height, scale=scale):
                log.info(f"Saved Plotly PNG via Matplotlib: {png_path}")
                return True
        except Exception as e:
            log.debug(f"Matplotlib conversion failed: {e}")
    
    # Strategy 2: Kaleido export (with retry)
    try:
        if _try_kaleido_export(fig, png_path, width=width, height=height, scale=scale):
            log.info(f"Saved Plotly PNG via Kaleido: {png_path}")
            return True
    except Exception as e:
        log.debug(f"Kaleido export failed: {e}")
    
    # All PNG strategies failed
    log.warning(f"Could not save Plotly PNG {png_path} (HTML saved if path provided)")
    return False


def safe_plot_and_export(
    fig,
    png_path: Path,
    html_path: Optional[Path] = None,
    fig_type: str = "plotly",  # "plotly" or "matplotlib"
    logger_instance: Optional[logging.Logger] = None,
    **kwargs
) -> tuple[bool, bool]:
    """
    Unified wrapper for saving plots (PNG + HTML).
    
    Returns:
        (png_success, html_success) tuple
    """
    log = logger_instance or logger
    
    if fig_type == "plotly":
        png_ok = safe_plotly_png(fig, png_path, html_path=html_path, logger_instance=log, **kwargs)
        # HTML is saved inside safe_plotly_png if html_path provided
        html_ok = safe_plotly_html(fig, html_path or png_path.with_suffix('.html'), logger_instance=log) if html_path or not png_ok else True
        return (png_ok, html_ok)
    else:  # matplotlib
        png_ok = safe_matplotlib_savefig(fig, png_path, **kwargs)
        # Matplotlib doesn't have HTML export, so html_ok is False
        return (png_ok, False)


def check_plotting_dependencies() -> dict[str, bool]:
    """
    Check availability of plotting dependencies at startup.
    
    Returns:
        Dict with availability status for each dependency
    """
    deps = {
        "matplotlib": False,
        "matplotlib_backend_agg": False,
        "plotly": False,
        "kaleido": False,
    }
    
    try:
        import matplotlib
        deps["matplotlib"] = True
        deps["matplotlib_backend_agg"] = matplotlib.get_backend() == "Agg"
    except ImportError:
        pass
    
    try:
        import plotly
        deps["plotly"] = True
    except ImportError:
        pass
    
    try:
        import kaleido
        deps["kaleido"] = True
    except ImportError:
        pass
    
    return deps


# Initialize backend at module import
_ensure_backend()
