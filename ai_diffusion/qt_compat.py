"""
Zero-dependency Qt5/Qt6 compatibility layer for krita-ai-diffusion.
Enables seamless execution on both Krita 5 (PyQt5) and Krita 6 (PyQt6).
"""
from __future__ import annotations

import sys
import types


def setup_qt_compat():
    """Configures PyQt compatibility so the plugin runs seamlessly on both PyQt5 and PyQt6."""
    try:
        import PyQt6  # noqa: F401
        # Native PyQt6 is available (Krita 6.x)
        return
    except ImportError:
        pass

    # Running under PyQt5 (Krita 5.x)
    from PyQt5 import QtCore, QtGui, QtWidgets, QtNetwork

    try:
        from PyQt5 import sip
    except ImportError:
        import sip  # type: ignore

    # Polyfill QAction in QtGui (in PyQt5, QAction lives in QtWidgets)
    if not hasattr(QtGui, "QAction"):
        QtGui.QAction = QtWidgets.QAction  # type: ignore

    # Polyfill QImage.sizeInBytes (in older PyQt5 builds, only byteCount exists)
    if not hasattr(QtGui.QImage, "sizeInBytes"):
        QtGui.QImage.sizeInBytes = QtGui.QImage.byteCount  # type: ignore

    # Polyfill QNetworkRequest attributes if needed
    if hasattr(QtNetwork, "QNetworkRequest"):
        req = QtNetwork.QNetworkRequest
        if not hasattr(req, "Attribute"):
            req.Attribute = req  # type: ignore
        if not hasattr(req, "RedirectPolicy"):
            req.RedirectPolicy = types.SimpleNamespace(NoLessSafeRedirectPolicy=True)  # type: ignore
            if hasattr(req, "FollowRedirectsAttribute"):
                req.Attribute.RedirectPolicyAttribute = req.FollowRedirectsAttribute  # type: ignore

    # Register virtual PyQt6 package in sys.modules
    qt6_mod = types.ModuleType("PyQt6")
    qt6_mod.QtCore = QtCore  # type: ignore
    qt6_mod.QtGui = QtGui  # type: ignore
    qt6_mod.QtWidgets = QtWidgets  # type: ignore
    qt6_mod.QtNetwork = QtNetwork  # type: ignore
    qt6_mod.sip = sip  # type: ignore

    sys.modules["PyQt6"] = qt6_mod
    sys.modules["PyQt6.QtCore"] = QtCore
    sys.modules["PyQt6.QtGui"] = QtGui
    sys.modules["PyQt6.QtWidgets"] = QtWidgets
    sys.modules["PyQt6.QtNetwork"] = QtNetwork
    sys.modules["PyQt6.sip"] = sip

    # Polyfill Krita DockPosition if needed
    try:
        from krita import DockWidgetFactoryBase

        if not hasattr(DockWidgetFactoryBase, "DockPosition"):
            DockWidgetFactoryBase.DockPosition = DockWidgetFactoryBase  # type: ignore
    except ImportError:
        pass


# Run setup immediately when this module is loaded
setup_qt_compat()
