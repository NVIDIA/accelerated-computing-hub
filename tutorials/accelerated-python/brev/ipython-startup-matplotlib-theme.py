"""Apply the Jupyter theme lazily when Matplotlib is first imported."""

import importlib.abc as _importlib_abc
import importlib.machinery as _importlib_machinery
import sys as _sys


def _apply_matplotlib_theme():
    from jupyter_dark_detect import is_dark
    from matplotlib import style

    style.use("dark_background" if is_dark() else "default")


class _MatplotlibThemeLoader(_importlib_abc.Loader):
    def __init__(self, loader, finder):
        self._loader = loader
        self._finder = finder

    def create_module(self, spec):
        if hasattr(self._loader, "create_module"):
            return self._loader.create_module(spec)
        return None

    def exec_module(self, module):
        try:
            self._loader.exec_module(module)
        finally:
            _sys.meta_path.remove(self._finder)
        module.__loader__ = self._loader
        module.__spec__.loader = self._loader
        _apply_matplotlib_theme()


class _MatplotlibThemeFinder(_importlib_abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != "matplotlib":
            return None

        spec = _importlib_machinery.PathFinder.find_spec(fullname, path, target)
        if spec is not None and spec.loader is not None:
            spec.loader = _MatplotlibThemeLoader(spec.loader, self)
        return spec


if "matplotlib" in _sys.modules:
    _apply_matplotlib_theme()
else:
    _sys.meta_path.insert(0, _MatplotlibThemeFinder())
