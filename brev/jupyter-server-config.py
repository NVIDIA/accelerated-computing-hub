import os

c = get_config() #noqa

c.ServerApp.ip = os.environ.get('JUPYTER_IP', '0.0.0.0')
c.ServerApp.certfile = os.environ.get('JUPYTER_HTTPS_CERT', '')
c.ServerApp.keyfile = os.environ.get('JUPYTER_HTTPS_KEY', '')

c.ServerApp.open_browser = False

c.ServerApp.allow_root = True
c.ServerApp.password = ''
c.IdentityProvider.token = ''

c.ServerApp.root_dir = '/'

c.ServerApp.terminado_settings = { 'shell_command': ['/bin/bash'] }

c.Application.log_level = 'INFO'
c.Application.logging_config = {
    "handlers": {
        "file": {
            "class": "logging.FileHandler",
            "level": "INFO",
            "filename": "/accelerated-computing-hub/logs/jupyter-server.log",
            "formatter": "console"
        }
    },
    "loggers": {
        "ServerApp": {
            "level": "INFO",
            "handlers": ["console", "file"]
        }
    }
}

c.ServerProxy.servers = {}
if os.environ.get('SELKIES_ENABLE_HTTPS', 'false').lower() != 'true':
    c.ServerProxy.servers = {
        "nsys": {
            "command": [],
            "port": 8080,
            "launcher_entry": {
                "title": "Nsight Systems",
                "category": "Console", # Must be Console or Notebook to render icons.
                "icon_path": "/accelerated-computing-hub/brev/icons/nsight_systems.svg",
            },
            "new_browser_tab": False,
        },
        "ncu": {
            "command": [],
            "port": 8081,
            "launcher_entry": {
                "title": "Nsight Compute",
                "category": "Console", # Must be Console or Notebook to render icons.
                "icon_path": "/accelerated-computing-hub/brev/icons/nsight_compute.svg",
            },
            "new_browser_tab": False,
        },
    }
