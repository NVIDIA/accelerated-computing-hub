c = get_config() #noqa

c.ServerApp.allow_root = True
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.open_browser = False
c.ServerApp.token = ''
c.ServerApp.password = ''
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
