from __future__ import annotations

import logging.config
from pathlib import Path
# import warnings
# warnings.filterwarnings("ignore")


def setup_logging(name, project_dir, log_file_name, config):
    """Setup logging configuration with dynamic log file naming and levels."""

    log_file_path = Path(project_dir, 'log', log_file_name)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    root_level = config['logging']['root_level']
    file_level = config['logging']['file_level']
    console_level = config['logging']['console_level']

    LOGGING_CONFIG = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'detailed': {
                'format': '%(levelname)s - %(pathname)s - %(asctime)s - %(filename)s'
                ' - %(lineno)d - %(module)s - %(name)s - %(funcName)s - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S',
            },
            'simple': {
                'format': '%(levelname)s - %(module)s - %(name)s - %(funcName)s - %(lineno)d - %(message)s',
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': console_level,
                'formatter': 'simple',
                'stream': 'ext://sys.stdout',
            },
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': file_level,
                'formatter': 'detailed',
                'filename': str(log_file_path),
                'maxBytes': 1000000,
                'backupCount': 5,
            },
        },
        'loggers': {
            'matplotlib': {
                'level': 'WARNING',
                'handlers': ['console', 'file'],
                'propagate': False
            },
            'PIL': {
                'level': 'WARNING',
                'handlers': ['console', 'file'],
                'propagate': False
            },
            'ssa': {
                'level': 'WARNING',
                'handlers': ['console', 'file'],
                'propagate': False
            },
            'numba.core.ssa': {
                'level': 'WARNING',
                'handlers': ['console', 'file'],
                'propagate': False
            },
            'numba.core.byteflow': {
                'level': 'WARNING',
                'handlers': ['console', 'file'],
                'propagate': False
            },
            'numba.core.interpreter': {
                'level': 'WARNING',
                'handlers': ['console', 'file'],
                'propagate': False
            },
        },
        'root': {
            'level': root_level,
            'handlers': ['console', 'file'],
        },
    }

    logging.config.dictConfig(LOGGING_CONFIG)
