# Development Setup

## Requirements

- Python: `^3.8`
- Poetry: `1.2`

This project uses [Poetry](https://python-poetry.org/docs/) for packaging and dependencies management. Please make sure it is installed on your system.

## Installation

```bash
# Git clone repository
poetry install
poetry shell
# Hack away
flask --app sketch_map_tool/app.py --debug run
```