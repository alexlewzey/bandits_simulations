#!/bin/bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
python baysian_bandits_animation.py