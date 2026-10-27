"""Launch with: python -m streamlit run app.py"""
from pathlib import Path
from afl_opposition.opposition_app import main

main(Path(__file__).resolve().parent)
