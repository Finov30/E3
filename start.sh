#!/bin/bash
python generate_sample_csv.py
uvicorn main:app --host 0.0.0.0 --port 8000 --reload 