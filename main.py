from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.indexroutes import user
from config.database import create_tables

app = FastAPI()

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Force la recréation des tables au démarrage
create_tables()

# Inclure les routes
app.include_router(user)

