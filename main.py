from fastapi import FastAPI
from routes.indexroutes import user
from config.database import create_tables
from config.monitoring import init_monitoring
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(
    title="FastAPI User Management",
    description="API pour la gestion des utilisateurs avec monitoring",
    version="2.0.0"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialisation du monitoring
instrumentator = Instrumentator().instrument(app)

# Routes
app.include_router(user)

# Création des tables au démarrage
create_tables()

# Exposer l'endpoint metrics explicitement
@app.on_event("startup")
async def startup():
    instrumentator.expose(app)

