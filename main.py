from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.indexroutes import user
from dotenv import load_dotenv
from config.database import create_tables

app = FastAPI(
    title="User Management API",
    description="API pour la gestion des utilisateurs",
    version="1.0.0",
    debug=False  # Désactiver en production
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # À restreindre en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

app.include_router(user)

@app.on_event("startup")
async def startup_event():
    create_tables()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

