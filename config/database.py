from sqlalchemy import create_engine, MetaData
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL', 'mysql+pymysql://root:root@mysql_db:3306/fastapi_db')

engine = create_engine(DATABASE_URL)
meta = MetaData()
conn = engine.connect()

def create_tables():
    try:
        # Supprime toutes les tables existantes
        meta.drop_all(engine)
        # Crée les nouvelles tables
        meta.create_all(engine)
        print("Tables créées avec succès")
    except Exception as e:
        print(f"Erreur lors de la création des tables: {e}")
