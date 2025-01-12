from sqlalchemy import create_engine, MetaData
import os
from dotenv import load_dotenv
import time
from sqlalchemy.exc import OperationalError

load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL', 'mysql+pymysql://root:root@mysql_db:3306/fastapi_db')

def get_db_connection(max_retries=5, retry_delay=5):
    for attempt in range(max_retries):
        try:
            engine = create_engine(DATABASE_URL)
            conn = engine.connect()
            print(f"Base de données connectée après {attempt + 1} tentative(s)")
            return engine, conn
        except OperationalError as e:
            if attempt < max_retries - 1:
                print(f"Tentative de connexion {attempt + 1} échouée. Nouvelle tentative dans {retry_delay} secondes...")
                time.sleep(retry_delay)
            else:
                raise Exception(f"Impossible de se connecter à la base de données après {max_retries} tentatives") from e

engine, conn = get_db_connection()
meta = MetaData()

def create_tables():
    try:
        # Supprime toutes les tables existantes
        meta.drop_all(engine)
        # Crée les nouvelles tables
        meta.create_all(engine)
        print("Tables créées avec succès")
    except Exception as e:
        print(f"Erreur lors de la création des tables: {e}")
