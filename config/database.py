from sqlalchemy import create_engine, MetaData, inspect
from sqlalchemy.exc import OperationalError
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL', "mysql+pymysql://root:rootpassword@localhost:3306/test")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
meta = MetaData()

def create_tables():
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()
    
    # Créer uniquement les tables manquantes
    tables_to_create = [table for table in meta.tables.values() if table.name not in existing_tables]
    
    if not tables_to_create:
        print("Toutes les tables existent déjà.")
        return

    try:
        meta.create_all(engine, tables=tables_to_create)
        print(f"Tables créées avec succès : {', '.join(table.name for table in tables_to_create)}")
    except OperationalError as e:
        print(f"Erreur lors de la création des tables : {e}")

# Appel pour créer les tables si elles n'existent pas
create_tables()

conn = engine.connect()
