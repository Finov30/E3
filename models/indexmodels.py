from sqlalchemy import Table, Column, ForeignKey
from models.usermodels import users, addresses 

# Si vous avez besoin d'exporter ces tables
__all__ = ['users', 'addresses']

